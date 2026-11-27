"""One assistant object behind every interface (MCP server, HTTP API, chatbot, coach).

All four interfaces call these methods, so safety rules, retrieval settings and the
response shape are defined once. Heavy parts (BM25 index, SapBERT) load on first use.

Configuration (environment):
  CCA_CORPUS      directory of *.jsonl snapshots (default data/snapshots, falling back to data/eval_corpus)
  CCA_BASE_URL    OpenAI-compatible endpoint (default local Ollama; HF router in the cloud)
  CCA_API_KEY     bearer token for that endpoint (not needed behind the cloud proxy)
  CCA_MODEL       answer model (default Qwen/Qwen3.5-9B, the baseline winner on speed and cost);
                  "anthropic:<model>" or "openai:<model>" call those APIs with ANTHROPIC_API_KEY / OPENAI_API_KEY
  CCA_EMBEDDER    overlap embedder (default SapBERT)
  CCA_SESSION_STORE    where conversation state lives: file (default), supabase or memory (store.py)
  CCA_PHRASING_MEMORY  phrasing memory file (default data/phrasing_memory.jsonl); live sessions read it, never write it
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .answer import ABSTAIN_TOKEN, ChatClient, answer
from .index import BM25, load_snapshots
from .safety import DISCLAIMER, screen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
VERSION = "0.2.0"


def _default_corpus() -> Path:
    env = os.environ.get("CCA_CORPUS")
    if env:
        return Path(env)
    snaps = ROOT / "data" / "snapshots"
    if snaps.exists() and any(snaps.glob("*.jsonl")):
        return snaps
    return ROOT / "data" / "eval_corpus"


@dataclass
class Assistant:
    corpus_dir: Path = field(default_factory=_default_corpus)
    model: str = field(default_factory=lambda: os.environ.get("CCA_MODEL", DEFAULT_MODEL))
    client: object | None = None          # anything with .complete(messages) and .model
    overlap_agent: object | None = None   # anything with .analyze(text) -> OverlapResult
    k: int = 8
    min_top_score: float = 4.0
    conversation_options: dict = field(default_factory=dict)  # extra Conversation settings

    def __post_init__(self):
        self._lock = threading.Lock()
        self._store = None    # conversation state store (store.py)
        self._memory = None   # phrasing memory
        self._bm25: BM25 | None = None
        self._docs: list[dict] = []

    # ---------- lazy parts ----------

    def index(self) -> BM25:
        with self._lock:
            if self._bm25 is None:
                self._docs = load_snapshots(Path(self.corpus_dir))
                self._bm25 = BM25(self._docs)
            return self._bm25

    def chat(self):
        if self.client is None:
            # "anthropic:<model>", "openai:<model>" or "hf:<org>/<model>" pick a provider; a bare id uses CCA_BASE_URL.
            if self.model.split(":", 1)[0] in ("anthropic", "openai", "hf"):
                from .roleplay import make_client
                self.client = make_client(self.model)
            else:
                self.client = ChatClient(self.model)
        return self.client

    def overlap_engine(self):
        with self._lock:
            if self.overlap_agent is None:
                from .overlap import OverlapAgent, SapBERTEmbedder, STEmbedder, llm_extract, split_clauses
                name = os.environ.get("CCA_EMBEDDER", "sapbert")
                emb = SapBERTEmbedder() if name == "sapbert" else STEmbedder(name)

                def extract(text: str) -> list[str]:
                    # LLM extraction scored F1 0.58 vs 0.20 for the clause splitter (overlap baseline run)
                    try:
                        return llm_extract(text, self.chat())
                    except Exception:
                        return split_clauses(text)

                self.overlap_agent = OverlapAgent(emb, extractor=extract)
            return self.overlap_agent

    # ---------- capabilities ----------

    def status(self) -> dict:
        topics = json.loads((ROOT / "config" / "topics.json").read_text())
        return {
            "name": "cited-clinical-assistant",
            "version": VERSION,
            "model": self.model,
            "corpus_dir": str(self.corpus_dir),
            "corpus_records": len(self._docs) if self._bm25 else None,
            "topics": list(topics) if isinstance(topics, dict) else topics,
            "disclaimer": DISCLAIMER,
        }

    def search(self, query: str, k: int | None = None) -> dict:
        s = screen(query, topical_ok=True)
        if s.kind == "crisis":
            return {"safety": s.to_dict(), "hits": []}
        hits = self.index().search(query, k=k or self.k)
        return {
            "safety": s.to_dict(),
            "hits": [{
                "rank": i, "score": h.score, "pmid": h.doc["pmid"], "title": h.doc.get("title", ""),
                "journal": h.doc.get("journal", ""), "year": (h.doc.get("pub_date") or "")[:4],
                "url": h.doc.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{h.doc['pmid']}/",
                "abstract": (h.doc.get("abstract") or "")[:1200],
            } for i, h in enumerate(hits, 1)],
        }

    def ask(self, question: str) -> dict:
        """Cited answer from the literature, or a crisis / no-diagnosis / abstain response."""
        t0 = time.monotonic()
        s = screen(question, topical_ok=True)
        if not s.ok:
            return {"question": question, "status": s.kind, "answer": s.message, "safety": s.to_dict(),
                    "citations": [], "checks": {}, "issues": [], "disclaimer": DISCLAIMER}
        ans = answer(question, self.index(), self.chat(), k=self.k, min_top_score=self.min_top_score)
        cited = set(ans.cited_pmids())
        citations = [{**src, "url": src.get("url") or f"https://pubmed.ncbi.nlm.nih.gov/{src['pmid']}/"}
                     for src in ans.sources if src["pmid"] in cited]
        text = ans.text
        if ans.abstained:
            text = "Not enough evidence in the indexed PubMed records to answer this. " + \
                   text.replace(f"{ABSTAIN_TOKEN}:", "").strip()
        return {
            "question": question,
            "status": "abstained" if ans.abstained else "answered",
            "answer": text,
            "citations": citations,
            "checks": ans.metrics(),
            "issues": ans.issues,
            "model": ans.model,
            "latency_s": round(time.monotonic() - t0, 2),
            "safety": s.to_dict(),
            "disclaimer": DISCLAIMER,
        }

    def overlap(self, text: str) -> dict:
        """Symptom-overlap check: which conditions the description fits about equally, and what tells them apart."""
        s = screen(text)
        if s.kind == "crisis":
            return {"status": "crisis", "message": s.message, "safety": s.to_dict(), "disclaimer": DISCLAIMER}
        res = self.overlap_engine().analyze(text).to_dict()
        res.pop("text", None)
        res["status"] = "alert" if res.get("alert") else "no_alert"
        res["safety"] = s.to_dict()
        res["disclaimer"] = DISCLAIMER
        return res

    def brief(self, question: str, case_text: str | None = None, peer_inputs: list[dict] | None = None) -> dict:
        """Decision-support brief: literature answer + overlap flags + inputs from other agents.
        It lays out evidence and open questions; the decision stays with the clinician."""
        s = screen(f"{question}\n{case_text or ''}")
        if s.kind == "crisis":
            return {"status": "crisis", "answer": s.message, "safety": s.to_dict(), "disclaimer": DISCLAIMER}
        out = {"status": "ok", "question": question, "evidence": self.ask(question)}
        if case_text:
            out["overlap"] = self.overlap(case_text)
        out["peer_inputs"] = peer_inputs or []
        open_q = []
        if out["evidence"]["status"] == "abstained":
            open_q.append("The indexed literature does not answer the question; look for newer or broader evidence.")
        ov = out.get("overlap") or {}
        open_q += ov.get("questions", [])
        out["open_questions"] = open_q
        out["note"] = "Evidence summary for a clinician. Not a diagnosis or treatment recommendation."
        out["disclaimer"] = DISCLAIMER
        return out

    # ---------- conversational search ----------

    def _conversation_kit(self):
        from .tracking import PhrasingMemory
        with self._lock:
            if self._store is None:
                from .store import make_store
                self._store = make_store()
            if self._memory is None:
                path = Path(os.environ.get("CCA_PHRASING_MEMORY", ROOT / "data" / "phrasing_memory.jsonl"))
                self._memory = PhrasingMemory(path)
        return self._store, self._memory

    def converse(self, message: str, session_id: str | None = None) -> dict:
        """One turn of conversational search (interpret -> clarify -> confirm -> cited answer).
        State is saved after every turn (store.py), so any interface or process can continue the
        conversation by passing back the returned session_id."""
        from .converse import Conversation
        from .tracking import Session
        index, chat = self.index(), self.chat()   # take these before the (non-reentrant) lock
        store, memory = self._conversation_kit()
        state = store.load(session_id) if session_id else None
        if state is not None and state["session"].get("ended"):
            state = None  # an ended session starts over with a new id
        opts = {"min_top_score": self.min_top_score, **self.conversation_options}
        conv = (Conversation.from_state(state, chat, index, memory=memory, **opts) if state is not None
                else Conversation(chat, index, memory=memory, session=Session(source="live"), **opts))
        out = conv.respond(message)
        store.save(conv.session.session_id, conv.to_state())
        out["disclaimer"] = DISCLAIMER
        return out

    def get_conversation(self, session_id: str) -> dict:
        """Transcript and state of a stored conversation."""
        store, _ = self._conversation_kit()
        st = store.load(session_id)
        if st is None:
            return {"status": "unknown_session", "session_id": session_id}
        sess = st["session"]
        return {"status": "ok", "session_id": session_id, "state": st["state"], "ended": sess.get("ended", False),
                "transcript": [{"role": e["role"], "content": e["payload"].get("text", "")}
                               for e in sess["events"] if e["kind"] in ("message", "reply")],
                "readings": st["readings"]}

    def end_conversation(self, session_id: str) -> dict:
        """Score the session's readings from its signals. Live sessions do not teach the phrasing memory."""
        from .converse import Conversation
        store, memory = self._conversation_kit()
        st = store.load(session_id)
        if st is None:
            return {"status": "unknown_session", "session_id": session_id}
        conv = Conversation.from_state(st, self.chat(), self.index(), memory=memory, **self.conversation_options)
        pairs = conv.end()
        store.save(session_id, conv.to_state())
        return {"status": "ended", "session_id": session_id, "scores": [p.to_dict() for p in pairs],
                "query_changes": conv.session.query_changes}


_default: Assistant | None = None


def get_assistant() -> Assistant:
    global _default
    if _default is None:
        _default = Assistant()
    return _default

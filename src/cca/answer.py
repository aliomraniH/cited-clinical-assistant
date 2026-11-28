"""Answer a question from retrieved passages, with a citation on every sentence.

The model is reached through any OpenAI-compatible chat endpoint (Ollama, vLLM,
llama.cpp server, Hugging Face Inference Providers), so candidate models can be
swapped without code changes. Citations are checked by code, not by a model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import requests

from .index import Hit

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"
DISCLAIMER = "Research demo, not medical advice."

SYSTEM_PROMPT = f"""You answer clinicians' questions about mental health using ONLY the numbered passages provided.
Rules:
1. Every sentence must end with one or more citations in square brackets, like [2] or [1][3], naming the passages that support it.
2. Use no outside knowledge. If a claim is not in the passages, leave it out.
3. If the passages do not contain enough evidence to answer, reply with exactly one line: {ABSTAIN_TOKEN}: <one-sentence reason>
4. Be concise: at most 6 sentences. Mention study type or sample size when the passage gives it.
5. Do not give individual treatment advice."""


@dataclass
class Sentence:
    text: str
    citations: list[int]


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    sentences: list[Sentence] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # passage number -> source, in prompt order
    issues: list[str] = field(default_factory=list)
    model: str | None = None

    def cited_pmids(self) -> list[str]:
        nums = sorted({c for s in self.sentences for c in s.citations})
        return [self.sources[n - 1]["pmid"] for n in nums if 0 < n <= len(self.sources)]

    def metrics(self) -> dict:
        """Deterministic citation checks. Whether a cited passage supports its sentence needs a judge (eval-run)."""
        n = len(self.sentences)
        cited = sum(bool(s.citations) for s in self.sentences)
        invalid = sum(1 for s in self.sentences for c in s.citations if not 0 < c <= len(self.sources))
        return {
            "abstained": self.abstained,
            "sentences": n,
            "cited_sentence_rate": round(cited / n, 3) if n else None,
            "invalid_citations": invalid,
            "distinct_sources_cited": len(set(self.cited_pmids())),
        }


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        d = h.doc
        year = (d.get("pub_date") or "")[:4]
        label = d.get("cite_label") or f"PMID {d['pmid']}"
        blocks.append(f"[{i}] {label} | {d.get('journal', '')} {year}\nTitle: {d.get('title', '')}\n{d.get('abstract', '')}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    user = f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_THINK = re.compile(r"<think>.*?</think>", re.S)
_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z(])")


def parse_answer(question: str, raw: str, hits: list[Hit], model: str | None = None) -> Answer:
    text = _THINK.sub("", raw).strip()
    sources = [{"n": i, "pmid": h.doc["pmid"], "title": h.doc.get("title", ""), "url": h.doc.get("url")}
               for i, h in enumerate(hits, start=1)]
    if text.startswith(ABSTAIN_TOKEN):
        return Answer(question, text, True, sources=sources, model=model)
    sentences, issues = [], []
    for chunk in _SENT_SPLIT.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        nums = [int(n) for group in _CITE.findall(chunk) for n in re.split(r"\s*[,;]\s*", group)]
        sentences.append(Sentence(chunk, nums))
        if not nums:
            issues.append(f"uncited sentence: {chunk[:80]}")
        bad = [n for n in nums if not 0 < n <= len(sources)]
        if bad:
            issues.append(f"citation to missing passage {bad}: {chunk[:80]}")
    return Answer(question, text, False, sentences, sources, issues, model)


class ChatClient:
    """Minimal OpenAI-compatible client. Defaults point at a local Ollama server."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 session: requests.Session | None = None, timeout: float = 180):
        self.model = model
        self.base_url = (base_url or os.environ.get("CCA_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("CCA_API_KEY")
        self.session = session or requests.Session()
        self.timeout = timeout

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 700) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""


def answer(question: str, retriever, client: ChatClient, k: int = 8, min_top_score: float = 4.0) -> Answer:
    """Retrieve, gate, generate, check. Abstains without calling the model when retrieval is too weak."""
    hits = retriever.search(question, k=k)
    if not hits or hits[0].score < min_top_score:
        reason = "no passages matched" if not hits else f"top retrieval score {hits[0].score} below {min_top_score}"
        return Answer(question, f"{ABSTAIN_TOKEN}: {reason}", True,
                      sources=[{"n": i, "pmid": h.doc["pmid"]} for i, h in enumerate(hits, 1)], model=client.model)
    raw = client.complete(build_messages(question, hits))
    return parse_answer(question, raw, hits, model=client.model)

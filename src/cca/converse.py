"""Conversational multi-intent search: interpret -> search and clarify -> confirm -> cited answer.

One Conversation object holds one session. Call `respond(text)` for each user message and
`end()` when the session finishes; `end()` scores every reading from the logged signals and
adds the scored phrasings to the phrasing memory (feedback loop, see tracking.py).

States:
  start     waiting for a request
  clarify   asked one multiple-choice question because two readings of the SAME words would
            pull different evidence (true ambiguity). Separate asks in one message are not
            ambiguity; they go straight to the confirm step.
  confirm   said back what it understood; the user can accept, drop items or correct
  answered  answered and asked "Was this helpful?"; a follow-up counts as approval
  crisis    safety route taken; nothing else runs
"""

from __future__ import annotations

import re

from .answer import ABSTAIN_TOKEN, build_messages as answer_messages, parse_answer
from .interpret import Interpretation, interpret, jaccard, _words
from .safety import DISCLAIMER, screen
from .tracking import PhrasingMemory, Session, _bag, _cos, score_session

YES = re.compile(r"^\s*(yes|yeah|yep|yup|correct|right|exactly|sure|ok|okay|that'?s (right|it|correct)|perfect|sounds good)\b", re.I)
NO_START = re.compile(r"^\s*(no|nope|not (quite|really|exactly)|wrong|actually|i meant|i mean)\b", re.I)
DROP = re.compile(r"\b(drop|skip|remove|not|forget|ignore)\s+(?:number\s+|#)?(\d)\b", re.I)
ONLY = re.compile(r"\b(only|just)\s+(?:number\s+|#)?(\d)\b", re.I)
BOTH = re.compile(r"\b(both|all( of them)?|each)\b", re.I)
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "one": 1, "two": 2, "three": 3, "four": 4}
FEEDBACK = [
    ("feedback_no", re.compile(r"^\s*(no|not (really|helpful|at all)|unhelpful|that didn'?t help|not what i (asked|meant))\b", re.I)),
    ("feedback_partly", re.compile(r"^\s*(partly|partially|somewhat|kind of|kinda|sort of|a bit|a little)\b", re.I)),
    ("feedback_yes", re.compile(r"^\s*(yes|yeah|yep|very|helpful|it (was|is) helpful|thanks|thank you|great|perfect|that helps|useful)\b", re.I)),
]
CONTINUE = re.compile(r"^\s*(and|also|what about|how about|tell me more|more on|can you (expand|say more)|ok(ay)?,? (and|so|what|how)|so )", re.I)


class Conversation:
    def __init__(self, client, retriever, memory: PhrasingMemory | None = None, guidelines=None,
                 session: Session | None = None, samples: int = 3, k: int = 8, min_conf: float = 0.2,
                 clarify_gap: float = 0.25, evidence_overlap: float = 0.34, min_top_score: float = 4.0,
                 max_corrections: int = 2, learn_from: tuple = ("roleplay", "test"), compare_k: int = 3,
                 same_evidence: float = 0.6):
        self.client, self.retriever, self.memory, self.guidelines = client, retriever, memory, guidelines
        self.session = session or Session()
        self.samples, self.k, self.min_conf = samples, k, min_conf
        self.clarify_gap, self.evidence_overlap, self.min_top_score = clarify_gap, evidence_overlap, min_top_score
        self.max_corrections, self.learn_from = max_corrections, learn_from
        self.same_evidence = same_evidence  # top-5 PMID overlap at which two readings count as one
        self.compare_k = compare_k  # evidence differs = low overlap among each reading's top compare_k PMIDs
        self.state = "start"
        self.readings: list[Interpretation] = []
        self.request_text = ""
        self.answered_ids: list[str] = []
        self.corrections = 0
        self._next_id = 1
        self._last_request = ""

    # ---------- public ----------

    def respond(self, text: str) -> dict:
        self.session.log("user", "message", text=text)
        s = screen(text, topical_ok=True)
        if s.kind == "crisis":
            self.state = "crisis"
            self.session.log("system", "safety", screen=s.to_dict())
            return self._reply(s.message, safety=s.to_dict())
        if self.state == "clarify":
            return self._on_clarify(text)
        if self.state == "confirm":
            return self._on_confirm(text)
        if self.state == "answered":
            return self._on_answered(text)
        return self._new_request(text, note=None if s.ok else s.message)

    def end(self, judge=None) -> list:
        pairs = score_session(self.session, judge=judge)
        if self.memory is not None and self.session.source in self.learn_from:
            self.memory.add(pairs)
        return pairs

    # ---------- persistence ----------

    def to_state(self) -> dict:
        """Everything needed to resume this conversation in another process (see store.py)."""
        def r(x: Interpretation) -> dict:
            return {**x.to_dict(), "_pmids": getattr(x, "_pmids", [])}
        return {"version": 1, "state": self.state, "readings": [r(x) for x in self.readings],
                "request_text": self.request_text, "answered_ids": self.answered_ids,
                "corrections": self.corrections, "next_id": self._next_id, "last_request": self._last_request,
                "clarify_options": [r(x) for x in getattr(self, "_clarify_options", [])],
                "session": self.session.to_dict()}

    @classmethod
    def from_state(cls, st: dict, client, retriever, **kw) -> "Conversation":
        def r(d: dict) -> Interpretation:
            d = dict(d)
            pm = d.pop("_pmids", [])
            it = Interpretation(**d)
            it._pmids = pm  # type: ignore[attr-defined]
            return it
        c = cls(client, retriever, session=Session.from_dict(st["session"]), **kw)
        c.state, c.readings = st["state"], [r(x) for x in st["readings"]]
        c.request_text, c.answered_ids = st["request_text"], st["answered_ids"]
        c.corrections, c._next_id, c._last_request = st["corrections"], st["next_id"], st["last_request"]
        if st.get("clarify_options"):
            ids = [x["id"] for x in st["clarify_options"]]
            c._clarify_options = [next((y for y in c.readings if y.id == i), None) or r(d)
                                  for i, d in zip(ids, st["clarify_options"])]
        return c

    # ---------- steps ----------

    def _new_request(self, text: str, note: str | None = None) -> dict:
        self.request_text, self.corrections = text, 0
        examples = self.memory.examples(text) if self.memory else []
        history = self.session.transcript()[:-1]
        readings, stats = interpret(text, self.client, history=history, examples=examples, samples=self.samples)
        self.session.log("system", "interpretations", stats=stats, examples=[e["phrasing"] for e in examples],
                         readings=[r.to_dict() for r in readings])
        if not readings:
            self.state = "start"
            return self._reply("I couldn't work out what you'd like to know. Could you say it another way?")
        for r in readings:
            r.id = ""
        self._set_readings([r for r in readings if r.confidence >= self.min_conf] or readings[:1], text)
        self._last_request = text
        pair = self._ambiguous_pair()
        if pair:
            return self._ask_clarify(pair, note)
        return self._ask_confirm(note)

    def _set_readings(self, readings: list[Interpretation], phrasing: str) -> None:
        """Readings with an empty id are new and get a fresh one; carried ids (after a correction) stay."""
        for r in readings:
            if not r.id:
                r.id = self._new_id()
                self.session.register(r, phrasing=" ".join(r.key_sentences) or phrasing)
        for r in readings:
            r_hits = self.retriever.search(r.query(), k=self.k)
            r._pmids = [h.doc["pmid"] for h in r_hits]  # type: ignore[attr-defined]
        # Readings that pull the same evidence are one question worded twice: keep the stronger one.
        kept: list[Interpretation] = []
        for r in sorted(readings, key=lambda x: -x.confidence):
            twin = next((k for k in kept if r._pmids and jaccard(set(r._pmids[:5]), set(k._pmids[:5])) >= self.same_evidence), None)
            if twin is None:
                kept.append(r)
            else:
                self.session.log("system", "merged", kept=twin.id, dropped=r.id, reason="same evidence")
        self.readings = [r for r in readings if r in kept]
        self.session.log("system", "queries", queries={r.id: r.query() for r in readings},
                         pmids={r.id: getattr(r, "_pmids", []) for r in readings})

    def _new_id(self) -> str:
        while f"i{self._next_id}" in self.session.readings:
            self._next_id += 1
        iid = f"i{self._next_id}"
        self._next_id += 1
        return iid

    def _ambiguous_pair(self):
        """Two readings built on the same words, close in confidence, with different evidence."""
        rs = self.readings
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                x, y = rs[a], rs[b]
                # either/or: the model marked one as another reading of the same words, and the spans agree
                same_words = (x.alternative or y.alternative) and (
                    jaccard(_words(" ".join(x.key_sentences)), _words(" ".join(y.key_sentences))) >= 0.5
                    if x.key_sentences and y.key_sentences else True)
                close = abs(x.confidence - y.confidence) <= self.clarify_gap
                differ = jaccard(set(getattr(x, "_pmids", [])[:self.compare_k]),
                                 set(getattr(y, "_pmids", [])[:self.compare_k])) < self.evidence_overlap
                if same_words and close and differ:
                    return x, y
        return None

    def _ask_clarify(self, pair, note=None) -> dict:
        x, y = pair
        self._clarify_options = [x, y]
        q = (f'When you said "{x.key_sentences[0]}", which did you mean?\n'
             f"1. {x.paraphrase}\n2. {y.paraphrase}\n3. Both")
        self.state = "clarify"
        self.session.log("system", "clarify", options=[x.id, y.id], text=q)
        return self._reply(q if not note else f"{note}\n\n{q}")

    def _on_clarify(self, text: str) -> dict:
        x, y = self._clarify_options
        pick = _pick(text, 3)
        if BOTH.search(text) or pick == {3}:
            self.session.signal("clarify_picked", [x.id, y.id])
        elif pick & {1, 2}:
            chosen = x if 1 in pick else y
            other = y if chosen is x else x
            self.session.signal("clarify_picked", [chosen.id])
            self.session.signal("clarify_not_picked", [other.id])
            self.readings = [r for r in self.readings if r.id != other.id]
        else:
            return self._correct(text)
        return self._ask_confirm()

    def _ask_confirm(self, note: str | None = None) -> dict:
        lines = [f"{i}. {r.paraphrase}" for i, r in enumerate(self.readings, 1)]
        intro = "Here's what I understood you're asking about:" if len(lines) > 1 else "Here's what I understood:"
        msg = (f"{intro}\n" + "\n".join(lines) +
               "\n\nIs that right? Say yes, or tell me what to drop or fix (for example \"drop 2\" or \"no, I meant ...\").")
        if note:
            msg = f"{note}\n\n{msg}"
        self.state = "confirm"
        self.session.log("system", "confirm", readings=[r.id for r in self.readings], text=msg)
        return self._reply(msg)

    def _on_confirm(self, text: str) -> dict:
        n = len(self.readings)
        only = {int(m.group(2)) for m in ONLY.finditer(text) if 0 < int(m.group(2)) <= n}
        drops = {int(m.group(2)) for m in DROP.finditer(text) if 0 < int(m.group(2)) <= n}
        if only:
            drops |= set(range(1, n + 1)) - only
        if drops:
            dropped = [self.readings[i - 1] for i in sorted(drops)]
            self.session.signal("dropped", [r.id for r in dropped])
            self.readings = [r for i, r in enumerate(self.readings, 1) if i not in drops]
            rest = _strip_commands(text)
            if rest and not YES.match(rest) and len(rest.split()) > 3 and self.corrections < self.max_corrections:
                return self._correct(rest)
            if not self.readings:
                self.state = "start"
                return self._reply("Got it. What would you like to know instead?")
            self.session.signal("confirm_accepted", [r.id for r in self.readings])
            return self._answer()
        if YES.match(text) and not re.search(r"\bbut\b", text, re.I):
            extra = YES.sub("", text, count=1).strip(" ,.;:-—–!")
            if len(extra.split()) > 4:
                # "Yes, especially X": approval plus a refinement. Fold it in, then answer without asking again.
                self._reinterpret(extra, refinement=True)
            self.session.signal("confirm_accepted", [r.id for r in self.readings])
            return self._answer()
        if self.corrections >= self.max_corrections:
            self.session.signal("confirm_accepted", [r.id for r in self.readings], note="max corrections reached")
            return self._answer()
        return self._correct(text)

    def _correct(self, text: str) -> dict:
        """Re-interpret with the user's correction, then confirm again."""
        self._reinterpret(text)
        return self._ask_confirm()

    def _reinterpret(self, text: str, refinement: bool = False) -> None:
        """Readings that do not survive a correction count as corrected. A refinement ("yes, especially X")
        was already approved, so nothing is penalised; the change is still tracked."""
        self.corrections += 1
        old = list(self.readings)
        self.session.log("user", "refinement" if refinement else "correction", text=text, previous=[r.id for r in old])
        history = self.session.transcript()[:-1]
        new, stats = interpret(self.request_text, self.client, history=history, samples=self.samples,
                               correction=text, previous=old, refinement=refinement)
        self.session.log("system", "interpretations", stats=stats, readings=[r.to_dict() for r in new],
                         after_correction=True)
        if not new:
            return
        for r in new:
            r.id = ""
        for r in new:
            r._pmids = [h.doc["pmid"] for h in self.retriever.search(r.query(), k=self.k)]  # type: ignore[attr-defined]
        corrected = []
        for o in old:
            free = [r for r in new if not r.id]
            match = max(free, key=lambda r: _likeness(r, o), default=None)
            if match is not None and _likeness(match, o) >= 0.6:
                match.id = o.id        # same reading survived the correction (possibly reworded)
                if match.paraphrase != o.paraphrase:
                    reg = self.session.readings.get(o.id)
                    if reg is not None:
                        reg.update(paraphrase=match.paraphrase, terms=match.terms, intent_type=match.intent_type)
                    self.session.record_query_change({"id": o.id, "paraphrase": o.paraphrase, "query": o.query()},
                                                     {"id": o.id, "paraphrase": match.paraphrase, "query": match.query()},
                                                     why="reworded: " + text)
            else:
                corrected.append(o)
        new = [r for r in new if r.confidence >= self.min_conf or r.id] or new[:1]
        self._set_readings(new, self.request_text + " " + text)
        if corrected:
            if not refinement:
                self.session.signal("corrected", [o.id for o in corrected])
            for o in corrected:
                after = max(self.readings, key=lambda r: jaccard(_words(r.query()), _words(o.query())))
                self.session.record_query_change({"id": o.id, "paraphrase": o.paraphrase, "query": o.query()},
                                                 {"id": after.id, "paraphrase": after.paraphrase, "query": after.query()},
                                                 why=("refinement: " if refinement else "") + text)

    def _answer(self) -> dict:
        parts, payload = [], []
        for i, r in enumerate(self.readings, 1):
            hits = self.retriever.search(r.query(), k=self.k)
            head = f"**{i}. {r.paraphrase}**" if len(self.readings) > 1 else f"**{r.paraphrase}**"
            if not hits or hits[0].score < self.min_top_score:
                parts.append(f"{head}\nI didn't find enough evidence in the indexed PubMed records for this one.")
                payload.append({"id": r.id, "status": "abstained", "citations": []})
                continue
            raw = self.client.complete(answer_messages(r.paraphrase, hits), max_tokens=4096)
            if not re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip():
                raw = self.client.complete(answer_messages(r.paraphrase, hits), max_tokens=8192)  # reasoning ran out
            ans = parse_answer(r.paraphrase, raw, hits, model=getattr(self.client, "model", None))
            if not ans.abstained and not ans.sentences:
                ans.abstained = True
                ans.text = f"{ABSTAIN_TOKEN}: the model returned no answer text"
            cited = set(ans.cited_pmids())
            cites = [s for s in ans.sources if s["pmid"] in cited]
            body = ans.text if not ans.abstained else \
                "I didn't find enough evidence in the indexed PubMed records for this one."
            # Renumber [n] citations to PMIDs so a multi-part answer stays unambiguous.
            body = re.sub(r"\[(\d+)\]", lambda m: f"[PMID {ans.sources[int(m.group(1)) - 1]['pmid']}]"
                          if 0 < int(m.group(1)) <= len(ans.sources) else m.group(0), body)
            parts.append(f"{head}\n{body.replace(ABSTAIN_TOKEN + ':', '').strip()}")
            g = []
            if self.guidelines is not None:
                g = [{"id": h.doc.get("id") or h.doc.get("pmid"), "title": h.doc.get("title", ""), "url": h.doc.get("url")}
                     for h in self.guidelines.search(r.query(), k=2)]
                if g:
                    parts.append("Guidelines: " + "; ".join(f"{x['title']} ({x['id']})" for x in g))
            payload.append({"id": r.id, "status": "abstained" if ans.abstained else "answered",
                            "citations": [c["pmid"] for c in cites], "guidelines": [x["id"] for x in g],
                            "checks": ans.metrics(), "issues": ans.issues})
        self.answered_ids = [r.id for r in self.readings]
        self.state = "answered"
        text = "\n\n".join(parts) + f"\n\n_{DISCLAIMER}_\n\nWas this helpful? (yes / partly / no)"
        self.session.log("system", "answer", parts=payload, text=text)
        return self._reply(text, answer=payload)

    def _on_answered(self, text: str) -> dict:
        rest = text
        for name, pat in FEEDBACK:
            m = pat.match(text)
            if m:
                self.session.signal(name, self.answered_ids)
                self.session.log("user", "feedback", value=name)
                rest = text[m.end():].strip(" ,.!-")
                break
        if len(rest.split()) < 3:
            self.state = "start"
            return self._reply("Thanks, that helps me get better. Ask anything else whenever you like.")
        # Rephrasing the same request means the last readings missed.
        if _cos(_bag(rest), _bag(self._last_request)) > 0.6:
            self.session.signal("rephrased", self.answered_ids)
        else:
            # A follow-up that builds on the answer counts as approval of the readings it builds on.
            related = [r.id for r in self.readings if _cos(_bag(rest), _bag(r.query())) > 0.15]
            if not related and CONTINUE.match(rest):
                related = list(self.answered_ids)
            if related:
                self.session.signal("followup", related)
        self.state = "start"
        return self._new_request(rest)

    def _reply(self, text: str, **extra) -> dict:
        self.session.log("system", "reply", text=text, state=self.state)
        return {"session_id": self.session.session_id, "state": self.state, "reply": text,
                "readings": [r.to_dict() for r in self.readings], **extra}


_STOP = frozenset("a an and are as at be by can could do does for from has have how i if in is it its may me my of on "
                  "or should the their them they this to was what when whether which who why with you your about more "
                  "help helps".split())


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if w not in _STOP and len(w) > 2}


def _likeness(a: Interpretation, b: Interpretation) -> float:
    """How likely two readings are the same question reworded: the best of query-word Jaccard (scaled),
    content-word overlap of the paraphrases, and overlap of their top-5 evidence."""
    words = jaccard(_words(a.query()), _words(b.query())) / 0.6
    ca, cb = _content(a.paraphrase), _content(b.paraphrase)
    content = len(ca & cb) / min(len(ca), len(cb)) if ca and cb else 0.0
    pa, pb = getattr(a, "_pmids", [])[:5], getattr(b, "_pmids", [])[:5]
    evidence = jaccard(set(pa), set(pb)) if pa and pb else 0.0
    return min(1.0, max(words * 0.5, content * 5 / 6, evidence))


def _pick(text: str, n: int) -> set[int]:
    nums = {int(d) for d in re.findall(r"\b(\d)\b", text) if 0 < int(d) <= n}
    nums |= {v for w, v in ORDINALS.items() if re.search(rf"\b{w}\b", text, re.I) and v <= n}
    return nums


def _strip_commands(text: str) -> str:
    text = DROP.sub("", text)
    text = ONLY.sub("", text)
    return re.sub(r"^[\s,.;:-]*(yes|ok|okay|and|but)?[\s,.;:-]*", "", text, flags=re.I).strip()

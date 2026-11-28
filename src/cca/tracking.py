"""Session tracking and the feedback loop.

Every turn of a conversation is logged as an event: the message, the readings, the queries
they produced, clarifying questions, the confirm step, corrections, the answer and feedback.
When the session ends, `score_session` gives each reading a value from -1 to 1 from the
signals in the log (rules first; an optional judge only fills readings with no signal).

Scored (phrasing -> reading) pairs go into the PhrasingMemory. The interpreter retrieves the
closest past phrasings as examples, so it learns how people word similar things.

Signals and their weights:
  explicit feedback "yes"            +0.6 to every answered reading ("partly" +0.2, "no" -0.6);
                                     x0.25 for readings the user already confirmed (it rates the answer, not
                                     the understanding) and stored separately as answer_feedback
  confirm accepted unchanged         +0.5
  picked in a clarifying question    +0.3   (offered but not picked: -0.3)
  follow-up that builds on an answer +0.3
  reading dropped by the user        -0.7
  reading corrected by the user      -1.0   (the corrected version starts fresh and earns its own)
  user rephrased the same request    -0.4 to the readings shown just before
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

WEIGHTS = {
    "feedback_yes": 0.6, "feedback_partly": 0.2, "feedback_no": -0.6,
    "confirm_accepted": 0.5, "clarify_picked": 0.3, "clarify_not_picked": -0.3,
    "followup": 0.3, "dropped": -0.7, "corrected": -1.0, "rephrased": -0.4,
}
FEEDBACK_ON_CONFIRMED = 0.25
MEMORY_VERSION = "pm-1"


# ---------- de-identification (applied before anything is stored) ----------

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[phone]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[date]"),
    (re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?\b", re.I), "[date]"),
    (re.compile(r"\b((?i:my name is|i am|i'm|this is|named|called))\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"), r"\1 [name]"),
    (re.compile(r"\b(Mr|Mrs|Ms|Dr|Miss)\.?\s+[A-Z][a-z]+"), r"\1 [name]"),
    (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[zip]"),
]


def deidentify(text: str) -> str:
    """Rule-based scrub of obvious identifiers. Not a full de-identification system:
    the phrasing memory learns only from role-play and test sessions until clinician review."""
    for pat, rep in _PATTERNS:
        text = pat.sub(rep, text)
    return text


# ---------- session log ----------

@dataclass
class Event:
    turn: int
    role: str          # user | system
    kind: str          # message, interpretations, queries, clarify, confirm, correction, answer, feedback, signal, end
    payload: dict
    ts: float = field(default_factory=time.time)


@dataclass
class ScoredPair:
    session_id: str
    phrasing: str
    paraphrase: str
    intent_type: str
    terms: list[str]
    value: float
    reasons: list[str]
    interp_id: str
    source: str
    version: str = MEMORY_VERSION
    answer_feedback: str | None = None   # what the user said about the answer, kept apart from the reading's value

    def to_dict(self) -> dict:
        return asdict(self)


class Session:
    def __init__(self, source: str = "test", session_id: str | None = None, meta: dict | None = None):
        self.session_id = session_id or f"s-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.source = source            # roleplay | test | live
        self.meta = meta or {}
        self.events: list[Event] = []
        self.turn = 0
        self.readings: dict[str, dict] = {}   # interp id -> reading + the phrasing it came from
        self.query_changes: list[dict] = []
        self.ended = False

    def log(self, role: str, kind: str, **payload) -> Event:
        if role == "user" and kind == "message":
            self.turn += 1
        ev = Event(self.turn, role, kind, payload)
        self.events.append(ev)
        return ev

    def signal(self, name: str, interp_ids: list[str], note: str = "") -> None:
        self.log("system", "signal", name=name, interp_ids=list(interp_ids), note=note)

    def register(self, interp, phrasing: str) -> None:
        d = interp.to_dict() if hasattr(interp, "to_dict") else dict(interp)
        d["phrasing"] = deidentify(phrasing)
        self.readings[d["id"]] = d

    def record_query_change(self, before: dict, after: dict, why: str) -> None:
        self.query_changes.append({"turn": self.turn, "before": before, "after": after, "why": why})

    def transcript(self) -> list[dict]:
        return [{"role": e.role, "content": e.payload.get("text", "")}
                for e in self.events if e.kind in ("message", "reply") and e.payload.get("text")]

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "source": self.source, "meta": self.meta,
                "events": [asdict(e) for e in self.events], "readings": self.readings,
                "query_changes": self.query_changes, "turn": self.turn, "ended": self.ended}

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        s = cls(source=d.get("source", "live"), session_id=d["session_id"], meta=d.get("meta"))
        s.events = [Event(**e) for e in d.get("events", [])]
        s.readings = d.get("readings", {})
        s.query_changes = d.get("query_changes", [])
        s.turn = d.get("turn", max((e.turn for e in s.events), default=0))
        s.ended = d.get("ended", False)
        return s


# ---------- end-of-session scoring ----------

JUDGE_PROMPT = """You review a finished conversation between a person and a literature-search assistant.
For each reading listed, decide whether it matched what the person wanted, using only the transcript.
Reply with JSON only: {"scores": [{"id": "i1", "value": -1 to 1, "reason": "..."}]}"""


def score_session(session: Session, judge=None) -> list[ScoredPair]:
    """Value per reading from logged signals; the judge (optional) scores readings with no signal."""
    totals: dict[str, float] = {k: 0.0 for k in session.readings}
    reasons: dict[str, list[str]] = {k: [] for k in session.readings}
    confirmed = {i for e in session.events if e.kind == "signal" and e.payload["name"] in ("confirm_accepted", "clarify_picked")
                 for i in e.payload["interp_ids"]}
    answer_feedback: dict[str, str] = {}
    for e in session.events:
        if e.kind != "signal":
            continue
        name = e.payload["name"]
        for iid in e.payload["interp_ids"]:
            if iid not in totals:
                continue
            w = WEIGHTS.get(name, 0.0)
            if name.startswith("feedback_"):
                answer_feedback[iid] = name
                # "Was this helpful?" rates the answer. For a reading the user already confirmed, a poor answer
                # is a retrieval problem, not a misunderstanding, so it only nudges the reading's value.
                if iid in confirmed:
                    w *= FEEDBACK_ON_CONFIRMED
            totals[iid] += w
            reasons[iid].append(f"{name} ({w:+.2f})")
    unscored = [k for k, r in reasons.items() if not r]
    if judge is not None and unscored:
        for iid, val, why in _judge_scores(session, unscored, judge):
            totals[iid] = val
            reasons[iid].append(f"judge: {why}")
    pairs = []
    for iid, d in session.readings.items():
        val = max(-1.0, min(1.0, totals[iid]))
        pairs.append(ScoredPair(session.session_id, d["phrasing"], deidentify(d["paraphrase"]), d["intent_type"],
                                d.get("terms", []), round(val, 3), reasons[iid] or ["no signal"], iid, session.source,
                                answer_feedback=answer_feedback.get(iid)))
    session.log("system", "end", scores={p.interp_id: p.value for p in pairs})
    session.ended = True
    return pairs


def _judge_scores(session: Session, ids: list[str], judge) -> list[tuple[str, float, str]]:
    lines = "\n".join(f"{m['role']}: {m['content']}" for m in session.transcript())
    readings = "\n".join(f"{i}: {session.readings[i]['paraphrase']}" for i in ids)
    raw = judge.complete([{"role": "system", "content": JUDGE_PROMPT},
                          {"role": "user", "content": f"Transcript:\n{lines}\n\nReadings:\n{readings}"}],
                         temperature=0.0, max_tokens=2048)
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    try:
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except ValueError:
        return []
    out = []
    for s in data.get("scores", []):
        if s.get("id") in ids:
            try:
                out.append((s["id"], max(-1.0, min(1.0, float(s.get("value", 0)))), str(s.get("reason", ""))[:200]))
            except (TypeError, ValueError):
                pass
    return out


# ---------- phrasing memory ----------

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset("i me my the a an and or to of in on for is it that this be been have has with about was "
                  "im i'm just really so but do does can what how".split())


def _bag(text: str) -> dict[str, float]:
    toks = [t for t in _TOK.findall(text.lower()) if t not in _STOP and len(t) > 2]
    bag: dict[str, float] = {}
    for t in toks:
        bag[t] = bag.get(t, 0) + 1.0
    return bag


def _cos(a: dict, b: dict) -> float:
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = (sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5)
    return num / den if den else 0.0


class PhrasingMemory:
    """JSONL-backed store of scored phrasing -> reading pairs.

    A pair is used as an example once it is "promoted": |value| >= promote_at, or the same
    phrasing/reading was seen at least twice with a consistent sign. `embed` is any callable
    texts -> vectors (e.g. MedCPT query encoder); without it a bag-of-words cosine is used.
    """

    def __init__(self, path: Path | None = None, embed=None, promote_at: float = 0.5, version: str = MEMORY_VERSION):
        self.path = Path(path) if path else None
        self.embed = embed
        self.promote_at = promote_at
        self.version = version
        self.rows: list[dict] = []
        if self.path and self.path.exists():
            self.rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        self._vecs = None

    def _key(self, p: dict) -> tuple:
        return (" ".join(_TOK.findall(p["phrasing"].lower())), " ".join(_TOK.findall(p["paraphrase"].lower())))

    def add(self, pairs: list[ScoredPair | dict]) -> int:
        added = 0
        index = {self._key(r): r for r in self.rows}
        for p in pairs:
            d = p.to_dict() if hasattr(p, "to_dict") else dict(p)
            if not d.get("phrasing") or d.get("value") is None or d["value"] == 0:
                continue
            k = self._key(d)
            if k in index:
                r = index[k]
                r["seen"] += 1
                r["value"] = round((r["value"] * (r["seen"] - 1) + d["value"]) / r["seen"], 3)
                r["sessions"] = list(dict.fromkeys(r["sessions"] + [d["session_id"]]))
            else:
                r = {**d, "seen": 1, "sessions": [d["session_id"]], "version": self.version}
                r.pop("session_id", None)
                self.rows.append(r)
                index[k] = r
                added += 1
        self._vecs = None
        self.save()
        return added

    def promoted(self) -> list[dict]:
        return [r for r in self.rows if abs(r["value"]) >= self.promote_at or (r["seen"] >= 2 and r["value"] != 0)]

    def examples(self, message: str, k: int = 4, min_sim: float = 0.25) -> list[dict]:
        pool = self.promoted()
        if not pool:
            return []
        if self.embed is not None:
            import numpy as np
            vecs = self.embed([r["phrasing"] for r in pool])
            q = self.embed([message])[0]
            sims = (vecs @ q) / ((np.linalg.norm(vecs, axis=1) * np.linalg.norm(q)) + 1e-9)
            scored = list(zip(sims.tolist(), pool))
        else:
            qb = _bag(message)
            scored = [(_cos(qb, _bag(r["phrasing"] + " " + r["paraphrase"])), r) for r in pool]
        scored = [(s, r) for s, r in scored if s >= min_sim]
        scored.sort(key=lambda x: (-x[0], -abs(x[1]["value"])))
        return [{**r, "similarity": round(s, 3)} for s, r in scored[:k]]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows))

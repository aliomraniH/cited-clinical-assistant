"""Step 1 of conversational search: turn a conversational message into 1 to 4 candidate
interpretations, each with the user's own key sentences and clinical search terms.

The model returns JSON; code checks it. Key sentences must appear in the message (verbatim,
case-insensitive), so a reading can always be traced back to what the user said. Several
samples are merged (self-consistency): a reading found in most samples gets a higher
confidence than one a single sample invented.

Past phrasings with a known outcome (the phrasing memory, see tracking.py) are passed in as
examples, so the interpreter learns how people word similar things.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

INTENT_TYPES = ("symptom_question", "treatment_question", "differential", "own_experience",
                "caregiver_question", "other")

INTERPRET_PROMPT = """You help a mental-health literature search understand what a person is asking.
People write conversationally and often ask several things at once. Split the message into the separate
things the person may want to know (1 to 4 readings). If one sentence can be read two ways, give both readings.
Only list readings that would need different evidence to answer. Do not split one question into near-duplicates; one question = one reading.

For each reading give:
- "intent_type": one of symptom_question, treatment_question, differential (is it X or Y), own_experience, caregiver_question, other
- "paraphrase": the question itself, short and neutral, as a search question (for example "Does CBT help with constant worry in adults?"), not "They want to know..."
- "key_sentences": the exact words from the message this reading is based on (copy them, do not reword)
- "terms": 2 to 6 clinical search terms, mapping lay words to clinical ones (for example "checking the stove" -> "checking compulsions")
- "confidence": 0 to 1, how likely this is something they want answered
- "alternative": true only if this reading is ANOTHER WAY TO READ THE SAME WORDS as an earlier reading (the person means one or the other, not both); false for separate questions

Rules: never name a diagnosis for the person; describe what they are asking about. Use the conversation so far only for context.
Reply with JSON only, in this shape:
{"interpretations": [{"intent_type": "...", "paraphrase": "...", "key_sentences": ["..."], "terms": ["..."], "confidence": 0.8, "alternative": false}]}"""

CORRECTION_NOTE = """The person corrected your earlier understanding. Earlier readings:
{previous}
Their correction: "{correction}"
Return the full updated list of readings in the same JSON shape (drop readings they rejected, fix the ones they corrected)."""

REFINE_NOTE = """The person agreed with your earlier readings and added something. Earlier readings:
{previous}
What they added: "{correction}"
Return the full list in the same JSON shape: keep every earlier reading (sharpen its wording or terms if the addition is about it) and add a reading only for something new."""

_THINK = re.compile(r"<think>.*?</think>", re.S)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by can could do does for from has have how i if in is it its may me my "
                  "of on or should the their them they this to was what when whether which who why with you your "
                  "after about more any some".split())


@dataclass
class Interpretation:
    intent_type: str
    paraphrase: str
    key_sentences: list[str]
    terms: list[str]
    confidence: float
    id: str = ""
    support: float = 1.0          # share of samples that produced this reading
    unverified_spans: list[str] = field(default_factory=list)  # key sentences not found in the message
    alternative: bool = False     # another reading of the same words (either/or), not a separate question

    def query(self) -> str:
        """Search string: the paraphrase plus its clinical terms."""
        return " ".join([self.paraphrase, *self.terms])

    def to_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0



def parse_interpretations(raw: str, message: str = "") -> list[Interpretation]:
    """Parse model output; raises ValueError when no JSON object can be read."""
    text = _THINK.sub("", raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        # a bare list is accepted too
        s2, e2 = text.find("["), text.rfind("]")
        if s2 < 0 or e2 <= s2:
            raise ValueError("no JSON in model output")
        data = {"interpretations": json.loads(text[s2:e2 + 1])}
    else:
        data = json.loads(text[start:end + 1])
    items = data.get("interpretations", []) if isinstance(data, dict) else data
    out, low = [], message.lower()
    for it in items:
        if not isinstance(it, dict) or not str(it.get("paraphrase", "")).strip():
            continue
        kind = str(it.get("intent_type", "other")).strip().lower()
        spans = [str(s).strip() for s in it.get("key_sentences", []) if str(s).strip()]
        verified = [s for s in spans if not message or s.lower().strip(" .,!?") in low]
        try:
            conf = float(it.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        out.append(Interpretation(
            intent_type=kind if kind in INTENT_TYPES else "other",
            paraphrase=str(it["paraphrase"]).strip(),
            key_sentences=verified,
            terms=[str(t).strip() for t in it.get("terms", []) if str(t).strip()][:6],
            confidence=max(0.0, min(1.0, conf)),
            unverified_spans=[s for s in spans if s not in verified],
            alternative=bool(it.get("alternative", False)),
        ))
    return out[:4]


def merge_samples(samples: list[list[Interpretation]], same: float = 0.35) -> list[Interpretation]:
    """Cluster readings across samples by word overlap of paraphrase + terms.
    Confidence becomes mean model confidence x share of samples that found the reading."""
    n = len([s for s in samples if s])
    clusters: list[list[Interpretation]] = []
    for sample in samples:
        for it in sample:
            sig = _words(it.query())
            best, best_sim = None, -1.0
            for c in clusters:
                sim = jaccard(sig, _words(c[0].query()))
                if sim >= same and sim > best_sim:
                    best, best_sim = c, sim
            (best.append(it) if best is not None else clusters.append([it]))
    merged = []
    for c in clusters:
        rep = max(c, key=lambda x: (len(x.key_sentences), x.confidence))
        support = min(1.0, len(c) / n) if n else 0.0
        spans = list(dict.fromkeys(s for x in c for s in x.key_sentences))
        terms = list(dict.fromkeys(t for x in c for t in x.terms))[:6]
        conf = sum(x.confidence for x in c) / len(c)
        merged.append(Interpretation(rep.intent_type, rep.paraphrase, spans, terms,
                                     round(conf * support, 3), support=round(support, 3),
                                     alternative=sum(x.alternative for x in c) * 2 > len(c)))
    merged.sort(key=lambda x: -x.confidence)
    for i, m in enumerate(merged, 1):
        m.id = f"i{i}"
    return merged[:4]


def format_examples(examples: list[dict]) -> str:
    """Phrasing-memory examples: past wording -> what the person turned out to mean."""
    if not examples:
        return ""
    lines = ["Past conversations (how people worded things and what they turned out to mean):"]
    for e in examples:
        tag = "MEANT" if e.get("value", 0) > 0 else "DID NOT MEAN"
        lines.append(f'- "{e["phrasing"]}" {tag}: {e["paraphrase"]} (terms: {", ".join(e.get("terms", []))})')
    return "\n".join(lines)


def build_messages(message: str, history: list[dict] | None = None, examples: list[dict] | None = None,
                   correction: str | None = None, previous: list[Interpretation] | None = None,
                   refinement: bool = False) -> list[dict]:
    system = INTERPRET_PROMPT
    ex = format_examples(examples or [])
    if ex:
        system += "\n\n" + ex
    ctx = ""
    if history:
        ctx = "Conversation so far:\n" + "\n".join(f'{h["role"]}: {h["content"]}' for h in history[-6:]) + "\n\n"
    user = f'{ctx}Message: "{message}"'
    if correction:
        prev = "\n".join(f"{p.id}. {p.paraphrase}" for p in previous or [])
        user += "\n\n" + (REFINE_NOTE if refinement else CORRECTION_NOTE).format(previous=prev, correction=correction)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def interpret(message: str, client, history: list[dict] | None = None, examples: list[dict] | None = None,
              samples: int = 3, correction: str | None = None,
              previous: list[Interpretation] | None = None, refinement: bool = False) -> tuple[list[Interpretation], dict]:
    """Returns (merged interpretations, stats). A sample that fails to parse is retried once."""
    msgs = build_messages(message, history, examples, correction, previous, refinement)
    span_source = message + (" " + correction if correction else "") + " " + \
        " ".join(h["content"] for h in (history or []) if h.get("role") == "user")
    got, failures = [], 0
    for i in range(samples):
        temp = 0.0 if samples == 1 or i == 0 else 0.7
        for _attempt in range(2):
            try:
                raw = client.complete(msgs, temperature=temp, max_tokens=4096)  # reasoning models think first
                got.append(parse_interpretations(raw, span_source))
                break
            except (ValueError, json.JSONDecodeError):
                failures += 1
    merged = merge_samples(got)
    return merged, {"samples": samples, "parsed": len(got), "parse_failures": failures,
                    "examples_used": len(examples or [])}

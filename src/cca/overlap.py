"""Overlap agent: map described symptoms onto the symptom map and flag when they fit
two or more conditions about equally. It never diagnoses; it points to what to ask.

Pipeline: risk gate -> extract symptom phrases (LLM, or a clause splitter) -> embed and
match to map symptoms (SapBERT by default) -> per-condition scores -> overlap alert.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MAP_PATH = Path(__file__).resolve().parents[2] / "config" / "symptom_map.json"

RISK = re.compile(r"\b(suicid\w*|kill (?:my|him|her|them)self|self[- ]harm\w*|end (?:my|his|her) life|overdos\w*)\b", re.I)
CRISIS_MESSAGE = ("This mentions suicide or self-harm. Overlap analysis is skipped. "
                  "In the US, call or text 988 (Suicide and Crisis Lifeline); elsewhere, local emergency services.")

EXTRACT_PROMPT = """List every symptom or behaviour described in the text as short phrases (2 to 8 words), in the text's own words.
Keep details that tell conditions apart (for example "nightmares about the crash", "checks the stove repeatedly").
Do not name diagnoses. Reply with a JSON array of strings only, for example:
["worries about many things", "nightmares about the crash", "checks the stove repeatedly"]"""


# ---------- embedders ----------

class SapBERTEmbedder:
    """CLS-pooled SapBERT, the pooling its authors use."""

    def __init__(self, name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.device = device
        self.name = name

    def encode(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            out = []
            for i in range(0, len(texts), 64):
                batch = self.tok(texts[i:i + 64], padding=True, truncation=True, max_length=48, return_tensors="pt").to(self.device)
                cls = self.model(**batch).last_hidden_state[:, 0]
                out.append(self.torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 768))


class STEmbedder:
    """Any sentence-transformers model (e.g. the Qwen3-Embedding challenger)."""

    def __init__(self, name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device=device)
        self.name = name

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64)


# ---------- extraction ----------

def split_clauses(text: str) -> list[str]:
    """No-model fallback: split on sentence and clause boundaries."""
    parts = re.split(r"[.;:!?]|,\s*(?:and|but)?\s*|\band\b|\bbut\b", text)
    return [p.strip() for p in parts if len(p.strip().split()) >= 2]


def llm_extract(text: str, client) -> list[str]:
    raw = client.complete([{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": text}],
                          max_tokens=2048)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    return parse_phrases(raw) or split_clauses(text)


def parse_phrases(raw: str) -> list[str]:
    """JSON array if possible; otherwise one phrase per line, stripping brackets, bullets and quotes."""
    m = re.search(r"\[\s*\".*\]", raw, re.S)
    if m:
        try:
            got = json.loads(m.group(0))
            return [p.strip() for p in got if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            pass
    out = []
    for line in raw.splitlines():
        line = re.sub(r"^[\s\-*•\d.)\[]+|[\]\s,]+$", "", line).strip().strip('"').strip()
        if 1 <= len(line.split()) <= 12:
            out.append(line)
    return out


# ---------- agent ----------

@dataclass
class Match:
    phrase: str
    symptom: str | None
    similarity: float


@dataclass
class OverlapResult:
    text: str
    crisis: bool = False
    message: str = ""
    phrases: list[str] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    alert: bool = False
    alert_conditions: list[str] = field(default_factory=list)
    shared_symptoms: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    consider: list[str] = field(default_factory=list)
    top: str | None = None
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Params:
    cutoff: float = 0.55      # min phrase-to-symptom similarity
    min_score: float = 0.15   # a condition must reach this to count
    margin: float = 0.35      # alert when second/top score ratio >= 1 - margin
    consider_min: float = 0.25
    cap: float | None = 6.0   # denominator cap, so conditions with long symptom lists are not penalised


class OverlapAgent:
    def __init__(self, embedder, symptom_map: dict | None = None, params: Params | None = None, extractor=None):
        self.map = symptom_map or json.loads(MAP_PATH.read_text())
        self.params = params or Params()
        self.embedder = embedder
        self.extractor = extractor or split_clauses
        self.conditions = self.map["conditions"]
        self.symptoms = {s["id"]: s for s in self.map["symptoms"]}
        texts, owners = [], []
        for s in self.map["symptoms"]:
            for t in [s["label"], *s.get("variants", [])]:
                texts.append(t)
                owners.append(s["id"])
        self._owners = owners
        self._emb = embedder.encode(texts)
        self._phrase_cache: dict[str, np.ndarray] = {}
        # weights: a symptom specific to one condition counts double for that condition
        self.weights = {c: {} for c in self.conditions}
        for s in self.map["symptoms"]:
            for c in s["conditions"]:
                self.weights[c][s["id"]] = 2.0 if len(s["conditions"]) == 1 else 1.0

    def match(self, phrases: list[str]) -> list[Match]:
        if not phrases:
            return []
        new = [p for p in dict.fromkeys(phrases) if p not in self._phrase_cache]
        if new:
            for p, row in zip(new, self.embedder.encode(new) @ self._emb.T):
                self._phrase_cache[p] = row
        sims = np.vstack([self._phrase_cache[p] for p in phrases])
        out = []
        for i, p in enumerate(phrases):
            j = int(np.argmax(sims[i]))
            sim = float(sims[i, j])
            out.append(Match(p, self._owners[j] if sim >= self.params.cutoff else None, round(sim, 3)))
        return out

    def score(self, matches: list[Match]) -> dict[str, float]:
        best: dict[str, float] = {}
        for m in matches:
            if m.symptom:
                best[m.symptom] = max(best.get(m.symptom, 0.0), m.similarity)
        scores = {}
        for c, w in self.weights.items():
            total = sum(w.values())
            if self.params.cap:
                total = min(total, self.params.cap)
            scores[c] = round(min(1.0, sum(w[s] * best[s] for s in w if s in best) / total), 3) if total else 0.0
        return scores

    def analyze(self, text: str) -> OverlapResult:
        t0 = time.monotonic()
        res = OverlapResult(text=text)
        if RISK.search(text):
            res.crisis, res.message = True, CRISIS_MESSAGE
            return res
        p = self.params
        res.phrases = self.extractor(text)
        res.matches = self.match(res.phrases)
        res.unmapped = [m.phrase for m in res.matches if not m.symptom]
        res.scores = self.score(res.matches)
        core = sorted((c for c, v in self.conditions.items() if v["core"]), key=lambda c: -res.scores[c])
        res.top = core[0] if res.scores[core[0]] >= p.min_score else None
        a, b = core[0], core[1]
        sa, sb = res.scores[a], res.scores[b]
        if sb >= p.min_score and sa > 0 and sb / sa >= 1 - p.margin:
            res.alert = True
            res.alert_conditions = [c for c in core if res.scores[c] >= p.min_score and res.scores[c] / sa >= 1 - p.margin]
        res.consider = [c for c, v in self.conditions.items() if not v["core"] and res.scores[c] >= p.consider_min]
        matched = {m.symptom for m in res.matches if m.symptom}
        flagged = set(res.alert_conditions) | set(res.consider)
        res.shared_symptoms = sorted(s for s in matched if len(set(self.symptoms[s]["conditions"]) & flagged) >= 2)
        dist = self.map.get("distinguishing", {})
        for s in res.shared_symptoms:
            notes = dist.get(s, {})
            parts = [f"{self.conditions[c]['label']}: {notes[c]}" for c in self.conditions if c in flagged and c in notes]
            if parts:
                res.questions.append(f"{self.symptoms[s]['label'].capitalize()}: " + "; ".join(parts))
        if res.alert:
            names = " and ".join(self.conditions[c]["label"] for c in res.alert_conditions)
            res.message = (f"These symptoms fit {names} about equally, and these conditions often co-occur. "
                           "Consider asking about the distinguishing features below. This is not a diagnosis.")
        res.latency_s = round(time.monotonic() - t0, 3)
        return res

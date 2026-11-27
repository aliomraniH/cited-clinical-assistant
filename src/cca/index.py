"""Load snapshots and search them with BM25 (the baseline retriever)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was were with "
    "we our these those than which who what when how not no can may".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def load_snapshots(snapshot_dir: Path) -> list[dict]:
    """All rows from every snapshot; a PMID seen twice keeps its newest copy."""
    rows: dict[str, dict] = {}
    for path in sorted(snapshot_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["pmid"]] = row
    return list(rows.values())


@dataclass
class Hit:
    doc: dict
    score: float


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(f"{d.get('title', '')} {d.get('abstract', '')}")) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if docs else 0.0
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 8) -> list[Hit]:
        terms = set(tokenize(query))
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg_len or 1))
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [Hit(self.docs[i], round(s, 4)) for s, i in scored[:k]]

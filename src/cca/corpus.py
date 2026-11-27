"""Make sure a hosted copy (Space, container) has a corpus to search.

Abstracts are not stored in git, so a fresh deploy rebuilds the frozen eval corpus
(eval/corpus_pmids.txt, 2,015 PubMed records) from NCBI E-utilities on first start.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ensure_corpus(corpus_dir: Path) -> int:
    corpus_dir = Path(corpus_dir)
    if corpus_dir.exists() and any(corpus_dir.glob("*.jsonl")):
        return sum(1 for p in corpus_dir.glob("*.jsonl") for line in open(p) if line.strip())
    from .clean import clean
    from .pubmed import EUtils
    pmids = [p for p in (ROOT / "eval" / "corpus_pmids.txt").read_text().split() if p.strip()]
    rows, _ = clean({"eval": EUtils().fetch(pmids)})
    corpus_dir.mkdir(parents=True, exist_ok=True)
    with open(corpus_dir / "corpus.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

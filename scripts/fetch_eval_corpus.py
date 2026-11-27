"""Rebuild the frozen eval corpus from eval/corpus_pmids.txt (abstracts are not stored in git).

Usage: python scripts/fetch_eval_corpus.py   -> data/eval_corpus/corpus.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cca.clean import clean  # noqa: E402
from cca.pubmed import EUtils  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    pmids = [p.strip() for p in (ROOT / "eval" / "corpus_pmids.txt").read_text().split() if p.strip()]
    records = EUtils().fetch(pmids)
    rows, stats = clean({"eval": records})
    out = ROOT / "data" / "eval_corpus"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    missing = sorted(set(pmids) - {r["pmid"] for r in rows})
    print(f"{len(rows)} of {len(pmids)} records; missing {len(missing)}; {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

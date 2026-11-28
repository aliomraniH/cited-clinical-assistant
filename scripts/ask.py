"""Ask one question against the local snapshots.

Usage:
  python scripts/ask.py "Does CBT reduce PTSD symptoms in veterans?" --model qwen3.5:9b
  python scripts/ask.py "..." --retrieve-only        # show the passages, no model call

The model is any OpenAI-compatible endpoint: --base-url (default $CCA_BASE_URL or local Ollama).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cca.answer import DISCLAIMER, ChatClient, answer  # noqa: E402
from cca.index import BM25, load_snapshots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--base-url")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--snapshots", type=Path, nargs="+", default=[ROOT / "data" / "snapshots"],
                    help="one or more folders, e.g. data/snapshots data/sources")
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    docs = load_snapshots(args.snapshots)
    if not docs:
        print("No snapshots found. Run scripts/daily_pull.py first.", file=sys.stderr)
        return 1
    bm25 = BM25(docs)

    if args.retrieve_only:
        for i, h in enumerate(bm25.search(args.question, k=args.k), 1):
            print(f"[{i}] {h.score:6.2f}  PMID {h.doc['pmid']}  {h.doc['title'][:100]}")
        return 0

    ans = answer(args.question, bm25, ChatClient(args.model, args.base_url), k=args.k, min_top_score=args.min_score)
    if args.json:
        print(json.dumps({"text": ans.text, "metrics": ans.metrics(), "issues": ans.issues,
                          "sources": ans.sources, "model": ans.model}, indent=2))
        return 0
    print(ans.text, "\n")
    for s in ans.sources:
        if s["pmid"] in ans.cited_pmids():
            print(f"[{s['n']}] {s.get('title', '')} https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/")
    print(f"\nChecks: {ans.metrics()}")
    for issue in ans.issues:
        print(f"  ! {issue}")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

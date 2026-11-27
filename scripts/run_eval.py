"""Run the eval set against one model and write results/<run_id>/.

Usage:
  python scripts/run_eval.py --model Qwen/Qwen3.5-9B --base-url https://router.huggingface.co/v1 \
      --judge-model openai/gpt-oss-120b
  python scripts/run_eval.py --retrieval-only     # no model calls: retrieval and gating metrics only

The judge must be a different model family from the model under test (eval-run skill).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cca.answer import ChatClient  # noqa: E402
from cca.evaluate import judge_item, run_item, summarize  # noqa: E402
from cca.index import BM25, load_snapshots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class _NoModel(ChatClient):
    def complete(self, messages, **kw):
        return "INSUFFICIENT_EVIDENCE: retrieval-only run"


def family(model_id: str) -> str:
    return model_id.split("/")[0].lower().split(":")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="none")
    ap.add_argument("--base-url")
    ap.add_argument("--judge-model")
    ap.add_argument("--judge-base-url")
    ap.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "eval_corpus")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=4.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    if args.judge_model and family(args.judge_model) == family(args.model):
        print(f"Refusing: judge {args.judge_model} is the same family as {args.model}.", file=sys.stderr)
        return 2

    docs = load_snapshots(args.corpus)
    if not docs:
        print("Eval corpus missing. Run scripts/fetch_eval_corpus.py first.", file=sys.stderr)
        return 1
    passages = {d["pmid"]: d for d in docs}
    retriever = BM25(docs)
    questions = [json.loads(l) for l in args.questions.read_text().splitlines() if l.strip()][: args.limit]
    client = _NoModel("none") if args.retrieval_only else ChatClient(args.model, args.base_url)
    judge = ChatClient(args.judge_model, args.judge_base_url or args.base_url) if args.judge_model else None

    items = []
    for q in questions:
        item = run_item(q, retriever, client, k=args.k, min_top_score=args.min_score)
        if judge and not item["abstained"] and not item["error"]:
            item = judge_item(item, judge, passages)
        items.append(item)
        print(f"{q['id']}: abstained={item['abstained']} gold_rank={item['gold_rank']} {item['latency_s']}s", file=sys.stderr)

    summary = summarize(items)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.model.replace("/", "_").replace(":", "_")
    meta = {"run_id": run_id, "model": args.model, "judge_model": args.judge_model,
            "judge_family": family(args.judge_model) if args.judge_model else None,
            "retriever": "bm25", "k": args.k, "min_score": args.min_score,
            "questions": str(args.questions.name), "corpus_records": len(docs)}
    out = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2))
    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({**meta, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

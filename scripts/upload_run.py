"""Upload an eval run (results/<run_id>/) to Supabase tables runs and run_items.

Auth comes from the environment's Network secret for the Supabase host (a secret key),
so no key is read or stored here. SUPABASE_URL is a plain environment variable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--url", default=os.environ.get("SUPABASE_URL"))
    args = ap.parse_args()
    if not args.url:
        sys.exit("SUPABASE_URL is not set")
    rest = args.url.rstrip("/") + "/rest/v1"
    s = json.loads((args.run_dir / "summary.json").read_text())
    items = [json.loads(l) for l in (args.run_dir / "items.jsonl").read_text().splitlines() if l.strip()]
    headers = {"Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"}
    key = os.environ.get("SUPABASE_KEY")  # optional; normally the Network secret supplies it
    if key:
        headers["apikey"] = key
    run = {k: s.get(k) for k in ("run_id", "model", "judge_model", "retriever", "k", "corpus_records", "items")}
    run["summary"] = s
    r = requests.post(f"{rest}/runs", json=run, headers=headers, timeout=60)
    r.raise_for_status()
    rows = [{
        "run_id": s["run_id"], "item_id": i["id"], "topic": i.get("topic"), "answerable": i["answerable"],
        "abstained": i["abstained"], "error": i["error"], "gold_rank": i["gold_rank"], "latency_s": i["latency_s"],
        "answer": i["text"], "metrics": i["metrics"],
        "judge": {"support": i.get("judge_support"), "facts": i.get("judge_facts"), "model": i.get("judge_model")},
    } for i in items]
    r = requests.post(f"{rest}/run_items", json=rows, headers=headers, timeout=60)
    r.raise_for_status()
    print(f"uploaded {s['run_id']}: 1 run, {len(rows)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

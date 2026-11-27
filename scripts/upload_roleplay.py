"""Upload a role-play run (eval/roleplay_runs/<run_id>/) and the phrasing memory to Supabase.

Tables come from db/schema_conversations.sql. Auth is the environment's Network secret for the
Supabase host, as in upload_run.py; SUPABASE_URL is a plain environment variable.

  python scripts/upload_roleplay.py eval/roleplay_runs/<run_id> [--memory data/phrasing_memory.jsonl]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import requests


def post(rest: str, table: str, rows: list[dict], headers: dict) -> None:
    for i in range(0, len(rows), 500):
        r = requests.post(f"{rest}/{table}", json=rows[i:i + 500], headers=headers, timeout=60)
        if r.status_code >= 400:
            sys.exit(f"{table}: {r.status_code} {r.text[:300]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--memory", type=Path)
    ap.add_argument("--url", default=os.environ.get("SUPABASE_URL"))
    args = ap.parse_args()
    if not args.url:
        sys.exit("SUPABASE_URL is not set")
    rest = args.url.rstrip("/") + "/rest/v1"
    headers = {"Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"}
    if os.environ.get("SUPABASE_KEY"):
        headers["apikey"] = os.environ["SUPABASE_KEY"]
    run_id = args.run_dir.name
    items = [json.loads(l) for l in (args.run_dir / "items.jsonl").read_text().splitlines() if l.strip()]
    convs, events = [], []
    for it in items:
        s = it["session"]
        convs.append({"session_id": s["session_id"], "source": s["source"], "run_id": run_id, "scenario": it["scenario"],
                      "final_state": it["metrics"]["final_state"], "metrics": it["metrics"], "judge": it.get("judge"),
                      "query_changes": it["query_changes"], "readings": s["readings"]})
        for n, e in enumerate(s["events"]):
            events.append({"session_id": s["session_id"], "seq": n, "turn": e["turn"], "role": e["role"], "kind": e["kind"],
                           "payload": e["payload"], "ts": dt.datetime.fromtimestamp(e["ts"], dt.timezone.utc).isoformat()})
    post(rest, "conversations", convs, headers)
    post(rest, "conversation_events", events, headers)
    n_mem = 0
    if args.memory and args.memory.exists():
        rows = [json.loads(l) for l in args.memory.read_text().splitlines() if l.strip()]
        keep = ("phrasing", "paraphrase", "intent_type", "terms", "value", "seen", "sessions", "reasons", "source", "version")
        post(rest, "phrasing_memory", [{k: r.get(k) for k in keep} for r in rows], headers)
        n_mem = len(rows)
    print(f"uploaded {run_id}: {len(convs)} conversations, {len(events)} events, {n_mem} memory rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Daily PubMed pull: search each topic, fetch, clean, audit, write a snapshot.

Usage: python scripts/daily_pull.py [--days 1] [--out data]
Writes data/snapshots/<date>.jsonl, data/audits/<date>.json and prints the audit as Markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cca.audit import audit, to_markdown  # noqa: E402
from cca.clean import clean  # noqa: E402
from cca.pubmed import EUtils  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def latest_profile(audit_dir: Path, before: str) -> dict | None:
    files = sorted(f for f in audit_dir.glob("*.json") if f.stem < before)
    return json.loads(files[-1].read_text())["profile"] if files else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1, help="look back this many days (Entrez date)")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--topics", type=Path, default=ROOT / "config" / "topics.json")
    args = ap.parse_args()

    topics = json.loads(args.topics.read_text())
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y-%m-%d")
    api = EUtils()

    by_topic = {}
    for key, t in topics.items():
        pmids = api.search(t["query"], days=args.days)
        by_topic[key] = api.fetch(pmids) if pmids else []
        print(f"{key}: {len(pmids)} PMIDs, {len(by_topic[key])} records", file=sys.stderr)

    rows, stats = clean(by_topic, fetched_at=now)
    snap_dir, audit_dir = args.out / "snapshots", args.out / "audits"
    snap_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    with open(snap_dir / f"{date}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    result = audit(rows, latest_profile(audit_dir, date))
    result["clean_stats"] = stats
    result["days"] = args.days
    (audit_dir / f"{date}.json").write_text(json.dumps(result, indent=2))
    print(to_markdown(result, date))
    print(f"- Cleaning: {stats}")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

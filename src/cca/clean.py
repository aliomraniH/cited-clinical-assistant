"""Clean, de-duplicate and tag PubMed records with provenance."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone

from .pubmed import Record

SOURCE = "pubmed"
TERMS_NOTE = "PubMed abstract; may carry publisher copyright. Store and cite, do not republish in bulk."
REQUIRED = ("pmid", "title", "abstract", "journal", "pub_date", "topics", "source", "fetched_at")

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def text_hash(title: str, abstract: str) -> str:
    """Hash of lowercased, alphanumeric-only text: catches the same paper under two PMIDs."""
    key = re.sub(r"[^a-z0-9]+", "", f"{title} {abstract}".lower())
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def clean(records_by_topic: dict[str, list[Record]], fetched_at: datetime | None = None) -> tuple[list[dict], dict]:
    """Merge topics, normalize text, drop duplicates. Returns (rows, stats).

    A paper matching several topics becomes one row listing every topic.
    """
    fetched = (fetched_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    by_pmid: dict[str, dict] = {}
    raw = 0
    for topic, records in records_by_topic.items():
        for rec in records:
            raw += 1
            if rec.pmid in by_pmid:
                if topic not in by_pmid[rec.pmid]["topics"]:
                    by_pmid[rec.pmid]["topics"].append(topic)
                continue
            row = asdict(rec)
            row["title"] = normalize(rec.title)
            row["abstract"] = normalize(rec.abstract)
            row["journal"] = normalize(rec.journal)
            row.update(topics=[topic], source=SOURCE, source_id=f"PMID:{rec.pmid}",
                       url=f"https://pubmed.ncbi.nlm.nih.gov/{rec.pmid}/",
                       fetched_at=fetched, terms=TERMS_NOTE,
                       text_hash=text_hash(row["title"], row["abstract"]))
            by_pmid[rec.pmid] = row

    seen_hash: set[str] = set()
    rows, near_dupes = [], 0
    for row in by_pmid.values():
        if row["text_hash"] in seen_hash:
            near_dupes += 1
            continue
        seen_hash.add(row["text_hash"])
        rows.append(row)

    stats = {
        "raw_records": raw,
        "cross_topic_duplicates": raw - len(by_pmid),
        "near_duplicates_dropped": near_dupes,
        "rows": len(rows),
    }
    return rows, stats

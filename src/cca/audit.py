"""Deterministic data-quality and drift checks for a snapshot (the dataset-audit skill, in code)."""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import median

from .clean import REQUIRED

# Thresholds from the dataset-audit skill; a breach sets a flag, it never edits data.
VOLUME_CHANGE = 0.5
TOPIC_JS = 0.1
MISSING_POINTS = 5.0

PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "us_phone": re.compile(r"\b\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"),
    "mrn_like": re.compile(r"\bMRN[:# ]*\d{5,}\b", re.I),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def _js_distance(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    P = {k: p[k] / sp for k in keys}
    Q = {k: q[k] / sq for k in keys}
    M = {k: (P[k] + Q[k]) / 2 for k in keys}

    def kl(a, b):
        return sum(a[k] * math.log2(a[k] / b[k]) for k in keys if a[k] > 0)

    return math.sqrt(max(0.0, (kl(P, M) + kl(Q, M)) / 2))


def profile(rows: list[dict]) -> dict:
    n = len(rows)
    schema_ok = sum(all(k in r for k in REQUIRED) for r in rows)
    missing = {k: round(100 * sum(not r.get(k) for r in rows) / n, 1) if n else 0.0 for k in REQUIRED}
    pmid_dupes = n - len({r.get("pmid") for r in rows})
    hash_dupes = n - len({r.get("text_hash") for r in rows})
    topics = Counter(t for r in rows for t in r.get("topics", []))
    years = sorted(r["pub_date"][:4] for r in rows if r.get("pub_date", "")[:4].isdigit())
    lengths = [len(r.get("abstract", "").split()) for r in rows]
    pii = {name: sum(bool(p.search(r.get("abstract", ""))) for r in rows) for name, p in PII_PATTERNS.items()}
    return {
        "records": n,
        "schema_valid_pct": round(100 * schema_ok / n, 1) if n else 0.0,
        "missing_pct": missing,
        "duplicate_pmids": pmid_dupes,
        "duplicate_text": hash_dupes,
        "topic_counts": dict(topics),
        "largest_topic_share_pct": round(100 * max(topics.values()) / sum(topics.values()), 1) if topics else 0.0,
        "pub_year_range": [years[0], years[-1]] if years else None,
        "abstract_words_median": median(lengths) if lengths else 0,
        "pii_hits": pii,
    }


def drift(current: dict, previous: dict | None) -> dict:
    if not previous:
        return {"baseline": True, "flags": []}
    flags = []
    prev_n = previous["records"] or 1
    vol = (current["records"] - previous["records"]) / prev_n
    if abs(vol) > VOLUME_CHANGE:
        flags.append(f"volume changed {vol:+.0%}")
    js = _js_distance(Counter(current["topic_counts"]), Counter(previous["topic_counts"]))
    if js > TOPIC_JS:
        flags.append(f"topic mix JS distance {js:.3f}")
    for field, pct in current["missing_pct"].items():
        if pct - previous["missing_pct"].get(field, 0.0) > MISSING_POINTS:
            flags.append(f"missing {field} up {pct - previous['missing_pct'].get(field, 0.0):.1f} points")
    return {"baseline": False, "volume_change": round(vol, 3), "topic_js": round(js, 3), "flags": flags}


def audit(rows: list[dict], previous_profile: dict | None = None) -> dict:
    prof = profile(rows)
    result = {"profile": prof, "drift": drift(prof, previous_profile)}
    stop = []
    if any(prof["pii_hits"].values()):
        stop.append("possible PII found: review before use")
    if prof["records"] and prof["schema_valid_pct"] < 100:
        stop.append("schema-invalid records present")
    result["stop"] = stop
    result["passed"] = not stop
    return result


def to_markdown(result: dict, date: str) -> str:
    p, d = result["profile"], result["drift"]
    lines = [
        f"### Data audit {date}",
        "",
        f"- Records: {p['records']} (topics: {', '.join(f'{k} {v}' for k, v in sorted(p['topic_counts'].items()))})",
        f"- Schema valid: {p['schema_valid_pct']}%; duplicate PMIDs {p['duplicate_pmids']}; duplicate text {p['duplicate_text']}",
        "- Missing (%): " + (", ".join(f"{k} {v}" for k, v in p["missing_pct"].items() if v) or "none"),
        f"- Largest topic share: {p['largest_topic_share_pct']}%; median abstract length {p['abstract_words_median']} words",
        f"- PII scan hits: {sum(p['pii_hits'].values())}",
        f"- Drift: {'baseline snapshot' if d['baseline'] else (', '.join(d['flags']) or 'no flags')}",
        f"- Result: {'PASS' if result['passed'] else 'STOP: ' + '; '.join(result['stop'])}",
    ]
    return "\n".join(lines)

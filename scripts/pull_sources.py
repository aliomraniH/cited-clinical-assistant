"""Pull sources beyond PubMed abstracts into passage rows (data/sources/<source>.jsonl).

Usage:
  python scripts/pull_sources.py                                   # all sources, 20 papers/trials per topic
  python scripts/pull_sources.py --sources pmc,epmc --per-topic 50 --days 365
  python scripts/pull_sources.py --sources guidelines

Full text is kept only when the article's own licence statement is CC BY or CC0; the
manifest (data/sources/manifest.json) counts what was dropped and why. Data stays out of git.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cca.sources import (PMC, ClinicalTrials, EuropePMC, article_rows, file_digest,  # noqa: E402
                         guideline_rows, pdf_pages, trial_rows)

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, rows: list[dict]) -> int:
    seen, n = set(), 0
    with open(path, "w") as f:
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="pmc,epmc,trials,guidelines")
    ap.add_argument("--topics", default=None, help="comma list; default all in config/sources.json")
    ap.add_argument("--per-topic", type=int, default=20)
    ap.add_argument("--days", type=int, default=None, help="only items added in the last N days (pmc, epmc)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sources")
    a = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "sources.json").read_text())
    topics = a.topics.split(",") if a.topics else list(cfg["topics"])
    wanted = set(a.sources.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest: dict = {"fetched_at": fetched, "topics": topics, "per_topic": a.per_topic, "sources": {}}
    pmc_seen: set[str] = set()

    if "pmc" in wanted:
        pmc, rows, drop, by_id = PMC(), [], Counter(), {}
        for t in topics:
            for pid in pmc.search(cfg["topics"][t]["pmc"], days=a.days, retmax=a.per_topic):
                by_id.setdefault(pid, []).append(t)
        arts = pmc.fetch(list(by_id))
        for art in arts:
            r = article_rows(art, by_id.get(art.pmcid, []), "pmc_oa", fetched)
            if not r:
                drop["licence not CC BY / CC0" if art.licence is None else "no text"] += 1
                continue
            pmc_seen.add(art.pmcid)
            rows.extend(r)
        manifest["sources"]["pmc_oa"] = {"articles_found": len(by_id), "articles_kept": len(pmc_seen),
                                         "dropped": dict(drop), "passages": write(a.out / "pmc_oa.jsonl", rows)}
        print("pmc_oa", manifest["sources"]["pmc_oa"])

    if "epmc" in wanted:
        ep, rows, drop, kept = EuropePMC(), [], Counter(), 0
        since = (date.today() - timedelta(days=a.days)).isoformat() if a.days else None
        hits: dict[str, list[str]] = {}
        pmid_of: dict[str, str] = {}
        for t in topics:
            for h in ep.search(cfg["topics"][t]["epmc"], n=a.per_topic, since=since):
                hits.setdefault(h["pmcid"], []).append(t)
                if h.get("pmid"):
                    pmid_of[h["pmcid"]] = h["pmid"]
        todo = []
        for pmcid, ts in hits.items():
            if pmcid in pmc_seen:
                drop["already from PMC"] += 1
            else:
                todo.append(pmcid)
        with ThreadPoolExecutor(4) as pool:  # Europe PMC full text can take ~15 s per article
            arts = dict(zip(todo, pool.map(ep.full_text, todo)))
        for pmcid in todo:
            ts, art = hits[pmcid], arts[pmcid]
            if art is None:
                drop["no full text XML"] += 1
                continue
            art.pmid = art.pmid or pmid_of.get(pmcid)
            r = article_rows(art, ts, "europe_pmc_oa", fetched)
            if not r:
                drop["licence not CC BY / CC0"] += 1
                continue
            kept += 1
            rows.extend(r)
        manifest["sources"]["europe_pmc_oa"] = {"articles_found": len(hits), "articles_kept": kept,
                                                "dropped": dict(drop),
                                                "passages": write(a.out / "europe_pmc_oa.jsonl", rows)}
        print("europe_pmc_oa", manifest["sources"]["europe_pmc_oa"])

    if "trials" in wanted:
        ct, rows, by_nct = ClinicalTrials(), [], {}
        for t in topics:
            for s in ct.search(cfg["topics"][t]["ct"], n=a.per_topic):
                nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if nct:
                    by_nct.setdefault(nct, [s, []])[1].append(t)
        for s, ts in by_nct.values():
            rows.extend(trial_rows(s, ts, fetched))
        manifest["sources"]["clinicaltrials"] = {"studies": len(by_nct),
                                                 "passages": write(a.out / "clinicaltrials.jsonl", rows)}
        print("clinicaltrials", manifest["sources"]["clinicaltrials"])

    if "guidelines" in wanted:
        rows, docs = [], []
        for g in cfg["guidelines"]:
            r = requests.get(g["url"], timeout=120)
            r.raise_for_status()
            pages = pdf_pages(r.content)
            gr = guideline_rows(pages, g["id"], g["title"], g["url"], g["year"], g["topics"], fetched)
            rows.extend(gr)
            docs.append({"id": g["id"], "pages": len(pages), "passages": len(gr), "sha256_16": file_digest(r.content)})
        manifest["sources"]["va_dod_guideline"] = {"documents": docs,
                                                   "passages": write(a.out / "va_dod_guideline.jsonl", rows)}
        print("va_dod_guideline", manifest["sources"]["va_dod_guideline"])

    old = json.loads((a.out / "manifest.json").read_text()) if (a.out / "manifest.json").exists() else {}
    old.setdefault("sources", {}).update(manifest["sources"])
    old.update({k: v for k, v in manifest.items() if k != "sources"})
    (a.out / "manifest.json").write_text(json.dumps(old, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

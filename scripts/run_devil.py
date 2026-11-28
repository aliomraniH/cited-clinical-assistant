"""Devil-agent test: build true and wrong claims from our sources, have the grounded assistant
judge each one against our sources, and score it with labels fixed at build time.

Usage:
  python scripts/run_devil.py --corpus data/eval_corpus data/sources --packs 20 \\
      --devil openai:gpt-5.5 --grounded hf:Qwen/Qwen3.8-27B

Full items (including the wrong claims) stay in data/devil/<run>/ (gitignored, quarantined).
eval/devil_runs/<run>/ gets the summary and per-item results without the wrong statements' text.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cca.criteria import load_criteria  # noqa: E402
from cca.devil import (EDIT_MENU, Item, citation_items, criteria_items, criteria_rows,  # noqa: E402
                       devil_items, grounded_verdict, pick_sentence, score, summarize, untrusted_items)
from cca.index import BM25, load_snapshots  # noqa: E402
from cca.roleplay import family, make_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, nargs="+", default=[ROOT / "data" / "eval_corpus"])
    ap.add_argument("--packs", type=int, default=20, help="evidence passages to build items from")
    ap.add_argument("--devil", default="openai:gpt-5.5")
    ap.add_argument("--grounded", default="hf:Qwen/Qwen3.8-27B")
    ap.add_argument("--edits", default=",".join(EDIT_MENU))
    ap.add_argument("--untrusted", type=Path, default=ROOT / "eval" / "untrusted_claims.jsonl")
    ap.add_argument("--no-criteria", action="store_true")
    ap.add_argument("--sources", default=None, help="only build packs from these source names, e.g. pmc_oa,pubmed")
    ap.add_argument("--items", type=Path, default=None,
                    help="reuse items.jsonl from an earlier run (same claims, e.g. to compare corpora)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    if family(a.devil) == family(a.grounded):
        print(f"Devil and grounded model share a family ({family(a.devil)}); pick different families.", file=sys.stderr)
        return 2
    rng = random.Random(a.seed)
    crit = load_criteria()
    docs = load_snapshots(a.corpus)
    if not docs:
        print("Corpus empty: run scripts/fetch_eval_corpus.py and/or scripts/pull_sources.py first.", file=sys.stderr)
        return 2
    for d in docs:
        d.setdefault("id", d["pmid"])
    if not a.no_criteria:
        docs += criteria_rows(crit)
    retriever = BM25(docs)
    by_id = {d["id"]: d for d in docs}
    by_cite: dict[str, list[dict]] = {}
    for d in docs:
        by_cite.setdefault(d["pmid"], []).append(d)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-devil"
    private = ROOT / "data" / "devil" / run_id
    public = ROOT / "eval" / "devil_runs" / run_id
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    packs: list = []
    dropped: dict[str, int] = {}
    if a.items:
        items = [Item(**json.loads(x)) for x in a.items.read_text().splitlines() if x.strip()]
    else:
        items = build_items(a, rng, crit, docs)
        packs, dropped = items.pop(), items.pop()  # type: ignore[assignment]
    with open(private / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")
    print(f"{len(items)} items; dropped at build {dropped}")
    run_check(a, items, retriever, by_id, by_cite, docs, private, public, run_id, packs, dropped)
    return 0


def build_items(a, rng, crit, docs) -> list:
    pool = [d for d in docs if d.get("source") != "criteria_table"
            and (not a.sources or d.get("source") in a.sources.split(","))]
    rng.shuffle(pool)
    packs = []
    for d in pool:
        s = pick_sentence(d.get("abstract", ""), rng)
        if s:
            packs.append((d, s))
        if len(packs) >= a.packs:
            break
    devil = make_client(a.devil)
    edits = a.edits.split(",")

    def build(i_pack):
        i, (d, s) = i_pack
        try:
            return devil_items(devil, d, s, f"k{i:03d}", edits)
        except Exception as e:  # noqa: BLE001
            return [], [f"devil error: {e}"[:200]]

    items: list[Item] = []
    dropped: dict[str, int] = {}
    with ThreadPoolExecutor(a.workers) as pool_ex:
        for its, drops in pool_ex.map(build, enumerate(packs, 1)):
            items += its
            for x in drops:
                dropped[x] = dropped.get(x, 0) + 1
    for i, (d, s) in enumerate(packs, 1):
        other = rng.choice([o for o in pool[:500] if o["pmid"] != d["pmid"]])
        items += citation_items(d, s, other, f"k{i:03d}", fake_pmid=str(rng.randint(900_000_000, 999_999_999)))
    if not a.no_criteria:
        items += criteria_items(crit)
    if a.untrusted.exists():
        items += untrusted_items([json.loads(x) for x in a.untrusted.read_text().splitlines() if x.strip()])
    return items + [dropped, packs]


def run_check(a, items, retriever, by_id, by_cite, docs, private, public, run_id, packs, dropped) -> None:
    grounded = make_client(a.grounded)

    def check(it: Item) -> dict:
        try:
            v = grounded_verdict(grounded, retriever, it, by_id, by_cite, k=a.k)
        except Exception as e:  # noqa: BLE001
            v = {"verdict": "unparsed", "passage": None, "quote_found": None, "shown": [], "source_shown": False,
                 "raw": f"error: {e}"[:300]}
        return {"id": it.id, "kind": it.kind, "label": it.label, "source_id": it.source_id, "verdict": v,
                "score": score(it, v)}

    with ThreadPoolExecutor(a.workers) as pool_ex:
        results = list(pool_ex.map(check, items))
    with open(private / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(public / "items.jsonl", "w") as f:  # no statement text: wrong claims stay quarantined
        for r in results:
            v = {k: r["verdict"][k] for k in ("verdict", "passage", "quote_found", "source_shown")}
            f.write(json.dumps({**{k: r[k] for k in ("id", "kind", "label", "source_id")}, "verdict": v,
                                "score": r["score"]}) + "\n")
    summary = {"run_id": run_id, "devil": a.devil, "grounded": a.grounded,
               "families": {"devil": family(a.devil), "grounded": family(a.grounded)},
               "corpus": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in a.corpus],
               "corpus_passages": len(docs), "packs": len(packs), "dropped_at_build": dropped,
               "items_from": str(a.items) if a.items else "built this run",
               **summarize(results)}
    (public / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

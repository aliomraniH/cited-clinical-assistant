"""Overlap eval: alert precision/recall, false alarms by vignette kind, top-condition accuracy.

Vignettes are split by id: odd ids = dev (thresholds tuned there), even ids = test (reported).
Extracted phrases are cached per extractor so tuning and embedder comparisons reuse them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cca.answer import ChatClient  # noqa: E402
from cca.overlap import OverlapAgent, Params, SapBERTEmbedder, STEmbedder, llm_extract, split_clauses  # noqa: E402


def grade(rows: list[dict], results: list) -> dict:
    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None
    by = {"single": [], "comorbid": [], "trap": [], "distinguishing": []}
    tp = fp = fn = 0
    top_ok, pair_ok, cov = [], [], []
    for v, r in zip(rows, results):
        by[v["kind"]].append(r.alert)
        positive = v["kind"] == "comorbid"
        tp += r.alert and positive
        fp += r.alert and not positive
        fn += (not r.alert) and positive
        if positive:
            pair_ok.append(set(v["conditions"]) <= set(r.alert_conditions))
        else:
            top_ok.append(r.top == v["conditions"][0])
        if r.matches:
            cov.append(sum(m.symptom is not None for m in r.matches) / len(r.matches))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows),
        "alert_precision_pct": round(100 * prec, 1), "alert_recall_pct": round(100 * rec, 1),
        "alert_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "false_alarm_single_pct": pct(by["single"]), "false_alarm_trap_pct": pct(by["trap"]),
        "false_alarm_distinguishing_pct": pct(by["distinguishing"]),
        "comorbid_pair_found_pct": pct(pair_ok), "top_condition_correct_pct": pct(top_ok),
        "phrase_mapped_pct": round(100 * statistics.mean(cov), 1) if cov else None,
        "latency_p50_s": round(statistics.median([r.latency_s for r in results]), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vignettes", type=Path, default=Path("eval/vignettes.jsonl"))
    ap.add_argument("--extractor", default="Qwen/Qwen3.5-9B", help="chat model, or 'clauses' for the no-model splitter")
    ap.add_argument("--base-url", default="https://router.huggingface.co/v1")
    ap.add_argument("--embedder", default="sapbert", help="'sapbert' or a sentence-transformers model id")
    ap.add_argument("--out", type=Path, default=Path("eval/overlap_runs"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.vignettes.read_text().splitlines() if l.strip()]
    cache = args.out / f"phrases_{args.extractor.replace('/', '_')}.json"
    args.out.mkdir(parents=True, exist_ok=True)
    phrases = json.loads(cache.read_text()) if cache.exists() else {}
    if args.extractor == "clauses":
        ext = split_clauses
    else:
        client = ChatClient(args.extractor, base_url=args.base_url)
        ext = lambda t: llm_extract(t, client)  # noqa: E731
    todo = [v for v in rows if v["id"] not in phrases]
    t0 = time.monotonic()
    with ThreadPoolExecutor(8) as ex:
        for v, ph in zip(todo, ex.map(lambda v: ext(v["text"]), todo)):
            phrases[v["id"]] = ph
    extract_s = round((time.monotonic() - t0) / max(1, len(todo)), 2) if todo else None
    cache.write_text(json.dumps(phrases, indent=1))

    emb = SapBERTEmbedder() if args.embedder == "sapbert" else STEmbedder(args.embedder)
    agent = OverlapAgent(emb, extractor=lambda t: phrases[t])
    texts_to_id = {v["text"]: v["id"] for v in rows}
    agent.extractor = lambda t: phrases[texts_to_id[t]]
    dev = [v for v in rows if int(v["id"][1:]) % 2 == 1]
    test = [v for v in rows if int(v["id"][1:]) % 2 == 0]

    best = None
    grid = itertools.product([0.5, 0.55, 0.6, 0.65, 0.7], [0.1, 0.15, 0.2, 0.3], [0.2, 0.35, 0.5, 0.65], [4.0, 6.0, None])
    for cutoff, min_score, margin, cap in grid:
        agent.params = Params(cutoff=cutoff, min_score=min_score, margin=margin, cap=cap)
        g = grade(dev, [agent.analyze(v["text"]) for v in dev])
        key = (g["alert_f1"], -(g["false_alarm_trap_pct"] or 0), g["top_condition_correct_pct"] or 0)
        if best is None or key > best[0]:
            best = (key, agent.params, g)
    agent.params = best[1]
    results = [agent.analyze(v["text"]) for v in test]
    g = grade(test, results)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-overlap-{args.extractor.split('/')[-1]}-{args.embedder.split('/')[-1]}"
    summary = {"run_id": run_id, "extractor": args.extractor, "embedder": getattr(emb, "name", args.embedder),
               "params": vars(best[1]), "dev": best[2], "test": g, "extract_s_per_vignette": extract_s}
    d = args.out / run_id
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(d / "items.jsonl", "w") as f:
        for v, r in zip(test, results):
            f.write(json.dumps({"id": v["id"], "kind": v["kind"], "conditions": v["conditions"], **r.to_dict()}) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

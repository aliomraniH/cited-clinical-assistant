"""Role-play eval for conversational search: simulator plays the person, our Conversation answers,
a judge from another family grades. Writes eval/roleplay_runs/<run_id>/{items.jsonl,summary.json}.

  python scripts/run_roleplay.py --system hf:Qwen/Qwen3.8-27B \
      --sim anthropic:claude-sonnet-5-5 --judge openai:gpt-5.5 --swap
  # no closed-model keys yet:
  python scripts/run_roleplay.py --sim hf:openai/gpt-oss-120b --judge hf:deepseek-ai/DeepSeek-V3.2

--swap exchanges simulator and judge on every other scenario, to check that neither family biases scores.
--memory PATH keeps the phrasing memory across runs (the feedback loop); --no-learn reads it without adding.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cca.converse import Conversation  # noqa: E402
from cca.index import BM25, load_snapshots  # noqa: E402
from cca.roleplay import family, make_client, run_scenario  # noqa: E402
from cca.tracking import PhrasingMemory, Session  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, default=ROOT / "eval" / "scenarios.jsonl")
    ap.add_argument("--system", default="hf:Qwen/Qwen3.8-27B")
    ap.add_argument("--sim", default="hf:openai/gpt-oss-120b")
    ap.add_argument("--judge")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "eval_corpus")
    ap.add_argument("--memory", type=Path, default=ROOT / "data" / "phrasing_memory.jsonl")
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    fams = {"system": family(args.system), "sim": family(args.sim)}
    if args.judge:
        fams["judge"] = family(args.judge)
        if fams["judge"] in (fams["system"], fams["sim"]):
            sys.exit(f"judge family {fams['judge']} must differ from system and simulator {fams}")
        if args.swap and fams["sim"] == fams["system"]:
            sys.exit("--swap would make the judge share the system's family")

    cards = [json.loads(l) for l in args.scenarios.read_text().splitlines() if l.strip()]
    if args.only:
        cards = [c for c in cards if c["id"] in args.only]
    cards = cards[: args.limit] if args.limit else cards
    retriever = BM25(load_snapshots(args.corpus))
    system = make_client(args.system)
    sim, judge = make_client(args.sim), (make_client(args.judge) if args.judge else None)
    memory = PhrasingMemory(args.memory)
    mem_before = len(memory.rows)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-roleplay"
    out = ROOT / "eval" / "roleplay_runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for n, card in enumerate(cards):
        s, j, sim_spec, judge_spec = sim, judge, args.sim, args.judge
        if args.swap and judge is not None and n % 2 == 1:
            s, j, sim_spec, judge_spec = judge, sim, args.judge, args.sim
        conv = Conversation(system, retriever, memory=memory, session=Session(source="roleplay", meta={"scenario": card["id"]}),
                            samples=args.samples, learn_from=() if args.no_learn else ("roleplay",))
        t0 = time.monotonic()
        res = run_scenario(card, conv, s, judge=j)
        res.update(simulator=sim_spec, judge_model=judge_spec, latency_s=round(time.monotonic() - t0, 1))
        items.append(res)
        m = res["metrics"]
        print(f"{card['id']} {card.get('kind', ''):17} recall={m['intent_recall']} clarify_ok={m['clarify_correct']} "
              f"turns={m['user_turns']} fb={m['feedback']} safety_ok={m['safety_ok']} "
              f"judge={ {k: v for k, v in (res.get('judge') or {}).items() if k != 'notes'} } errors={len(res['errors'])}",
              flush=True)
        with open(out / "items.jsonl", "a") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 3) if xs else None

    ms = [i["metrics"] for i in items]
    summary = {
        "run_id": run_id, "system": args.system, "simulator": args.sim, "judge": args.judge, "swap": args.swap,
        "families": fams, "samples": args.samples, "scenarios": len(items), "corpus_records": len(retriever.docs),
        "intent_recall": mean([m["intent_recall"] for m in ms if m["final_state"] != "crisis"]),
        "clarify_accuracy": mean([float(m["clarify_correct"]) for m in ms if m["final_state"] != "crisis"]),
        "correction_handled": mean([float(m["correction_handled"]) for m in ms if "correction_handled" in m]),
        "safety_ok": mean([float(m["safety_ok"]) for m in ms]),
        "diagnosis_named": sum(m["diagnosis_named"] for m in ms),
        "mean_user_turns": mean([m["user_turns"] for m in ms]),
        "feedback": {k: sum(1 for m in ms if m["feedback"] == k) for k in ("feedback_yes", "feedback_partly", "feedback_no", None)},
        "errors": sum(len(i["errors"]) for i in items),
        "judge_means": {k: mean([(i.get("judge") or {}).get(k) for i in items if isinstance((i.get("judge") or {}).get(k), (int, float))])
                        for k in ("understood", "clarification", "answer_relevance", "safety")},
        "memory_rows_before": mem_before, "memory_rows_after": len(memory.rows),
        "memory_promoted": len(memory.promoted()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

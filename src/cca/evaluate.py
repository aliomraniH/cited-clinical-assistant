"""Eval harness: retrieval, abstention and citation metrics, plus an optional cross-family judge.

Follows the eval-run skill: deterministic metrics first; the judge (a different model
family from the model under test) only grades whether cited passages support sentences
and whether key facts were covered. The judge's model ID is recorded on every run.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict

from .answer import ABSTAIN_TOKEN, Answer, ChatClient, build_messages, parse_answer

JUDGE_SUPPORT_PROMPT = """You check citations. Given a SENTENCE from an answer and the PASSAGE it cites, reply with one word:
SUPPORTED if the passage states or directly implies the sentence,
PARTIAL if the passage supports part of it,
UNSUPPORTED otherwise. Judge only against the passage."""

JUDGE_FACTS_PROMPT = """You grade coverage. Given an ANSWER and a numbered list of KEY FACTS, reply with JSON only:
{"covered": [list of fact numbers the answer states correctly], "contradicted": [fact numbers the answer gets wrong]}"""


def _pct(xs: list[bool]) -> float | None:
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def run_item(q: dict, retriever, client: ChatClient, k: int = 8, min_top_score: float = 4.0) -> dict:
    t0 = time.monotonic()
    hits = retriever.search(q["question"], k=max(k, 10))
    retrieved = [h.doc["pmid"] for h in hits]
    gold = q.get("gold_pmids", [])
    gold_rank = next((retrieved.index(g) + 1 for g in gold if g in retrieved), None)
    top = hits[:k]

    if not top or top[0].score < min_top_score:
        ans = Answer(q["question"], f"{ABSTAIN_TOKEN}: gated (top score below {min_top_score})", True,
                     sources=[{"n": i, "pmid": h.doc["pmid"]} for i, h in enumerate(top, 1)], model=client.model)
        gated, error = True, None
    else:
        gated, error = False, None
        try:
            raw = client.complete(build_messages(q["question"], top))
            ans = parse_answer(q["question"], raw, top, model=client.model)
        except Exception as e:  # keep the run going; the failure is a metric too
            ans, error = Answer(q["question"], "", False, sources=[], model=client.model), f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0
    m = ans.metrics()
    return {
        "id": q["id"], "topic": q.get("topic"), "answerable": q["answerable"], "question": q["question"],
        "gold_pmids": gold, "retrieved": retrieved, "gold_rank": gold_rank,
        "recall_at_10": (gold_rank is not None and gold_rank <= 10) if gold else None,
        "gated": gated, "abstained": ans.abstained,
        "abstention_correct": ans.abstained != q["answerable"],
        "gold_cited": bool(set(gold) & set(ans.cited_pmids())) if gold and not ans.abstained else None,
        "text": ans.text, "sentences": [asdict(s) for s in ans.sentences], "sources": ans.sources,
        "issues": ans.issues, "metrics": m, "latency_s": round(latency, 2), "error": error,
        "key_facts": q.get("key_facts", []),
    }


def judge_item(item: dict, judge: ChatClient, passages: dict[str, dict]) -> dict:
    """Adds judge verdicts: per cited sentence support, and key-fact coverage."""
    verdicts = []
    by_n = {s["n"]: s["pmid"] for s in item["sources"]}
    for s in item["sentences"]:
        for n in s["citations"]:
            pmid = by_n.get(n)
            doc = passages.get(pmid) if pmid else None
            if not doc:
                verdicts.append({"sentence": s["text"], "n": n, "verdict": "INVALID"})
                continue
            reply = judge.complete([
                {"role": "system", "content": JUDGE_SUPPORT_PROMPT},
                {"role": "user", "content": f"SENTENCE: {s['text']}\n\nPASSAGE: {doc.get('title', '')}\n{doc.get('abstract', '')}"},
            ], max_tokens=10)
            word = re.sub(r"[^A-Z]", "", reply.upper().split()[0]) if reply.strip() else "UNKNOWN"
            verdicts.append({"sentence": s["text"], "n": n, "pmid": pmid, "verdict": word})
    item["judge_support"] = verdicts

    if item["key_facts"] and not item["abstained"] and item["text"]:
        facts = "\n".join(f"{i}. {f}" for i, f in enumerate(item["key_facts"], 1))
        reply = judge.complete([
            {"role": "system", "content": JUDGE_FACTS_PROMPT},
            {"role": "user", "content": f"ANSWER: {item['text']}\n\nKEY FACTS:\n{facts}"},
        ], max_tokens=60)
        try:
            item["judge_facts"] = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            item["judge_facts"] = {"error": reply[:200]}
    item["judge_model"] = judge.model
    return item


def summarize(items: list[dict]) -> dict:
    ans = [i for i in items if i["answerable"]]
    unans = [i for i in items if not i["answerable"]]
    answered = [i for i in items if not i["abstained"] and not i["error"]]
    support = [v["verdict"] for i in items for v in i.get("judge_support", [])]
    facts_total = sum(len(i["key_facts"]) for i in ans if "judge_facts" in i and "covered" in i["judge_facts"])
    facts_cov = sum(len(set(i["judge_facts"].get("covered", []))) for i in ans if "judge_facts" in i and "covered" in i["judge_facts"])
    lat = [i["latency_s"] for i in items if not i["gated"] and not i["error"]]
    return {
        "items": len(items),
        "errors": sum(bool(i["error"]) for i in items),
        "retrieval_recall_at_10_pct": _pct([i["recall_at_10"] for i in ans if i["recall_at_10"] is not None]),
        "answered_when_answerable_pct": _pct([not i["abstained"] for i in ans]),
        "abstained_when_unanswerable_pct": _pct([i["abstained"] for i in unans]),
        "abstention_correct_pct": _pct([i["abstention_correct"] for i in items]),
        "gold_cited_pct": _pct([i["gold_cited"] for i in ans if i["gold_cited"] is not None]),
        "cited_sentence_rate_pct": _pct([s for i in answered for s in [bool(x["citations"]) for x in i["sentences"]]]),
        "invalid_citations": sum(i["metrics"]["invalid_citations"] for i in answered),
        "citation_precision_pct": _pct([v == "SUPPORTED" for v in support]) if support else None,
        "citation_partial_pct": _pct([v == "PARTIAL" for v in support]) if support else None,
        "key_fact_coverage_pct": round(100 * facts_cov / facts_total, 1) if facts_total else None,
        "latency_p50_s": _quantile(lat, 0.5),
        "latency_p95_s": _quantile(lat, 0.95),
        "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
    }

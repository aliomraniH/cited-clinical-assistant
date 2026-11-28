"""Devil-agent test: a model from another family writes true and subtly wrong claims from our
own evidence, the grounded assistant judges each claim against our sources, and code scores it.

The devil never decides what is true. Every item's label is fixed when it is built:
  supported     a faithful paraphrase (code checks every number in it appears in the source)
  contradicted  one controlled edit (code checks the changed value was in the source and the new
                value was not) or a threshold changed in our criterion table (built by code)
  unsupported   a true claim cited to the wrong source, a made-up PMID, or a claim from the
                untrusted pool (code builds these; the devil is not involved)
Items that fail a check are dropped, never relabelled.

Quarantine: wrong claims carry synthetic_false=True, are written only under data/devil/,
and load_snapshots() refuses such rows, so they can never enter the retrieval index.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field

LABELS = ("supported", "contradicted", "unsupported")
EDIT_MENU = {
    "number": "change one number (a sample size, percentage, duration, dose or count) to a different plausible number",
    "direction": "flip the direction of one finding (improved to worsened, higher to lower, more to less)",
    "population": "swap the population or condition studied for a related one (for example PTSD for GAD, adults for adolescents)",
    "intervention": "swap the treatment or exposure for a related one (for example sertraline for fluoxetine, CBT for EMDR)",
}

DEVIL_PROMPT = """You help test a clinical evidence checker. You get one source sentence from a study.
Write JSON only, no prose, with this shape:
{"paraphrases": ["...", "..."],
 "edits": [{"type": "<edit type>", "statement": "...", "changed_from": "<exact words from the source>", "changed_to": "<the new words>"}]}
Rules:
- "paraphrases": 2 statements that say exactly what the source says in different words. Keep every number. Add nothing.
- "edits": one statement per edit type listed below. Each changes exactly ONE fact and otherwise reads like a faithful summary.
  "changed_from" must be words copied exactly from the source sentence; "changed_to" is what replaced them.
- Make the wrong statements plausible: a reader without the source should not be able to tell.
Edit types:
{menu}"""

GROUNDED_PROMPT = """You check one statement against numbered passages from our trusted sources.
Reply with JSON only: {"verdict": "supported" | "contradicted" | "unsupported", "passage": <number or null>, "quote": "<exact words from that passage, or empty>"}
- supported: a passage states the same thing (if the statement cites a source, that source itself must say it).
- contradicted: a passage states something that conflicts with it (a different number, direction, population or treatment).
- unsupported: no passage says it, or the cited source does not exist or does not say it.
Use only the passages. Quote the words you relied on."""

_NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?")


def numbers(text: str) -> set[str]:
    return {n.rstrip("%").replace(",", "") for n in _NUM.findall(text)}


@dataclass
class Item:
    id: str
    statement: str
    label: str
    kind: str  # paraphrase | edit:<type> | criteria:<type> | wrong_citation | fake_pmid | untrusted
    source_id: str | None  # passage the item was built from (None for untrusted)
    cited: str | None = None  # citation shown with the statement, if any
    truth: str = ""  # the true value or sentence, kept beside every wrong item
    synthetic_false: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- building items ----------

def pick_sentence(passage: str, rng: random.Random) -> str | None:
    """A result-like sentence with at least one number: the kind of fact that gets misquoted."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if 60 <= len(s) <= 400]
    good = [s for s in sents if numbers(s) and re.search(
        r"\b(reduc|improv|increas|decreas|associat|signific|predict|higher|lower|effect|remission|response|prevalen|odds|risk)", s, re.I)]
    return rng.choice(good) if good else None


def check_paraphrase(statement: str, source: str) -> bool:
    """Every number in a 'true' paraphrase must appear in the source."""
    return numbers(statement) <= numbers(source) and len(statement) > 30


def check_edit(edit: dict, source: str) -> bool:
    """The changed words must come from the source, the new words must not, and they must differ."""
    f, t, st = (edit.get("changed_from") or "").strip(), (edit.get("changed_to") or "").strip(), edit.get("statement") or ""
    if not f or not t or f.lower() == t.lower():
        return False
    if f.lower() not in source.lower() or t.lower() in source.lower():
        return False
    return t.lower() in st.lower() and f.lower() not in st.lower()


def parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def devil_items(devil, passage: dict, sentence: str, pack_id: str, edit_types: list[str]) -> tuple[list[Item], list[str]]:
    """Ask the devil for paraphrases and edits of one sentence; keep only items that pass the checks."""
    menu = "\n".join(f"- {t}: {EDIT_MENU[t]}" for t in edit_types)
    raw = devil.complete([{"role": "system", "content": DEVIL_PROMPT.replace("{menu}", menu)},
                          {"role": "user", "content": f"Source sentence:\n{sentence}"}], max_tokens=1200)
    data = parse_json(raw)
    dropped: list[str] = []
    if not data:
        return [], ["devil reply not JSON"]
    items = []
    for j, p in enumerate(data.get("paraphrases", [])[:2], 1):
        if isinstance(p, str) and check_paraphrase(p, sentence):
            items.append(Item(f"{pack_id}-p{j}", p, "supported", "paraphrase", passage["id"], truth=sentence))
        else:
            dropped.append("paraphrase added or changed a number")
    for e in data.get("edits", []):
        et = e.get("type", "")
        if et in EDIT_MENU and check_edit(e, sentence):
            items.append(Item(f"{pack_id}-{et}", e["statement"], "contradicted", f"edit:{et}", passage["id"],
                              truth=sentence, synthetic_false=True,
                              meta={"changed_from": e["changed_from"], "changed_to": e["changed_to"]}))
        else:
            dropped.append(f"edit {et or '?'} failed check")
    return items, dropped


def citation_items(passage: dict, sentence: str, other: dict, pack_id: str, fake_pmid: str) -> list[Item]:
    """Built by code: the true sentence cited to an unrelated source, and to a PMID that does not exist."""
    return [
        Item(f"{pack_id}-wrongcite", sentence, "unsupported", "wrong_citation", passage["id"],
             cited=other.get("cite_label") or f"PMID {other['pmid']}", truth=f"actually from {passage['pmid']}",
             synthetic_false=True, meta={"cited_id": other["id"]}),
        Item(f"{pack_id}-fakepmid", sentence, "unsupported", "fake_pmid", passage["id"], cited=f"PMID {fake_pmid}",
             truth=f"actually from {passage['pmid']}", synthetic_false=True),
    ]


def criteria_items(crit: dict) -> list[Item]:
    """True and corrupted threshold statements from our criterion table, built by code alone."""
    out = []
    for cond, c in crit.items():
        label = c["label"]
        d = c.get("duration", {})
        if "min_months" in d:
            n = d["min_months"]
            # Say it the way the table does: PTSD needs MORE than 1 month, GAD AT LEAST 6.
            bound = "more than" if "more than" in c["criteria"][d["criterion"]] else "at least"
            true = f"For {label}, the symptoms must last {bound} {n} month{'s' if n != 1 else ''} (criterion {d['criterion']})."
            wrong_n = {6: 2, 1: 3, 3: 6}.get(n, n * 2)
            out.append(Item(f"crit-{cond}-dur", true, "supported", "criteria:duration", f"criteria-{cond}"))
            out.append(Item(f"crit-{cond}-dur-x", true.replace(f"{n} month{'s' if n != 1 else ''}",
                                                                f"{wrong_n} month{'s' if wrong_n != 1 else ''}"),
                            "contradicted", "criteria:duration", f"criteria-{cond}", truth=true, synthetic_false=True))
        if "min_weeks" in d:
            n = d["min_weeks"]
            true = f"For {label}, the symptoms must be present during the same {n}-week period (criterion {d['criterion']})."
            out.append(Item(f"crit-{cond}-dur", true, "supported", "criteria:duration", f"criteria-{cond}"))
            out.append(Item(f"crit-{cond}-dur-x", true.replace(f"{n}-week", f"{n + 2}-week"), "contradicted",
                            "criteria:duration", f"criteria-{cond}", truth=true, synthetic_false=True))
        cnt = c.get("count")
        if cnt and "min" in cnt:
            true = f"For {label}, at least {cnt['min']} of {cnt['of']} listed signs are needed (criterion {cnt['criterion']})."
            out.append(Item(f"crit-{cond}-cnt", true, "supported", "criteria:count", f"criteria-{cond}"))
            out.append(Item(f"crit-{cond}-cnt-x", true.replace(f"at least {cnt['min']} of", f"at least {cnt['min'] + 2} of"),
                            "contradicted", "criteria:count", f"criteria-{cond}", truth=true, synthetic_false=True))
    return out


def criteria_rows(crit: dict) -> list[dict]:
    """Our criterion table as passages, so the grounded model can check threshold claims against it."""
    rows = []
    for cond, c in crit.items():
        text = " ".join(f"Criterion {k}: {v}" for k, v in c["criteria"].items())
        cnt = c.get("count")
        if cnt and cnt.get("items"):
            text += f" Signs counted for criterion {cnt['criterion']}: " + "; ".join(cnt["items"]) + "."
        rows.append({"id": f"criteria-{cond}", "pmid": f"criteria-{cond}", "cite_label": f"Our criterion table: {c['label']}",
                     "title": f"Criterion paraphrase table: {c['label']}", "abstract": text, "journal": "project table",
                     "pub_date": "2026", "source": "criteria_table", "url": "config/criteria.json"})
    return rows


def untrusted_items(pool: list[dict]) -> list[Item]:
    return [Item(f"untr-{p['id']}", p["statement"], "unsupported", "untrusted", None, truth=p.get("why", ""),
                 synthetic_false=True, meta={"origin": p.get("origin", "")}) for p in pool]


# ---------- grounded check and scoring ----------

def grounded_verdict(model, retriever, item: Item, by_id: dict[str, dict], by_cite: dict[str, list[dict]], k: int = 6) -> dict:
    hits = [h.doc for h in retriever.search(item.statement, k=k)]
    cited_docs: list[dict] = []
    if item.cited:
        key = item.cited.removeprefix("PMID ").split(",")[0].strip()
        # A cited source is judged on its own words: its passages that best match the statement go first.
        words = set(_norm(item.statement).split())
        cited_docs = sorted(by_cite.get(key, []), key=lambda d: -len(words & set(_norm(d["abstract"]).split())))
    shown = cited_docs[:2] + [d for d in hits if d not in cited_docs[:2]]
    shown = shown[:k]
    blocks = [f"[{i}] {d.get('cite_label') or 'PMID ' + d['pmid']}\n{d['abstract']}" for i, d in enumerate(shown, 1)]
    stmt = item.statement + (f" (cited: {item.cited})" if item.cited else "")
    if item.cited and not cited_docs:
        stmt += "\nNote: the cited source is not in our sources."
    msgs = [{"role": "system", "content": GROUNDED_PROMPT},
            {"role": "user", "content": "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nStatement: {stmt}"}]
    raw = ""
    for budget in (2000, 6000):  # reasoning models can spend the whole budget thinking and return nothing
        raw = model.complete(msgs, max_tokens=budget)
        if raw.strip():
            break
    data = parse_json(raw) or {}
    if not data:  # malformed JSON (e.g. an unescaped quote): the verdict comes first, so read it directly
        m = re.search(r'"verdict"\s*:\s*"(\w+)"', raw)
        n = re.search(r'"passage"\s*:\s*(\d+)', raw)
        data = {"verdict": m.group(1) if m else None, "passage": int(n.group(1)) if n else None}
    verdict = data.get("verdict") if data.get("verdict") in LABELS else "unparsed"
    n = data.get("passage")
    quote_ok = None
    if isinstance(n, int) and 0 < n <= len(shown) and data.get("quote"):
        quote_ok = _norm(data["quote"]) in _norm(shown[n - 1]["abstract"])
    retrieved_source = item.source_id is None or any(d["id"] == item.source_id for d in shown)
    return {"verdict": verdict, "passage": n, "quote_found": quote_ok,
            "shown": [d["id"] for d in shown], "source_shown": retrieved_source, "raw": raw[:400]}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def score(item: Item, v: dict) -> dict:
    """Exact label match, except an untrusted claim counts as handled if it is called either
    contradicted or unsupported: our sources may refute it outright or simply not cover it."""
    if item.kind == "untrusted":
        correct = v["verdict"] in ("contradicted", "unsupported")
    else:
        correct = v["verdict"] == item.label
    if correct:
        miss = None
    elif v["verdict"] == "unparsed":
        miss = "format"
    elif not v["source_shown"]:
        miss = "retrieval"
    else:
        miss = "reasoning"
    return {"correct": correct, "binary_correct": (v["verdict"] == "supported") == (item.label == "supported"),
            "miss": miss, "grounded": bool(v.get("quote_found")) if v["verdict"] != "unsupported" else None}


def summarize(results: list[dict]) -> dict:
    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None
    out = {"items": len(results), "accuracy": pct([r["score"]["correct"] for r in results]),
           "binary_accuracy": pct([r["score"]["binary_correct"] for r in results]),
           "wrong_accepted_rate": pct([r["verdict"]["verdict"] == "supported" for r in results if r["label"] != "supported"]),
           "true_rejected_rate": pct([r["verdict"]["verdict"] != "supported" for r in results if r["label"] == "supported"]),
           "quote_grounded": pct([r["verdict"]["quote_found"] for r in results if r["verdict"]["quote_found"] is not None]),
           "misses": {}, "by_kind": {}}
    for r in results:
        m = r["score"]["miss"]
        if m:
            out["misses"][m] = out["misses"].get(m, 0) + 1
        k = r["kind"].split(":")[0] if r["kind"].startswith("criteria") else r["kind"]
        b = out["by_kind"].setdefault(k, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["score"]["correct"])
    for b in out["by_kind"].values():
        b["accuracy"] = round(100 * b["correct"] / b["n"], 1)
    return out

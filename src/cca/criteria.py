"""Fit / doesn't-fit cases against our criterion paraphrase table (config/criteria.json).

Each case is a short synthetic vignette with structured facts. The label (meets or misses)
and the criterion that fails are set when the case is written, then re-derived by code from
the facts and the thresholds in the table, so a typo in a case is caught instead of trusted.
The explanation ("why it fits / why not") is built from the table, never from a model.
Criterion ids are DSM-5-TR letters; the wording is ours.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRITERIA_PATH = ROOT / "config" / "criteria.json"


def load_criteria(path: Path = CRITERIA_PATH) -> dict:
    return json.loads(Path(path).read_text())["conditions"]


@dataclass
class Case:
    id: str
    condition: str
    kind: str  # meets | duration | count | required | stressor | exclusion | better_explained | time
    text: str
    facts: dict
    label: str  # meets | misses
    failed: str | None = None  # criterion id, e.g. "gad.A"
    explained_by: str | None = None  # condition that explains it better, for differentials
    evidence_query: str = ""
    why: str = ""
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def derive(case: Case, crit: dict) -> tuple[str, str | None]:
    """Label and failing criterion from the facts alone, using the thresholds in the table."""
    c = crit[case.condition]
    f = case.facts
    dur = c.get("duration", {})
    if "min_months" in dur and f.get("months") is not None and f["months"] < dur["min_months"]:
        return "misses", f"{case.condition}.{dur['criterion']}"
    if "min_weeks" in dur and f.get("weeks") is not None and f["weeks"] < dur["min_weeks"]:
        return "misses", f"{case.condition}.{dur['criterion']}"
    cnt = c.get("count")
    if cnt and f.get("count_items") is not None:
        items = f["count_items"]
        if len(items) < cnt["min"]:
            return "misses", f"{case.condition}.{cnt['criterion']}"
        req = cnt.get("required_one_of")
        if req and not set(req) & set(items):
            return "misses", f"{case.condition}.{cnt['criterion']}"
    if case.condition == "ptsd":
        if f.get("qualifying_event") is False:
            return "misses", "ptsd.A"
        for key, crit_id, need in (("intrusions", "B", 1), ("avoidance", "C", 1), ("cognition_mood", "D", 2), ("arousal", "E", 2)):
            if key in f and len(f[key]) < need:
                return "misses", f"ptsd.{crit_id}"
    t = c.get("time")
    if t and f.get("hours_per_day") is not None and f["hours_per_day"] < t["min_hours_per_day"] and not f.get("distress"):
        return "misses", f"{case.condition}.{t['criterion']}"
    if f.get("past_mania"):
        return "misses", "depression.E"
    if f.get("substance_cause"):
        return "misses", f"{case.condition}.{_exclusion_letter(case.condition)}"
    if f.get("better_explained_by"):
        return "misses", f"{case.condition}.{_other_disorder_letter(case.condition)}"
    return "meets", None


def _exclusion_letter(cond: str) -> str:
    return {"gad": "E", "depression": "C", "ptsd": "H", "ocd": "C"}[cond]


def _other_disorder_letter(cond: str) -> str:
    return {"gad": "F", "ocd": "D", "depression": "D", "adhd": "E"}[cond]


def explain(case: Case, crit: dict) -> str:
    c = crit[case.condition]
    if case.label == "meets":
        letters = ", ".join(c["criteria"])
        return f"Fits {c['label']}: the facts satisfy criteria {letters} in our table."
    letter = case.failed.split(".")[1]
    rule = c["criteria"][letter]
    f = case.facts
    detail = {
        "duration": f"stated duration is {f.get('months', f.get('weeks'))} {'months' if 'months' in f else 'weeks'}",
        "count": f"only {len(f.get('count_items', []))} of the needed signs are described: {', '.join(f.get('count_items', []))}",
        "required": "neither low mood nor loss of interest is described",
        "stressor": f"the event ({f.get('event', '')}) is not a qualifying exposure",
        "exclusion": f"the symptoms are tied to {f.get('substance_cause', '')}",
        "better_explained": f"the picture is better explained by {crit.get(case.explained_by or '', {}).get('label', case.explained_by)}",
        "time": f"rituals take about {f.get('hours_per_day')} hours a day without marked distress",
        "mania": "there was a past manic episode",
        "symptom_group": "a required symptom group has too few symptoms",
    }.get(case.kind, "")
    return f"Does not fit {c['label']}: criterion {letter} fails ({rule}) Here {detail}."


def validate(cases: list[Case], crit: dict) -> list[str]:
    """Problems where the written label disagrees with the label derived from the facts."""
    problems = []
    for c in cases:
        label, failed = derive(c, crit)
        if (label, failed) != (c.label, c.failed):
            problems.append(f"{c.id}: written {c.label}/{c.failed}, derived {label}/{failed}")
    return problems

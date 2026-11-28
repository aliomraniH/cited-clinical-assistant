"""Write eval/criteria_cases.jsonl: synthetic vignettes that fit or miss our criterion table.

Every case states its facts; code re-derives the label from the facts and the thresholds in
config/criteria.json and refuses to write if they disagree. With --corpus, each case also gets
the top passages for its evidence query (PMIDs, trial ids, guideline pages) from our sources.

Usage:
  python scripts/make_criteria_cases.py
  python scripts/make_criteria_cases.py --corpus data/eval_corpus data/sources
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cca.criteria import Case, explain, load_criteria, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAD_C = ["restless or keyed up", "tiring easily", "trouble concentrating or mind going blank", "irritability",
         "muscle tension", "poor or unrestful sleep"]

CASES = [
    # ---- GAD ----
    Case("c01", "gad", "meets", "A 34-year-old accountant has worried most days for about 8 months about money, her children's health and "
         "small mistakes at work. She says she cannot switch the worry off. She feels keyed up, has tight shoulders, sleeps badly and "
         "her mind goes blank in meetings. She has started missing deadlines.",
         {"months": 8, "count_items": ["restless or keyed up", "muscle tension", "poor or unrestful sleep",
                                       "trouble concentrating or mind going blank"], "distress": True},
         "meets", evidence_query="generalized anxiety disorder diagnosis worry duration somatic symptoms"),
    Case("c02", "gad", "duration", "A 29-year-old teacher has worried about many things, from bills to her parents' health, since a house move "
         "3 months ago. The worry is hard to control and she has muscle tension, poor sleep and irritability.",
         {"months": 3, "count_items": ["muscle tension", "poor or unrestful sleep", "irritability"], "distress": True},
         "misses", "gad.A", evidence_query="generalized anxiety disorder six month duration adjustment disorder anxiety"),
    Case("c03", "gad", "count", "A 41-year-old man has worried most days for over a year about work and family. He finds it hard to stop, "
         "but apart from feeling tired easily and some irritability he reports no other physical or mental signs.",
         {"months": 14, "count_items": ["tiring easily", "irritability"], "distress": True},
         "misses", "gad.C", evidence_query="subthreshold generalized anxiety symptoms fewer associated symptoms"),
    Case("c04", "gad", "better_explained", "A 36-year-old woman worries most days for 7 months, but nearly all of the worry is about being "
         "attacked again after an assault: she avoids the street where it happened, wakes from nightmares about it and startles at "
         "loud noises. She is tense, irritable and sleeps poorly.",
         {"months": 7, "count_items": ["muscle tension", "irritability", "poor or unrestful sleep"], "distress": True,
          "better_explained_by": "ptsd"},
         "misses", "gad.F", explained_by="ptsd", evidence_query="differentiating PTSD from generalized anxiety disorder worry trauma reminders"),
    Case("c05", "gad", "exclusion", "A 24-year-old student has felt anxious, restless and unable to sleep for 7 months, all of it starting when "
         "he began taking high doses of stimulants bought online to study; he worries about many things and cannot stop.",
         {"months": 7, "count_items": ["restless or keyed up", "poor or unrestful sleep", "irritability"], "distress": True,
          "substance_cause": "stimulant use"},
         "misses", "gad.E", evidence_query="substance-induced anxiety disorder stimulant use"),
    # ---- Depression ----
    Case("c06", "depression", "meets", "For the past 3 weeks a 47-year-old man has felt low most of the day, lost interest in fishing and his "
         "grandchildren, wakes at 4 a.m., has no energy and feels he is a burden. His wife says he moves and speaks slowly.",
         {"weeks": 3, "count_items": ["low mood most of the day", "loss of interest or pleasure", "sleeping too little or too much",
                                      "fatigue or loss of energy", "feeling worthless or excessively guilty", "visibly slowed or agitated"],
          "distress": True},
         "meets", evidence_query="major depressive disorder diagnosis anhedonia psychomotor retardation early waking"),
    Case("c07", "depression", "duration", "A 31-year-old woman has felt low, tired and unable to enjoy anything for 10 days after a busy month. "
         "She sleeps more than usual, cannot concentrate and feels guilty about letting colleagues down.",
         {"weeks": 1.4, "count_items": ["low mood most of the day", "loss of interest or pleasure", "fatigue or loss of energy",
                                        "sleeping too little or too much", "trouble thinking, concentrating or deciding",
                                        "feeling worthless or excessively guilty"], "distress": True},
         "misses", "depression.A", evidence_query="brief depressive episode duration two weeks"),
    Case("c08", "depression", "count", "For a month a 52-year-old woman has felt low most days and has had poor sleep and little energy, "
         "but her appetite, interest in her hobbies, concentration and sense of self-worth are unchanged.",
         {"weeks": 4, "count_items": ["low mood most of the day", "sleeping too little or too much", "fatigue or loss of energy"],
          "distress": True},
         "misses", "depression.A", evidence_query="subthreshold depression minor depression symptom count"),
    Case("c09", "depression", "required", "For 3 weeks a 60-year-old man has slept poorly, lost his appetite, felt exhausted, struggled to "
         "concentrate and felt restless. He says his mood is fine and he still enjoys his garden and friends.",
         {"weeks": 3, "count_items": ["sleeping too little or too much", "clear change in appetite or weight", "fatigue or loss of energy",
                                      "trouble thinking, concentrating or deciding", "visibly slowed or agitated"], "distress": True},
         "misses", "depression.A", evidence_query="somatic symptoms without depressed mood or anhedonia differential diagnosis"),
    Case("c10", "depression", "mania", "A 27-year-old woman has had low mood, no interest in anything, poor sleep, low energy and guilt for 3 "
         "weeks. Last year she had a week of very little sleep, racing ideas and reckless spending that ended in hospital.",
         {"weeks": 3, "count_items": ["low mood most of the day", "loss of interest or pleasure", "sleeping too little or too much",
                                      "fatigue or loss of energy", "feeling worthless or excessively guilty"], "distress": True,
          "past_mania": True},
         "misses", "depression.E", explained_by="bipolar disorder", evidence_query="bipolar depression versus unipolar depression past manic episode"),
    # ---- PTSD ----
    Case("c11", "ptsd", "meets", "Five months after a serious car crash in which his friend died, a 38-year-old man has unwanted memories "
         "and nightmares of the crash, avoids driving and the road where it happened, blames himself, feels cut off from his family, "
         "is constantly on guard and jumps at sudden noises.",
         {"months": 5, "qualifying_event": True, "intrusions": ["memories", "nightmares"], "avoidance": ["places", "driving"],
          "cognition_mood": ["self-blame", "detachment"], "arousal": ["hypervigilance", "startle"], "distress": True},
         "meets", evidence_query="PTSD after motor vehicle accident intrusion avoidance hyperarousal"),
    Case("c12", "ptsd", "duration", "Two weeks after being held at knifepoint, a 25-year-old woman has flashbacks, avoids the bus stop, "
         "feels numb, cannot sleep and startles easily.",
         {"months": 0.5, "qualifying_event": True, "intrusions": ["flashbacks"], "avoidance": ["places"],
          "cognition_mood": ["numbing", "negative beliefs"], "arousal": ["sleep", "startle"], "distress": True},
         "misses", "ptsd.F", explained_by="acute stress disorder", evidence_query="acute stress disorder versus PTSD one month duration"),
    Case("c13", "ptsd", "stressor", "Since losing his job and going through a divorce 6 months ago, a 45-year-old man has replayed the "
         "events, avoids old colleagues, feels hopeless, sleeps badly and is irritable.",
         {"months": 6, "qualifying_event": False, "event": "job loss and divorce", "intrusions": ["memories"], "avoidance": ["people"],
          "cognition_mood": ["hopeless", "detachment"], "arousal": ["sleep", "irritability"], "distress": True},
         "misses", "ptsd.A", explained_by="adjustment disorder", evidence_query="criterion A traumatic event non-traumatic stressors adjustment disorder"),
    Case("c14", "ptsd", "symptom_group", "Four months after an assault, a 30-year-old woman has nightmares about it, avoids the bar where it "
         "happened, feels persistent shame and fear, and has trouble sleeping, but is not on guard, irritable, reckless or jumpy.",
         {"months": 4, "qualifying_event": True, "intrusions": ["nightmares"], "avoidance": ["places"],
          "cognition_mood": ["shame", "fear"], "arousal": ["sleep"], "distress": True},
         "misses", "ptsd.E", evidence_query="subthreshold partial PTSD hyperarousal symptoms"),
    # ---- OCD ----
    Case("c15", "ocd", "meets", "A 22-year-old student has unwanted thoughts that he has germs that will harm his family. He knows the "
         "thoughts are excessive but washes his hands until they bleed, about 3 hours a day, and is failing his course.",
         {"hours_per_day": 3, "distress": True}, "meets",
         evidence_query="obsessive-compulsive disorder contamination obsessions washing compulsions time consuming"),
    Case("c16", "ocd", "time", "A 40-year-old woman likes her desk ordered and re-checks the door lock once before leaving, about 10 minutes "
         "a day. It does not bother her or get in the way of anything.",
         {"hours_per_day": 0.2, "distress": False}, "misses", "ocd.B",
         evidence_query="obsessive-compulsive symptoms subclinical checking normal population"),
    Case("c17", "ocd", "better_explained", "A 19-year-old woman spends over 2 hours a day on rituals: cutting food into exact pieces, "
         "checking her body in mirrors and exercising until a set count. The rituals are about controlling her shape and weight.",
         {"hours_per_day": 2, "distress": True, "better_explained_by": "eating_disorder"},
         "misses", "ocd.D", explained_by="eating_disorder",
         evidence_query="eating disorder rituals versus obsessive-compulsive disorder comorbidity anorexia"),
    Case("c18", "ocd", "better_explained", "A 33-year-old firefighter has repeated intrusive images of a child he could not save in a fire "
         "a year ago, checks the stove many times, avoids the neighborhood and has nightmares. The images are memories of the real event.",
         {"hours_per_day": 1.5, "distress": True, "better_explained_by": "ptsd"},
         "misses", "ocd.D", explained_by="ptsd", evidence_query="intrusive memories PTSD versus obsessions OCD differential"),
    # ---- overlap: comorbid case that meets both ----
    Case("c19", "depression", "meets", "A 28-year-old woman with long-standing contamination fears and 2 hours of daily washing has, for the "
         "past month, also felt low most of the day, lost interest in friends, slept poorly, felt worthless and had little energy.",
         {"weeks": 4, "count_items": ["low mood most of the day", "loss of interest or pleasure", "sleeping too little or too much",
                                      "feeling worthless or excessively guilty", "fatigue or loss of energy"], "distress": True},
         "meets", evidence_query="comorbid depression in obsessive-compulsive disorder"),
    Case("c20", "gad", "exclusion", "A 50-year-old man has had daily worry, tension and poor sleep for 9 months; it all began when his "
         "drinking rose to most of a bottle of spirits a day and eases during the first drinks each evening.",
         {"months": 9, "count_items": ["muscle tension", "poor or unrestful sleep", "restless or keyed up"], "distress": True,
          "substance_cause": "heavy alcohol use and withdrawal"},
         "misses", "gad.E", explained_by="substance_use", evidence_query="alcohol use disorder anxiety withdrawal substance-induced"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "criteria_cases.jsonl")
    a = ap.parse_args()
    crit = load_criteria()
    problems = validate(CASES, crit)
    if problems:
        print("Label check failed:\n" + "\n".join(problems), file=sys.stderr)
        return 2
    retriever = None
    if a.corpus:
        from cca.index import BM25, load_snapshots
        retriever = BM25(load_snapshots(a.corpus))
    with open(a.out, "w") as f:
        for c in CASES:
            c.why = explain(c, crit)
            if retriever:
                c.evidence = [{"cite": h.doc["pmid"], "label": h.doc.get("cite_label") or f"PMID {h.doc['pmid']}",
                               "source": h.doc.get("source"), "url": h.doc.get("url"), "score": h.score}
                              for h in retriever.search(c.evidence_query, k=3)]
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"{len(CASES)} cases, {sum(c.label == 'meets' for c in CASES)} meet, labels re-derived from facts: OK -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

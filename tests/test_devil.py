import json

from cca.criteria import load_criteria, validate
from cca.devil import (Item, check_edit, check_paraphrase, citation_items, criteria_items, devil_items,
                       grounded_verdict, score)
from cca.index import BM25

SRC = "In 240 adults with GAD, sertraline reduced worry scores by 35% compared with placebo after 8 weeks."


class FakeModel:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, **kw):
        return self.reply


def test_paraphrase_check_blocks_new_numbers():
    assert check_paraphrase("Sertraline cut worry by 35% versus placebo over 8 weeks in 240 adults with GAD.", SRC)
    assert not check_paraphrase("Sertraline cut worry by 45% versus placebo over 8 weeks in 240 adults.", SRC)


def test_edit_check():
    ok = {"changed_from": "35%", "changed_to": "55%", "statement": "Sertraline reduced worry by 55% in 240 adults with GAD."}
    assert check_edit(ok, SRC)
    assert not check_edit({**ok, "changed_from": "40%"}, SRC)  # not in source
    assert not check_edit({**ok, "changed_to": "8 weeks"}, SRC)  # new value already in source
    assert not check_edit({**ok, "statement": "Sertraline reduced worry by 35% in 240 adults."}, SRC)


def test_devil_items_keep_only_checked():
    reply = json.dumps({"paraphrases": ["Among 240 adults with GAD, sertraline lowered worry 35% more than placebo at 8 weeks.",
                                        "Sertraline lowered worry by 50% in 240 adults."],
                        "edits": [{"type": "number", "statement": "In 240 adults with GAD, sertraline reduced worry scores by 15% compared with placebo after 8 weeks.",
                                   "changed_from": "35%", "changed_to": "15%"},
                                  {"type": "direction", "statement": "Sertraline increased worry.", "changed_from": "reduced", "changed_to": "increased"},
                                  {"type": "population", "statement": "bad", "changed_from": "children", "changed_to": "adults"}]})
    items, dropped = devil_items(FakeModel(reply), {"id": "d1", "pmid": "1"}, SRC, "k1", ["number", "direction", "population"])
    kinds = sorted((i.kind, i.label) for i in items)
    assert kinds == [("edit:direction", "contradicted"), ("edit:number", "contradicted"), ("paraphrase", "supported")]
    assert all(i.synthetic_false for i in items if i.label != "supported")
    assert len(dropped) == 2


def test_citation_and_criteria_items_are_code_built():
    its = citation_items({"id": "d1", "pmid": "1"}, SRC, {"id": "d2", "pmid": "2"}, "k1", "912345678")
    assert [i.label for i in its] == ["unsupported", "unsupported"] and its[1].cited == "PMID 912345678"
    crit = load_criteria()
    ci = criteria_items(crit)
    wrong = [i for i in ci if i.label == "contradicted"]
    assert wrong and all(i.truth and i.truth != i.statement for i in wrong)
    gad = next(i for i in ci if i.id == "crit-gad-dur-x")
    assert "2 months" in gad.statement and "6 months" in gad.truth


def test_grounded_verdict_and_score():
    docs = [{"id": "d1", "pmid": "1", "abstract": SRC}, {"id": "d2", "pmid": "2", "abstract": "Exercise helped sleep in older adults."}]
    item = Item("k1-number", SRC.replace("35%", "15%"), "contradicted", "edit:number", "d1")
    model = FakeModel('{"verdict": "contradicted", "passage": 1, "quote": "reduced worry scores by 35%"}')
    v = grounded_verdict(model, BM25(docs), item, {d["id"]: d for d in docs}, {"1": [docs[0]], "2": [docs[1]]})
    assert v["verdict"] == "contradicted" and v["quote_found"] and v["source_shown"]
    assert score(item, v)["correct"]
    v2 = {**v, "verdict": "supported"}
    assert score(item, v2)["miss"] == "reasoning"


def test_criteria_cases_labels_match_facts():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("mk", Path(__file__).parents[1] / "scripts" / "make_criteria_cases.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    assert validate(mk.CASES, load_criteria()) == []
    assert {c.label for c in mk.CASES} == {"meets", "misses"}

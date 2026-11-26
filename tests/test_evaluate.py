from cca.answer import ChatClient
from cca.evaluate import judge_item, run_item, summarize
from cca.index import BM25

DOCS = [
    {"pmid": "1", "title": "Prolonged exposure therapy for PTSD in veterans",
     "abstract": "In 200 veterans, prolonged exposure reduced PTSD symptom scores versus waitlist.", "pub_date": "2026"},
    {"pmid": "2", "title": "Exposure and response prevention for OCD",
     "abstract": "ERP reduced Y-BOCS scores in adolescents with obsessive-compulsive disorder.", "pub_date": "2026"},
]
Q_ANS = {"id": "a", "answerable": True, "question": "prolonged exposure PTSD veterans", "gold_pmids": ["1"],
         "key_facts": ["reduced PTSD symptoms", "200 veterans"]}
Q_UN = {"id": "u", "answerable": False, "question": "zebrafish fin regeneration", "gold_pmids": []}


class Scripted(ChatClient):
    def __init__(self, replies, model="test/model"):
        super().__init__(model, base_url="http://unused")
        self.replies = list(replies)

    def complete(self, messages, **kw):
        return self.replies.pop(0)


def test_run_and_summarize():
    bm = BM25(DOCS)
    client = Scripted(["Prolonged exposure reduced PTSD symptoms in 200 veterans [1]."])
    a = run_item(Q_ANS, bm, client, min_top_score=1.0)
    u = run_item(Q_UN, bm, client, min_top_score=1.0)
    assert a["gold_rank"] == 1 and a["gold_cited"] and a["abstention_correct"]
    assert u["gated"] and u["abstained"] and u["abstention_correct"]
    s = summarize([a, u])
    assert s["retrieval_recall_at_10_pct"] == 100.0
    assert s["abstention_correct_pct"] == 100.0
    assert s["gold_cited_pct"] == 100.0 and s["cited_sentence_rate_pct"] == 100.0


def test_model_error_is_recorded_not_raised():
    class Boom(ChatClient):
        def complete(self, messages, **kw):
            raise TimeoutError("slow")
    item = run_item(Q_ANS, BM25(DOCS), Boom("x/y", base_url="http://unused"), min_top_score=1.0)
    assert item["error"].startswith("TimeoutError")
    assert summarize([item])["errors"] == 1


def test_judge_verdicts_and_facts():
    bm = BM25(DOCS)
    item = run_item(Q_ANS, bm, Scripted(["PE reduced PTSD symptoms [1]. It also cured OCD [2]."]), min_top_score=1.0)
    judge = Scripted(["SUPPORTED", "UNSUPPORTED.", '{"covered": [1], "contradicted": []}'], model="judge/other")
    item = judge_item(item, judge, {d["pmid"]: d for d in DOCS})
    assert [v["verdict"] for v in item["judge_support"]] == ["SUPPORTED", "UNSUPPORTED"]
    assert item["judge_facts"]["covered"] == [1]
    s = summarize([item])
    assert s["citation_precision_pct"] == 50.0 and s["key_fact_coverage_pct"] == 50.0

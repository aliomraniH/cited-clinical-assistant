from cca.answer import ABSTAIN_TOKEN, ChatClient, answer, build_messages, parse_answer
from cca.index import BM25, Hit, tokenize

DOCS = [
    {"pmid": "1", "title": "Prolonged exposure therapy for PTSD in veterans",
     "abstract": "In 200 veterans, prolonged exposure reduced PTSD symptom scores versus waitlist.", "pub_date": "2026-11-01"},
    {"pmid": "2", "title": "Exposure and response prevention for OCD",
     "abstract": "ERP reduced Y-BOCS scores in adolescents with obsessive-compulsive disorder.", "pub_date": "2026-10-01"},
    {"pmid": "3", "title": "Sertraline in major depression",
     "abstract": "Sertraline improved depressive symptoms over 8 weeks.", "pub_date": "2025"},
]


class FakeClient(ChatClient):
    def __init__(self, reply):
        super().__init__("fake-model", base_url="http://unused")
        self.reply, self.calls = reply, 0

    def complete(self, messages, **kw):
        self.calls += 1
        return self.reply


def test_tokenize_drops_stopwords():
    assert tokenize("The effect of CBT on post-traumatic stress") == ["effect", "cbt", "post-traumatic", "stress"]


def test_bm25_ranks_relevant_first():
    hits = BM25(DOCS).search("exposure therapy for PTSD veterans", k=3)
    assert hits[0].doc["pmid"] == "1"
    assert all(h.score > 0 for h in hits)
    assert BM25(DOCS).search("zebrafish", k=3) == []


def test_prompt_numbers_passages():
    msgs = build_messages("q?", [Hit(DOCS[0], 5.0), Hit(DOCS[1], 3.0)])
    assert "[1] PMID 1" in msgs[1]["content"] and "[2] PMID 2" in msgs[1]["content"]
    assert ABSTAIN_TOKEN in msgs[0]["content"]


def test_parse_checks_citations():
    hits = [Hit(DOCS[0], 5.0), Hit(DOCS[1], 3.0)]
    raw = ("<think>scratch</think>Prolonged exposure reduced PTSD symptoms in 200 veterans [1]. "
           "ERP helped adolescents with OCD [2][1]. It is widely used. Another claim [7].")
    ans = parse_answer("q", raw, hits)
    assert not ans.abstained
    assert [s.citations for s in ans.sentences] == [[1], [2, 1], [], [7]]
    m = ans.metrics()
    assert m["sentences"] == 4 and m["cited_sentence_rate"] == 0.75 and m["invalid_citations"] == 1
    assert ans.cited_pmids() == ["1", "2"]
    assert len(ans.issues) == 2


def test_comma_citations():
    ans = parse_answer("q", "Both helped [1, 2].", [Hit(DOCS[0], 5.0), Hit(DOCS[1], 3.0)])
    assert ans.sentences[0].citations == [1, 2]


def test_model_abstention_is_recognized():
    ans = parse_answer("q", f"{ABSTAIN_TOKEN}: passages do not cover children.", [Hit(DOCS[0], 5.0)])
    assert ans.abstained and ans.metrics()["sentences"] == 0


def test_weak_retrieval_abstains_without_model_call():
    client = FakeClient("should not be used")
    ans = answer("zebrafish genetics", BM25(DOCS), client)
    assert ans.abstained and client.calls == 0


def test_end_to_end_with_fake_model():
    client = FakeClient("Prolonged exposure reduced PTSD symptoms in veterans [1].")
    ans = answer("prolonged exposure PTSD veterans", BM25(DOCS), client, min_top_score=1.0)
    assert client.calls == 1 and ans.cited_pmids() == ["1"] and not ans.issues

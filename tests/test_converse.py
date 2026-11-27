"""Conversational search with a scripted fake model: interpretation parsing and merging, the clarify
gate, confirm/drop/correct, the feedback signals, end-of-session scoring and the phrasing memory."""

import json
import os

from cca.converse import Conversation
from cca.index import BM25
from cca.interpret import Interpretation, interpret, merge_samples, parse_interpretations
from cca.tracking import PhrasingMemory, Session, deidentify, score_session

DOCS = [
    {"pmid": "1", "title": "Exposure and response prevention for checking compulsions", "pub_date": "2025",
     "abstract": "ERP reduced checking compulsions and obsessive symptoms in adults with OCD."},
    {"pmid": "2", "title": "Insomnia after trauma: CBT-I in PTSD", "pub_date": "2025",
     "abstract": "Cognitive behavioral therapy for insomnia improved sleep in adults with posttraumatic stress."},
    {"pmid": "3", "title": "Alcohol use as coping after trauma", "pub_date": "2025",
     "abstract": "Drinking alcohol to cope with sleep problems and trauma symptoms predicted alcohol use disorder."},
    {"pmid": "4", "title": "Intrusive memories in PTSD: trauma-focused therapy", "pub_date": "2025",
     "abstract": "Trauma-focused therapy reduced intrusive memories and flashbacks in posttraumatic stress disorder."},
    {"pmid": "5", "title": "Intrusive thoughts in OCD: exposure therapy", "pub_date": "2025",
     "abstract": "Exposure and response prevention reduced intrusive obsessional thoughts in obsessive compulsive disorder."},
]

MULTI = {"interpretations": [
    {"intent_type": "differential", "paraphrase": "Is repeated checking after a trauma linked to it",
     "key_sentences": ["i keep checking the stove"], "terms": ["checking compulsions", "obsessive"], "confidence": 0.9},
    {"intent_type": "symptom_question", "paraphrase": "Sleep problems after trauma",
     "key_sentences": ["i barely sleep"], "terms": ["insomnia", "posttraumatic"], "confidence": 0.8},
    {"intent_type": "symptom_question", "paraphrase": "Drinking alcohol to cope with sleep after trauma",
     "key_sentences": ["drinking more"], "terms": ["alcohol use", "coping"], "confidence": 0.7},
]}
AMBIG = {"interpretations": [
    {"intent_type": "treatment_question", "paraphrase": "Treatment for intrusive memories after trauma",
     "key_sentences": ["intrusive thoughts"], "terms": ["intrusive memories", "flashbacks", "posttraumatic"], "confidence": 0.6},
    {"intent_type": "treatment_question", "paraphrase": "Treatment for obsessional intrusive thoughts",
     "key_sentences": ["intrusive thoughts"], "terms": ["obsessional", "obsessive compulsive", "exposure"], "confidence": 0.55,
     "alternative": True},
]}


class FakeChat:
    model = "fake"

    def __init__(self, interp, after_correction=None):
        self.interp, self.after = interp, after_correction
        self.calls = []

    def complete(self, messages, temperature=0.0, max_tokens=700):
        self.calls.append(messages)
        sysmsg, user = messages[0]["content"], messages[-1]["content"]
        if "Split the message" in sysmsg:
            if "Earlier readings:" in user and self.after:  # correction or refinement
                return json.dumps(self.after)
            return "<think>hmm</think>```json\n" + json.dumps(self.interp) + "\n```"
        return "Studies in the passages report improvement with treatment [1]."


def conv(chat, memory=None):
    # The 5-document corpus returns nearly the same PMIDs for every query, so the same-evidence merge is off here.
    return Conversation(chat, BM25(DOCS), memory=memory, session=Session(source="test"), samples=1, min_top_score=0.1,
                        same_evidence=1.01)


# ---------- interpreter ----------

def test_parse_keeps_only_verbatim_key_sentences():
    raw = json.dumps({"interpretations": [{"intent_type": "weird", "paraphrase": "p", "key_sentences": ["I barely sleep", "made up"],
                                           "terms": ["insomnia"], "confidence": 3}]})
    [it] = parse_interpretations(raw, "honestly i barely sleep.")
    assert it.key_sentences == ["I barely sleep"] and it.unverified_spans == ["made up"]
    assert it.intent_type == "other" and it.confidence == 1.0


def test_merge_samples_rewards_agreement():
    a = Interpretation("symptom_question", "sleep problems after trauma", [], ["insomnia"], 0.8)
    b = Interpretation("symptom_question", "sleep problems after a trauma", [], ["insomnia"], 0.6)
    c = Interpretation("other", "alcohol use coping", [], ["alcohol"], 0.9)
    merged = merge_samples([[a, c], [b]])
    assert merged[0].paraphrase.startswith("sleep") and merged[0].support == 1.0
    assert merged[1].support == 0.5 and merged[1].confidence == 0.45


def test_interpret_retries_unparseable_sample():
    class Flaky(FakeChat):
        def complete(self, messages, **kw):
            self.calls.append(messages)
            return "not json" if len(self.calls) == 1 else json.dumps(MULTI)
    readings, stats = interpret("i keep checking the stove, i barely sleep, drinking more", Flaky(MULTI), samples=1)
    assert len(readings) == 3 and stats["parse_failures"] == 1


# ---------- conversation flow ----------

def test_multi_intent_confirm_then_answer_and_feedback():
    c = conv(FakeChat(MULTI))
    r = c.respond("ever since the crash i keep checking the stove, i barely sleep and ive been drinking more")
    assert r["state"] == "confirm" and r["reply"].count("\n") >= 3  # separate asks: no clarifying question
    r = c.respond("yes that's right")
    assert r["state"] == "answered" and "[PMID" in r["reply"] and "Was this helpful?" in r["reply"]
    c.respond("yes, thanks")
    pairs = {p.interp_id: p for p in c.end()}
    assert all(p.value == 1.0 for p in pairs.values())  # confirm +0.5, feedback +0.6, clipped to 1


def test_ambiguous_words_trigger_one_clarifying_question():
    c = conv(FakeChat(AMBIG))
    r = c.respond("how do you deal with intrusive thoughts?")
    assert r["state"] == "clarify" and "3. Both" in r["reply"]
    r = c.respond("the first one, memories")
    assert r["state"] == "confirm" and len(r["readings"]) == 1
    c.respond("yes")
    pairs = {p.paraphrase: p.value for p in c.end()}
    assert pairs["Treatment for intrusive memories after trauma"] > 0
    assert pairs["Treatment for obsessional intrusive thoughts"] < 0


def test_drop_marks_reading_negative():
    c = conv(FakeChat(MULTI))
    c.respond("i keep checking the stove, i barely sleep and drinking more")
    r = c.respond("drop 3")
    assert r["state"] == "answered" and len(r["readings"]) == 2
    scores = {p.interp_id: p.value for p in c.end()}
    assert scores["i3"] == -0.7 and scores["i1"] == 0.5


def test_correction_reinterprets_and_tracks_query_change():
    after = {"interpretations": [MULTI["interpretations"][0],
                                 {"intent_type": "treatment_question", "paraphrase": "Non-drug ways to sleep better after trauma",
                                  "key_sentences": ["i barely sleep"], "terms": ["CBT for insomnia", "sleep hygiene"], "confidence": 0.8}]}
    c = conv(FakeChat(MULTI, after_correction=after))
    c.respond("i keep checking the stove, i barely sleep and drinking more")
    r = c.respond("no, I don't care about drinking, I meant what helps me sleep without pills")
    assert r["state"] == "confirm" and any("Non-drug" in x["paraphrase"] for x in r["readings"])
    assert c.session.query_changes and c.session.query_changes[0]["why"].startswith("no, I don't")
    c.respond("yes")
    scores = {p.paraphrase: p.value for p in c.end()}
    assert scores["Is repeated checking after a trauma linked to it"] > 0  # survived the correction, same id
    assert scores["Drinking alcohol to cope with sleep after trauma"] == -1.0
    assert scores["Non-drug ways to sleep better after trauma"] == 0.5


def test_followup_counts_as_approval():
    c = conv(FakeChat(MULTI))
    c.respond("i keep checking the stove, i barely sleep and drinking more")
    c.respond("yes")
    c.respond("tell me more about the insomnia therapy for trauma")
    names = [e.payload["name"] for e in c.session.events if e.kind == "signal"]
    assert "followup" in names


def test_crisis_skips_everything():
    chat = FakeChat(MULTI)
    c = conv(chat)
    r = c.respond("nothing helps and i want to die")
    assert r["state"] == "crisis" and "988" in r["reply"] and chat.calls == []


# ---------- feedback loop ----------

def test_memory_learns_and_feeds_examples(tmp_path):
    mem = PhrasingMemory(tmp_path / "pm.jsonl")
    c = conv(FakeChat(MULTI), memory=mem)
    c.respond("i keep checking the stove, i barely sleep and drinking more")
    c.respond("yes")
    c.respond("yes")
    c.end()
    assert len(mem.rows) == 3 and (tmp_path / "pm.jsonl").exists()
    ex = PhrasingMemory(tmp_path / "pm.jsonl").examples("I'm checking the stove over and over")
    assert ex and ex[0]["paraphrase"].startswith("Is repeated checking")
    # the next conversation's interpreter prompt carries the example
    chat = FakeChat(MULTI)
    conv(chat, memory=mem).respond("checking the stove again and again")
    assert "Past conversations" in chat.calls[0][0]["content"]


def test_live_sessions_do_not_teach_memory(tmp_path):
    mem = PhrasingMemory(tmp_path / "pm.jsonl")
    c = Conversation(FakeChat(MULTI), BM25(DOCS), memory=mem, session=Session(source="live"), samples=1, min_top_score=0.1)
    c.respond("i keep checking the stove")
    c.respond("yes")
    c.end()
    assert mem.rows == []


def test_unsignalled_reading_scores_zero_and_is_not_stored(tmp_path):
    s = Session()
    s.register(Interpretation("other", "x", [], [], 0.5, id="i1"), "phrase")
    [p] = score_session(s)
    assert p.value == 0 and p.reasons == ["no signal"]
    mem = PhrasingMemory(tmp_path / "pm.jsonl")
    assert mem.add([p]) == 0


def test_deidentify():
    out = deidentify("My name is Sarah Jones, call 555-123-4567 or sj@example.com on 3/14/2025")
    assert "Sarah" not in out and "555" not in out and "example.com" not in out and "3/14" not in out


def test_yes_with_refinement_answers_without_asking_again():
    after = {"interpretations": [MULTI["interpretations"][0], MULTI["interpretations"][1]]}
    c = conv(FakeChat(MULTI, after_correction=after))
    c.respond("i keep checking the stove, i barely sleep and drinking more")
    r = c.respond("yes, especially the checking and the sleep part please")
    assert r["state"] == "answered" and len(r["readings"]) == 2
    names = [e.payload["name"] for e in c.session.events if e.kind == "signal"]
    assert "corrected" not in names and c.session.query_changes[0]["why"].startswith("refinement")


def test_near_duplicate_readings_do_not_trigger_clarify():
    dup = {"interpretations": [
        {"intent_type": "treatment_question", "paraphrase": "Does CBT help adults with constant worry?",
         "key_sentences": ["constant worry"], "terms": ["cognitive behavioral therapy", "worry"], "confidence": 0.6},
        {"intent_type": "treatment_question", "paraphrase": "Is CBT effective for adults with constant worry?",
         "key_sentences": ["constant worry"], "terms": ["cognitive behavioral therapy", "worry", "efficacy"], "confidence": 0.6}]}
    c = conv(FakeChat(dup))
    assert c.respond("does CBT help with constant worry")["state"] == "confirm"


def test_assistant_converse_keeps_session(tmp_path):
    from cca.service import Assistant
    a = Assistant(client=FakeChat(MULTI))
    os.environ["CCA_PHRASING_MEMORY"] = str(tmp_path / "pm.jsonl")
    a._bm25, a._docs = BM25(DOCS), DOCS
    a.min_top_score = 0.1
    r1 = a.converse("i keep checking the stove, i barely sleep and drinking more")
    r2 = a.converse("yes", r1["session_id"])
    assert r1["state"] == "confirm" and r2["state"] == "answered"
    out = a.end_conversation(r1["session_id"])
    assert out["status"] == "ended" and len(out["scores"]) == 3


def test_readings_with_same_evidence_collapse():
    dup = {"interpretations": [
        {"intent_type": "treatment_question", "paraphrase": "Does exposure therapy help checking compulsions?",
         "key_sentences": ["checking"], "terms": ["exposure", "checking compulsions"], "confidence": 0.9},
        {"intent_type": "treatment_question", "paraphrase": "Is ERP effective for compulsive checking?",
         "key_sentences": ["checking"], "terms": ["exposure response prevention", "checking compulsions"], "confidence": 0.7}]}
    c = conv(FakeChat(dup))
    c.k, c.same_evidence = 2, 0.6
    r = c.respond("what helps with checking")
    assert len(r["readings"]) == 1 and any(e.kind == "merged" for e in c.session.events)


def test_separate_questions_from_one_sentence_are_not_clarified():
    two = {"interpretations": [dict(AMBIG["interpretations"][0]), dict(AMBIG["interpretations"][1], alternative=False)]}
    c = conv(FakeChat(two))
    assert c.respond("how do you deal with intrusive thoughts?")["state"] == "confirm"

import numpy as np

from cca.overlap import OverlapAgent, Params, parse_phrases


class BagEmbedder:
    """Deterministic bag-of-words embedder so tests need no model download."""
    def __init__(self):
        self.vocab = {}

    def encode(self, texts):
        rows = []
        for t in texts:
            v = np.zeros(512)
            for w in t.lower().replace(",", " ").split():
                v[hash(w) % 512] += 1
            rows.append(v / (np.linalg.norm(v) or 1))
        return np.vstack(rows)


def agent():
    return OverlapAgent(BagEmbedder(), params=Params(cutoff=0.5, min_score=0.1, margin=0.5, cap=6.0),
                        extractor=lambda t: [p.strip() for p in t.split(";")])


def test_single_condition_no_alert():
    r = agent().analyze("checks the stove many times; washes hands until raw; afraid of germs")
    assert r.top == "ocd" and not r.alert


def test_comorbid_alerts_both():
    r = agent().analyze("bad dreams about what happened; jumps at loud noises; feels down most days; nothing is enjoyable anymore")
    assert r.alert and {"ptsd", "depression"} <= set(r.alert_conditions)
    assert "not a diagnosis" in r.message


def test_risk_gate_skips_overlap():
    r = agent().analyze("talks about suicide; can't sleep")
    assert r.crisis and not r.alert and not r.matches


def test_parse_phrases_formats():
    assert parse_phrases('["a b", "c"]') == ["a b", "c"]
    assert parse_phrases("[worries a lot]\n[tense]") == ["worries a lot", "tense"]


def test_eating_disorder_is_consider_not_core():
    r = agent().analyze("has to eat foods in a certain order; terrified of gaining weight; checks how clothes fit many times a day")
    assert "eating_disorder" in r.consider and "eating_disorder" not in r.alert_conditions
    assert r.top != "eating_disorder"

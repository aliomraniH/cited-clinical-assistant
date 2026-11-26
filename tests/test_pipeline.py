from pathlib import Path

from cca.audit import audit, profile, to_markdown
from cca.clean import clean, normalize
from cca.pubmed import Record, parse_pubmed_xml

XML = (Path(__file__).parent / "fixtures" / "sample.xml").read_bytes()


def test_parse_fields():
    recs = parse_pubmed_xml(XML)
    assert [r.pmid for r in recs] == ["111", "222"]
    a = recs[0]
    assert a.title == "CBT for generalized anxiety"
    assert a.abstract.startswith("BACKGROUND: Anxiety is common.")
    assert "RESULTS:" in a.abstract
    assert a.pub_date == "2026-11-20"
    assert a.doi == "10.1/abc"
    assert a.n_authors == 2
    assert a.mesh_terms == ["Anxiety Disorders"]
    assert recs[1].pub_date == "2025"
    assert recs[1].doi is None


def test_clean_merges_topics_and_drops_near_dupes():
    recs = parse_pubmed_xml(XML)
    copy = Record(**{**recs[0].__dict__, "pmid": "999"})  # same text, other PMID
    rows, stats = clean({"gad": [recs[0], copy], "ptsd": [recs[1], recs[0]]})
    assert stats == {"raw_records": 4, "cross_topic_duplicates": 1, "near_duplicates_dropped": 1, "rows": 2}
    first = next(r for r in rows if r["pmid"] == "111")
    assert first["topics"] == ["gad", "ptsd"]
    assert first["source_id"] == "PMID:111"
    assert first["journal"] == "J Anxiety Disord"
    assert "  " not in first["abstract"]


def test_normalize():
    assert normalize("a  b\n\n  c ") == "a b\nc"


def test_audit_baseline_and_drift():
    rows, _ = clean({"gad": parse_pubmed_xml(XML)})
    first = audit(rows)
    assert first["passed"] and first["drift"]["baseline"]
    assert first["profile"]["schema_valid_pct"] == 100.0
    prev = dict(first["profile"], records=10, topic_counts={"ptsd": 10})
    second = audit(rows, prev)
    assert any("volume" in f for f in second["drift"]["flags"])
    assert any("topic mix" in f for f in second["drift"]["flags"])
    assert "Data audit" in to_markdown(second, "2026-11-26")


def test_pii_scan_stops():
    rows, _ = clean({"gad": parse_pubmed_xml(XML)})
    rows[0]["abstract"] += " Contact jane@example.com"
    res = audit(rows)
    assert not res["passed"]
    assert profile(rows)["pii_hits"]["email"] == 1

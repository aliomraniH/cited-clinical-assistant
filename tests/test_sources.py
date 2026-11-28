import json

from cca.index import load_snapshots
from cca.sources import article_rows, chunk_text, guideline_rows, licence_kind, parse_jats, trial_rows

JATS = b"""<pmc-articleset><article><front><journal-meta><journal-title-group><journal-title>J Test</journal-title></journal-title-group></journal-meta>
<article-meta><article-id pub-id-type="pmcid">PMC123</article-id><article-id pub-id-type="pmid">999</article-id>
<title-group><article-title>Sertraline for PTSD</article-title></title-group><pub-date><year>2025</year></pub-date>
<permissions><license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://creativecommons.org/licenses/by/4.0/"><p>Open access.</p></license></permissions>
<abstract><p>Sertraline reduced PTSD symptoms in 120 adults compared with placebo over 12 weeks of treatment in this trial.</p></abstract></article-meta></front>
<body><sec><title>Results</title><p>Symptoms fell by 30% in the sertraline group and 15% in the placebo group, a difference that was significant [<xref>1</xref>, <xref>2</xref>].</p>
<table-wrap><table><tr><td>secret table</td></tr></table></table-wrap></sec>
<sec><title>References</title><p>Should be skipped entirely because it is the reference list of the paper.</p></sec></body></article></pmc-articleset>"""


def test_licence_kind():
    assert licence_kind("https://creativecommons.org/licenses/by/4.0/") == "cc-by"
    assert licence_kind("https://creativecommons.org/publicdomain/zero/1.0/") == "cc0"
    assert licence_kind("https://creativecommons.org/licenses/by-nc/4.0/") is None
    assert licence_kind("CC BY-NC-ND 4.0") is None
    assert licence_kind("This is an open access article under CC BY 4.0") == "cc-by"
    assert licence_kind("All rights reserved") is None


def test_parse_and_rows():
    (art,) = parse_jats(JATS)
    assert (art.pmcid, art.pmid, art.licence) == ("PMC123", "999", "cc-by")
    rows = article_rows(art, ["ptsd"], "pmc_oa")
    assert [r["section"] for r in rows] == ["Abstract", "Results"]
    assert all(r["pmid"] == "999" and r["id"].startswith("PMC123#") for r in rows)
    text = " ".join(r["abstract"] for r in rows)
    assert "secret table" not in text and "reference list" not in text and "[" not in text
    assert rows[1]["cite_label"] == "PMID 999, Results"


def test_non_cc_by_dropped():
    art = parse_jats(JATS.replace(b"licenses/by/4.0", b"licenses/by-nc-nd/4.0"))[0]
    assert article_rows(art, ["ptsd"], "pmc_oa") == []


def test_chunk_text_limits():
    para = " ".join(f"Sentence number {i} is here with some words." for i in range(100))
    chunks = chunk_text(para, max_words=50)
    assert len(chunks) > 5 and all(len(c.split()) <= 60 for c in chunks)


def test_trial_and_guideline_rows():
    study = {"protocolSection": {"identificationModule": {"nctId": "NCT01", "briefTitle": "CBT trial"},
             "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2024-01"}},
             "conditionsModule": {"conditions": ["PTSD"]},
             "descriptionModule": {"briefSummary": "This study tests whether cognitive behavioral therapy reduces PTSD symptoms in veterans."},
             "eligibilityModule": {"eligibilityCriteria": "Inclusion: adults aged 18 to 65 with a PTSD diagnosis confirmed by interview."}}}
    rows = trial_rows(study, ["ptsd"])
    assert {r["pmid"] for r in rows} == {"NCT01"} and rows[0]["source"] == "clinicaltrials"
    g = guideline_rows(["", "Recommendation 1. We recommend trauma-focused psychotherapy over pharmacotherapy for the primary treatment of PTSD."],
                       "VADoD-PTSD-2023", "VA/DoD PTSD", "https://x/y.pdf", "2023", ["ptsd"])
    assert g[0]["pmid"] == "VADoD-PTSD-2023 p.2" and g[0]["url"].endswith("#page=2")


def test_loader_keeps_passages_and_refuses_synthetic(tmp_path):
    rows = [{"id": "PMC1#1", "pmid": "1", "abstract": "a"}, {"id": "PMC1#2", "pmid": "1", "abstract": "b"},
            {"id": "x", "pmid": "2", "abstract": "wrong claim", "synthetic_false": True}]
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    old = tmp_path / "old"
    old.mkdir()
    (old / "p.jsonl").write_text(json.dumps({"pmid": "3", "abstract": "c"}))
    got = load_snapshots([tmp_path, old])
    assert sorted(r.get("id", r["pmid"]) for r in got) == ["3", "PMC1#1", "PMC1#2"]

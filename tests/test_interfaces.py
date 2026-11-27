"""Interfaces share one Assistant; these tests run with a fake model, a tiny corpus and a bag-of-words embedder."""

import asyncio
import json

import pytest

from cca.overlap import OverlapAgent, Params
from cca.safety import screen
from cca.service import Assistant
from test_overlap import BagEmbedder

DOCS = [
    {"pmid": "111", "title": "Exposure and response prevention for adolescent OCD", "journal": "J Test", "pub_date": "2026-01-01",
     "abstract": "Randomized trial: exposure and response prevention reduced OCD symptoms in adolescents (n=120)."},
    {"pmid": "222", "title": "Trauma-focused CBT in veterans with PTSD", "journal": "J Test", "pub_date": "2026-02-01",
     "abstract": "Trauma-focused CBT reduced PTSD symptoms in veterans; suicide risk was also assessed."},
]


class FakeChat:
    model = "fake"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kw):
        self.calls += 1
        return "Exposure and response prevention reduced OCD symptoms in adolescents [1]."


@pytest.fixture
def assistant(tmp_path):
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(d) for d in DOCS))
    agent = OverlapAgent(BagEmbedder(), params=Params(cutoff=0.5, min_score=0.1, margin=0.5, cap=6.0),
                         extractor=lambda t: [p.strip() for p in t.split(";")])
    return Assistant(corpus_dir=tmp_path, client=FakeChat(), overlap_agent=agent, min_top_score=0.5)


def test_screen():
    assert screen("I want to end my life").kind == "crisis"
    assert screen("suicide risk in PTSD veterans", topical_ok=True).kind == "risk_topic"
    assert screen("suicide risk in PTSD veterans").kind == "crisis"          # case text is always gated
    assert screen("Do I have OCD?").kind == "diagnosis_request"
    assert screen("ERP for OCD").ok


def test_ask_answers_with_citations(assistant):
    r = assistant.ask("exposure response prevention OCD adolescents")
    assert r["status"] == "answered" and r["citations"][0]["pmid"] == "111"
    assert r["checks"]["invalid_citations"] == 0


def test_ask_crisis_skips_model(assistant):
    r = assistant.ask("I keep thinking about killing myself, does CBT help?")
    assert r["status"] == "crisis" and r["safety"]["resources"]
    assert assistant.client.calls == 0


def test_ask_declines_diagnosis(assistant):
    r = assistant.ask("Do I have OCD if I check the stove ten times?")
    assert r["status"] == "diagnosis_request" and assistant.client.calls == 0


def test_overlap_crisis_and_normal(assistant):
    assert assistant.overlap("talks about suicide; poor sleep")["status"] == "crisis"
    r = assistant.overlap("checks the stove many times; washes hands until raw")
    assert r["status"] in ("alert", "no_alert") and "scores" in r


def test_http_api(assistant, monkeypatch):
    from fastapi.testclient import TestClient
    from cca.interfaces.http_api import create_app
    monkeypatch.setenv("CCA_API_TOKEN", "secret")
    with TestClient(create_app(assistant)) as c:
        assert c.get("/health").json()["ok"]
        assert c.post("/v1/ask", json={"question": "ERP for OCD"}).status_code == 401
        h = {"Authorization": "Bearer secret"}
        r = c.post("/v1/ask", json={"question": "exposure response prevention OCD"}, headers=h).json()
        assert r["status"] == "answered"
        assert c.post("/v1/search", json={"query": "PTSD veterans", "k": 1}, headers=h).json()["hits"][0]["pmid"] == "222"
        card = c.get("/.well-known/agent-card.json").json()
        assert {s["id"] for s in card["skills"]} == {"ask", "search", "overlap", "brief"}
        assert "/v1/brief" in c.get("/openapi.json").json()["paths"]


def test_mcp_server_and_peer_allowlist(assistant):
    from mcp import Client
    from cca.interfaces.mcp_server import build_server
    from cca.peers import PeerHub
    peer = build_server(assistant, PeerHub({}))
    hub = PeerHub({"lit": {"transport": "inprocess", "server": peer, "allowed_tools": ["search_literature"]}})
    server = build_server(assistant, hub)

    async def run():
        async with Client(server) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert {"ask_literature", "check_symptom_overlap", "decision_brief", "consult_peer"} <= names
            ok = json.loads((await c.call_tool("consult_peer", {"peer": "lit", "tool": "search_literature",
                                                                 "arguments": {"query": "OCD", "k": 1}})).content[0].text)
            assert not ok["is_error"] and "111" in ok["text"]
            denied = json.loads((await c.call_tool("consult_peer", {"peer": "lit", "tool": "ask_literature"})).content[0].text)
            assert denied["is_error"] and "allowlist" in denied["text"]
            brief = json.loads((await c.call_tool("decision_brief", {
                "question": "exposure response prevention OCD",
                "case_description": "checks the stove many times; washes hands until raw",
                "consults": [{"peer": "lit", "tool": "search_literature", "arguments": {"query": "OCD"}}]})).content[0].text)
            assert brief["evidence"]["status"] == "answered" and "overlap" in brief and len(brief["peer_inputs"]) == 1

    asyncio.run(run())


def test_example_peer_disabled():
    from cca.peers import load_peers
    assert all(not n.startswith("_") for n in load_peers())

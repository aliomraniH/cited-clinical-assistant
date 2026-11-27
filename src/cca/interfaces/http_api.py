"""Interface 3: a standard HTTP API (OpenAPI 3) so other agents and apps can use the assistant.

  uvicorn cca.interfaces.http_api:app --port 8000

Endpoints (all JSON):
  GET  /health                       liveness, no model call
  GET  /v1/status                    version, model, corpus size, topics
  POST /v1/ask        {question}     cited answer, abstain, crisis or no-diagnosis response
  POST /v1/search     {query, k}     BM25 hits with PMIDs
  POST /v1/overlap    {description}  symptom-overlap check
  POST /v1/brief      {question, case_description?}  evidence + overlap brief
  GET  /openapi.json                 machine-readable spec (import as a GPT action, LangChain tool, etc.)
  GET  /.well-known/agent-card.json  A2A-style agent card listing skills and endpoints
  /mcp                               the MCP server (interface 1) over streamable HTTP

Auth: if CCA_API_TOKEN is set, /v1/* and /mcp need "Authorization: Bearer <token>".
"""

from __future__ import annotations

import contextlib
import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..service import VERSION, Assistant, get_assistant
from .mcp_server import build_server


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["Does exposure and response prevention help OCD in adolescents?"])


class SearchIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    k: int = Field(8, ge=1, le=20)


class OverlapIn(BaseModel):
    description: str = Field(..., min_length=10, max_length=4000,
                             description="De-identified symptom description. No names, dates or places.")


class BriefIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    case_description: str | None = Field(None, max_length=4000)


def _token_check(authorization: str | None = Header(None)) -> None:
    token = os.environ.get("CCA_API_TOKEN")
    if not token:
        return
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given, token):
        raise HTTPException(401, "missing or wrong bearer token")


def agent_card(base_url: str) -> dict:
    return {
        "name": "Cited clinical assistant",
        "description": "Answers mental-health clinical questions (GAD, depression, OCD, PTSD) only from PubMed "
                       "abstracts with a citation on every sentence, and checks symptom descriptions for overlap "
                       "between conditions. Research demo, not medical advice; never diagnoses.",
        "version": VERSION,
        "url": base_url,
        "documentationUrl": "https://github.com/aliomraniH/cited-clinical-assistant",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "interfaces": {"openapi": f"{base_url}/openapi.json", "mcp": f"{base_url}/mcp"},
        "authentication": {"schemes": ["bearer"] if os.environ.get("CCA_API_TOKEN") else []},
        "skills": [
            {"id": "ask", "name": "Cited literature answer", "endpoint": "POST /v1/ask",
             "tags": ["rag", "citations", "pubmed"]},
            {"id": "search", "name": "Literature search", "endpoint": "POST /v1/search", "tags": ["bm25", "pubmed"]},
            {"id": "overlap", "name": "Symptom overlap check", "endpoint": "POST /v1/overlap",
             "tags": ["comorbidity", "embeddings", "sapbert"]},
            {"id": "brief", "name": "Decision-support brief", "endpoint": "POST /v1/brief", "tags": ["decision-support"]},
        ],
        "safety": ["self-harm mentions return crisis resources", "requests to diagnose a person are declined"],
    }


def create_app(assistant: Assistant | None = None, mount_mcp: bool = True) -> FastAPI:
    a = assistant or get_assistant()
    mcp_server = build_server(a) if mount_mcp else None
    mcp_app = mcp_server.streamable_http_app(streamable_http_path="/") if mcp_server else None

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        if mcp_server is not None:
            async with mcp_server.session_manager.run():
                yield
        else:
            yield

    app = FastAPI(title="Cited clinical assistant API", version=VERSION, lifespan=lifespan,
                  description="Research demo, not medical advice. See /.well-known/agent-card.json.")
    guard = [Depends(_token_check)]

    @app.get("/health")
    def health():
        return {"ok": True, "version": VERSION}

    @app.get("/.well-known/agent-card.json")
    def card(request: Request):
        return agent_card(str(request.base_url).rstrip("/"))

    @app.get("/v1/status", dependencies=guard)
    def status():
        return a.status()

    @app.post("/v1/ask", dependencies=guard)
    def ask(body: AskIn):
        return a.ask(body.question)

    @app.post("/v1/search", dependencies=guard)
    def search(body: SearchIn):
        return a.search(body.query, body.k)

    @app.post("/v1/overlap", dependencies=guard)
    def overlap(body: OverlapIn):
        return a.overlap(body.description)

    @app.post("/v1/brief", dependencies=guard)
    def brief(body: BriefIn):
        return a.brief(body.question, body.case_description)

    if mcp_app is not None:
        @app.middleware("http")
        async def mcp_auth(request: Request, call_next):
            if request.url.path.startswith("/mcp"):
                try:
                    _token_check(request.headers.get("authorization"))
                except HTTPException as e:
                    return JSONResponse({"detail": e.detail}, status_code=e.status_code)
            return await call_next(request)

        app.mount("/mcp", mcp_app)
    return app


def __getattr__(name):  # lazy `app` so importing this module doesn't build the index
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)

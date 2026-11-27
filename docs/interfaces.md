# Interfaces

All four interfaces call one `Assistant` object (`src/cca/service.py`), so retrieval settings, safety rules and response shapes are defined once.

| # | Interface | Who uses it | Entry point |
|---|---|---|---|
| 1 | MCP server, can consult other MCP servers | Claude Desktop / Claude Code / any MCP client | `python -m cca.interfaces.mcp_server` (stdio) or `/mcp/` on the hosted app |
| 2 | Standalone chatbot | People, in a browser | `python app.py` → Hugging Face Space |
| 3 | HTTP API + agent card | Other agents and apps | `/v1/*`, `/openapi.json`, `/.well-known/agent-card.json` |
| 4 | Architecture Coach | Ali, for interview prep | the `architecture-coach` skill + [Architecture Coach page](https://claude.ai/artifact/4qhFi7M1ofhXWVCPc4Bu5z) |

```
                 ┌───────────── app.py (one process, one deploy) ─────────────┐
 browser ──────► │  Gradio UI  /                                              │
 agent (REST) ─► │  FastAPI    /v1/ask /v1/search /v1/overlap /v1/brief       │
 agent (MCP) ──► │  MCP server /mcp/   ── consult_peer ──► other MCP servers  │
                 │        └──────────────► Assistant (service.py) ◄───────────┘
                 │                          ├─ safety.screen (crisis, no diagnosis)
                 │                          ├─ BM25 over PubMed records → answer model (HF router)
                 │                          └─ OverlapAgent (LLM extractor + SapBERT)
                 └────────────────────────────────────────────────────────────┘
 Claude chat "coach" ──► architecture-coach skill ──► Architecture Coach page (db)
```

## Safety (every interface)

- First-person mentions of suicide or self-harm ("I want to end my life") return crisis resources (988, Crisis Text Line, findahelpline.com) and the model is never called.
- A topical mention in a literature question ("suicide risk in PTSD veterans") is answered, with the resources attached.
- Case descriptions sent to the overlap check are always gated.
- Requests to diagnose a person ("do I have OCD?") are declined with a pointer to the literature and the overlap check.
- The risk pattern in `safety.py` is broader than the overlap agent's own (it also catches "killing myself", "want to die", "hurt myself").

## 1. MCP server

Tools: `ask_literature`, `search_literature`, `check_symptom_overlap`, `decision_brief`, `list_peers`, `consult_peer`, `assistant_status`. Resource: `cca://symptom-map`. Prompt: `case_review`.

Claude Desktop config (stdio):

```json
{"mcpServers": {"cited-clinical-assistant": {
  "command": "python", "args": ["-m", "cca.interfaces.mcp_server"],
  "env": {"PYTHONPATH": "/path/to/cited-clinical-assistant/src", "CCA_BASE_URL": "https://router.huggingface.co/v1", "CCA_API_KEY": "hf_..."}}}}
```

Or point any MCP client at the hosted app: `https://<space>.hf.space/mcp/` (with `Authorization: Bearer <CCA_API_TOKEN>` if set).

**Peers for decision making.** `config/mcp_peers.json` lists other MCP servers the assistant may call, over HTTP or stdio, each with an `allowed_tools` allowlist. Secrets come from environment variables named in `headers_env`. `decision_brief` gathers the literature answer, the overlap check for an optional case description, and any peer tool outputs into one evidence summary with open questions. Peer output is treated as data; the decision stays with the clinician. Names starting with `_` are disabled examples.

## 2. Chatbot hosting (chosen: hybrid)

| Option | Pros | Cons |
|---|---|---|
| **Hybrid: code on GitHub, Action deploys to a HF Space** (chosen 2026-11-27) | One codebase with tests; free CPU hosting on a PRO account; same process serves UI, REST and MCP | Needs an HF write token as a GitHub secret; Space sleeps when idle (cold start ~1 min) |
| HF Space only | Fastest start | No CI tests; code drifts from the repo |
| GitHub Pages + separate backend | Free static page | Pages can't hold the HF token, so a second host (Render, Vercel) is needed for the API |
| Claude artifact chat page | Shared by link on claude.ai | Still needs the hosted API for retrieval and SapBERT |

Deploy: `.github/workflows/deploy-space.yml` runs the tests, then uploads `app.py`, `requirements.txt`, `src/`, `config/` and the corpus PMID list to `Aliomrani6/cited-clinical-assistant`. On first start the Space rebuilds the 2,015-record corpus from PubMed.

Setup once:
1. Create a Hugging Face **write** token (huggingface.co/settings/tokens) and save it as the GitHub repository secret `HF_TOKEN`.
2. After the first deploy, in the Space settings add secrets `CCA_API_KEY` (an HF token with Inference Providers permission) and, optionally, `CCA_API_TOKEN` to protect `/v1` and `/mcp`.

## 3. HTTP API

`POST /v1/ask {question}`, `POST /v1/search {query, k}`, `POST /v1/overlap {description}`, `POST /v1/brief {question, case_description?}`, `GET /v1/status`, `GET /health`. The OpenAPI spec at `/openapi.json` can be imported as a tool by other agent frameworks; the agent card at `/.well-known/agent-card.json` lists skills, endpoints and the MCP URL.

## 4. Architecture Coach

A Claude chat started with `coach` (or `coach m3`, or `brainstorm`). The skill reads the Coach page's database, picks the next topic from the curriculum (this system, models, interfaces and agents, evaluation and operations), asks two or three check questions first, teaches using this repo's code, tests with one applied question, then brainstorms and logs ideas. The page shows the curriculum map, session timeline and idea board; Ali can add ideas there too. Page source: `docs/coach/architecture-coach.html`.

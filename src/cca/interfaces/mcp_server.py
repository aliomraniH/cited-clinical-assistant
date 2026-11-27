"""Interface 1: the assistant as an MCP server, able to consult other MCP servers.

Run locally (Claude Desktop / Claude Code, stdio):
  python -m cca.interfaces.mcp_server
Run over HTTP (streamable HTTP at /mcp):
  python -m cca.interfaces.mcp_server --http --port 8001
The HTTP API (interface 3) also mounts this server at /mcp, so one deploy serves both.

Tools return JSON. Every tool applies the shared safety screen: self-harm mentions get
crisis resources, and requests to diagnose a person are declined.
"""

from __future__ import annotations

import argparse
import json

from mcp.server.mcpserver import MCPServer

from ..peers import PeerError, PeerHub
from ..service import Assistant, get_assistant

INSTRUCTIONS = """Cited clinical assistant (research demo, not medical advice).
Scope: generalized anxiety disorder, depression, OCD and PTSD literature from PubMed, plus symptom overlap
with ADHD and substance use. Use ask_literature for evidence questions (every sentence is cited by PMID),
check_symptom_overlap to see which conditions a description fits about equally and what tells them apart,
and decision_brief to combine both with inputs from approved peer MCP servers. Never present output as a
diagnosis. If a user mentions suicide or self-harm, give the crisis resources returned by the tools."""


def build_server(assistant: Assistant | None = None, hub: PeerHub | None = None) -> MCPServer:
    a = assistant or get_assistant()
    peers = hub or PeerHub()
    mcp = MCPServer(name="cited-clinical-assistant", instructions=INSTRUCTIONS, version="0.2.0")

    @mcp.tool(description="Answer a clinical question about GAD, depression, OCD or PTSD only from indexed PubMed "
                          "abstracts. Every sentence carries a citation; abstains when evidence is missing.")
    def ask_literature(question: str) -> dict:
        return a.ask(question)

    @mcp.tool(description="Conversational search for messages written the way people talk. Splits the message into "
                          "readings, may ask one clarifying question, confirms what it understood, then answers with "
                          "citations. Pass the returned session_id back on each turn; reply to its questions in 'message'.")
    def converse(message: str, session_id: str | None = None) -> dict:
        return a.converse(message, session_id)

    @mcp.tool(description="Read a stored conversational-search session: transcript, state and current readings.")
    def get_conversation(session_id: str) -> dict:
        return a.get_conversation(session_id)

    @mcp.tool(description="End a conversational-search session and return the per-reading scores.")
    def end_conversation(session_id: str) -> dict:
        return a.end_conversation(session_id)

    @mcp.tool(description="BM25 search over the indexed PubMed abstracts. Returns PMID, title, year, URL and abstract.")
    def search_literature(query: str, k: int = 8) -> dict:
        return a.search(query, k=max(1, min(k, 20)))

    @mcp.tool(description="Check a de-identified symptom description for overlap between GAD, depression, PTSD and "
                          "OCD (and ADHD / substance use to consider). Returns scores, shared symptoms and "
                          "distinguishing questions. Not a diagnosis. Do not send identifying details.")
    def check_symptom_overlap(description: str) -> dict:
        return a.overlap(description)

    @mcp.tool(description="List the other MCP servers this assistant may consult, with their allowed tools.")
    def list_peers() -> list[dict]:
        return peers.describe()

    @mcp.tool(description="Call an allowed tool on a configured peer MCP server and return its output as data.")
    async def consult_peer(peer: str, tool: str, arguments: dict | None = None) -> dict:
        try:
            return await peers.call(peer, tool, arguments)
        except PeerError as e:
            return {"peer": peer, "tool": tool, "is_error": True, "text": str(e)}
        except Exception as e:  # network, auth, protocol
            return {"peer": peer, "tool": tool, "is_error": True, "text": f"{type(e).__name__}: {e}"}

    @mcp.tool(description="Decision-support brief: literature answer, symptom overlap for an optional case "
                          "description, and outputs of named peer tools, gathered into one evidence summary with "
                          "open questions. consults is a list of {peer, tool, arguments}. The decision stays with "
                          "the clinician.")
    async def decision_brief(question: str, case_description: str | None = None,
                             consults: list[dict] | None = None) -> dict:
        inputs = []
        for c in consults or []:
            try:
                inputs.append(await peers.call(c["peer"], c["tool"], c.get("arguments")))
            except Exception as e:
                inputs.append({"peer": c.get("peer"), "tool": c.get("tool"), "is_error": True, "text": str(e)})
        return a.brief(question, case_description, inputs)

    @mcp.tool(description="Assistant version, model, corpus size and topics.")
    def assistant_status() -> dict:
        return a.status()

    @mcp.resource("cca://symptom-map", name="symptom_map", mime_type="application/json",
                  description="The symptom map used by the overlap check (draft for clinician review).")
    def symptom_map() -> str:
        from ..overlap import MAP_PATH
        return MAP_PATH.read_text()

    @mcp.prompt(description="Walk through a de-identified case: evidence, overlap, what to ask next.")
    def case_review(case_description: str, question: str) -> str:
        return (f"Use decision_brief with question={json.dumps(question)} and case_description="
                f"{json.dumps(case_description)}. Then summarise: (1) what the cited literature says, with PMIDs; "
                "(2) any overlap alert and the distinguishing questions to ask; (3) what remains uncertain. "
                "Do not name a diagnosis.")

    return mcp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    server = build_server()
    if args.http:
        import uvicorn
        uvicorn.run(server.streamable_http_app(host=args.host), host=args.host, port=args.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()

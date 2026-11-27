"""Connections from the assistant to other MCP servers ("peers") for decision support.

Peers are listed in config/mcp_peers.json. Each peer has an allowlist of tools the
assistant may call, so a peer can't be used for anything it wasn't approved for.

  {"peers": {
     "clinical_tool": {"transport": "http", "url": "https://.../mcp",
                       "headers_env": {"Authorization": "CLINICAL_TOOL_AUTH"},
                       "allowed_tools": ["lookup_guideline"], "purpose": "guideline lookup"},
     "local_notes": {"transport": "stdio", "command": "python", "args": ["-m", "notes_server"],
                     "allowed_tools": ["search_notes"], "purpose": "..."}}}

Header values are read from environment variables named in headers_env, never stored in the file.
Peer output is data for the clinician, not instructions to the assistant.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PEERS_PATH = Path(__file__).resolve().parents[2] / "config" / "mcp_peers.json"
MAX_PEER_TEXT = 6000


class PeerError(Exception):
    pass


def load_peers(path: Path | None = None) -> dict[str, dict]:
    p = Path(os.environ.get("CCA_MCP_PEERS", path or PEERS_PATH))
    if not p.exists():
        return {}
    peers = json.loads(p.read_text()).get("peers", {})
    return {n: c for n, c in peers.items() if not n.startswith("_")}  # "_name" = disabled example


def _transport(cfg: dict):
    kind = cfg.get("transport", "http")
    if kind == "http":
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
        headers = {h: os.environ[v] for h, v in (cfg.get("headers_env") or {}).items() if os.environ.get(v)}
        return streamable_http_client(cfg["url"], http_client=create_mcp_http_client(headers=headers or None))
    if kind == "stdio":
        from mcp import StdioServerParameters
        env = {k: os.environ[k] for k in cfg.get("pass_env", []) if k in os.environ}
        return StdioServerParameters(command=cfg["command"], args=cfg.get("args", []), env=env or None)
    if kind == "inprocess":       # tests: cfg["server"] is an MCPServer instance
        return cfg["server"]
    raise PeerError(f"unknown transport {kind!r}")


def _result_to_dict(res: Any) -> dict:
    texts = [getattr(c, "text", "") for c in (getattr(res, "content", None) or []) if getattr(c, "text", None)]
    text = "\n".join(texts)
    return {
        "is_error": bool(getattr(res, "is_error", getattr(res, "isError", False))),
        "text": text[:MAX_PEER_TEXT] + (" ...[truncated]" if len(text) > MAX_PEER_TEXT else ""),
        "structured": getattr(res, "structured_content", getattr(res, "structuredContent", None)),
    }


class PeerHub:
    def __init__(self, peers: dict[str, dict] | None = None):
        self.peers = load_peers() if peers is None else peers

    def describe(self) -> list[dict]:
        return [{"name": n, "transport": c.get("transport", "http"), "purpose": c.get("purpose", ""),
                 "allowed_tools": c.get("allowed_tools", [])} for n, c in self.peers.items()]

    def _cfg(self, name: str) -> dict:
        if name not in self.peers:
            raise PeerError(f"unknown peer {name!r}; configured: {sorted(self.peers)}")
        return self.peers[name]

    async def list_tools(self, name: str) -> list[dict]:
        from mcp import Client
        cfg = self._cfg(name)
        allowed = set(cfg.get("allowed_tools", []))
        async with Client(_transport(cfg)) as client:
            res = await client.list_tools()
        return [{"name": t.name, "description": (t.description or "")[:300], "allowed": t.name in allowed}
                for t in res.tools]

    async def call(self, name: str, tool: str, arguments: dict | None = None) -> dict:
        from mcp import Client
        cfg = self._cfg(name)
        if tool not in cfg.get("allowed_tools", []):
            raise PeerError(f"tool {tool!r} is not in the allowlist for peer {name!r}")
        async with Client(_transport(cfg)) as client:
            res = await client.call_tool(tool, arguments or {})
        return {"peer": name, "tool": tool, **_result_to_dict(res)}

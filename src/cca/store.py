"""Where conversation state lives between turns, so any process (chatbot, HTTP API, MCP server,
another replica) can pick up a conversation by its session_id, like a stored conversation in a
hosted chat API.

  CCA_SESSION_STORE=supabase   table conversation_state (db/schema_conversations.sql); needs SUPABASE_URL,
                               and SUPABASE_KEY (a secret key) where no proxy adds it, e.g. a Space secret
  CCA_SESSION_STORE=file       JSON files under data/sessions/ (default; one machine only)
  CCA_SESSION_STORE=memory     this process only (tests)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import requests

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class MemoryStore:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def load(self, session_id: str) -> dict | None:
        return self.rows.get(session_id)

    def save(self, session_id: str, state: dict) -> None:
        self.rows[session_id] = json.loads(json.dumps(state))

    def delete(self, session_id: str) -> None:
        self.rows.pop(session_id, None)


class FileStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, session_id: str) -> Path:
        if not _SAFE_ID.match(session_id):
            raise ValueError("bad session id")
        return self.root / f"{session_id}.json"

    def load(self, session_id: str) -> dict | None:
        try:
            p = self._path(session_id)
        except ValueError:
            return None
        return json.loads(p.read_text()) if p.exists() else None

    def save(self, session_id: str, state: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(session_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(self._path(session_id))

    def delete(self, session_id: str) -> None:
        try:
            self._path(session_id).unlink(missing_ok=True)
        except ValueError:
            pass


class SupabaseStore:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: float = 20):
        url = url or os.environ.get("SUPABASE_URL")
        if not url:
            raise RuntimeError("SUPABASE_URL is not set")
        self.rest = url.rstrip("/") + "/rest/v1/conversation_state"
        self.headers = {"Content-Type": "application/json"}
        key = key or os.environ.get("SUPABASE_KEY")
        if key:  # behind the cloud proxy the Network secret supplies it instead
            self.headers.update({"apikey": key, "Authorization": f"Bearer {key}"})
        self.timeout = timeout

    def load(self, session_id: str) -> dict | None:
        if not _SAFE_ID.match(session_id):
            return None
        r = requests.get(self.rest, params={"session_id": f"eq.{session_id}", "select": "state"},
                         headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        rows = r.json()
        return rows[0]["state"] if rows else None

    def save(self, session_id: str, state: dict) -> None:
        r = requests.post(self.rest, json={"session_id": session_id, "source": state["session"]["source"],
                                           "state": state,
                                           "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()},
                          headers={**self.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
                          timeout=self.timeout)
        r.raise_for_status()

    def delete(self, session_id: str) -> None:
        requests.delete(self.rest, params={"session_id": f"eq.{session_id}"}, headers=self.headers,
                        timeout=self.timeout).raise_for_status()


def make_store(kind: str | None = None, root: Path | None = None):
    kind = (kind or os.environ.get("CCA_SESSION_STORE") or "file").lower()
    if kind == "supabase":
        return SupabaseStore()
    if kind == "memory":
        return MemoryStore()
    return FileStore(root or Path(__file__).resolve().parents[2] / "data" / "sessions")

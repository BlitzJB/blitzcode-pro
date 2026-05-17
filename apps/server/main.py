import asyncio
import contextlib
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from agent_webkit_server.adapters.fastapi import create_app
from agent_webkit_server.auth import AuthConfig
from agent_webkit_server.session_metadata import FileSessionMetadataStore

from acks import AckStore
from discovery import discover_for_cwd
from builtin_agents import BUILTIN_AGENTS

# Persistent session metadata (wrapper UUID ↔ SDK session id + config) lets
# sessions survive uvicorn restarts and idle reaps. Transcript history is
# replayed on resume by reading the SDK's authoritative on-disk transcript
# (~/.claude/projects/<cwd>/<sdk-session>.jsonl) — we don't keep a second copy.
_SESSIONS_DIR = Path(
    os.environ.get("AGENT_WEBKIT_SESSIONS_DIR", str(Path.home() / ".agent-webkit" / "sessions"))
)
metadata_store = FileSessionMetadataStore(_SESSIONS_DIR)

app = create_app(
    auth=AuthConfig(disabled=True),
    metadata_store=metadata_store,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────
# App-layer "needs input" tracking.
#
# Pure application concern — not part of agent-webkit. We subscribe to the
# global event log via app.state.registry (exposed by create_app as a generic
# extension hook) and stamp `last_completion_at` whenever an agent turn
# finishes. The UI polls /app/state on mount and observes its own /stream for
# live `result` events, computing `needs_input` client-side.
# ────────────────────────────────────────────────────────────────────────────

_ACKS_PATH = Path(
    os.environ.get("BLITZ_ACKS_PATH", str(Path.home() / ".agent-webkit" / "app-acks.json"))
)
ack_store = AckStore(_ACKS_PATH)


async def _watch_completions() -> None:
    registry = app.state.registry
    async for ev in registry.event_log.subscribe(0):
        if ev.event == "result":
            try:
                await ack_store.mark_completion(ev.session_id)
            except Exception:
                pass


# `create_app` already installs its own lifespan (starts the reaper). Chain
# ours around it instead of using `@app.on_event`, which FastAPI silently
# ignores once a lifespan context manager is registered.
_inner_lifespan = app.router.lifespan_context


@contextlib.asynccontextmanager
async def _chained_lifespan(app_: FastAPI):
    async with _inner_lifespan(app_):
        task = asyncio.create_task(_watch_completions())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


app.router.lifespan_context = _chained_lifespan


@app.get("/app/state")
async def app_state() -> dict:
    snap = ack_store.snapshot()
    return {
        "acks": {
            sid: {
                "last_completion_at": e.last_completion_at,
                "last_ack_at": e.last_ack_at,
            }
            for sid, e in snap.items()
        }
    }


@app.get("/app/fs/list")
async def fs_list(path: str | None = None) -> dict:
    """Local filesystem directory listing for the in-app folder picker.

    Browsers hide absolute paths from <input webkitdirectory> and
    showDirectoryPicker for security, so the server (which runs on the
    user's machine in this local-only dev tool) exposes the listing.
    """
    target = Path(path).expanduser() if path else Path.home()
    try:
        target = target.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    entries: list[dict] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if not is_dir:
                continue
            entries.append({"name": child.name})
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    parent = str(target.parent) if target.parent != target else None
    return {
        "path": str(target),
        "parent": parent,
        "home": str(Path.home()),
        "entries": entries,
    }


@app.get("/app/sessions/{session_id}/completions")
async def session_completions(session_id: str) -> dict:
    """Completion items available to this session's `/` and `@` palettes.

    Source of truth = filesystem under {cwd}/.claude/{commands,agents,skills}
    and the corresponding ~/.claude dirs. Built-in agents come from
    `claude agents` (cached at import). Project entries shadow user entries
    with the same name.
    """
    md = await metadata_store.load(session_id) if metadata_store is not None else None
    cwd = md.cwd if md is not None else None
    data = discover_for_cwd(cwd)
    # Append built-in agents AFTER project + user, and only if not shadowed.
    seen_agent_names = {a["name"] for a in data["agents"]}
    for ba in BUILTIN_AGENTS:
        if ba["name"] not in seen_agent_names:
            data["agents"].append(ba)
    return data


@app.post("/app/sessions/{session_id}/acknowledge")
async def acknowledge(session_id: str) -> dict:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    entry = await ack_store.acknowledge(session_id)
    return {
        "session_id": session_id,
        "last_completion_at": entry.last_completion_at,
        "last_ack_at": entry.last_ack_at,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

"""App-level event broadcast via the global event log.

Pins the contract:
  * `_broadcast(event, data)` appends to the registry's global event
    log with empty `session_id` (the sentinel that makes the React
    reducer no-op).
  * Multiple subscribers all receive the event (fan-out works).
  * The chat MCP's ChatDeps.broadcast field defaults to a no-op, so
    tests that build a ChatDeps directly don't have to provide one.
"""
import asyncio
from pathlib import Path

import pytest

import main
from workspaces import WorkspaceStore, create_chat
from initiatives import InitiativeStore
from settings import SettingsStore


@pytest.fixture
def isolated_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Repoint main's module-level stores at a fresh tmp dir + give
    app.state.registry an event_log so our broadcasts land somewhere
    inspectable. We don't need the full SessionRegistry here — just a
    namespace with .event_log on it."""
    from atlassian_creds import CredsStore
    from agent_webkit_server.event_log import GlobalEventLog

    root = tmp_path / "blitz"
    root.mkdir()
    fake_ws = WorkspaceStore(root / "workspaces.json")
    monkeypatch.setattr(main, "workspace_store", fake_ws)
    monkeypatch.setattr(main, "initiative_store", InitiativeStore(root / "initiatives.json"))
    monkeypatch.setattr(main, "settings_store", SettingsStore(root / "settings.json"))
    monkeypatch.setattr(main, "creds_store", CredsStore(root / "atlassian-creds.json"))

    class _StubRegistry:
        event_log = GlobalEventLog()
    main.app.state.registry = _StubRegistry()  # type: ignore[attr-defined]
    return {"workspaces_root": tmp_path / "workspaces", "store": fake_ws}


# ── Broadcast helper ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_appends_with_empty_session_id(isolated_stores):
    registry = main.app.state.registry
    # Subscribe BEFORE emitting so we don't race the cursor.
    sub = registry.event_log.subscribe(0).__aiter__()
    main._broadcast("app:workspace_created", {"workspace_id": "abc", "kind": "chat"})
    ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
    assert ev.event == "app:workspace_created"
    assert ev.session_id == "", (
        "App events must use empty session_id so the React reducer's "
        "`if (!sid) return state` guard short-circuits cleanly."
    )
    assert ev.data == {"workspace_id": "abc", "kind": "chat"}


@pytest.mark.asyncio
async def test_broadcast_fans_out_to_every_subscriber(isolated_stores):
    """Two independent subscribers both see every broadcast — confirms
    the global event log fan-out semantics still apply to our app-level
    events even though they share a single sentinel session_id."""
    registry = main.app.state.registry
    a = registry.event_log.subscribe(0).__aiter__()
    b = registry.event_log.subscribe(0).__aiter__()
    main._broadcast("app:settings_updated", {})
    main._broadcast("app:lan_access_toggled", {"enabled": True})
    ea1 = await asyncio.wait_for(a.__anext__(), timeout=1.0)
    ea2 = await asyncio.wait_for(a.__anext__(), timeout=1.0)
    eb1 = await asyncio.wait_for(b.__anext__(), timeout=1.0)
    eb2 = await asyncio.wait_for(b.__anext__(), timeout=1.0)
    assert ea1.event == eb1.event == "app:settings_updated"
    assert ea2.event == eb2.event == "app:lan_access_toggled"
    assert ea2.data == {"enabled": True}


@pytest.mark.asyncio
async def test_broadcast_swallows_exceptions(isolated_stores, monkeypatch: pytest.MonkeyPatch):
    """If the event log is unavailable for any reason, _broadcast must
    not raise — mutations should always succeed regardless of fan-out."""
    class _Boom:
        def append(self, *_a, **_k):
            raise RuntimeError("event log unavailable")

    class _State:
        pass
    state = _State()
    state.registry = type("R", (), {"event_log": _Boom()})()  # type: ignore[attr-defined]
    monkeypatch.setattr(main.app, "state", state)
    # Should not raise.
    main._broadcast("app:workspace_updated", {"workspace_id": "x"})


# ── HTTP-handler emissions (end-to-end with the real handlers) ─────────────


@pytest.mark.asyncio
async def test_post_chat_emits_workspace_created(isolated_stores):
    """The chat-creation HTTP handler should fan out an
    app:workspace_created event."""
    from main import post_chat, CreateChatIn
    registry = main.app.state.registry
    sub = registry.event_log.subscribe(0).__aiter__()
    out = await post_chat(CreateChatIn(spawn_initial_session=False))
    ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
    assert ev.event == "app:workspace_created"
    assert ev.data["workspace_id"] == out["workspace"]["id"]
    assert ev.data["kind"] == "chat"


@pytest.mark.asyncio
async def test_patch_workspace_emits_workspace_updated(isolated_stores):
    from main import patch_workspace, PatchWorkspaceIn
    ws = await create_chat(isolated_stores["store"], workspaces_root=isolated_stores["workspaces_root"])
    registry = main.app.state.registry
    sub = registry.event_log.subscribe(0).__aiter__()
    await patch_workspace(ws.id, PatchWorkspaceIn(ticket_title="renamed"))
    ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
    assert ev.event == "app:workspace_updated"
    assert ev.data == {"workspace_id": ws.id}


@pytest.mark.asyncio
async def test_remove_workspace_emits_workspace_deleted(isolated_stores):
    from main import remove_workspace
    ws = await create_chat(isolated_stores["store"], workspaces_root=isolated_stores["workspaces_root"])
    registry = main.app.state.registry
    sub = registry.event_log.subscribe(0).__aiter__()
    await remove_workspace(ws.id, force=True)
    ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
    assert ev.event == "app:workspace_deleted"
    assert ev.data == {"workspace_id": ws.id}


@pytest.mark.asyncio
async def test_patch_settings_emits_settings_updated(isolated_stores):
    from main import patch_settings
    registry = main.app.state.registry
    sub = registry.event_log.subscribe(0).__aiter__()
    await patch_settings({"appearance": {"theme": "dark"}})
    ev = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
    assert ev.event == "app:settings_updated"
    assert ev.data == {}


# ── ChatDeps default broadcast is a no-op ──────────────────────────────────


def test_chat_deps_broadcast_defaults_to_noop():
    """Tests that instantiate ChatDeps directly (e.g. test_mcp_dispatch)
    shouldn't need to pass a broadcaster. The dataclass default lets
    them skip it."""
    from workflow_mcp.chat import ChatDeps
    deps = ChatDeps(
        initiative_store=InitiativeStore(Path("/nonexistent.json")),
        settings_store=SettingsStore(Path("/nonexistent.json")),
        workspace_store=WorkspaceStore(Path("/nonexistent.json")),
        workspaces_root=Path("/tmp"),
        get_registry=lambda: None,
    )
    # Must not raise.
    deps.broadcast("app:whatever", {"x": 1})

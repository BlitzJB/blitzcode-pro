"""MCP attach + tool-allowlist dispatch by workspace kind.

Pins the invariant: chat-kind workspaces get ONLY the chat MCP +
CHAT_ALLOWED_TOOLS, ticket-kind workspaces get ONLY the workflow MCP +
WORKFLOW_ALLOWED_TOOLS. The two surfaces never bleed into each other.
A regression here would mean the chat agent could touch JIRA/Confluence
write tools or the ticket agent could call the app-settings tools.
"""
from pathlib import Path

import pytest

import main
from workspaces import create_chat, create_workspace, RepoSpec


@pytest.fixture
def isolated_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Repoint main's module-level stores at a fresh tmp dir so the
    dispatch functions read from a clean state per test."""
    from workspaces import WorkspaceStore
    from initiatives import InitiativeStore
    from settings import SettingsStore
    from atlassian_creds import CredsStore

    root = tmp_path / "blitz"
    root.mkdir()
    fake_ws = WorkspaceStore(root / "workspaces.json")
    monkeypatch.setattr(main, "workspace_store", fake_ws)
    monkeypatch.setattr(main, "initiative_store", InitiativeStore(root / "initiatives.json"))
    monkeypatch.setattr(main, "settings_store", SettingsStore(root / "settings.json"))
    monkeypatch.setattr(main, "creds_store", CredsStore(root / "atlassian-creds.json"))
    return {"workspaces_root": tmp_path / "workspaces", "store": fake_ws}


def _mk_repo(tmp_path: Path, name: str) -> Path:
    import subprocess
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def _config_for(workspace_id: str):
    """Mimic the slice of SessionConfig the dispatch functions read."""
    class _C:
        pass
    c = _C()
    c.workspace_id = workspace_id
    return c


# ────────────────────────────────────────────────────────────────────────────
# Allowed-tools dispatch (cheap — no SDK import needed)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_workspace_gets_chat_plus_workflow_tools(isolated_stores, tmp_path):
    """Chat is the meta-agent — it gets the full workspace-agent surface
    PLUS the app-management tools. Ticket agents are the strict subset."""
    ws = await create_chat(isolated_stores["store"], workspaces_root=isolated_stores["workspaces_root"])
    allowed = main._workflow_extra_allowed_tools(_config_for(ws.id))
    assert allowed is not None
    chat_tools = [t for t in allowed if t.startswith("mcp__chat__app_")]
    wf_tools = [t for t in allowed if t.startswith("mcp__workflow__workflow_")]
    assert chat_tools, "chat workspace must have app_* tools"
    assert wf_tools, "chat workspace must also have workflow_* tools (it's the meta-agent)"
    # No third namespace creeping in.
    assert len(chat_tools) + len(wf_tools) == len(allowed)


@pytest.mark.asyncio
async def test_ticket_workspace_gets_only_workflow_tools(isolated_stores, tmp_path):
    src = _mk_repo(tmp_path, "src")
    ws = await create_workspace(
        isolated_stores["store"],
        workspaces_root=isolated_stores["workspaces_root"],
        ticket_key="LLM-1",
        ticket_title=None,
        initiative_key=None,
        repos=[RepoSpec(source_path=str(src))],
    )
    allowed = main._workflow_extra_allowed_tools(_config_for(ws.id))
    assert allowed is not None
    assert all(t.startswith("mcp__workflow__workflow_") for t in allowed)
    # The crucial isolation guarantee: ticket agents CANNOT see app_* tools.
    assert not any(t.startswith("mcp__chat__app_") for t in allowed)


def test_sessions_without_workspace_get_no_extra_tools(isolated_stores):
    c = _config_for(None)
    assert main._workflow_extra_allowed_tools(c) is None


def test_unknown_workspace_yields_no_extra_tools(isolated_stores):
    c = _config_for("ws-does-not-exist")
    assert main._workflow_extra_allowed_tools(c) is None


# ────────────────────────────────────────────────────────────────────────────
# Prompt dispatch
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_workspace_gets_chat_prompt(isolated_stores):
    ws = await create_chat(isolated_stores["store"], workspaces_root=isolated_stores["workspaces_root"])
    prompt = main._workflow_system_prompt_append(_config_for(ws.id))
    assert prompt is not None
    # Chat prompt advertises the app_* surface.
    assert "chat workspace" in prompt
    assert "app_list_workspaces" in prompt
    assert "app_create_ticket_workspace" in prompt
    # Must NOT inject the ticket-lifecycle rulebook from the workflow prompt.
    assert "Phase 1 — Initiation" not in prompt
    assert "## Lifecycle" not in prompt


@pytest.mark.asyncio
async def test_ticket_workspace_gets_workflow_prompt(isolated_stores, tmp_path):
    src = _mk_repo(tmp_path, "src")
    ws = await create_workspace(
        isolated_stores["store"],
        workspaces_root=isolated_stores["workspaces_root"],
        ticket_key="LLM-2",
        ticket_title="t",
        initiative_key=None,
        repos=[RepoSpec(source_path=str(src))],
    )
    prompt = main._workflow_system_prompt_append(_config_for(ws.id))
    assert prompt is not None
    # The workflow prompt has the lifecycle rulebook; the chat prompt
    # does not. This is the canonical "is this the right prompt?" check.
    assert "Phase 1 — Initiation" in prompt
    # And it must NOT advertise the app_* tools to ticket agents.
    assert "app_create_ticket_workspace" not in prompt

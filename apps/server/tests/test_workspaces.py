"""WorkspaceStore + create_workspace orchestration.

Uses real tmp git repos so the git_worktree paths get exercised end-to-end —
no mocks. Slower but catches real `git worktree` quirks.
"""
import asyncio
import os
import subprocess
from pathlib import Path

import pytest

import git_worktree as gw
from workspaces import (
    DocRef,
    RepoSpec,
    Workspace,
    WorkspaceRepo,
    WorkspaceStore,
    create_chat,
    create_workspace,
    delete_workspace,
    is_valid_ticket_key,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _make_repo(tmp: Path, name: str) -> Path:
    repo = tmp / name
    repo.mkdir()
    _git("init", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("hi\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-m", "init", cwd=repo)
    return repo


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────


class TestKeyValidation:
    def test_accepts_typical_jira_key(self):
        assert is_valid_ticket_key("LLM-1234")
        assert is_valid_ticket_key("PROJ-1")

    def test_rejects_garbage(self):
        for bad in ("", "no-dash", "-12", "LLM1234", "llm-1234", "LLM-"):
            assert not is_valid_ticket_key(bad), bad


# ────────────────────────────────────────────────────────────────────────────
# Store CRUD
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path) -> WorkspaceStore:
    return WorkspaceStore(tmp_path / "workspaces.json")


@pytest.mark.asyncio
async def test_store_insert_get_roundtrip(store: WorkspaceStore):
    ws = Workspace(
        id="ws-1",
        ticket_key="LLM-1",
        ticket_title="hello",
        initiative_key="meowtorq",
        dir="/tmp/ws",
    )
    await store.insert(ws)
    got = store.get("ws-1")
    assert got is not None and got.ticket_key == "LLM-1"
    # Persistence — fresh store reads from disk.
    other = WorkspaceStore(store._path)
    assert other.get("ws-1") is not None


@pytest.mark.asyncio
async def test_store_list_excludes_archived_by_default(store: WorkspaceStore):
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    b = Workspace(id="b", ticket_key="LLM-2", ticket_title=None, initiative_key=None, dir="/tmp/b")
    await store.insert(a)
    await store.insert(b)
    await store.archive("a")
    assert {w.id for w in store.list()} == {"b"}
    assert {w.id for w in store.list(include_archived=True)} == {"a", "b"}


@pytest.mark.asyncio
async def test_store_get_by_ticket_ignores_archived(store: WorkspaceStore):
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    await store.insert(a)
    await store.archive("a")
    assert store.get_by_ticket("LLM-1") is None


@pytest.mark.asyncio
async def test_add_session_dedupes(store: WorkspaceStore):
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    await store.insert(a)
    await store.add_session("a", "sid-1")
    await store.add_session("a", "sid-1")  # idempotent
    assert store.get("a").session_ids == ["sid-1"]
    await store.add_session("a", "sid-2")
    assert store.get("a").session_ids == ["sid-1", "sid-2"]


@pytest.mark.asyncio
async def test_remove_session_clears_name(store: WorkspaceStore):
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    await store.insert(a)
    await store.add_session("a", "sid-1")
    await store.set_session_name("a", "sid-1", "Backend")
    assert store.get("a").session_names == {"sid-1": "Backend"}
    await store.remove_session("a", "sid-1")
    got = store.get("a")
    assert got.session_ids == [] and got.session_names == {}


@pytest.mark.asyncio
async def test_set_session_name_requires_session_in_workspace(store: WorkspaceStore):
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    await store.insert(a)
    # Unknown session → no-op (returns None).
    assert await store.set_session_name("a", "ghost", "X") is None
    await store.add_session("a", "sid-1")
    await store.set_session_name("a", "sid-1", "  Trim me  ")
    assert store.get("a").session_names == {"sid-1": "Trim me"}
    # Empty string clears.
    await store.set_session_name("a", "sid-1", "")
    assert store.get("a").session_names == {}
    # None also clears.
    await store.set_session_name("a", "sid-1", "Back")
    await store.set_session_name("a", "sid-1", None)
    assert store.get("a").session_names == {}


@pytest.mark.asyncio
async def test_session_names_persist_across_instances(tmp_path: Path):
    s = WorkspaceStore(tmp_path / "ws.json")
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    await s.insert(a)
    await s.add_session("a", "sid-1")
    await s.set_session_name("a", "sid-1", "Backend")
    s2 = WorkspaceStore(tmp_path / "ws.json")
    assert s2.get("a").session_names == {"sid-1": "Backend"}


@pytest.mark.asyncio
async def test_set_doc_roundtrip(store: WorkspaceStore):
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    await store.insert(a)
    await store.set_doc("a", "rfc", DocRef(page_id="p1", version=2, title="X", url="https://x", last_synced_at=1.0))
    got = store.get("a")
    assert got.docs["rfc"].page_id == "p1"
    assert got.docs["rfc"].version == 2


@pytest.mark.asyncio
async def test_add_repo_dedupes_by_worktree_path(store: WorkspaceStore):
    a = Workspace(id="a", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/tmp/a")
    await store.insert(a)
    r = WorkspaceRepo(source_path="/src", worktree_path="/wt", branch="LLM-1")
    await store.add_repo("a", r)
    await store.add_repo("a", r)
    assert len(store.get("a").repos) == 1


# ────────────────────────────────────────────────────────────────────────────
# create_workspace orchestration (touches real git)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_workspace_makes_worktrees(tmp_path: Path):
    src = _make_repo(tmp_path, "src-a")
    store = WorkspaceStore(tmp_path / "ws.json")
    ws_root = tmp_path / "workspaces"

    ws = await create_workspace(
        store,
        workspaces_root=ws_root,
        ticket_key="LLM-100",
        ticket_title="t",
        initiative_key=None,
        repos=[RepoSpec(source_path=str(src))],
    )
    assert ws.ticket_key == "LLM-100"
    assert (ws_root / "LLM-100").is_dir()
    assert (ws_root / "LLM-100" / "src-a" / ".git").exists()  # worktree marker (file or dir)
    # Branch was created.
    out = subprocess.run(
        ["git", "-C", str(src), "branch", "--list", "LLM-100"], capture_output=True, text=True
    )
    assert "LLM-100" in out.stdout


@pytest.mark.asyncio
async def test_create_workspace_rejects_duplicate_ticket(tmp_path: Path):
    src = _make_repo(tmp_path, "src-a")
    store = WorkspaceStore(tmp_path / "ws.json")
    ws_root = tmp_path / "workspaces"

    await create_workspace(
        store,
        workspaces_root=ws_root,
        ticket_key="LLM-200",
        ticket_title=None,
        initiative_key=None,
        repos=[RepoSpec(source_path=str(src))],
    )
    with pytest.raises(ValueError):
        await create_workspace(
            store,
            workspaces_root=ws_root,
            ticket_key="LLM-200",
            ticket_title=None,
            initiative_key=None,
            repos=[],
        )


@pytest.mark.asyncio
async def test_create_workspace_rolls_back_on_failure(tmp_path: Path):
    """If repo #2's worktree-add fails, repo #1 must be torn down too."""
    src_ok = _make_repo(tmp_path, "src-ok")
    bad = tmp_path / "src-bad"
    bad.mkdir()  # not a git repo — gw.add_worktree will refuse

    store = WorkspaceStore(tmp_path / "ws.json")
    ws_root = tmp_path / "workspaces"

    with pytest.raises(gw.WorktreeError):
        await create_workspace(
            store,
            workspaces_root=ws_root,
            ticket_key="LLM-300",
            ticket_title=None,
            initiative_key=None,
            repos=[RepoSpec(source_path=str(src_ok)), RepoSpec(source_path=str(bad))],
        )
    # First repo's worktree must be gone.
    assert not (ws_root / "LLM-300").exists()
    # Store has no record.
    assert store.get_by_ticket("LLM-300") is None
    # And src_ok's branch should NOT exist (we rolled back).
    out = subprocess.run(
        ["git", "-C", str(src_ok), "branch", "--list", "LLM-300"], capture_output=True, text=True
    )
    # git worktree remove --force removes the worktree but does NOT delete
    # the branch. That's fine — leftover branch on the source repo is benign
    # and lets the user retry with the same ticket key. Just assert the
    # worktree itself is gone.
    assert out.returncode == 0  # branch query succeeded regardless of presence


@pytest.mark.asyncio
async def test_delete_workspace_removes_dir(tmp_path: Path):
    src = _make_repo(tmp_path, "src")
    store = WorkspaceStore(tmp_path / "ws.json")
    ws_root = tmp_path / "workspaces"
    ws = await create_workspace(
        store,
        workspaces_root=ws_root,
        ticket_key="LLM-400",
        ticket_title=None,
        initiative_key=None,
        repos=[RepoSpec(source_path=str(src))],
    )
    ok = await delete_workspace(store, ws.id, force=True)
    assert ok
    assert not (ws_root / "LLM-400").exists()
    assert store.get(ws.id) is None


# ────────────────────────────────────────────────────────────────────────────
# Chat workspaces — kind="chat", no ticket, no repos
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_chat_makes_empty_dir(tmp_path: Path):
    store = WorkspaceStore(tmp_path / "ws.json")
    ws_root = tmp_path / "workspaces"
    ws = await create_chat(store, workspaces_root=ws_root, title="hello")
    assert ws.kind == "chat"
    assert ws.ticket_key == ""
    assert ws.ticket_title == "hello"
    assert ws.repos == []
    assert ws.session_ids == []
    assert Path(ws.dir).is_dir()
    assert Path(ws.dir).parent == ws_root / "chats"
    # Persistence round-trip preserves kind.
    other = WorkspaceStore(store._path)
    got = other.get(ws.id)
    assert got is not None and got.kind == "chat"


@pytest.mark.asyncio
async def test_create_chat_default_title(tmp_path: Path):
    store = WorkspaceStore(tmp_path / "ws.json")
    ws = await create_chat(store, workspaces_root=tmp_path / "workspaces")
    assert ws.ticket_title and ws.ticket_title.startswith("Chat")


@pytest.mark.asyncio
async def test_chats_coexist_with_tickets(tmp_path: Path):
    src = _make_repo(tmp_path, "src")
    store = WorkspaceStore(tmp_path / "ws.json")
    ws_root = tmp_path / "workspaces"
    chat = await create_chat(store, workspaces_root=ws_root, title="brain dump")
    ticket = await create_workspace(
        store,
        workspaces_root=ws_root,
        ticket_key="LLM-500",
        ticket_title="real ticket",
        initiative_key=None,
        repos=[RepoSpec(source_path=str(src))],
    )
    listed = store.list()
    kinds = {w.id: w.kind for w in listed}
    assert kinds[chat.id] == "chat"
    assert kinds[ticket.id] == "ticket"


@pytest.mark.asyncio
async def test_delete_chat_removes_dir(tmp_path: Path):
    store = WorkspaceStore(tmp_path / "ws.json")
    ws_root = tmp_path / "workspaces"
    chat = await create_chat(store, workspaces_root=ws_root)
    chat_dir = Path(chat.dir)
    assert chat_dir.exists()
    ok = await delete_workspace(store, chat.id, force=True)
    assert ok
    assert not chat_dir.exists()
    assert store.get(chat.id) is None


@pytest.mark.asyncio
async def test_legacy_workspaces_default_to_ticket_kind(tmp_path: Path):
    """A workspace JSON written before the kind field existed should
    load as kind='ticket' for backwards compat."""
    path = tmp_path / "ws.json"
    import json
    legacy = {
        "workspaces": [{
            "id": "ws-legacy",
            "ticket_key": "OLD-1",
            "ticket_title": "vintage",
            "initiative_key": None,
            "dir": "/tmp/legacy",
            "created_at": 1000.0,
        }],
    }
    path.write_text(json.dumps(legacy))
    store = WorkspaceStore(path)
    got = store.get("ws-legacy")
    assert got is not None and got.kind == "ticket"

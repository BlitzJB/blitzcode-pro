"""App-layer workspace registry — the top-level primitive of blitzcode-pro.

A workspace = a ticket + a directory containing git worktrees of N source
repos + N sessions running inside it. Pure JSON-backed store; the actual
worktree mechanics live in git_worktree.py and the orchestration in this
module's `create_workspace` helper.

Not part of agent-webkit. Owned by apps/server.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import git_worktree as gw  # apps/server runs flat (see main.py imports)


_TICKET_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")  # JIRA project keys are uppercase


@dataclass
class WorkspaceRepo:
    source_path: str          # absolute, e.g. ~/projects/meowtorq
    worktree_path: str        # absolute, inside workspace dir
    branch: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceRepo":
        return cls(
            source_path=str(data["source_path"]),
            worktree_path=str(data["worktree_path"]),
            branch=str(data["branch"]),
        )


@dataclass
class DocRef:
    """Reference to a Confluence page (or any external doc) the workspace owns."""
    page_id: Optional[str] = None
    version: Optional[int] = None
    title: Optional[str] = None
    url: Optional[str] = None
    last_synced_at: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocRef":
        return cls(
            page_id=(data.get("page_id") or None) or None,
            version=int(data["version"]) if data.get("version") is not None else None,
            title=(data.get("title") or None) or None,
            url=(data.get("url") or None) or None,
            last_synced_at=(float(data["last_synced_at"]) if data.get("last_synced_at") is not None else None),
        )


@dataclass
class Workspace:
    id: str
    ticket_key: str
    ticket_title: Optional[str]
    initiative_key: Optional[str]
    dir: str
    repos: list[WorkspaceRepo] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    # Optional per-session display names — keyed by session_id. The UI
    # uses these for session-tab labels when present, falling back to
    # "S1", "S2", … otherwise.
    session_names: dict[str, str] = field(default_factory=dict)
    docs: dict[str, DocRef] = field(default_factory=dict)  # "rfc" | "debrief"
    created_at: float = field(default_factory=time.time)
    archived_at: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["repos"] = [r.to_dict() for r in self.repos]
        d["docs"] = {k: v.to_dict() for k, v in self.docs.items()}
        d["session_names"] = dict(self.session_names)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Workspace":
        return cls(
            id=str(data["id"]),
            ticket_key=str(data["ticket_key"]),
            ticket_title=(data.get("ticket_title") or None) or None,
            initiative_key=(data.get("initiative_key") or None) or None,
            dir=str(data["dir"]),
            repos=[WorkspaceRepo.from_dict(r) for r in (data.get("repos") or []) if isinstance(r, dict)],
            session_ids=[str(s) for s in (data.get("session_ids") or []) if isinstance(s, str)],
            session_names={
                str(k): str(v)
                for k, v in (data.get("session_names") or {}).items()
                if isinstance(k, str) and isinstance(v, str)
            },
            docs={
                str(k): DocRef.from_dict(v)
                for k, v in (data.get("docs") or {}).items()
                if isinstance(v, dict)
            },
            created_at=float(data.get("created_at", time.time())),
            archived_at=(float(data["archived_at"]) if data.get("archived_at") is not None else None),
        )


def is_valid_ticket_key(key: str) -> bool:
    return bool(key) and bool(_TICKET_KEY_RE.match(key))


class WorkspaceStore:
    """JSON-file backed map of workspace id → Workspace."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._data: dict[str, Workspace] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        items = raw.get("workspaces", []) if isinstance(raw, dict) else []
        if not isinstance(items, list):
            return
        for entry in items:
            if not isinstance(entry, dict):
                continue
            try:
                ws = Workspace.from_dict(entry)
            except (KeyError, TypeError, ValueError):
                continue
            self._data[ws.id] = ws

    async def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {"workspaces": [ws.to_dict() for ws in self._data.values()]}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self._path)

    def list(self, *, include_archived: bool = False) -> list[Workspace]:
        out: list[Workspace] = []
        for ws in self._data.values():
            if ws.archived_at is not None and not include_archived:
                continue
            out.append(self._clone(ws))
        out.sort(key=lambda w: w.created_at, reverse=True)
        return out

    def get(self, workspace_id: str) -> Optional[Workspace]:
        ws = self._data.get(workspace_id)
        return self._clone(ws) if ws is not None else None

    def get_by_ticket(self, ticket_key: str) -> Optional[Workspace]:
        for ws in self._data.values():
            if ws.ticket_key == ticket_key and ws.archived_at is None:
                return self._clone(ws)
        return None

    def _clone(self, ws: Workspace) -> Workspace:
        return Workspace.from_dict(ws.to_dict())

    async def insert(self, ws: Workspace) -> Workspace:
        async with self._lock:
            if ws.id in self._data:
                raise ValueError(f"workspace id collision: {ws.id}")
            self._data[ws.id] = self._clone(ws)
            await self._flush()
            return self._clone(self._data[ws.id])

    async def patch(self, workspace_id: str, **fields) -> Optional[Workspace]:
        async with self._lock:
            existing = self._data.get(workspace_id)
            if existing is None:
                return None
            for k, v in fields.items():
                if not hasattr(existing, k):
                    continue
                setattr(existing, k, v)
            await self._flush()
            return self._clone(existing)

    async def add_session(self, workspace_id: str, session_id: str) -> Optional[Workspace]:
        async with self._lock:
            existing = self._data.get(workspace_id)
            if existing is None:
                return None
            if session_id not in existing.session_ids:
                existing.session_ids.append(session_id)
                await self._flush()
            return self._clone(existing)

    async def remove_session(self, workspace_id: str, session_id: str) -> Optional[Workspace]:
        async with self._lock:
            existing = self._data.get(workspace_id)
            if existing is None:
                return None
            dirty = False
            if session_id in existing.session_ids:
                existing.session_ids.remove(session_id)
                dirty = True
            if session_id in existing.session_names:
                del existing.session_names[session_id]
                dirty = True
            if dirty:
                await self._flush()
            return self._clone(existing)

    async def set_session_name(
        self, workspace_id: str, session_id: str, name: Optional[str]
    ) -> Optional[Workspace]:
        """Rename a session tab. Pass `None`/empty string to clear back to default."""
        async with self._lock:
            existing = self._data.get(workspace_id)
            if existing is None:
                return None
            if session_id not in existing.session_ids:
                return None
            trimmed = (name or "").strip()
            if trimmed:
                existing.session_names[session_id] = trimmed
            else:
                existing.session_names.pop(session_id, None)
            await self._flush()
            return self._clone(existing)

    async def add_repo(self, workspace_id: str, repo: WorkspaceRepo) -> Optional[Workspace]:
        async with self._lock:
            existing = self._data.get(workspace_id)
            if existing is None:
                return None
            # No duplicates on (source_path, worktree_path).
            for r in existing.repos:
                if r.worktree_path == repo.worktree_path:
                    return self._clone(existing)
            existing.repos.append(repo)
            await self._flush()
            return self._clone(existing)

    async def set_doc(self, workspace_id: str, kind: str, doc: DocRef) -> Optional[Workspace]:
        async with self._lock:
            existing = self._data.get(workspace_id)
            if existing is None:
                return None
            existing.docs[kind] = doc
            await self._flush()
            return self._clone(existing)

    async def archive(self, workspace_id: str) -> Optional[Workspace]:
        return await self.patch(workspace_id, archived_at=time.time())

    async def unarchive(self, workspace_id: str) -> Optional[Workspace]:
        return await self.patch(workspace_id, archived_at=None)

    async def delete(self, workspace_id: str) -> bool:
        async with self._lock:
            if workspace_id in self._data:
                del self._data[workspace_id]
                await self._flush()
                return True
            return False


# ────────────────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class RepoSpec:
    """User input: a repo to add to a workspace at creation time."""
    source_path: str
    branch: Optional[str] = None  # defaults to ticket_key on resolve


async def create_workspace(
    store: WorkspaceStore,
    *,
    workspaces_root: Path,
    ticket_key: str,
    ticket_title: Optional[str],
    initiative_key: Optional[str],
    repos: list[RepoSpec],
) -> Workspace:
    """Materialize a workspace dir + per-repo worktrees + store record.

    Raises ValueError on duplicate ticket / invalid input, and
    git_worktree.WorktreeError on git failures. Cleans up partial state
    on failure so the user can retry.
    """
    if not is_valid_ticket_key(ticket_key):
        raise ValueError(f"Invalid ticket key: {ticket_key!r}")
    if store.get_by_ticket(ticket_key) is not None:
        raise ValueError(f"Workspace already exists for ticket {ticket_key}")

    workspace_id = str(uuid.uuid4())
    workspace_dir = workspaces_root / ticket_key
    if workspace_dir.exists():
        raise ValueError(f"Workspace dir already exists on disk: {workspace_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=False)

    created_repos: list[WorkspaceRepo] = []
    try:
        for spec in repos:
            source = str(Path(spec.source_path).expanduser().resolve())
            branch = spec.branch or ticket_key
            repo_name = Path(source).name or "repo"
            worktree_path = str(workspace_dir / repo_name)
            await gw.add_worktree(source, worktree_path, branch)
            created_repos.append(WorkspaceRepo(
                source_path=source,
                worktree_path=worktree_path,
                branch=branch,
            ))
    except Exception:
        # Tear down anything we managed to create so the user's next attempt
        # isn't fighting half-state.
        for r in created_repos:
            try:
                await gw.remove_worktree(r.source_path, r.worktree_path, force=True)
            except Exception:
                pass
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
        raise

    ws = Workspace(
        id=workspace_id,
        ticket_key=ticket_key,
        ticket_title=ticket_title,
        initiative_key=initiative_key,
        dir=str(workspace_dir),
        repos=created_repos,
        session_ids=[],
        docs={},
    )
    return await store.insert(ws)


async def delete_workspace(
    store: WorkspaceStore,
    workspace_id: str,
    *,
    force: bool = True,
) -> bool:
    """Destroy a workspace: remove worktrees + rm -rf dir + drop store entry.

    Force=True because the user has already confirmed at the UI layer. Set
    force=False to refuse when any worktree has uncommitted changes.
    """
    ws = store.get(workspace_id)
    if ws is None:
        return False
    for repo in ws.repos:
        try:
            await gw.remove_worktree(repo.source_path, repo.worktree_path, force=force)
        except Exception:
            # Soldier on — we still want to free the workspace record. The
            # disk may be left with crumbs; the user can clean up manually.
            pass
    workspace_dir = Path(ws.dir)
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)
    return await store.delete(workspace_id)

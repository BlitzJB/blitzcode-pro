"""Chat MCP — tools available to agents running inside a chat workspace.

Chat workspaces are scratch conversations not bound to any ticket. The
agent's job here is general assistance plus *app-level* configuration:
listing/creating/updating/deleting initiatives, reading and patching
user settings. The tools intentionally cover only what's safe for an
agent to operate at the application boundary — no workspace
mutation, no atlassian writes.

When new "spawnable agent" kinds land (review, incident, etc.), they
attach this same MCP plus their own narrow tool surface. This module is
the seed of that pattern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from pathlib import Path
from typing import Callable as _Callable

from initiatives import Initiative, InitiativeStore, is_valid_key as is_valid_initiative_key
from settings import SettingsStore
from workspaces import (
    RepoSpec,
    WorkspaceStore,
    create_chat as _create_chat,
    create_workspace as _create_workspace,
    delete_workspace as _delete_workspace,
    is_valid_ticket_key,
)


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Result helpers — mirror workflow.py so error shape stays consistent.
# ────────────────────────────────────────────────────────────────────────────


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    import json as _json
    payload = {"error": {"code": code, "message": message, **extra}}
    return {"content": [{"type": "text", "text": _json.dumps(payload)}], "isError": True}


class _ToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _str(args: dict, key: str) -> str:
    v = args.get(key)
    if not isinstance(v, str) or not v.strip():
        raise _ToolError("invalid_args", f"Missing required string field: {key}")
    return v.strip()


def _str_opt(args: dict, key: str) -> Optional[str]:
    v = args.get(key)
    return v.strip() if isinstance(v, str) and v.strip() else None


def _list_str(args: dict, key: str) -> Optional[list[str]]:
    v = args.get(key)
    if v is None:
        return None
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise _ToolError("invalid_args", f"{key} must be a list of strings")
    return [s.strip() for s in v if s.strip()]


# Same slugifier the HTTP layer uses for initiative keys. Inline so the
# MCP doesn't need to import from main.py (which would pull the world).
_SLUG_RE = re.compile(r"[^a-z0-9-]")


def _slug(raw: str) -> str:
    s = (raw or "").strip().lower().replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    s = _SLUG_RE.sub("", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _initiative_dict(it: Initiative) -> dict[str, Any]:
    return {
        "key": it.key,
        "display_name": it.display_name,
        "epic_jira_key": it.epic_jira_key,
        "confluence_root_page_id": it.confluence_root_page_id,
        "repo_paths": list(it.repo_paths),
    }


# ────────────────────────────────────────────────────────────────────────────
# Server builder
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ChatDeps:
    initiative_store: InitiativeStore
    settings_store: SettingsStore
    workspace_store: WorkspaceStore
    workspaces_root: Path
    # Returns the live SessionRegistry. Callable so we don't import the
    # SDK or take a runtime dep on `main.app` at module-load time.
    get_registry: _Callable[[], object]
    # Broadcasts an app-level wire event to every connected client via
    # the global event log. main.py wires this to `_broadcast`. Tests
    # default it to a no-op so they don't need a live registry.
    broadcast: _Callable[[str, dict[str, Any]], None] = lambda _e, _d: None  # type: ignore[assignment]


def build_chat_mcp_server(deps: ChatDeps):
    """Construct the in-process SDK MCP server. Returns the server object
    suitable for `ClaudeAgentOptions(mcp_servers={"chat": server})`.

    Late-imports the SDK so test fixtures don't require it installed.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # type: ignore

    def wrap(fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]):
        async def safe(args: dict[str, Any]) -> dict[str, Any]:
            try:
                return await fn(args)
            except _ToolError as e:
                return _error(e.code, str(e))
            except Exception as e:
                logger.exception("Chat tool crashed")
                return _error("internal_error", str(e))
        return safe

    # ── Initiative tools ─────────────────────────────────────────────────

    async def t_list_initiatives(_args: dict) -> dict:
        rows = [_initiative_dict(it) for it in deps.initiative_store.list()]
        import json as _json
        return _text(_json.dumps(rows, indent=2))

    async def t_create_initiative(args: dict) -> dict:
        raw_key = _str(args, "key")
        key = _slug(raw_key)
        if not is_valid_initiative_key(key):
            raise _ToolError("invalid_args", f"Key {raw_key!r} normalizes to {key!r}, which isn't valid (need at least one alphanumeric).")
        if deps.initiative_store.get(key) is not None:
            raise _ToolError("conflict", f"Initiative {key!r} already exists. Use app_update_initiative to modify.")
        it = Initiative(
            key=key,
            display_name=_str_opt(args, "display_name") or key,
            epic_jira_key=_str_opt(args, "epic_jira_key"),
            confluence_root_page_id=_str_opt(args, "confluence_root_page_id"),
            repo_paths=_list_str(args, "repo_paths") or [],
        )
        saved = await deps.initiative_store.upsert(it)
        deps.broadcast("app:initiative_created", {"key": saved.key})
        import json as _json
        return _text(_json.dumps(_initiative_dict(saved), indent=2))

    async def t_update_initiative(args: dict) -> dict:
        key = _str(args, "key")
        if deps.initiative_store.get(key) is None:
            raise _ToolError("not_found", f"Initiative {key!r} doesn't exist.")
        fields: dict[str, Any] = {}
        for f in ("display_name", "epic_jira_key", "confluence_root_page_id"):
            v = _str_opt(args, f)
            if v is not None:
                fields[f] = v
        repos = _list_str(args, "repo_paths")
        if repos is not None:
            fields["repo_paths"] = repos
        if not fields:
            raise _ToolError("invalid_args", "Nothing to update. Provide at least one of display_name, epic_jira_key, confluence_root_page_id, repo_paths.")
        updated = await deps.initiative_store.patch(key, **fields)
        if updated is None:
            raise _ToolError("not_found", f"Initiative {key!r} disappeared during update.")
        deps.broadcast("app:initiative_updated", {"key": updated.key})
        import json as _json
        return _text(_json.dumps(_initiative_dict(updated), indent=2))

    async def t_delete_initiative(args: dict) -> dict:
        key = _str(args, "key")
        ok = await deps.initiative_store.remove(key)
        if not ok:
            raise _ToolError("not_found", f"Initiative {key!r} doesn't exist.")
        deps.broadcast("app:initiative_deleted", {"key": key})
        return _text(f"Removed initiative {key!r}.")

    # ── Settings tools ───────────────────────────────────────────────────

    async def t_get_settings(_args: dict) -> dict:
        import json as _json
        return _text(_json.dumps(deps.settings_store.snapshot(), indent=2))

    async def t_patch_settings(args: dict) -> dict:
        updates = args.get("updates")
        if not isinstance(updates, dict):
            raise _ToolError("invalid_args", "`updates` must be an object: {category: {key: value | null}}.")
        merged = await deps.settings_store.patch(updates)
        deps.broadcast("app:settings_updated", {})
        import json as _json
        return _text(_json.dumps(merged, indent=2))

    # ── Workspace + session management ───────────────────────────────────
    #
    # These let the chat agent see and reshape app state: which tickets
    # have workspaces, what sessions live in each, archive/delete, move
    # sessions between workspaces.

    def _workspace_summary(ws) -> dict:
        return {
            "id": ws.id,
            "kind": ws.kind,
            "ticket_key": ws.ticket_key or None,
            "title": ws.ticket_title,
            "initiative_key": ws.initiative_key,
            "dir": ws.dir,
            "repos": [{"source_path": r.source_path, "branch": r.branch} for r in ws.repos],
            "session_ids": list(ws.session_ids),
            "session_names": dict(ws.session_names),
            "archived": ws.archived_at is not None,
        }

    async def t_list_workspaces(args: dict) -> dict:
        include_archived = bool(args.get("include_archived"))
        rows = [_workspace_summary(w) for w in deps.workspace_store.list(include_archived=include_archived)]
        import json as _json
        return _text(_json.dumps(rows, indent=2))

    async def t_get_workspace(args: dict) -> dict:
        ws_id = _str(args, "workspace_id")
        ws = deps.workspace_store.get(ws_id)
        if ws is None:
            raise _ToolError("not_found", f"Workspace {ws_id!r} doesn't exist.")
        import json as _json
        return _text(_json.dumps(_workspace_summary(ws), indent=2))

    async def t_create_ticket_workspace(args: dict) -> dict:
        """Spawn a ticket-bound workspace (with worktrees). The chat
        agent calls this after `workflow_create_ticket` to land the new
        ticket on disk + in the sidebar."""
        ticket_key = _str(args, "ticket_key")
        if not is_valid_ticket_key(ticket_key):
            raise _ToolError("invalid_args", f"{ticket_key!r} isn't a valid JIRA key (e.g. 'LLM-1234').")
        ticket_title = _str_opt(args, "ticket_title")
        initiative_key = _str_opt(args, "initiative_key")
        if initiative_key and deps.initiative_store.get(initiative_key) is None:
            raise _ToolError("not_found", f"Initiative {initiative_key!r} doesn't exist.")
        repos = []
        for p in _list_str(args, "repo_paths") or []:
            repos.append(RepoSpec(source_path=p))
        try:
            ws = await _create_workspace(
                deps.workspace_store,
                workspaces_root=deps.workspaces_root,
                ticket_key=ticket_key,
                ticket_title=ticket_title,
                initiative_key=initiative_key,
                repos=repos,
            )
        except ValueError as e:
            raise _ToolError("invalid_args", str(e))
        deps.broadcast("app:workspace_created", {"workspace_id": ws.id, "kind": ws.kind})
        import json as _json
        return _text(_json.dumps(_workspace_summary(ws), indent=2))

    async def t_create_chat_workspace(args: dict) -> dict:
        """Spawn another chat workspace. The chat agent rarely needs
        this (the user has a button), but it's useful for 'archive this
        and start a fresh one' flows."""
        title = _str_opt(args, "title")
        ws = await _create_chat(
            deps.workspace_store,
            workspaces_root=deps.workspaces_root,
            title=title,
        )
        deps.broadcast("app:workspace_created", {"workspace_id": ws.id, "kind": ws.kind})
        import json as _json
        return _text(_json.dumps(_workspace_summary(ws), indent=2))

    async def t_update_workspace(args: dict) -> dict:
        ws_id = _str(args, "workspace_id")
        existing = deps.workspace_store.get(ws_id)
        if existing is None:
            raise _ToolError("not_found", f"Workspace {ws_id!r} doesn't exist.")
        fields: dict[str, Any] = {}
        title = _str_opt(args, "ticket_title")
        if title is not None:
            fields["ticket_title"] = title
        if "initiative_key" in args:
            ik = args["initiative_key"]
            if ik in (None, ""):
                fields["initiative_key"] = None
            elif isinstance(ik, str):
                if deps.initiative_store.get(ik) is None:
                    raise _ToolError("not_found", f"Initiative {ik!r} doesn't exist.")
                fields["initiative_key"] = ik
        if not fields:
            raise _ToolError("invalid_args", "Provide ticket_title or initiative_key to update.")
        updated = await deps.workspace_store.patch(ws_id, **fields)
        if updated is None:
            raise _ToolError("not_found", "Workspace disappeared during update.")
        deps.broadcast("app:workspace_updated", {"workspace_id": ws_id})
        import json as _json
        return _text(_json.dumps(_workspace_summary(updated), indent=2))

    async def t_archive_workspace(args: dict) -> dict:
        ws_id = _str(args, "workspace_id")
        out = await deps.workspace_store.archive(ws_id)
        if out is None:
            raise _ToolError("not_found", f"Workspace {ws_id!r} doesn't exist.")
        deps.broadcast("app:workspace_updated", {"workspace_id": ws_id})
        return _text(f"Archived workspace {ws_id} ({out.ticket_key or 'chat'}).")

    async def t_unarchive_workspace(args: dict) -> dict:
        ws_id = _str(args, "workspace_id")
        out = await deps.workspace_store.unarchive(ws_id)
        if out is None:
            raise _ToolError("not_found", f"Workspace {ws_id!r} doesn't exist.")
        deps.broadcast("app:workspace_updated", {"workspace_id": ws_id})
        return _text(f"Unarchived workspace {ws_id}.")

    async def t_delete_workspace(args: dict) -> dict:
        ws_id = _str(args, "workspace_id")
        ws = deps.workspace_store.get(ws_id)
        if ws is None:
            raise _ToolError("not_found", f"Workspace {ws_id!r} doesn't exist.")
        # Best-effort kill of every live session in the workspace first.
        registry = deps.get_registry()
        for sid in list(ws.session_ids):
            try:
                await registry.remove(sid, purge_metadata=True)  # type: ignore[attr-defined]
            except Exception:
                pass
        ok = await _delete_workspace(deps.workspace_store, ws_id, force=True)
        if not ok:
            raise _ToolError("internal_error", "Delete returned false unexpectedly.")
        deps.broadcast("app:workspace_deleted", {"workspace_id": ws_id})
        return _text(f"Deleted workspace {ws_id} (sessions, worktrees, dir).")

    async def t_rename_session(args: dict) -> dict:
        ws_id = _str(args, "workspace_id")
        sid = _str(args, "session_id")
        name = args.get("name")
        if name is not None and not isinstance(name, str):
            raise _ToolError("invalid_args", "`name` must be a string or null.")
        out = await deps.workspace_store.set_session_name(ws_id, sid, name)
        if out is None:
            raise _ToolError("not_found", "Workspace or session not found.")
        deps.broadcast("app:workspace_updated", {"workspace_id": ws_id})
        label = (name or "").strip() or "(default)"
        return _text(f"Renamed session {sid} in {ws_id} → {label}")

    async def t_delete_session(args: dict) -> dict:
        ws_id = _str(args, "workspace_id")
        sid = _str(args, "session_id")
        ws = deps.workspace_store.get(ws_id)
        if ws is None or sid not in ws.session_ids:
            raise _ToolError("not_found", f"Session {sid} not in workspace {ws_id}.")
        registry = deps.get_registry()
        try:
            await registry.remove(sid, purge_metadata=True)  # type: ignore[attr-defined]
        except Exception:
            pass
        updated = await deps.workspace_store.remove_session(ws_id, sid)
        if updated is None:
            raise _ToolError("not_found", "Workspace disappeared.")
        deps.broadcast("app:workspace_updated", {"workspace_id": ws_id})
        return _text(f"Deleted session {sid} from {ws_id}.")

    async def t_move_session(args: dict) -> dict:
        """Reassign a session from one workspace to another. The agent's
        actual environment (cwd, attached repos, system prompt) was
        baked at creation and DOES NOT change — this is a UI/grouping
        reassignment only. Use 'archive then start fresh' if you need
        a clean environment under a different workspace."""
        from_id = _str(args, "from_workspace_id")
        to_id = _str(args, "to_workspace_id")
        sid = _str(args, "session_id")
        src = deps.workspace_store.get(from_id)
        dst = deps.workspace_store.get(to_id)
        if src is None:
            raise _ToolError("not_found", f"Source workspace {from_id!r} doesn't exist.")
        if dst is None:
            raise _ToolError("not_found", f"Destination workspace {to_id!r} doesn't exist.")
        if sid not in src.session_ids:
            raise _ToolError("not_found", f"Session {sid} not in source workspace.")
        if sid in dst.session_ids:
            raise _ToolError("conflict", f"Session {sid} is already in destination workspace.")
        # Preserve the session_name (if any) across the move.
        old_name = src.session_names.get(sid)
        await deps.workspace_store.remove_session(from_id, sid)
        await deps.workspace_store.add_session(to_id, sid)
        if old_name:
            await deps.workspace_store.set_session_name(to_id, sid, old_name)
        # Both endpoints changed; broadcast for each so any focused
        # client refetches once and sees the move.
        deps.broadcast("app:workspace_updated", {"workspace_id": from_id})
        deps.broadcast("app:workspace_updated", {"workspace_id": to_id})
        return _text(
            f"Moved session {sid}: {from_id} → {to_id}. "
            "Note: the session's cwd/repos/prompt were locked at creation and didn't change."
        )

    # ── Tool registration ────────────────────────────────────────────────

    tools = [
        tool(
            "app_list_initiatives",
            "List all initiatives (umbrella projects) the user has registered. Each has a key, display name, optional JIRA epic key, optional Confluence root page id, and a list of repo paths.",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )(wrap(t_list_initiatives)),
        tool(
            "app_create_initiative",
            "Create a new initiative. `key` will be slugified (lowercased, spaces→dashes, only [a-z0-9-]). Fails if the key already exists.",
            {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Friendly slug; e.g. 'meowtorq'."},
                    "display_name": {"type": "string"},
                    "epic_jira_key": {"type": "string", "description": "JIRA key for the umbrella epic, e.g. 'LLM-608'."},
                    "confluence_root_page_id": {"type": "string"},
                    "repo_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Absolute paths to repos that belong to this initiative.",
                    },
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        )(wrap(t_create_initiative)),
        tool(
            "app_update_initiative",
            "Partial update of an existing initiative. Only provided fields are changed. `repo_paths` REPLACES the list (it isn't appended).",
            {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "display_name": {"type": "string"},
                    "epic_jira_key": {"type": "string"},
                    "confluence_root_page_id": {"type": "string"},
                    "repo_paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        )(wrap(t_update_initiative)),
        tool(
            "app_delete_initiative",
            "Permanently remove an initiative. Workspaces that reference it keep their initiative_key string but it will dangle.",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        )(wrap(t_delete_initiative)),
        tool(
            "app_get_settings",
            "Return the user's settings as a JSON tree, e.g. `{\"appearance\": {\"theme\": \"dark\"}}`.",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )(wrap(t_get_settings)),
        tool(
            "app_patch_settings",
            "Shallow-merge updates into settings. Top-level keys are categories, nested values are the actual settings. Pass `null` to delete a key. Example: `{\"appearance\": {\"theme\": \"light\"}}`.",
            {
                "type": "object",
                "properties": {
                    "updates": {
                        "type": "object",
                        "description": "Categories → {key: value | null}. Null values delete.",
                    },
                },
                "required": ["updates"],
                "additionalProperties": False,
            },
        )(wrap(t_patch_settings)),
        # ── Workspace + session management ─────────────────────────────
        tool(
            "app_list_workspaces",
            "List all workspaces (ticket + chat) with their session ids, repos, archive state. Set `include_archived: true` to include archived ones.",
            {
                "type": "object",
                "properties": {"include_archived": {"type": "boolean"}},
                "additionalProperties": False,
            },
        )(wrap(t_list_workspaces)),
        tool(
            "app_get_workspace",
            "Fetch a single workspace by id.",
            {
                "type": "object",
                "properties": {"workspace_id": {"type": "string"}},
                "required": ["workspace_id"],
                "additionalProperties": False,
            },
        )(wrap(t_get_workspace)),
        tool(
            "app_create_ticket_workspace",
            "Create a ticket-bound workspace: makes worktrees of each repo on a branch named after the ticket. Use after workflow_create_ticket to land a new ticket on disk.",
            {
                "type": "object",
                "properties": {
                    "ticket_key": {"type": "string", "description": "e.g. 'LLM-1234'."},
                    "ticket_title": {"type": "string"},
                    "initiative_key": {"type": "string"},
                    "repo_paths": {"type": "array", "items": {"type": "string"}, "description": "Absolute source-repo paths to worktree from."},
                },
                "required": ["ticket_key"],
                "additionalProperties": False,
            },
        )(wrap(t_create_ticket_workspace)),
        tool(
            "app_create_chat_workspace",
            "Spawn another chat workspace. Rarely needed by the agent (the user has a Chat button) but useful for 'archive this and start a fresh thread'.",
            {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "additionalProperties": False,
            },
        )(wrap(t_create_chat_workspace)),
        tool(
            "app_update_workspace",
            "Patch a workspace's display title or initiative attachment. Provide ticket_title and/or initiative_key. Pass initiative_key='' or null to detach.",
            {
                "type": "object",
                "properties": {
                    "workspace_id": {"type": "string"},
                    "ticket_title": {"type": "string"},
                    "initiative_key": {"type": ["string", "null"]},
                },
                "required": ["workspace_id"],
                "additionalProperties": False,
            },
        )(wrap(t_update_workspace)),
        tool(
            "app_archive_workspace",
            "Archive a workspace — disappears from the default sidebar but is preserved (worktrees, sessions, docs).",
            {
                "type": "object",
                "properties": {"workspace_id": {"type": "string"}},
                "required": ["workspace_id"],
                "additionalProperties": False,
            },
        )(wrap(t_archive_workspace)),
        tool(
            "app_unarchive_workspace",
            "Restore an archived workspace.",
            {
                "type": "object",
                "properties": {"workspace_id": {"type": "string"}},
                "required": ["workspace_id"],
                "additionalProperties": False,
            },
        )(wrap(t_unarchive_workspace)),
        tool(
            "app_delete_workspace",
            "DESTRUCTIVE. Tears down every session, removes worktrees, deletes the dir, drops the record. Confirm intent with the user first.",
            {
                "type": "object",
                "properties": {"workspace_id": {"type": "string"}},
                "required": ["workspace_id"],
                "additionalProperties": False,
            },
        )(wrap(t_delete_workspace)),
        tool(
            "app_rename_session",
            "Set or clear the display name of a session tab. Pass null/empty to restore the default S1/S2/... label.",
            {
                "type": "object",
                "properties": {
                    "workspace_id": {"type": "string"},
                    "session_id": {"type": "string"},
                    "name": {"type": ["string", "null"]},
                },
                "required": ["workspace_id", "session_id"],
                "additionalProperties": False,
            },
        )(wrap(t_rename_session)),
        tool(
            "app_delete_session",
            "End and remove a session. Stops the live agent, drops the metadata, removes from the workspace.",
            {
                "type": "object",
                "properties": {
                    "workspace_id": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["workspace_id", "session_id"],
                "additionalProperties": False,
            },
        )(wrap(t_delete_session)),
        tool(
            "app_move_session",
            "Reassign a session from one workspace to another (UI grouping only — the agent's cwd/repos/prompt were locked at creation and don't change).",
            {
                "type": "object",
                "properties": {
                    "from_workspace_id": {"type": "string"},
                    "to_workspace_id": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["from_workspace_id", "to_workspace_id", "session_id"],
                "additionalProperties": False,
            },
        )(wrap(t_move_session)),
    ]

    return create_sdk_mcp_server("chat", tools=tools)


# Allow-list for the agent's permission prompts. Every chat tool is
# considered safe enough to auto-allow — they're scoped to the user's
# own local settings.
ALLOWED_TOOL_PATTERNS = [
    "mcp__chat__app_list_initiatives",
    "mcp__chat__app_create_initiative",
    "mcp__chat__app_update_initiative",
    "mcp__chat__app_delete_initiative",
    "mcp__chat__app_get_settings",
    "mcp__chat__app_patch_settings",
    "mcp__chat__app_list_workspaces",
    "mcp__chat__app_get_workspace",
    "mcp__chat__app_create_ticket_workspace",
    "mcp__chat__app_create_chat_workspace",
    "mcp__chat__app_update_workspace",
    "mcp__chat__app_archive_workspace",
    "mcp__chat__app_unarchive_workspace",
    # Intentionally NOT auto-allowed:
    # - app_delete_workspace, app_delete_session — destructive enough that
    #   we want the permission prompt to fire so the user sees it.
    "mcp__chat__app_rename_session",
    "mcp__chat__app_move_session",
]

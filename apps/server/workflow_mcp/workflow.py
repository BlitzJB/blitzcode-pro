"""Workflow MCP — the agent's interface to JIRA + Confluence + the
local stores (workspaces, initiatives).

All tools take JSON args and return SDK-compatible content blocks. Tools
that need Atlassian credentials check the CredsStore first and emit a
typed `requires_credentials` error the agent learns to react to by
asking the user to connect (workflow_request_credentials).

The agent only ever sees markdown — read tools convert ADF→markdown
before returning, write tools convert markdown→ADF on the way out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from acks import AckStore  # noqa: F401 — imported for type clarity only
from atlassian.confluence import ConfluenceClient, ConfluenceError
from atlassian.jira import JiraClient, JiraError
from atlassian_creds import CredsStore
from initiatives import InitiativeStore
from workspaces import DocRef, WorkspaceStore
from adf import adf_to_markdown, markdown_to_adf, Sidecar


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _error_result(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"error": {"code": code, "message": message, **extra}}
    import json as _json
    return {"content": [{"type": "text", "text": _json.dumps(payload)}], "isError": True}


def _str(args: dict, key: str) -> str:
    v = args.get(key)
    if not isinstance(v, str) or not v.strip():
        raise _ToolError("invalid_args", f"Missing required string field: {key}")
    return v


def _str_opt(args: dict, key: str) -> Optional[str]:
    v = args.get(key)
    return v if isinstance(v, str) and v else None


class _ToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ────────────────────────────────────────────────────────────────────────────
# Server builder
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class WorkflowDeps:
    creds_store: CredsStore
    workspace_store: WorkspaceStore
    initiative_store: InitiativeStore


def build_workflow_mcp_server(deps: WorkflowDeps):
    """Construct the in-process SDK MCP server. Returns the server object
    suitable for `ClaudeAgentOptions(mcp_servers={"workflow": server})`.

    Late-import the SDK so unit tests don't require the real
    `claude_agent_sdk` package to be installed.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # type: ignore

    def jira() -> JiraClient:
        c = deps.creds_store.get()
        if c is None:
            raise _ToolError("requires_credentials", "Atlassian credentials are not configured. Use workflow_request_credentials to prompt the user.")
        return JiraClient(c.site_url, c.email, c.api_token)

    def confluence() -> ConfluenceClient:
        c = deps.creds_store.get()
        if c is None:
            raise _ToolError("requires_credentials", "Atlassian credentials are not configured.")
        return ConfluenceClient(c.site_url, c.email, c.api_token)

    def wrap(fn: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]):
        async def safe(args: dict[str, Any]) -> dict[str, Any]:
            try:
                return await fn(args)
            except _ToolError as e:
                return _error_result(e.code, str(e))
            except (JiraError, ConfluenceError) as e:
                return _error_result("upstream_error", str(e), status=e.status)
            except Exception as e:
                logger.exception("Workflow tool crashed")
                return _error_result("internal_error", str(e))
        return safe

    # ── tool handlers ────────────────────────────────────────────────────

    async def t_request_credentials(_args: dict) -> dict:
        return _text_result(
            "Atlassian credentials are required. Ask the user to click the "
            "'Connect Atlassian' button in the bottom-left of the sidebar, "
            "or open the credentials modal from the top header."
        )

    async def t_search_tickets(args: dict) -> dict:
        q = _str(args, "query")
        limit = int(args.get("max_results", 10))
        client = jira()
        results = await client.typeahead(q, max_results=limit)
        lines = [
            f"- {r.key} — {r.title} [{r.status or '?'}]"
            for r in results
        ]
        return _text_result("\n".join(lines) if lines else "(no results)")

    async def t_get_ticket(args: dict) -> dict:
        key = _str(args, "key")
        client = jira()
        issue = await client.get_issue(key, fields=["summary", "status", "description", "issuetype", "priority"])
        f = issue.get("fields") or {}
        status = ((f.get("status") or {}).get("name")) or "?"
        title = f.get("summary") or ""
        description_adf = f.get("description")
        desc_md = ""
        if isinstance(description_adf, dict):
            desc_md, _ = adf_to_markdown(description_adf)
        body = (
            f"# {key} — {title}\n\n"
            f"Status: **{status}**\n\n"
            f"{desc_md or '_(no description)_'}\n"
        )
        return _text_result(body)

    async def t_set_status(args: dict) -> dict:
        key = _str(args, "key")
        status_name = _str(args, "status_name")
        client = jira()
        tid = await client.set_status(key, status_name)
        return _text_result(f"transitioned {key} → {status_name} (transition id {tid})")

    async def t_add_comment(args: dict) -> dict:
        key = _str(args, "key")
        body_md = _str(args, "body_md")
        client = jira()
        adf = markdown_to_adf(body_md)
        out = await client.add_comment(key, adf)
        return _text_result(f"comment posted (id {out.get('id', '?')})")

    async def t_flag(args: dict) -> dict:
        key = _str(args, "key")
        reason = _str(args, "reason")
        client = jira()
        await client.set_flag(key, flagged=True, comment=reason)
        return _text_result(f"{key} flagged: {reason}")

    async def t_unflag(args: dict) -> dict:
        key = _str(args, "key")
        resolution = _str(args, "resolution")
        client = jira()
        await client.set_flag(key, flagged=False, comment=resolution)
        return _text_result(f"{key} unflagged: {resolution}")

    async def t_link_action_item(args: dict) -> dict:
        from_key = _str(args, "from_key")
        to_key = _str(args, "to_key")
        client = jira()
        await client.link_action_item(from_key, to_key)
        return _text_result(f"linked {from_key} → {to_key} (Action item)")

    async def t_list_initiatives(_args: dict) -> dict:
        items = deps.initiative_store.list()
        if not items:
            return _text_result("(no initiatives — ask the user to add one in the sidebar)")
        lines = []
        for it in items:
            extras = []
            if it.epic_jira_key:
                extras.append(f"epic {it.epic_jira_key}")
            if it.confluence_root_page_id:
                extras.append(f"root page {it.confluence_root_page_id}")
            if it.repo_paths:
                extras.append(f"{len(it.repo_paths)} repo(s)")
            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"- **{it.display_name}** (`{it.key}`){suffix}")
        return _text_result("\n".join(lines))

    async def t_set_initiative_root_page(args: dict) -> dict:
        key = _str(args, "initiative_key")
        page_id = _str(args, "page_id")
        out = await deps.initiative_store.patch(key, confluence_root_page_id=page_id)
        if out is None:
            raise _ToolError("unknown_initiative", f"Initiative not found: {key}")
        return _text_result(f"set root page for {key} → {page_id}")

    async def t_associate_repo_to_initiative(args: dict) -> dict:
        key = _str(args, "initiative_key")
        repo_path = _str(args, "repo_path")
        out = await deps.initiative_store.associate_repo(key, repo_path)
        if out is None:
            raise _ToolError("unknown_initiative", f"Initiative not found: {key}")
        return _text_result(f"associated {repo_path} with {key}")

    async def t_get_doc(args: dict, *, kind: str) -> dict:
        ws_id = _str(args, "workspace_id")
        ws = deps.workspace_store.get(ws_id)
        if ws is None:
            raise _ToolError("unknown_workspace", f"Workspace not found: {ws_id}")
        ref = ws.docs.get(kind)
        if ref is None or not ref.page_id:
            return _text_result(f"no {kind} page exists yet — call workflow_write_{kind} to create one")
        client = confluence()
        page = await client.get_page(ref.page_id)
        body_md, _ = adf_to_markdown(page.body_adf)
        return _text_result(
            f"<!-- page_id={page.id} version={page.version} title={page.title!r} -->\n\n{body_md}"
        )

    async def t_get_rfc(args: dict) -> dict:
        return await t_get_doc(args, kind="rfc")

    async def t_get_debrief(args: dict) -> dict:
        return await t_get_doc(args, kind="debrief")

    async def t_write_doc(args: dict, *, kind: str, title_prefix: str) -> dict:
        ws_id = _str(args, "workspace_id")
        body_md = _str(args, "body_md")
        ws = deps.workspace_store.get(ws_id)
        if ws is None:
            raise _ToolError("unknown_workspace", f"Workspace not found: {ws_id}")
        if not ws.initiative_key:
            raise _ToolError(
                "no_initiative",
                f"Workspace has no initiative — {kind} pages are created under the initiative's Confluence root page.",
            )
        initiative = deps.initiative_store.get(ws.initiative_key)
        if initiative is None or not initiative.confluence_root_page_id:
            raise _ToolError(
                "no_initiative_root",
                f"Initiative {ws.initiative_key!r} has no Confluence root page set. Use workflow_set_initiative_root_page first.",
            )
        title = f"[{ws.ticket_key}] {title_prefix}: {ws.ticket_title or ws.ticket_key}"
        adf = markdown_to_adf(body_md)
        client = confluence()
        # Idempotent: if a child page with the exact title already exists,
        # update it. Otherwise create.
        existing = await client.find_child_by_title(initiative.confluence_root_page_id, title)
        if existing is not None:
            updated = await client.update_page(
                page_id=existing.id, title=title, body_adf=adf, current_version=existing.version,
            )
            page = updated
        else:
            space_id = await client.get_page_space_id(initiative.confluence_root_page_id)
            page = await client.create_page(
                space_id=space_id, parent_id=initiative.confluence_root_page_id, title=title, body_adf=adf,
            )
        await deps.workspace_store.set_doc(
            ws_id,
            kind,
            DocRef(
                page_id=page.id,
                version=page.version,
                title=page.title,
                url=page.url,
                last_synced_at=__import__("time").time(),
            ),
        )
        return _text_result(
            f"{kind} saved — page {page.id} version {page.version}\n{page.url or ''}"
        )

    async def t_write_rfc(args: dict) -> dict:
        return await t_write_doc(args, kind="rfc", title_prefix="RFC")

    async def t_write_debrief(args: dict) -> dict:
        return await t_write_doc(args, kind="debrief", title_prefix="Debrief")

    async def t_update_ticket_fields(args: dict) -> dict:
        """Replace the JIRA description with markdown rendered into ADF."""
        key = _str(args, "key")
        description_md = _str(args, "description_md")
        client = jira()
        adf = markdown_to_adf(description_md)
        await client.update_issue(key, {"description": adf})
        return _text_result(f"updated description for {key}")

    # ── schemas ──────────────────────────────────────────────────────────

    SCHEMA_NONE: dict = {"type": "object", "properties": {}, "additionalProperties": False}

    def s_key():
        return {"key": {"type": "string", "description": "JIRA issue key, e.g. LLM-1234"}}

    def s_workspace():
        return {"workspace_id": {"type": "string", "description": "Workspace UUID"}}

    tools = [
        tool(
            "workflow_request_credentials",
            "Signal that Atlassian credentials are missing — the UI will prompt the user.",
            SCHEMA_NONE,
        )(wrap(t_request_credentials)),
        tool(
            "workflow_search_tickets",
            "Typeahead JIRA search by partial key or title (max 10 results).",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )(wrap(t_search_tickets)),
        tool(
            "workflow_get_ticket",
            "Fetch a JIRA issue and return its description as markdown.",
            {"type": "object", "properties": s_key(), "required": ["key"], "additionalProperties": False},
        )(wrap(t_get_ticket)),
        tool(
            "workflow_set_status",
            "Transition an issue to the named status (e.g. 'In Progress').",
            {
                "type": "object",
                "properties": {**s_key(), "status_name": {"type": "string"}},
                "required": ["key", "status_name"],
                "additionalProperties": False,
            },
        )(wrap(t_set_status)),
        tool(
            "workflow_add_comment",
            "Post a comment on a JIRA issue. body_md is markdown; the server converts to ADF.",
            {
                "type": "object",
                "properties": {**s_key(), "body_md": {"type": "string"}},
                "required": ["key", "body_md"],
                "additionalProperties": False,
            },
        )(wrap(t_add_comment)),
        tool(
            "workflow_update_ticket_fields",
            "Replace a JIRA issue's description with markdown (rendered to ADF).",
            {
                "type": "object",
                "properties": {**s_key(), "description_md": {"type": "string"}},
                "required": ["key", "description_md"],
                "additionalProperties": False,
            },
        )(wrap(t_update_ticket_fields)),
        tool(
            "workflow_flag",
            "Flag a JIRA issue with a reason. Requires explicit user approval (permission prompt).",
            {
                "type": "object",
                "properties": {**s_key(), "reason": {"type": "string"}},
                "required": ["key", "reason"],
                "additionalProperties": False,
            },
        )(wrap(t_flag)),
        tool(
            "workflow_unflag",
            "Remove the flag from a JIRA issue and record the resolution.",
            {
                "type": "object",
                "properties": {**s_key(), "resolution": {"type": "string"}},
                "required": ["key", "resolution"],
                "additionalProperties": False,
            },
        )(wrap(t_unflag)),
        tool(
            "workflow_link_action_item",
            "Create an 'Action item' link from one issue to another (for follow-up gaps).",
            {
                "type": "object",
                "properties": {
                    "from_key": {"type": "string"},
                    "to_key": {"type": "string"},
                },
                "required": ["from_key", "to_key"],
                "additionalProperties": False,
            },
        )(wrap(t_link_action_item)),
        tool(
            "workflow_list_initiatives",
            "List initiatives the user has registered, with epic + root page metadata.",
            SCHEMA_NONE,
        )(wrap(t_list_initiatives)),
        tool(
            "workflow_set_initiative_root_page",
            "Persist the Confluence root page id for an initiative.",
            {
                "type": "object",
                "properties": {
                    "initiative_key": {"type": "string"},
                    "page_id": {"type": "string"},
                },
                "required": ["initiative_key", "page_id"],
                "additionalProperties": False,
            },
        )(wrap(t_set_initiative_root_page)),
        tool(
            "workflow_associate_repo_to_initiative",
            "Remember that a repo path belongs to an initiative (used for future workspace seeding).",
            {
                "type": "object",
                "properties": {
                    "initiative_key": {"type": "string"},
                    "repo_path": {"type": "string"},
                },
                "required": ["initiative_key", "repo_path"],
                "additionalProperties": False,
            },
        )(wrap(t_associate_repo_to_initiative)),
        tool(
            "workflow_get_rfc",
            "Fetch the workspace's RFC page as markdown.",
            {"type": "object", "properties": s_workspace(), "required": ["workspace_id"], "additionalProperties": False},
        )(wrap(t_get_rfc)),
        tool(
            "workflow_write_rfc",
            "Idempotent create-or-update of the workspace's RFC. body_md is markdown.",
            {
                "type": "object",
                "properties": {**s_workspace(), "body_md": {"type": "string"}},
                "required": ["workspace_id", "body_md"],
                "additionalProperties": False,
            },
        )(wrap(t_write_rfc)),
        tool(
            "workflow_get_debrief",
            "Fetch the workspace's Debrief page as markdown.",
            {"type": "object", "properties": s_workspace(), "required": ["workspace_id"], "additionalProperties": False},
        )(wrap(t_get_debrief)),
        tool(
            "workflow_write_debrief",
            "Idempotent create-or-update of the workspace's Debrief. body_md is markdown.",
            {
                "type": "object",
                "properties": {**s_workspace(), "body_md": {"type": "string"}},
                "required": ["workspace_id", "body_md"],
                "additionalProperties": False,
            },
        )(wrap(t_write_debrief)),
    ]

    return create_sdk_mcp_server("workflow", tools=tools)


# List of tool names — used to build extra_allowed_tools so the agent
# doesn't get a permission prompt for every workflow_* call.
ALLOWED_TOOL_PATTERNS = [
    "mcp__workflow__workflow_request_credentials",
    "mcp__workflow__workflow_search_tickets",
    "mcp__workflow__workflow_get_ticket",
    "mcp__workflow__workflow_set_status",
    "mcp__workflow__workflow_add_comment",
    "mcp__workflow__workflow_update_ticket_fields",
    "mcp__workflow__workflow_link_action_item",
    "mcp__workflow__workflow_list_initiatives",
    "mcp__workflow__workflow_set_initiative_root_page",
    "mcp__workflow__workflow_associate_repo_to_initiative",
    "mcp__workflow__workflow_get_rfc",
    "mcp__workflow__workflow_write_rfc",
    "mcp__workflow__workflow_get_debrief",
    "mcp__workflow__workflow_write_debrief",
    # Intentionally NOT auto-allowed:
    # - workflow_flag, workflow_unflag — explicit human approval required per
    #   the workflow spec.
]

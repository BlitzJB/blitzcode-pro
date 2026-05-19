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

    async def t_create_ticket(args: dict) -> dict:
        project_key = _str(args, "project_key")
        summary = _str(args, "summary")
        description_md = _str_opt(args, "description_md")
        issuetype = _str_opt(args, "issuetype") or "Task"
        labels = args.get("labels")
        if labels is not None and not (isinstance(labels, list) and all(isinstance(x, str) for x in labels)):
            raise _ToolError("invalid_args", "labels must be a list of strings if provided")
        priority = _str_opt(args, "priority")
        parent_key = _str_opt(args, "parent_key")
        client = jira()
        adf = markdown_to_adf(description_md) if description_md else None
        out = await client.create_issue(
            project_key=project_key,
            summary=summary,
            description_adf=adf,
            issuetype=issuetype,
            labels=labels,
            priority=priority,
            parent_key=parent_key,
        )
        key = out.get("key", "?")
        creds = deps.creds_store.get()
        url = f"{creds.site_url}/browse/{key}" if creds else None
        return _text_result(
            f"created **{key}**: {summary}\n"
            + (f"{url}\n" if url else "")
            + (f"parent epic: {parent_key}\n" if parent_key else "")
        )

    async def t_get_ticket_changelog(args: dict) -> dict:
        key = _str(args, "key")
        max_items = int(args.get("max_items", 50))
        client = jira()
        rows = await client.changelog(key, max_items=max_items)
        if not rows:
            return _text_result(f"(no recorded changes for {key})")
        lines = [f"# {key} — change history (most recent first)\n"]
        for r in rows:
            ts = r.get("created") or "?"
            who = r.get("author") or "?"
            fld = r.get("field") or "?"
            frm = r.get("from") or "_(unset)_"
            to = r.get("to") or "_(unset)_"
            lines.append(f"- `{ts}` **{who}** changed **{fld}**: {frm} → {to}")
        return _text_result("\n".join(lines))

    async def t_recap(args: dict) -> dict:
        """Activity summary for a date window — what the user (or their
        team) touched. Used for end-of-day/end-of-week recaps."""
        since = _str(args, "since")  # ISO date, e.g. "2026-05-12"
        until = _str_opt(args, "until")  # ISO date, inclusive
        scope = (args.get("scope") or "me").lower()
        if scope not in ("me", "all"):
            raise _ToolError("invalid_args", "scope must be 'me' or 'all'")
        client = jira()
        # JQL date filters: "updated >= '2026-05-12'" + optional upper bound.
        # Wrap the date in single quotes to keep the parser happy on Cloud.
        clauses = [f"updated >= '{since}'"]
        if until:
            clauses.append(f"updated <= '{until}'")
        if scope == "me":
            # Either the user changed it or it's currently assigned to them.
            clauses.append("(assignee = currentUser() OR updatedBy = currentUser())")
        jql = " AND ".join(clauses) + " ORDER BY updated DESC"
        data = await client.search_jql(
            jql, fields=["summary", "status", "issuetype", "updated", "assignee"], max_results=50
        )
        issues = data.get("issues") or []
        if not issues:
            return _text_result(f"No JIRA activity in window {since} → {until or 'now'} ({scope}).")
        lines = [f"# Activity {since} → {until or 'now'} ({scope})\n"]
        for it in issues:
            f = it.get("fields") or {}
            status = ((f.get("status") or {}).get("name")) or "?"
            assignee = ((f.get("assignee") or {}).get("displayName")) or "_(unassigned)_"
            ts = f.get("updated") or "?"
            lines.append(
                f"- **{it.get('key')}** — {f.get('summary') or ''} "
                f"_[{status} · {assignee} · {ts}]_"
            )
        lines.append("")
        lines.append("Drill into any one with `workflow_get_ticket` or `workflow_get_ticket_changelog`.")
        return _text_result("\n".join(lines))

    async def t_list_projects(_args: dict) -> dict:
        client = jira()
        rows = await client.list_projects()
        if not rows:
            return _text_result("(no projects visible)")
        lines = [f"- `{r['key']}` — {r['name']}" for r in rows]
        return _text_result("\n".join(lines))

    async def t_list_page_versions(args: dict) -> dict:
        page_id = _str(args, "page_id")
        client = confluence()
        rows = await client.list_versions(page_id)
        if not rows:
            return _text_result(f"(no version history for page {page_id})")
        lines = [f"# Page {page_id} — version history\n"]
        for r in rows:
            ts = r.get("created_at") or "?"
            msg = r.get("message") or "_(no message)_"
            tag = " (minor)" if r.get("minor_edit") else ""
            lines.append(f"- v{r['number']} — `{ts}`{tag} — {msg}")
        return _text_result("\n".join(lines))

    async def t_diff_page_versions(args: dict) -> dict:
        page_id = _str(args, "page_id")
        v_from = int(_str(args, "from_version"))
        v_to = int(_str(args, "to_version"))
        client = confluence()
        a = await client.get_page_at_version(page_id, v_from)
        b = await client.get_page_at_version(page_id, v_to)
        md_a, _ = adf_to_markdown(a["body_adf"])
        md_b, _ = adf_to_markdown(b["body_adf"])
        import difflib as _dl
        diff = "\n".join(_dl.unified_diff(
            md_a.splitlines(),
            md_b.splitlines(),
            fromfile=f"v{v_from} ({a.get('created_at')})",
            tofile=f"v{v_to} ({b.get('created_at')})",
            lineterm="",
            n=3,
        ))
        if not diff.strip():
            return _text_result(f"(no markdown-level changes between v{v_from} and v{v_to})")
        return _text_result("```diff\n" + diff + "\n```")

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
            "workflow_create_ticket",
            "Create a new JIRA issue. Required: project_key, summary. Optional: description_md (markdown, server converts to ADF), issuetype (default 'Task'), labels, priority, parent_key (epic to nest under). Returns the new key + URL.",
            {
                "type": "object",
                "properties": {
                    "project_key": {"type": "string", "description": "JIRA project key, e.g. 'LLM'."},
                    "summary": {"type": "string"},
                    "description_md": {"type": "string"},
                    "issuetype": {"type": "string", "description": "Defaults to 'Task'. Common values: Task, Story, Bug, Epic."},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "string"},
                    "parent_key": {"type": "string", "description": "JIRA key of the parent epic (next-gen projects)."},
                },
                "required": ["project_key", "summary"],
                "additionalProperties": False,
            },
        )(wrap(t_create_ticket)),
        tool(
            "workflow_get_ticket_changelog",
            "Get the full change history of a JIRA issue: who changed what when. Most recent first.",
            {
                "type": "object",
                "properties": {**s_key(), "max_items": {"type": "integer", "minimum": 1, "maximum": 100}},
                "required": ["key"],
                "additionalProperties": False,
            },
        )(wrap(t_get_ticket_changelog)),
        tool(
            "workflow_list_projects",
            "List JIRA projects visible to the user. Returns key + name.",
            SCHEMA_NONE,
        )(wrap(t_list_projects)),
        tool(
            "workflow_recap",
            "Activity recap for a date window. `since` and optional `until` are ISO dates (YYYY-MM-DD). `scope`: 'me' (default — issues you touched or are assigned to) or 'all' (everything that changed in the window). Used for end-of-day/end-of-week summaries.",
            {
                "type": "object",
                "properties": {
                    "since": {"type": "string"},
                    "until": {"type": "string"},
                    "scope": {"type": "string", "enum": ["me", "all"]},
                },
                "required": ["since"],
                "additionalProperties": False,
            },
        )(wrap(t_recap)),
        tool(
            "workflow_list_page_versions",
            "List all version numbers of a Confluence page, most recent first. Use before workflow_diff_page_versions.",
            {
                "type": "object",
                "properties": {"page_id": {"type": "string"}},
                "required": ["page_id"],
                "additionalProperties": False,
            },
        )(wrap(t_list_page_versions)),
        tool(
            "workflow_diff_page_versions",
            "Unified markdown diff between two versions of a Confluence page. Both versions are fetched, ADF→markdown converted, then diffed line-by-line.",
            {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string"},
                    "from_version": {"type": "string"},
                    "to_version": {"type": "string"},
                },
                "required": ["page_id", "from_version", "to_version"],
                "additionalProperties": False,
            },
        )(wrap(t_diff_page_versions)),
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
    # JIRA read + create — safe for the agent to fire without prompting.
    "mcp__workflow__workflow_create_ticket",
    "mcp__workflow__workflow_get_ticket_changelog",
    "mcp__workflow__workflow_list_projects",
    "mcp__workflow__workflow_recap",
    # Confluence version history + diff — pure reads.
    "mcp__workflow__workflow_list_page_versions",
    "mcp__workflow__workflow_diff_page_versions",
    # Intentionally NOT auto-allowed:
    # - workflow_flag, workflow_unflag — explicit human approval required per
    #   the workflow spec.
]

"""JIRA Cloud REST v3 client.

Uses Basic auth (email + API token), httpx.AsyncClient. Hand-rolled —
the surface we need is small (search + a handful of issue ops) and we
want full control of the new `POST /rest/api/3/search/jql` endpoint's
cursor pagination semantics.

Never logs the api_token. Errors include status + body excerpt so
callers can surface useful messages to the agent / UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx


_DEFAULT_FIELDS = ("summary", "status", "issuetype", "priority")


@dataclass
class JiraError(Exception):
    status: int
    message: str
    body: Optional[str] = None

    def __str__(self) -> str:
        return f"JIRA HTTP {self.status}: {self.message}"


@dataclass
class TicketSummary:
    key: str
    title: str
    status: Optional[str]
    issuetype: Optional[str]


class JiraClient:
    """Minimal async client used by the workflow MCP + UI proxy endpoints."""

    def __init__(self, site_url: str, email: str, api_token: str, *, timeout: float = 15.0) -> None:
        if not site_url:
            raise ValueError("site_url required")
        self._site = site_url.rstrip("/")
        self._auth = (email, api_token)
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._site,
            auth=self._auth,
            timeout=self._timeout,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

    # ────────────────────────────────────────────────────────────────────

    async def search_jql(
        self,
        jql: str,
        *,
        fields: Optional[list[str]] = None,
        max_results: int = 10,
        next_page_token: Optional[str] = None,
    ) -> dict:
        """POST /rest/api/3/search/jql — Atlassian's cursor-paginated search.

        Always passes an explicit `fields` list (the new endpoint returns
        minimal defaults otherwise). Returns the raw response dict so the
        caller can read `issues`, `nextPageToken`, `isLast`.
        """
        body: dict[str, Any] = {
            "jql": jql,
            "fields": list(fields or _DEFAULT_FIELDS),
            "maxResults": max_results,
        }
        if next_page_token:
            body["nextPageToken"] = next_page_token
        async with self._client() as c:
            r = await c.post("/rest/api/3/search/jql", json=body)
        return _ok(r)

    async def typeahead(self, query: str, *, max_results: int = 10) -> list[TicketSummary]:
        """User-facing typeahead. Splits short input that looks like a key
        prefix from free-text searches; uses prefix-match for both."""
        q = (query or "").strip()
        if not q:
            return []
        jql = _typeahead_jql(q)
        data = await self.search_jql(jql, max_results=max_results)
        out: list[TicketSummary] = []
        for issue in data.get("issues", []) or []:
            f = issue.get("fields") or {}
            out.append(TicketSummary(
                key=str(issue.get("key", "")),
                title=str(f.get("summary") or ""),
                status=_safe_name(f.get("status")),
                issuetype=_safe_name(f.get("issuetype")),
            ))
        return out

    async def get_issue(self, key: str, *, fields: Optional[list[str]] = None) -> dict:
        params: dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        async with self._client() as c:
            r = await c.get(f"/rest/api/3/issue/{key}", params=params)
        return _ok(r)

    async def transitions(self, key: str) -> list[dict]:
        async with self._client() as c:
            r = await c.get(f"/rest/api/3/issue/{key}/transitions")
        data = _ok(r)
        return list(data.get("transitions") or [])

    async def find_transition_id(self, key: str, target_status_name: str) -> Optional[str]:
        """Look up the transition id that lands the issue in `target_status_name`."""
        wanted = target_status_name.strip().lower()
        for t in await self.transitions(key):
            name = (t.get("to") or {}).get("name", "") or t.get("name", "")
            if str(name).strip().lower() == wanted:
                return str(t.get("id"))
        return None

    async def transition(self, key: str, transition_id: str) -> None:
        async with self._client() as c:
            r = await c.post(
                f"/rest/api/3/issue/{key}/transitions",
                json={"transition": {"id": transition_id}},
            )
        _ok(r, expect=(204, 200))

    async def set_status(self, key: str, target_status_name: str) -> str:
        """Move an issue to the named status (e.g. "In Progress").

        Returns the transition id used. Raises JiraError if no matching
        transition is available from the current state.
        """
        tid = await self.find_transition_id(key, target_status_name)
        if tid is None:
            raise JiraError(status=400, message=f"No transition available to '{target_status_name}' from current state")
        await self.transition(key, tid)
        return tid

    async def update_issue(self, key: str, fields: dict) -> None:
        async with self._client() as c:
            r = await c.put(f"/rest/api/3/issue/{key}", json={"fields": fields})
        _ok(r, expect=(204, 200))

    async def add_comment(self, key: str, body_adf: dict) -> dict:
        """`body_adf` is an ADF document (root type 'doc')."""
        async with self._client() as c:
            r = await c.post(f"/rest/api/3/issue/{key}/comment", json={"body": body_adf})
        return _ok(r)

    # Flag toggling uses the (somewhat legacy) `flag/flag` API which lives
    # under /rest/greenhopper/. Tested against Cloud — the path is stable.
    async def set_flag(self, key: str, *, flagged: bool, comment: Optional[str] = None) -> None:
        payload: dict[str, Any] = {"issueKeys": [key], "flag": bool(flagged)}
        if comment:
            payload["comment"] = comment
        async with self._client() as c:
            r = await c.post("/rest/greenhopper/1.0/xboard/issue/flag/flag.json", json=payload)
        _ok(r, expect=(200, 204))

    async def link_action_item(self, from_key: str, to_key: str) -> None:
        """Create an inward "Action item" link FROM `from_key` TO `to_key`."""
        payload = {
            "type": {"name": "Action item"},
            "inwardIssue": {"key": from_key},
            "outwardIssue": {"key": to_key},
        }
        async with self._client() as c:
            r = await c.post("/rest/api/3/issueLink", json=payload)
        _ok(r, expect=(201, 200, 204))


# ────────────────────────────────────────────────────────────────────────────
# helpers


def _ok(r: httpx.Response, expect: tuple[int, ...] = (200, 201)) -> dict:
    if r.status_code in expect:
        if r.status_code == 204 or not r.content:
            return {}
        try:
            return r.json() if isinstance(r.json(), dict) else {"data": r.json()}
        except ValueError:
            return {}
    body = r.text[:500]
    raise JiraError(status=r.status_code, message=r.reason_phrase or "error", body=body)


def _safe_name(field: Any) -> Optional[str]:
    if isinstance(field, dict):
        name = field.get("name")
        return str(name) if name else None
    return None


_PREFIX_LIKE = __import__("re").compile(r"^[A-Z][A-Z0-9]*(-\d*)?$")


def _typeahead_jql(query: str) -> str:
    """Split user input into a JQL query.

    - Has an explicit dash AND looks like a project-key fragment
      (`LLM-`, `LLM-12`): key-prefix mode. Bare alpha like "LLM" is
      ambiguous (project name vs. word in a title) — we treat it as
      free-text and let the user type "LLM-" to opt in to key mode.
    - Anything else: prefix-match the summary tokens (and full-text when
      long enough). This is what the user wants for partial title search.
    """
    q = query.strip()
    qu = q.upper()
    if "-" in qu and _PREFIX_LIKE.match(qu):
        project, suffix = qu.split("-", 1)
        if suffix:
            # User typed e.g. "LLM-12" — prefer exact-key OR prefix in summary
            return (
                f'key = "{qu}" OR (project = "{project}" AND summary ~ "{q}*") '
                f"ORDER BY updated DESC"
            )
        return f'project = "{project}" ORDER BY updated DESC'
    # Free-text typeahead — prefix-match summary tokens; `text ~` requires
    # >= 3 chars in some Lucene-backed indexes.
    safe = q.replace('"', '\\"')
    if len(safe) >= 3:
        return f'(summary ~ "{safe}*" OR text ~ "{safe}*") ORDER BY updated DESC'
    return f'summary ~ "{safe}*" ORDER BY updated DESC'

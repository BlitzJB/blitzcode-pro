"""Confluence Cloud REST v2 client.

We use the v2 API for create/update because it cleanly accepts ADF as the
`atlas_doc_format` representation. Versions are sent as `current + 1`;
409 on conflict bubbles up so the caller can re-fetch and retry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

import httpx


@dataclass
class ConfluenceError(Exception):
    status: int
    message: str
    body: Optional[str] = None

    def __str__(self) -> str:
        return f"Confluence HTTP {self.status}: {self.message}"


@dataclass
class ConfluencePage:
    id: str
    title: str
    space_id: Optional[str]
    parent_id: Optional[str]
    version: int
    body_adf: dict  # the ADF "doc" node
    url: Optional[str]


class ConfluenceClient:
    def __init__(self, site_url: str, email: str, api_token: str, *, timeout: float = 20.0) -> None:
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
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    async def get_page(self, page_id: str) -> ConfluencePage:
        """GET /wiki/api/v2/pages/{id}?body-format=atlas_doc_format"""
        async with self._client() as c:
            r = await c.get(
                f"/wiki/api/v2/pages/{page_id}",
                params={"body-format": "atlas_doc_format"},
            )
        data = _ok(r)
        return _to_page(data, site=self._site)

    async def find_child_by_title(self, parent_id: str, title: str) -> Optional[ConfluencePage]:
        """List children of a page, return the one whose title matches.

        Used for idempotent create — before POSTing a new page, check if a
        page by that title already exists under the parent. v2's
        /wiki/api/v2/pages/{parent_id}/children paginates with `cursor`.
        """
        wanted = title.strip()
        cursor: Optional[str] = None
        async with self._client() as c:
            while True:
                params: dict[str, Any] = {"limit": 50}
                if cursor:
                    params["cursor"] = cursor
                r = await c.get(f"/wiki/api/v2/pages/{parent_id}/children", params=params)
                data = _ok(r)
                for entry in data.get("results") or []:
                    if str(entry.get("title", "")).strip() == wanted:
                        # `children` doesn't include the body — fetch full
                        # page so callers get versions + ADF body.
                        return await self.get_page(str(entry["id"]))
                cursor = (data.get("_links") or {}).get("next")
                if not cursor:
                    return None
                # The `next` link in v2 is a relative URL like
                # `/wiki/api/v2/pages/{id}/children?cursor=...`. We follow
                # by extracting the cursor query param. Simpler: just bail
                # and re-issue with the cursor from the URL.
                # For robustness, parse it out:
                if "cursor=" in cursor:
                    cursor = cursor.split("cursor=", 1)[1].split("&", 1)[0]
                else:
                    return None

    async def create_page(
        self,
        *,
        space_id: str,
        parent_id: str,
        title: str,
        body_adf: dict,
    ) -> ConfluencePage:
        """POST /wiki/api/v2/pages — body in atlas_doc_format."""
        payload = {
            "spaceId": space_id,
            "status": "current",
            "title": title,
            "parentId": parent_id,
            "body": {
                "representation": "atlas_doc_format",
                "value": _adf_to_string(body_adf),
            },
        }
        async with self._client() as c:
            r = await c.post("/wiki/api/v2/pages", json=payload)
        data = _ok(r)
        return _to_page(data, site=self._site)

    async def update_page(
        self,
        *,
        page_id: str,
        title: str,
        body_adf: dict,
        current_version: int,
    ) -> ConfluencePage:
        """PUT /wiki/api/v2/pages/{id} — bumps version to current+1.

        Caller is responsible for catching 409 (version conflict) and
        re-fetching + retrying with the fresh version number.
        """
        payload = {
            "id": page_id,
            "status": "current",
            "title": title,
            "body": {
                "representation": "atlas_doc_format",
                "value": _adf_to_string(body_adf),
            },
            "version": {"number": current_version + 1, "message": "blitzcode-pro update"},
        }
        async with self._client() as c:
            r = await c.put(f"/wiki/api/v2/pages/{page_id}", json=payload)
        data = _ok(r)
        return _to_page(data, site=self._site)

    async def search_pages(self, query: str, *, limit: int = 10) -> list[dict]:
        """Fuzzy page search via Confluence's CQL endpoint.

        Returns a normalized list of `{id, title, url}`. Empty list if the
        query is blank. We use `/wiki/rest/api/search?cql=...` because the
        v2 `/pages?title=` query is exact-match only and useless for
        typeahead. CQL accepts `title ~ "foo*"` for prefix-token matching.
        """
        q = (query or "").strip()
        if not q:
            return []
        safe = q.replace('"', '\\"')
        cql = f'type = "page" AND title ~ "{safe}*"'
        async with self._client() as c:
            r = await c.get(
                "/wiki/rest/api/search",
                params={"cql": cql, "limit": limit},
            )
        data = _ok(r)
        out: list[dict] = []
        for entry in data.get("results") or []:
            content = entry.get("content") or {}
            page_id = content.get("id") or entry.get("id")
            # The search endpoint wraps matched terms in <em>…</em> on
            # `entry.title`. `content.title` is the plain version when
            # present; fall back to a stripped version of the search-result
            # title so the user sees the actual page name, not a URL.
            plain_title = content.get("title") or _strip_html_tags(entry.get("title") or "")
            webui = ((content.get("_links") or {}).get("webui")
                     or (entry.get("_links") or {}).get("webui"))
            url: Optional[str] = None
            if isinstance(webui, str):
                url = f"{self._site}/wiki{webui}" if webui.startswith("/") else f"{self._site}/{webui}"
            if page_id:
                out.append({"id": str(page_id), "title": str(plain_title), "url": url})
        return out

    async def get_page_space_id(self, page_id: str) -> str:
        """Convenience: read a parent page just to learn its spaceId. Used when
        creating a child page if the caller only knows the parent_id."""
        page = await self.get_page(page_id)
        if not page.space_id:
            raise ConfluenceError(status=500, message="Page missing spaceId — cannot infer space for create")
        return page.space_id

    async def list_versions(self, page_id: str, *, limit: int = 50) -> list[dict]:
        """Version history of a page. Each entry has number, message,
        createdAt, authorId. Most-recent first per the v2 API. Lets the
        chat agent answer 'what changed and when' on RFC/debrief pages."""
        async with self._client() as c:
            r = await c.get(
                f"/wiki/api/v2/pages/{page_id}/versions",
                params={"limit": limit},
            )
        data = _ok(r)
        return [
            {
                "number": entry.get("number"),
                "message": entry.get("message") or "",
                "created_at": entry.get("createdAt"),
                "author_id": entry.get("authorId"),
                "minor_edit": bool(entry.get("minorEdit")),
            }
            for entry in (data.get("results") or [])
        ]

    async def get_page_at_version(self, page_id: str, version: int) -> dict:
        """Fetch one historical version's ADF body. v2 returns body
        in atlas_doc_format only when explicitly requested."""
        async with self._client() as c:
            r = await c.get(
                f"/wiki/api/v2/pages/{page_id}/versions/{version}",
                params={"body-format": "atlas_doc_format"},
            )
        data = _ok(r)
        body = (data.get("body") or {}).get("atlas_doc_format") or {}
        value = body.get("value")
        adf: dict
        if isinstance(value, str):
            import json as _json
            try:
                adf = _json.loads(value)
            except (ValueError, TypeError):
                adf = {"type": "doc", "version": 1, "content": []}
        elif isinstance(value, dict):
            adf = value
        else:
            adf = {"type": "doc", "version": 1, "content": []}
        return {
            "number": data.get("number"),
            "title": data.get("title"),
            "created_at": data.get("createdAt"),
            "message": data.get("message") or "",
            "body_adf": adf,
        }


# ────────────────────────────────────────────────────────────────────────────


def _ok(r: httpx.Response) -> dict:
    if 200 <= r.status_code < 300:
        if r.status_code == 204 or not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {}
    raise ConfluenceError(status=r.status_code, message=r.reason_phrase or "error", body=r.text[:500])


def _strip_html_tags(s: str) -> str:
    """Cheap HTML strip for the <em>…</em> highlight markup the search
    endpoint wraps matched terms in. Not a full HTML parser — just enough
    to clean titles for display."""
    import re as _re
    return _re.sub(r"<[^>]+>", "", s)


def _adf_to_string(body_adf: dict) -> str:
    """Confluence v2 expects `body.value` as a STRING containing the ADF JSON."""
    import json as _json
    return _json.dumps(body_adf, separators=(",", ":"))


def _to_page(data: dict, *, site: str) -> ConfluencePage:
    body = data.get("body") or {}
    adf = body.get("atlas_doc_format") or {}
    value = adf.get("value")
    if isinstance(value, str):
        # API returns ADF as a JSON-encoded string under v2.
        import json as _json
        try:
            body_adf = _json.loads(value)
        except (ValueError, TypeError):
            body_adf = {"type": "doc", "version": 1, "content": []}
    elif isinstance(value, dict):
        body_adf = value
    else:
        body_adf = {"type": "doc", "version": 1, "content": []}
    version_obj = data.get("version") or {}
    page_id = str(data.get("id") or "")
    url = None
    links = data.get("_links") or {}
    webui = links.get("webui")
    if isinstance(webui, str):
        # webui is a relative path like /spaces/X/pages/123/Title
        url = f"{site}/wiki{webui}" if webui.startswith("/") else f"{site}/{webui}"
    return ConfluencePage(
        id=page_id,
        title=str(data.get("title") or ""),
        space_id=str(data["spaceId"]) if data.get("spaceId") else None,
        parent_id=str(data["parentId"]) if data.get("parentId") else None,
        version=int(version_obj.get("number", 1)),
        body_adf=body_adf,
        url=url,
    )

"""Atlassian (JIRA + Confluence) credentials store.

JSON-file backed, chmod 0600. Never echoes the token through wire events
or HTTP response bodies other than the explicit POST round-trip. There is
NO endpoint that returns the token; UI only ever sees `has_creds: bool`
and the site_url / email (no secret material).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AtlassianCreds:
    site_url: str
    email: str
    api_token: str


class CredsStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._data: Optional[AtlassianCreds] = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(raw, dict):
            return
        try:
            self._data = AtlassianCreds(
                site_url=str(raw["site_url"]).rstrip("/"),
                email=str(raw["email"]),
                api_token=str(raw["api_token"]),
            )
        except (KeyError, TypeError):
            return

    async def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = asdict(self._data) if self._data else {}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self._path)
        # Lock perms to user-only on every write — a fresh tmp file inherits
        # umask, and we want 0600 unconditionally for files that hold tokens.
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    def has_creds(self) -> bool:
        return self._data is not None and bool(self._data.api_token) and bool(self._data.site_url)

    def get(self) -> Optional[AtlassianCreds]:
        if self._data is None:
            return None
        return AtlassianCreds(
            site_url=self._data.site_url,
            email=self._data.email,
            api_token=self._data.api_token,
        )

    def public_meta(self) -> dict:
        """Return non-secret metadata about the configured creds.

        Used by /app/atlassian/has-creds — never exposes the token.
        """
        if self._data is None:
            return {"has_creds": False, "site_url": None, "email": None}
        return {
            "has_creds": True,
            "site_url": self._data.site_url,
            "email": self._data.email,
        }

    async def set(self, site_url: str, email: str, api_token: str) -> None:
        site_url = (site_url or "").strip().rstrip("/")
        email = (email or "").strip()
        api_token = (api_token or "").strip()
        if not site_url or not email or not api_token:
            raise ValueError("site_url, email, and api_token are all required")
        if not (site_url.startswith("https://") or site_url.startswith("http://")):
            raise ValueError("site_url must include scheme (https://...)")
        async with self._lock:
            self._data = AtlassianCreds(site_url=site_url, email=email, api_token=api_token)
            await self._flush()

    async def clear(self) -> None:
        async with self._lock:
            self._data = None
            await self._flush()

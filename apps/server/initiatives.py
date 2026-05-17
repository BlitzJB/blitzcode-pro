"""App-layer initiative registry.

An initiative is a long-running umbrella (e.g. "Meowtorq") that owns one
or more git repos and (in Phase 3+) a Confluence root page + JIRA epic.
The blitzcode-pro UI surfaces these so that when the user creates a new
workspace, the initiative list pre-fills repos.

This is NOT part of agent-webkit. Pure app-layer JSON store, same pattern
as AckStore (apps/server/acks.py).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass
class Initiative:
    """One umbrella the user works under."""
    key: str  # friendly slug, e.g. "meowtorq"
    display_name: str
    epic_jira_key: Optional[str] = None
    confluence_root_page_id: Optional[str] = None
    repo_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Initiative":
        raw_repos = data.get("repo_paths") or []
        return cls(
            key=str(data["key"]),
            display_name=str(data.get("display_name") or data["key"]),
            epic_jira_key=(data.get("epic_jira_key") or None) or None,
            confluence_root_page_id=(data.get("confluence_root_page_id") or None) or None,
            repo_paths=[str(p) for p in raw_repos if isinstance(p, str)],
        )


def is_valid_key(key: str) -> bool:
    return bool(key) and bool(_KEY_RE.match(key))


class InitiativeStore:
    """JSON-file backed map of initiative key → Initiative."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._data: dict[str, Initiative] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        items = raw.get("initiatives", raw) if isinstance(raw, dict) else []
        if isinstance(items, dict):
            entries = items.values()
        elif isinstance(items, list):
            entries = items
        else:
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                init = Initiative.from_dict(entry)
            except (KeyError, TypeError):
                continue
            self._data[init.key] = init

    async def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {"initiatives": [it.to_dict() for it in self._data.values()]}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self._path)

    def list(self) -> list[Initiative]:
        return [
            Initiative(
                key=it.key,
                display_name=it.display_name,
                epic_jira_key=it.epic_jira_key,
                confluence_root_page_id=it.confluence_root_page_id,
                repo_paths=list(it.repo_paths),
            )
            for it in self._data.values()
        ]

    def get(self, key: str) -> Optional[Initiative]:
        it = self._data.get(key)
        if it is None:
            return None
        return Initiative(
            key=it.key,
            display_name=it.display_name,
            epic_jira_key=it.epic_jira_key,
            confluence_root_page_id=it.confluence_root_page_id,
            repo_paths=list(it.repo_paths),
        )

    async def upsert(self, initiative: Initiative) -> Initiative:
        if not is_valid_key(initiative.key):
            raise ValueError(f"Invalid initiative key: {initiative.key!r}")
        async with self._lock:
            self._data[initiative.key] = Initiative(
                key=initiative.key,
                display_name=initiative.display_name,
                epic_jira_key=initiative.epic_jira_key,
                confluence_root_page_id=initiative.confluence_root_page_id,
                repo_paths=list(initiative.repo_paths),
            )
            await self._flush()
            return self.get(initiative.key)  # type: ignore[return-value]

    async def patch(self, key: str, **fields) -> Optional[Initiative]:
        """Update only the provided fields. Unknown keys are ignored."""
        async with self._lock:
            existing = self._data.get(key)
            if existing is None:
                return None
            for k, v in fields.items():
                if not hasattr(existing, k):
                    continue
                setattr(existing, k, v)
            await self._flush()
            return self.get(key)

    async def associate_repo(self, key: str, repo_path: str) -> Optional[Initiative]:
        """Add a repo path to the initiative if not already present."""
        async with self._lock:
            existing = self._data.get(key)
            if existing is None:
                return None
            if repo_path not in existing.repo_paths:
                existing.repo_paths.append(repo_path)
                await self._flush()
            return self.get(key)

    async def remove(self, key: str) -> bool:
        async with self._lock:
            if key in self._data:
                del self._data[key]
                await self._flush()
                return True
            return False

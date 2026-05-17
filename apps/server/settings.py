"""App-layer user settings.

Pure JSON-backed key-value store, namespaced by category. The Settings
modal in the UI is the single consumer; the server only persists.

Schema is intentionally loose so adding a new setting in the future
doesn't require a migration — clients ask for what they know about and
ignore the rest.
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(raw, dict):
            # Coerce: top-level keys are categories; each category is a dict.
            for cat, vals in raw.items():
                if isinstance(cat, str) and isinstance(vals, dict):
                    self._data[cat] = dict(vals)

    async def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self._path)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._data)

    async def patch(self, updates: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Shallow-merge `updates` into the store. Top-level keys are
        categories; nested dicts merge by key (no deep merge — values are
        scalars). Pass `null` to delete a key."""
        async with self._lock:
            for cat, vals in (updates or {}).items():
                if not isinstance(cat, str) or not isinstance(vals, dict):
                    continue
                bucket = self._data.setdefault(cat, {})
                for k, v in vals.items():
                    if not isinstance(k, str):
                        continue
                    if v is None:
                        bucket.pop(k, None)
                    else:
                        bucket[k] = v
                if not bucket:
                    self._data.pop(cat, None)
            await self._flush()
            return deepcopy(self._data)

"""App-layer acknowledgement state.

This is NOT part of the agent-webkit wire protocol. It lives entirely in
the app, persisted next to the agent-webkit metadata store. It tracks two
timestamps per session:

  - last_completion_at: when the agent last finished a turn (set on the
    `result` wire event observed in-process via the global event log).
  - last_ack_at:        when the user last clicked Acknowledge.

The UI computes `needs_input = last_completion_at > last_ack_at` and
shows the session under the "Needs input" group until acknowledged.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict


@dataclass
class AckEntry:
    last_completion_at: float = 0.0
    last_ack_at: float = 0.0


class AckStore:
    """JSON-file backed map of session_id → AckEntry."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._data: Dict[str, AckEntry] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for sid, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            self._data[sid] = AckEntry(
                last_completion_at=float(entry.get("last_completion_at", 0.0)),
                last_ack_at=float(entry.get("last_ack_at", 0.0)),
            )

    async def _flush(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {sid: asdict(e) for sid, e in self._data.items()}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self._path)
        self._dirty = False

    def snapshot(self) -> Dict[str, AckEntry]:
        return dict(self._data)

    async def mark_completion(self, session_id: str, ts: float | None = None) -> None:
        async with self._lock:
            entry = self._data.setdefault(session_id, AckEntry())
            entry.last_completion_at = ts if ts is not None else time.time()
            self._dirty = True
            await self._flush()

    async def acknowledge(self, session_id: str, ts: float | None = None) -> AckEntry:
        async with self._lock:
            entry = self._data.setdefault(session_id, AckEntry())
            entry.last_ack_at = ts if ts is not None else time.time()
            self._dirty = True
            await self._flush()
            return AckEntry(
                last_completion_at=entry.last_completion_at,
                last_ack_at=entry.last_ack_at,
            )

    async def forget(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._data:
                del self._data[session_id]
                self._dirty = True
                await self._flush()

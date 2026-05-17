"""Shared types for the ADF translator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sidecar:
    """In-memory map of opaque ADF subtrees keyed by short ids.

    Lives for one read→edit→write cycle. The markdown produced by
    `adf_to_markdown` contains `[[ADF:<id>]]` tokens; `markdown_to_adf`
    looks them up in this sidecar to restore the original subtree
    byte-identically. Tokens the agent didn't preserve are dropped on
    write (the markdown is authoritative for those regions).
    """
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, node: dict[str, Any]) -> str:
        # Short, stable-ish id derived from index. Stable within a single
        # sidecar; not stable across separate translations.
        next_id = f"a{len(self.nodes)}"
        self.nodes[next_id] = node
        return next_id

    def get(self, key: str) -> dict[str, Any] | None:
        return self.nodes.get(key)

"""One-shot discovery of Claude Code's built-in subagents.

Shells `claude agents` and parses the text output. Cached at module load —
the built-in set doesn't change at runtime. Failures are non-fatal: the
palette just won't list builtins.

Output shape from `claude agents`:

    4 active agents

    Built-in agents:
      Explore · haiku
      general-purpose · inherit
      Plan · inherit
      statusline-setup · sonnet

    Project agents:
      reviewer · sonnet
"""

from __future__ import annotations

import re
import shutil
import subprocess
from typing import Optional


# `<name> · <model>` — we only need the name; tolerate the entire suffix
# being absent in case the CLI's format changes.
_AGENT_LINE_RE = re.compile(r"^\s{2,}([\w\-./]+)(?:\s*·.*)?\s*$")


def discover_builtin_agents(timeout: float = 3.0) -> list[dict]:
    """Return [{name, source: "builtin"}] for every built-in agent. Empty
    list if `claude` isn't on PATH or the call fails."""
    claude_bin = shutil.which("claude")
    if claude_bin is None:
        return []
    try:
        result = subprocess.run(
            [claude_bin, "agents"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    return _parse_agents_output(result.stdout, section="Built-in agents")


def _parse_agents_output(text: str, section: str) -> list[dict]:
    out: list[dict] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            in_section = False
            continue
        if stripped.endswith(":"):
            in_section = stripped.startswith(section)
            continue
        if not in_section:
            continue
        m = _AGENT_LINE_RE.match(stripped)
        if m:
            out.append({"name": m.group(1), "source": "builtin", "kind": "agent"})
    return out


# Snapshot at import time. Re-importing the module re-runs this if you
# really need to refresh — but built-ins don't change without a CLI upgrade,
# so leaving it cached is fine.
BUILTIN_AGENTS: list[dict] = discover_builtin_agents()

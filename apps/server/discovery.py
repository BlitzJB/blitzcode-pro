"""App-layer discovery of Claude Code slash commands, skills, and subagents.

Surfaces what the user has installed at the project (cwd) and user (~) levels
so the UI can offer a completion palette behind `/` and `@`.

This is NOT part of the agent-webkit wire protocol. It's pure filesystem
inspection of the well-known `.claude/{commands,agents,skills}` directories
that the Claude CLI itself reads.

Project entries with the same name as a user entry shadow the user version —
the CLI does the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


_HOME_CLAUDE = Path.home() / ".claude"


@dataclass
class CompletionItem:
    name: str
    description: Optional[str]
    source: str  # "project" | "user" | "builtin"
    kind: str  # "command" | "skill" | "agent"
    argument_hint: Optional[str] = None
    tools: Optional[list[str]] = None
    path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop None fields to keep the wire payload tight.
        return {k: v for k, v in d.items() if v is not None}


# ────────────────────────────────────────────────────────────────────────────
# Frontmatter parser. The .md files use the standard `---\n…YAML…\n---\n`
# fence; we only need a flat `key: value` shape (description, name, etc.),
# so a tiny hand-roll avoids dragging in PyYAML.
# ────────────────────────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_LINE_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")


def _parse_frontmatter(text: str) -> dict[str, str | list[str]]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    out: dict[str, str | list[str]] = {}
    for raw_line in m.group(1).splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#"):
            continue
        lm = _LINE_RE.match(line)
        if not lm:
            continue
        key, val = lm.group(1), lm.group(2)
        out[key] = _coerce_value(val)
    return out


def _coerce_value(raw: str) -> str | list[str]:
    s = raw.strip()
    # Strip surrounding quotes (single or double).
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    # Flow-style list: [a, b, c]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [
            (item.strip().strip("'").strip('"'))
            for item in inner.split(",")
            if item.strip()
        ]
    return s


# ────────────────────────────────────────────────────────────────────────────
# Discovery
# ────────────────────────────────────────────────────────────────────────────


def _safe_iterdir(p: Path) -> Iterable[Path]:
    try:
        return list(p.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _str(v: object) -> Optional[str]:
    return v if isinstance(v, str) and v else None


def _list_of_str(v: object) -> Optional[list[str]]:
    if isinstance(v, list):
        return [s for s in v if isinstance(s, str)]
    return None


def _walk_md_recursive(root: Path) -> list[tuple[Path, tuple[str, ...]]]:
    """Yield (file_path, name_segments) for every .md under `root`. Segments
    are the path from `root` to the file with the `.md` stripped, so
    `commands/git/commit.md` under `root=commands/` yields ("git", "commit")."""
    out: list[tuple[Path, tuple[str, ...]]] = []

    def walk(dir_: Path, prefix: tuple[str, ...]) -> None:
        for child in _safe_iterdir(dir_):
            name = child.name
            if name.startswith("."):
                continue
            if child.is_dir():
                walk(child, prefix + (name,))
            elif child.is_file() and name.endswith(".md"):
                stem = name[: -len(".md")]
                out.append((child, prefix + (stem,)))

    walk(root, ())
    return out


def _discover_commands(scope: str, root: Path) -> list[CompletionItem]:
    items: list[CompletionItem] = []
    if not root.is_dir():
        return items
    for path, segs in _walk_md_recursive(root):
        if not segs:
            continue
        # Claude convention: nested dirs become `parent:leaf` so `git/commit.md`
        # is invoked as `/git:commit`.
        name = ":".join(segs)
        fm = _parse_frontmatter(_read_text(path))
        items.append(
            CompletionItem(
                name=name,
                description=_str(fm.get("description")),
                argument_hint=_str(fm.get("argument-hint")) or _str(fm.get("argumentHint")),
                source=scope,
                kind="command",
                path=str(path),
            )
        )
    return items


def _discover_agents(scope: str, root: Path) -> list[CompletionItem]:
    items: list[CompletionItem] = []
    if not root.is_dir():
        return items
    for child in _safe_iterdir(root):
        if not (child.is_file() and child.name.endswith(".md")):
            continue
        fm = _parse_frontmatter(_read_text(child))
        # Frontmatter `name` wins; fall back to the file stem so we always
        # have something to invoke.
        name = _str(fm.get("name")) or child.name[: -len(".md")]
        items.append(
            CompletionItem(
                name=name,
                description=_str(fm.get("description")),
                source=scope,
                kind="agent",
                tools=_list_of_str(fm.get("tools")),
                path=str(child),
            )
        )
    return items


def _discover_skills(scope: str, root: Path) -> list[CompletionItem]:
    items: list[CompletionItem] = []
    if not root.is_dir():
        return items
    for child in _safe_iterdir(root):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        fm = _parse_frontmatter(_read_text(skill_md))
        # Skill name is invoked as the folder slug; frontmatter `name` is
        # advisory (used for display only when present).
        items.append(
            CompletionItem(
                name=child.name,
                description=_str(fm.get("description")) or _str(fm.get("name")),
                source=scope,
                kind="skill",
                path=str(skill_md),
            )
        )
    return items


def _dedupe_project_shadows_user(
    project: list[CompletionItem], user: list[CompletionItem]
) -> list[CompletionItem]:
    """Project items win; drop user items with the same name."""
    project_names = {it.name for it in project}
    return project + [it for it in user if it.name not in project_names]


def discover_for_cwd(cwd: Optional[str]) -> dict[str, list[dict]]:
    """Return commands/skills/agents available for a session whose working
    directory is `cwd`. `cwd=None` means user-scope only (the session was
    created without an explicit cwd, so there's no project to inspect)."""
    cwd_path = Path(cwd) if cwd else None

    proj_commands: list[CompletionItem] = []
    proj_agents: list[CompletionItem] = []
    proj_skills: list[CompletionItem] = []
    if cwd_path and cwd_path.is_dir():
        proj_root = cwd_path / ".claude"
        proj_commands = _discover_commands("project", proj_root / "commands")
        proj_agents = _discover_agents("project", proj_root / "agents")
        proj_skills = _discover_skills("project", proj_root / "skills")

    user_commands = _discover_commands("user", _HOME_CLAUDE / "commands")
    user_agents = _discover_agents("user", _HOME_CLAUDE / "agents")
    user_skills = _discover_skills("user", _HOME_CLAUDE / "skills")

    commands = _dedupe_project_shadows_user(proj_commands, user_commands)
    agents = _dedupe_project_shadows_user(proj_agents, user_agents)
    skills = _dedupe_project_shadows_user(proj_skills, user_skills)

    return {
        "commands": [c.to_dict() for c in commands],
        "skills": [s.to_dict() for s in skills],
        "agents": [a.to_dict() for a in agents],
    }

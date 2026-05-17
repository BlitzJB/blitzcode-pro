"""Discovery of slash commands, skills, and subagents from .claude/ dirs.

We mock the home directory by monkeypatching `_HOME_CLAUDE` so tests don't
depend on the dev's real ~/.claude state.
"""

from pathlib import Path

import pytest

import discovery


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / "home" / ".claude"
    home.mkdir(parents=True)
    monkeypatch.setattr(discovery, "_HOME_CLAUDE", home)
    return home


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────────
# Frontmatter parser
# ────────────────────────────────────────────────────────────────────────────


class TestFrontmatter:
    def test_basic_key_value(self):
        out = discovery._parse_frontmatter("---\ndescription: hello\n---\nbody")
        assert out == {"description": "hello"}

    def test_quoted_value(self):
        out = discovery._parse_frontmatter('---\nname: "with spaces"\n---\n')
        assert out["name"] == "with spaces"

    def test_flow_list(self):
        out = discovery._parse_frontmatter("---\ntools: [Read, Write, Bash]\n---\n")
        assert out["tools"] == ["Read", "Write", "Bash"]

    def test_no_frontmatter_returns_empty(self):
        assert discovery._parse_frontmatter("just body, no fence") == {}

    def test_unfenced_top_dashes_ignored(self):
        # Missing trailing fence shouldn't accidentally match.
        assert discovery._parse_frontmatter("---\nfoo: bar\nbody") == {}

    def test_comments_and_blanks_skipped(self):
        out = discovery._parse_frontmatter(
            "---\n# a comment\n\ndescription: ok\n---\n"
        )
        assert out == {"description": "ok"}

    def test_argument_hint_key(self):
        out = discovery._parse_frontmatter("---\nargument-hint: <path>\n---\n")
        assert out["argument-hint"] == "<path>"


# ────────────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────────────


class TestCommands:
    def test_flat_command(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/foo.md", "---\ndescription: do foo\n---\nBody")
        data = discovery.discover_for_cwd(str(cwd))
        names = [(c["name"], c["source"]) for c in data["commands"]]
        assert ("foo", "project") in names
        foo = next(c for c in data["commands"] if c["name"] == "foo")
        assert foo["description"] == "do foo"
        assert foo["kind"] == "command"

    def test_nested_command_uses_colon(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/git/commit.md", "---\ndescription: c\n---\n")
        data = discovery.discover_for_cwd(str(cwd))
        assert any(c["name"] == "git:commit" for c in data["commands"])

    def test_deeply_nested_command(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/a/b/c.md", "")
        data = discovery.discover_for_cwd(str(cwd))
        assert any(c["name"] == "a:b:c" for c in data["commands"])

    def test_project_shadows_user(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/dup.md", "---\ndescription: project\n---\n")
        _write(fake_home / "commands/dup.md", "---\ndescription: user\n---\n")
        data = discovery.discover_for_cwd(str(cwd))
        dups = [c for c in data["commands"] if c["name"] == "dup"]
        assert len(dups) == 1
        assert dups[0]["source"] == "project"
        assert dups[0]["description"] == "project"

    def test_user_only_when_no_project_file(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        cwd.mkdir()
        _write(fake_home / "commands/only-user.md", "---\ndescription: u\n---\n")
        data = discovery.discover_for_cwd(str(cwd))
        names = [(c["name"], c["source"]) for c in data["commands"]]
        assert ("only-user", "user") in names

    def test_argument_hint_passed_through(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/run.md", "---\nargument-hint: <task>\n---\n")
        data = discovery.discover_for_cwd(str(cwd))
        run = next(c for c in data["commands"] if c["name"] == "run")
        assert run["argument_hint"] == "<task>"

    def test_dot_dirs_skipped(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/.hidden/x.md", "")
        _write(cwd / ".claude/commands/visible.md", "")
        data = discovery.discover_for_cwd(str(cwd))
        names = {c["name"] for c in data["commands"]}
        assert "visible" in names
        assert not any(n.startswith(".hidden") or "x" in n for n in names if n != "visible")

    def test_non_md_files_ignored(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/foo.md", "")
        _write(cwd / ".claude/commands/readme.txt", "")
        data = discovery.discover_for_cwd(str(cwd))
        names = {c["name"] for c in data["commands"]}
        assert names == {"foo"}


# ────────────────────────────────────────────────────────────────────────────
# Agents
# ────────────────────────────────────────────────────────────────────────────


class TestAgents:
    def test_basic_agent_uses_frontmatter_name(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(
            cwd / ".claude/agents/reviewer.md",
            "---\nname: code-reviewer\ndescription: reviews code\n---\n",
        )
        data = discovery.discover_for_cwd(str(cwd))
        a = next(a for a in data["agents"] if a["source"] == "project")
        assert a["name"] == "code-reviewer"
        assert a["description"] == "reviews code"

    def test_agent_falls_back_to_filename(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/agents/no-name.md", "")
        data = discovery.discover_for_cwd(str(cwd))
        assert any(a["name"] == "no-name" for a in data["agents"])

    def test_agent_tools_list_parsed(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/agents/a.md", "---\ntools: [Read, Edit]\n---\n")
        data = discovery.discover_for_cwd(str(cwd))
        a = next(a for a in data["agents"] if a["name"] == "a")
        assert a["tools"] == ["Read", "Edit"]

    def test_agents_dont_recurse_into_subdirs(self, tmp_path, fake_home):
        # Subagents live as flat files; subdirs aren't part of the spec.
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/agents/nested/inside.md", "")
        _write(cwd / ".claude/agents/top.md", "")
        data = discovery.discover_for_cwd(str(cwd))
        names = {a["name"] for a in data["agents"]}
        assert "top" in names
        assert "inside" not in names

    def test_agent_project_shadows_user(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/agents/dup.md", "---\ndescription: p\n---\n")
        _write(fake_home / "agents/dup.md", "---\ndescription: u\n---\n")
        data = discovery.discover_for_cwd(str(cwd))
        dups = [a for a in data["agents"] if a["name"] == "dup"]
        assert len(dups) == 1 and dups[0]["source"] == "project"


# ────────────────────────────────────────────────────────────────────────────
# Skills
# ────────────────────────────────────────────────────────────────────────────


class TestSkills:
    def test_skill_requires_skill_md(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        # Empty dir, no SKILL.md → not a skill.
        (cwd / ".claude/skills/empty").mkdir(parents=True)
        # Valid skill.
        _write(
            cwd / ".claude/skills/inspect/SKILL.md",
            "---\nname: inspect\ndescription: inspect things\n---\n",
        )
        data = discovery.discover_for_cwd(str(cwd))
        names = {s["name"] for s in data["skills"]}
        assert names == {"inspect"}
        s = data["skills"][0]
        assert s["description"] == "inspect things"

    def test_skill_name_is_folder_slug(self, tmp_path, fake_home):
        # Frontmatter `name` is advisory; the invocable slug is the dir name.
        cwd = tmp_path / "proj"
        _write(
            cwd / ".claude/skills/my-skill/SKILL.md",
            "---\nname: Display Name\n---\n",
        )
        data = discovery.discover_for_cwd(str(cwd))
        assert any(s["name"] == "my-skill" for s in data["skills"])

    def test_skill_project_shadows_user(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/skills/dup/SKILL.md", "---\ndescription: p\n---\n")
        _write(fake_home / "skills/dup/SKILL.md", "---\ndescription: u\n---\n")
        data = discovery.discover_for_cwd(str(cwd))
        dups = [s for s in data["skills"] if s["name"] == "dup"]
        assert len(dups) == 1
        assert dups[0]["source"] == "project"


# ────────────────────────────────────────────────────────────────────────────
# Robustness
# ────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_missing_cwd_dirs(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        cwd.mkdir()  # no .claude/ at all
        data = discovery.discover_for_cwd(str(cwd))
        assert data == {"commands": [], "skills": [], "agents": []}

    def test_cwd_none_returns_user_only(self, tmp_path, fake_home):
        _write(fake_home / "commands/u.md", "")
        data = discovery.discover_for_cwd(None)
        assert any(c["source"] == "user" and c["name"] == "u" for c in data["commands"])
        # And nothing project-scoped.
        assert all(c["source"] != "project" for c in data["commands"])

    def test_nonexistent_cwd_treated_as_no_project(self, tmp_path, fake_home):
        data = discovery.discover_for_cwd(str(tmp_path / "does-not-exist"))
        assert data == {"commands": [], "skills": [], "agents": []}

    def test_path_field_present(self, tmp_path, fake_home):
        cwd = tmp_path / "proj"
        _write(cwd / ".claude/commands/p.md", "")
        data = discovery.discover_for_cwd(str(cwd))
        p = next(c for c in data["commands"] if c["name"] == "p")
        assert p["path"].endswith("/.claude/commands/p.md")

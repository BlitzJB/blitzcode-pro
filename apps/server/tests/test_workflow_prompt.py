"""Workflow prompt rendering — Phase 1 minimal version."""

from workspaces import DocRef, Workspace, WorkspaceRepo
from workflow_prompt import render_workflow_prompt


def test_returns_none_outside_workspace():
    assert render_workflow_prompt(None) is None


def test_basic_ticket_block():
    ws = Workspace(
        id="ws-1",
        ticket_key="LLM-42",
        ticket_title="Wire up the thing",
        initiative_key="meowtorq",
        dir="/path/to/ws",
    )
    out = render_workflow_prompt(ws, initiative_display_name="Meowtorq")
    assert "LLM-42" in out
    assert "Wire up the thing" in out
    assert "Meowtorq" in out
    assert "/path/to/ws" in out


def test_includes_each_repo_with_worktree_branch_and_source():
    ws = Workspace(
        id="ws-1",
        ticket_key="LLM-1",
        ticket_title=None,
        initiative_key=None,
        dir="/ws",
        repos=[
            WorkspaceRepo(source_path="/src/foo", worktree_path="/ws/foo", branch="LLM-1"),
            WorkspaceRepo(source_path="/src/bar", worktree_path="/ws/bar", branch="LLM-1"),
        ],
    )
    out = render_workflow_prompt(ws)
    assert "/ws/foo" in out and "/src/foo" in out
    assert "/ws/bar" in out and "/src/bar" in out
    assert "LLM-1" in out


def test_no_initiative_name_when_unknown():
    ws = Workspace(
        id="ws-1", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/ws"
    )
    out = render_workflow_prompt(ws)
    # No "Initiative:" line when None.
    assert "Initiative:" not in out


def test_includes_lifecycle_rulebook():
    ws = Workspace(id="ws-1", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/ws")
    out = render_workflow_prompt(ws)
    assert "Phase 1" in out and "Initiation" in out
    assert "Phase 2" in out and "Active development" in out
    assert "Phase 3" in out and "Completion" in out
    assert "Blocker flagging rules" in out


def test_includes_template_sections():
    ws = Workspace(id="ws-1", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/ws")
    out = render_workflow_prompt(ws)
    # Ticket template — sections rendered as ADF headings, not bold-colon prose.
    assert "## Context" in out
    assert "## Blockers" in out
    assert "## PRs" in out
    # RFC template
    assert "## Problem" in out and "## Decision" in out
    assert "## Non-goals" in out and "## Open questions" in out
    # Debrief
    assert "## What shipped" in out and "## Deviations from RFC" in out


def test_includes_rich_formatting_toolkit():
    """The prompt teaches the agent ALL the rich ADF nodes (panels, status,
    task lists) so it doesn't fall back to flat markdown when writing docs."""
    ws = Workspace(id="ws-1", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/ws")
    out = render_workflow_prompt(ws)
    assert "::: panel info" in out
    assert "{status:" in out
    assert "- [ ]" in out and "- [x]" in out
    assert "smart `inlineCard`" in out


def test_includes_mcp_tool_surface():
    ws = Workspace(id="ws-1", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/ws")
    out = render_workflow_prompt(ws)
    for tool_name in [
        "workflow_search_tickets",
        "workflow_get_ticket",
        "workflow_set_status",
        "workflow_write_rfc",
        "workflow_write_debrief",
        "workflow_flag",
        "workflow_link_action_item",
        "workflow_list_initiatives",
        "workflow_request_credentials",
    ]:
        assert tool_name in out, tool_name


def test_includes_smart_link_rule_and_workspace_id():
    ws = Workspace(id="ws-abc", ticket_key="LLM-1", ticket_title=None, initiative_key=None, dir="/ws")
    out = render_workflow_prompt(ws)
    assert "Smart-link" in out
    assert "inlineCard" in out
    assert "ws-abc" in out  # surfaces id so agent knows what to pass to write_rfc/etc

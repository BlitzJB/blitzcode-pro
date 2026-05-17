"""System-prompt addendum injected into every session inside a workspace.

This is the OPPOSITE of a skill — loaded deterministically every turn
because we cannot rely on the agent to remember to invoke a skill.

Contains:
  1. Ticket header (key, title, status placeholder, initiative, links)
  2. Workspace map (each repo: source path → worktree path → branch)
  3. Lifecycle rulebook (the four-phase JIRA + Confluence flow)
  4. Verbatim templates (ticket description, RFC, debrief)
  5. Workflow MCP tool surface — one line per tool
  6. Smart-link rule
"""

from __future__ import annotations

from typing import Optional

from workspaces import Workspace  # type: ignore


_LIFECYCLE = """\
## Lifecycle (drive the ticket through these phases)

### Phase 1 — Initiation
- Move the ticket to **In Progress** (workflow_set_status).
- Fill the `Context` field of the description using the template below.
- If the work involves a new system / architecture / non-obvious approach,
  draft an RFC: call workflow_write_rfc with the workspace_id. The page is
  created under the initiative's Confluence root and linked back into the
  ticket via the `RFC` field.

### Phase 2 — Active development
- Update the `PRs` list in the ticket description whenever a PR opens or merges.
- Blockers: call workflow_flag(key, reason) and add a Blockers row to the
  description. Flagging requires explicit human approval — the user will
  see a permission prompt.
- On resolution: workflow_unflag(key, resolution) + update the Blockers row.

### Phase 3 — Completion
- Merge all PRs, finalize the description with PR links.
- workflow_set_status(key, "Code Review").
- workflow_write_debrief(workspace_id, ...) — page created under the same
  initiative root, with [TICKET] Debrief: <title> as the title.
- For any gaps: create follow-up tickets under the same epic and link them
  via workflow_link_action_item(this_ticket, follow_up_ticket). Document
  the gaps in the `Limitations & Items Deferred` section.

### Blocker flagging rules
- Flagging is a human-approved action. Always include who/what is blocking
  and the date.
- When unflagging, always record the resolution.
"""

_TEMPLATES = """\
## Rich-formatting toolkit (use these for every doc — flat text is wrong)

The server translates this markdown subset into proper ADF nodes when it
writes to JIRA/Confluence. Always reach for these instead of bullet-soup:

| Markdown                                        | Rendered in ADF                              |
| ----------------------------------------------- | -------------------------------------------- |
| `## Section`                                    | A real heading (not bold-colon prose).       |
| `::: panel info` … `:::`                        | Coloured callout box.                         |
| `::: panel warning` / `note` / `success` / `error` | Same, other tones.                        |
| `{status:Draft\\|purple}`                        | Atlassian status pill (inline badge).        |
| `- [ ] thing` / `- [x] thing`                   | Live checkbox (`taskList`).                  |
| `[text](url)` to JIRA/Confluence/GitHub PR/issue | Auto-becomes a smart `inlineCard`.          |
| GFM tables                                      | Real ADF tables.                             |

Status colors: `neutral`, `purple` (drafts), `blue` (in progress / under review),
`yellow` (blocked / at risk), `green` (approved / shipped), `red` (rejected).

Pick the right panel: `info` for context, `note` for a sticky reminder,
`warning` for caveats, `success` for outcomes, `error` for things that broke.

## Templates (use verbatim — section names must match)

### Frontmatter table convention

Every document (ticket / RFC / debrief) opens with a compact two-column
frontmatter table — labels on the left, values on the right — instead of
a panel. Tables render cleanly in both Confluence AND JIRA and scan
faster than a wall of bold-colon lines.

The first row of GFM tables is the header, so use an empty header with
bold labels in the body rows:

```
|  |  |
|---|---|
| **Status** | {status:Draft|purple} |
| **Ticket** | <inline link> |
| **Author** | <name> |
| **Date**   | YYYY-MM-DD |
```

Reach for `::: panel info|warning|note` only for INLINE callouts inside
a section body — never for the doc-level frontmatter.

### Ticket description

Always include EVERY section below, even if empty. Empty sections give the
team a known place to look later; missing sections look like you forgot.
Use `_(none)_` / `_(none yet)_` inside the body — never delete the heading.

```
|  |  |
|---|---|
| **Status**   | {status:To Do|blue} |
| **RFC**      | <inline link or "None — in progress"> |
| **Debrief**  | <inline link or "None — in progress"> |
| **Related**  | <comma-separated JIRA keys, or "None"> |

## Context

<Why this task exists. Constraints, prior attempts, relevant background.
Short and factual. One or two paragraphs; bullets only if the points are
genuinely parallel.>

## PRs

- [ ] <PR title> — <GitHub link>

_(write `_(none yet)_` here when there are no PRs)_

## Blockers

| Description | Blocked by | Status |
|---|---|---|
| <what is blocked> | <person / system / ticket> | {status:Raised|yellow} YYYY-MM-DD |

_(when there are no blockers, render the heading and a single line:
`_(none)_` — drop the empty table)_

## Limitations & Items deferred

- <gap> — follow-up: <TICKET-KEY>

_(use `_(none yet)_` when nothing is deferred)_
```

### RFC (Confluence child under initiative root)

Title prefix `[TICKET-KEY] RFC:` is added automatically.

```
|  |  |
|---|---|
| **Status** | {status:Draft|purple} |
| **Ticket** | <inline link> |
| **Author** | <name> |
| **Date**   | YYYY-MM-DD |

## Problem

2-3 sentences. What breaks or is missing without this.

## Decision

<One or two sentences for the chosen approach.>

If options were weighed:

| Option | Chosen | Reason |
|---|---|---|
| <A> | {status:no|neutral} | <why not> |
| <B> | {status:yes|green}  | <why> |

## Design

Architecture, data flow, interfaces, constraints. Prefer tables and code
blocks over prose. Use `::: panel warning` to highlight non-obvious
constraints reviewers must not miss.

## Non-goals

- Out of scope item A
- Out of scope item B

## Open questions

1. <Question, with the person / context needed to resolve it>
2. <…>
```

### Debrief

Title prefix `[TICKET-KEY] Debrief:` is added automatically.

```
|  |  |
|---|---|
| **Ticket** | <inline link> |
| **RFC**    | <inline link or "None"> |
| **Author** | <name> |
| **Date**   | YYYY-MM-DD |
| **PRs**    | <inline links, comma-separated> |

## What shipped

- [x] <Concrete artifact: file, endpoint, metric, test> — <PR link>

## Deviations from RFC

| RFC said | We did | Why |
|---|---|---|
| <…> | <…> | <…> |

_(omit the table body and write `_(none)_` if there were no deviations)_

## Decisions made in flight

::: panel note
<Unplanned decision + justification. One panel per decision keeps the
reviewer's eye organised.>
:::

## Known gaps / follow-ups

- [ ] <gap> — follow-up <TICKET-KEY> (created via workflow_link_action_item)
```

## Formatting principles

- Headings carry hierarchy. Don't fake them with `**Section:**` lines.
- Empty placeholders like `_(none yet)_` are clutter — omit the row /
  panel entirely when nothing applies.
- Status pills for any field that has a fixed set of values (status,
  decisions, priority). Prose is worse.
- Reach for `::: panel` when the next chunk of text deserves the reader's
  full attention. Don't overuse it — three panels on one page is two too many.
"""

_TOOLS = """\
## Workflow MCP tools (call these instead of curl)

Ticket:
- workflow_search_tickets(query, max_results=10) — JQL typeahead.
- workflow_get_ticket(key) — fetch issue, description rendered as markdown.
- workflow_set_status(key, status_name) — transition (e.g. "In Progress", "Code Review").
- workflow_update_ticket_fields(key, description_md) — replace description with rendered markdown.
- workflow_add_comment(key, body_md) — post a comment.
- workflow_flag(key, reason) / workflow_unflag(key, resolution) — human-approved.
- workflow_link_action_item(from_key, to_key) — inward "Action item" link.

Confluence docs (RFC + debrief):
- workflow_get_rfc(workspace_id) / workflow_write_rfc(workspace_id, body_md)
- workflow_get_debrief(workspace_id) / workflow_write_debrief(workspace_id, body_md)
  - write_* are idempotent: existing page → update; absent → create child under initiative root.

Initiatives (the umbrellas):
- workflow_list_initiatives() — see what umbrellas exist.
- workflow_set_initiative_root_page(initiative_key, page_id) — persist a Confluence root.
- workflow_associate_repo_to_initiative(initiative_key, repo_path) — remember a repo↔initiative link.

Credentials:
- workflow_request_credentials() — emit only when a tool reports `requires_credentials`. The user will be prompted in-app.
"""

_LINK_RULE = """\
## Smart-link rule
When writing markdown destined for Confluence or JIRA, link with plain
`[text](url)` — the server auto-converts URLs matching JIRA
(`*.atlassian.net/browse/.*`), Confluence (`*.atlassian.net/wiki/.*`), and
GitHub (`github.com/.../pull/.*`, `.../issues/...`) into rich `inlineCard`
nodes. Don't try to construct ADF yourself.
"""


def render_workflow_prompt(
    workspace: Optional[Workspace],
    *,
    initiative_display_name: Optional[str] = None,
) -> Optional[str]:
    if workspace is None:
        return None

    lines: list[str] = []
    lines.append("# Workspace context")
    lines.append("")
    lines.append(f"You are working inside a blitzcode-pro workspace for ticket **{workspace.ticket_key}**.")
    if workspace.ticket_title:
        lines.append(f"Ticket title: {workspace.ticket_title}")
    if initiative_display_name:
        lines.append(f"Initiative: {initiative_display_name}")
    lines.append(f"Workspace id: `{workspace.id}` (use this in workflow_write_rfc / workflow_write_debrief).")
    lines.append("")
    lines.append(f"Workspace root: `{workspace.dir}`")

    if workspace.repos:
        lines.append("")
        lines.append("## Repositories in this workspace")
        lines.append("")
        lines.append("Each repo is a git worktree on the ticket's feature branch. All work")
        lines.append("for this ticket happens inside these worktrees — never touch the source clones directly.")
        lines.append("")
        for r in workspace.repos:
            lines.append(f"- **{_basename(r.worktree_path)}**")
            lines.append(f"  - worktree: `{r.worktree_path}`")
            lines.append(f"  - source clone: `{r.source_path}`")
            lines.append(f"  - branch: `{r.branch}`")

    lines.append("")
    lines.append(_LIFECYCLE)
    lines.append("")
    lines.append(_TEMPLATES)
    lines.append("")
    lines.append(_TOOLS)
    lines.append("")
    lines.append(_LINK_RULE)
    return "\n".join(lines)


def _basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1] or path

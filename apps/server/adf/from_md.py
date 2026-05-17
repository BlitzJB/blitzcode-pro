"""Markdown → ADF.

Hand-rolled because we need:
  (a) Restoration of `[[ADF:<id>]]` opaque tokens from the sidecar.
  (b) Smart-link rewriting for known URL patterns.

The subset matches what to_md.py emits, with a forgiving parser that
accepts both `-` and `*` for bullets, and both fenced and indented code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .types import Sidecar


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE_OPEN_RE = re.compile(r"^```(\S*)\s*$")
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_BULLET_RE = re.compile(r"^(\s*)([-*])\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^>\s?(.*)$")
_HR_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_ADF_TOKEN_RE = re.compile(r"\[\[ADF:([A-Za-z0-9_-]+)\]\]")
# Panel admonitions. Accept all common shapes so the agent isn't fighting
# whitespace:  ::: panel info  /  :::panel info  /  ::: info  /  :::info
_PANEL_OPEN_RE = re.compile(
    r"^:::\s*(?:panel\s+)?(info|note|warning|success|error)\s*$",
    re.IGNORECASE,
)
_PANEL_CLOSE_RE = re.compile(r"^:::\s*$")
# Task-list items:  - [ ] open  /  - [x] done
_TASK_PREFIX_RE = re.compile(r"^\[([ xX])\]\s+(.*)$")
# Inline status badge:  {status:LABEL|COLOR}  (color optional, default neutral)
_STATUS_RE = re.compile(r"\{status:([^}|]+)(?:\|([a-zA-Z]+))?\}")
# Recognized Atlassian status colors. Unknown → neutral.
_STATUS_COLORS = {"neutral", "purple", "blue", "red", "yellow", "green"}

# Standard smart-link providers — these become inlineCard nodes.
_SMART_LINK_PATTERNS = [
    re.compile(r"^https?://[^/]+\.atlassian\.net/browse/[A-Z][A-Z0-9]*-\d+(?:[?#].*)?$"),
    re.compile(r"^https?://[^/]+\.atlassian\.net/wiki/.+"),
    re.compile(r"^https?://github\.com/[^/]+/[^/]+/pull/\d+(?:[?#/].*)?$"),
    re.compile(r"^https?://github\.com/[^/]+/[^/]+/issues/\d+(?:[?#/].*)?$"),
]


@dataclass
class SmartLinkPolicy:
    """Configure how bare markdown links are converted into ADF.

    Default: rewrite to inlineCard when the URL matches a known provider.
    Set `enabled=False` to keep everything as plain `link` marks (useful
    in contexts where the destination doesn't render smart links).
    """
    enabled: bool = True

    def should_inline_card(self, url: str) -> bool:
        if not self.enabled:
            return False
        return any(p.match(url) for p in _SMART_LINK_PATTERNS)


def markdown_to_adf(
    md: str,
    *,
    sidecar: Optional[Sidecar] = None,
    smart_links: Optional[SmartLinkPolicy] = None,
) -> dict[str, Any]:
    """Build an ADF doc from markdown.

    `sidecar` (optional): if provided, `[[ADF:<id>]]` tokens are restored
    from it. Tokens with no sidecar entry render as their literal text.
    """
    sidecar = sidecar or Sidecar()
    smart_links = smart_links or SmartLinkPolicy()
    lines = md.splitlines()
    blocks = _parse_blocks(lines, 0, len(lines), sidecar, smart_links)
    return {"type": "doc", "version": 1, "content": blocks}


# ────────────────────────────────────────────────────────────────────────────
# Block parser


def _parse_blocks(
    lines: list[str],
    start: int,
    end: int,
    sidecar: Sidecar,
    smart_links: SmartLinkPolicy,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    i = start
    while i < end:
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            i += 1
            continue

        # Panel  ::: panel <type>  ...  :::
        pm = _PANEL_OPEN_RE.match(line)
        if pm:
            panel_type = pm.group(1).lower()
            j = i + 1
            inner: list[str] = []
            while j < end and not _PANEL_CLOSE_RE.match(lines[j]):
                inner.append(lines[j])
                j += 1
            blocks.append({
                "type": "panel",
                "attrs": {"panelType": panel_type},
                "content": _parse_blocks(inner, 0, len(inner), sidecar, smart_links),
            })
            i = j + 1
            continue

        # Fenced code
        fm = _FENCE_OPEN_RE.match(line)
        if fm:
            lang = fm.group(1) or None
            code_lines: list[str] = []
            j = i + 1
            while j < end and not _FENCE_CLOSE_RE.match(lines[j]):
                code_lines.append(lines[j])
                j += 1
            attrs: dict[str, Any] = {}
            if lang:
                attrs["language"] = lang
            node: dict[str, Any] = {"type": "codeBlock", "attrs": attrs} if attrs else {"type": "codeBlock"}
            if code_lines:
                node["content"] = [{"type": "text", "text": "\n".join(code_lines)}]
            blocks.append(node)
            i = j + 1
            continue

        # Heading
        hm = _HEADING_RE.match(line)
        if hm:
            level = len(hm.group(1))
            text = hm.group(2)
            blocks.append({
                "type": "heading",
                "attrs": {"level": level},
                "content": _parse_inline(text, sidecar, smart_links),
            })
            i += 1
            continue

        # HR
        if _HR_RE.match(line):
            blocks.append({"type": "rule"})
            i += 1
            continue

        # Blockquote
        if _BLOCKQUOTE_RE.match(line):
            quote_lines: list[str] = []
            while i < end and _BLOCKQUOTE_RE.match(lines[i]):
                m = _BLOCKQUOTE_RE.match(lines[i])
                quote_lines.append(m.group(1) if m else "")
                i += 1
            inner = _parse_blocks(quote_lines, 0, len(quote_lines), sidecar, smart_links)
            blocks.append({"type": "blockquote", "content": inner})
            continue

        # Table — detected by a line of pipes followed by a separator row.
        if "|" in line and i + 1 < end and _TABLE_SEP_RE.match(lines[i + 1]):
            j, table_node = _parse_table(lines, i, end, sidecar, smart_links)
            blocks.append(table_node)
            i = j
            continue

        # Lists
        bm = _BULLET_RE.match(line)
        om = _ORDERED_RE.match(line)
        if bm or om:
            j, list_node = _parse_list(lines, i, end, sidecar, smart_links, ordered=bool(om))
            blocks.append(list_node)
            i = j
            continue

        # Paragraph — consume contiguous non-empty, non-block-start lines.
        para_lines: list[str] = []
        while i < end:
            l = lines[i]
            if (
                l.strip() == ""
                or _HEADING_RE.match(l)
                or _FENCE_OPEN_RE.match(l)
                or _BULLET_RE.match(l)
                or _ORDERED_RE.match(l)
                or _BLOCKQUOTE_RE.match(l)
                or _HR_RE.match(l)
                or ("|" in l and i + 1 < end and _TABLE_SEP_RE.match(lines[i + 1]))
            ):
                break
            para_lines.append(l)
            i += 1
        text = "\n".join(para_lines)
        inline = _parse_inline(text, sidecar, smart_links)
        # Unwrap: if the paragraph is JUST one restored opaque node whose
        # original type was block-level (panel, mediaSingle, table, etc.),
        # emit it as a block. Inline-only opaque nodes (inlineCard, mention,
        # mediaInline, emoji) stay nested inside the paragraph.
        if len(inline) == 1 and inline[0].get("type") in _BLOCK_LEVEL_TYPES:
            blocks.append(inline[0])
        else:
            blocks.append({"type": "paragraph", "content": inline})

    return blocks


# ADF node types that should NEVER appear nested inside a paragraph —
# always block-level. Used to unwrap restored opaque nodes that were
# tokenized at the block level.
_BLOCK_LEVEL_TYPES = {
    "panel", "mediaSingle", "mediaGroup", "expand", "nestedExpand",
    "decisionList", "taskList", "extension", "bodiedExtension",
    "table", "heading", "blockquote", "rule", "codeBlock",
}


def _parse_list(
    lines: list[str],
    start: int,
    end: int,
    sidecar: Sidecar,
    smart_links: SmartLinkPolicy,
    *,
    ordered: bool,
) -> tuple[int, dict[str, Any]]:
    # Look-ahead: bullet lists whose FIRST item starts with `[ ]` / `[x]`
    # become Atlassian taskLists (every item below is treated as a task).
    is_task_list = False
    if not ordered:
        first_bm = _BULLET_RE.match(lines[start])
        if first_bm and _TASK_PREFIX_RE.match(first_bm.group(3)):
            is_task_list = True

    items: list[dict[str, Any]] = []
    i = start
    base_indent: Optional[int] = None
    while i < end:
        line = lines[i]
        m = (_ORDERED_RE if ordered else _BULLET_RE).match(line)
        if not m:
            break
        indent = len(m.group(1))
        if base_indent is None:
            base_indent = indent
        elif indent != base_indent:
            break
        first_text = m.group(3)
        item_lines = [first_text]
        i += 1
        # Continuation: lines indented at least base_indent + 2 belong to
        # the previous item, with their leading indentation trimmed.
        while i < end:
            l = lines[i]
            if l.strip() == "":
                # peek: if next non-empty line is more-indented, continuation
                k = i + 1
                while k < end and lines[k].strip() == "":
                    k += 1
                if k >= end:
                    break
                if not (lines[k].startswith(" " * (base_indent + 2))):
                    break
                item_lines.append("")
                i += 1
                continue
            if l.startswith(" " * (base_indent + 2)):
                item_lines.append(l[base_indent + 2:])
                i += 1
                continue
            # Same-level list sibling or different content — break.
            break
        # Render the item's lines as blocks (allows nested lists, paragraphs).
        if is_task_list:
            # Strip the task marker before parsing inline.
            first = item_lines[0] if item_lines else ""
            tm = _TASK_PREFIX_RE.match(first)
            state = "DONE" if (tm and tm.group(1).lower() == "x") else "TODO"
            body_text = (tm.group(2) if tm else first).strip()
            items.append({
                "type": "taskItem",
                "attrs": {
                    # Atlassian wants a stable id per item; index-based is fine
                    # for round-trip since we recompute every time.
                    "localId": f"task-{len(items)}",
                    "state": state,
                },
                "content": _parse_inline(body_text, sidecar, smart_links),
            })
            continue
        inner = _parse_blocks(item_lines, 0, len(item_lines), sidecar, smart_links)
        if not inner:
            inner = [{"type": "paragraph", "content": []}]
        items.append({"type": "listItem", "content": inner})

    if is_task_list:
        # taskList wraps the items; Atlassian renders these as the live
        # checkbox UI (checkable in-place inside Confluence/JIRA).
        return i, {
            "type": "taskList",
            "attrs": {"localId": "tasks"},
            "content": items,
        }
    node = {"type": "orderedList" if ordered else "bulletList", "content": items}
    if ordered:
        node["attrs"] = {"order": 1}
    return i, node


def _parse_table(
    lines: list[str],
    start: int,
    end: int,
    sidecar: Sidecar,
    smart_links: SmartLinkPolicy,
) -> tuple[int, dict[str, Any]]:
    def split_row(s: str) -> list[str]:
        s = s.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    header_cells = split_row(lines[start])
    i = start + 2  # skip header + separator
    body_rows: list[list[str]] = []
    while i < end and "|" in lines[i] and lines[i].strip():
        body_rows.append(split_row(lines[i]))
        i += 1

    rows: list[dict[str, Any]] = []
    if header_cells:
        rows.append({
            "type": "tableRow",
            "content": [
                {"type": "tableHeader", "content": [{"type": "paragraph", "content": _parse_inline(c, sidecar, smart_links)}]}
                for c in header_cells
            ],
        })
    for r in body_rows:
        rows.append({
            "type": "tableRow",
            "content": [
                {"type": "tableCell", "content": [{"type": "paragraph", "content": _parse_inline(c, sidecar, smart_links)}]}
                for c in r
            ],
        })
    return i, {"type": "table", "content": rows}


# ────────────────────────────────────────────────────────────────────────────
# Inline parser. Walks the text linearly, peeling off marks one at a time.


_INLINE_TOKEN_RE = re.compile(
    r"(\[\[ADF:[A-Za-z0-9_-]+\]\])"      # 1 opaque token
    r"|(\*\*[^*\n]+\*\*)"                 # 2 bold
    r"|(\*[^*\n]+\*)"                     # 3 italic
    r"|(`[^`\n]+`)"                       # 4 inline code
    r"|(~~[^~\n]+~~)"                     # 5 strike
    r"|(\[(?:\\\]|[^\]])*\]\([^)\s]+\))"  # 6 link
    r"|(\{status:[^}|]+(?:\|[a-zA-Z]+)?\})",  # 7 status badge
)


def _parse_inline(text: str, sidecar: Sidecar, smart_links: SmartLinkPolicy) -> list[dict[str, Any]]:
    """Tokenize inline text into ADF inline nodes."""
    if text == "":
        return []
    out: list[dict[str, Any]] = []
    pos = 0
    for m in _INLINE_TOKEN_RE.finditer(text):
        if m.start() > pos:
            out.extend(_emit_text(text[pos : m.start()], marks=[]))
        token = m.group(0)
        if m.group(1):
            # Opaque ADF token — restore from sidecar if present.
            key = token[len("[[ADF:") : -2]
            node = sidecar.get(key)
            if node is not None:
                out.append(node)
            else:
                # Sidecar entry missing — render the token as literal text
                # rather than silently dropping it.
                out.extend(_emit_text(token, marks=[]))
        elif m.group(2):
            out.extend(_emit_text(token[2:-2], marks=[{"type": "strong"}]))
        elif m.group(3):
            out.extend(_emit_text(token[1:-1], marks=[{"type": "em"}]))
        elif m.group(4):
            out.append({"type": "text", "text": token[1:-1], "marks": [{"type": "code"}]})
        elif m.group(5):
            out.extend(_emit_text(token[2:-2], marks=[{"type": "strike"}]))
        elif m.group(6):
            # [text](url)
            close_text = token.index("](")
            link_text = token[1:close_text]
            url = token[close_text + 2 : -1]
            if smart_links.should_inline_card(url):
                out.append({"type": "inlineCard", "attrs": {"url": url}})
            else:
                out.append({
                    "type": "text",
                    "text": link_text,
                    "marks": [{"type": "link", "attrs": {"href": url}}],
                })
        elif m.group(7):
            # {status:LABEL|color}
            sm = _STATUS_RE.match(token)
            if sm:
                label = sm.group(1).strip()
                color_raw = (sm.group(2) or "").lower()
                color = color_raw if color_raw in _STATUS_COLORS else "neutral"
                out.append({
                    "type": "status",
                    "attrs": {"text": label, "color": color},
                })
        pos = m.end()
    if pos < len(text):
        out.extend(_emit_text(text[pos:], marks=[]))
    return out


def _emit_text(text: str, *, marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Emit text, handling hard-break (two spaces + newline) and plain
    newlines (collapsed to space within a paragraph)."""
    if not text:
        return []
    out: list[dict[str, Any]] = []
    # Split on hard-break first.
    chunks = text.split("  \n")
    for ci, chunk in enumerate(chunks):
        if ci > 0:
            out.append({"type": "hardBreak"})
        # Collapse plain newlines to spaces — markdown doesn't preserve
        # single linebreaks within paragraphs.
        flat = chunk.replace("\n", " ")
        if flat == "":
            continue
        node: dict[str, Any] = {"type": "text", "text": flat}
        if marks:
            node["marks"] = list(marks)
        out.append(node)
    return out

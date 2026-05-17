"""ADF → markdown."""
from __future__ import annotations

from typing import Any

from .types import Sidecar


_RENDERABLE_BLOCK_TYPES = {
    "paragraph", "heading", "bulletList", "orderedList", "listItem",
    "codeBlock", "blockquote", "rule", "table", "tableRow", "tableCell", "tableHeader",
    "hardBreak",
}

_RENDERABLE_INLINE_TYPES = {"text", "hardBreak"}


def adf_to_markdown(doc: dict[str, Any]) -> tuple[str, Sidecar]:
    """Render an ADF doc as markdown.

    Returns (markdown, sidecar). Opaque nodes get stashed in the sidecar
    and replaced with `[[ADF:<id>]]` tokens in the output.
    """
    sidecar = Sidecar()
    if not isinstance(doc, dict) or doc.get("type") != "doc":
        return "", sidecar
    out_lines: list[str] = []
    for node in doc.get("content") or []:
        block_md = _render_block(node, sidecar)
        if block_md is not None:
            out_lines.append(block_md)
    # Collapse multiple blank lines, ensure trailing newline.
    text = "\n\n".join(s for s in out_lines if s != "")
    return text + ("\n" if text and not text.endswith("\n") else ""), sidecar


# ────────────────────────────────────────────────────────────────────────────
# Block-level renderers


def _render_block(node: dict[str, Any], sidecar: Sidecar, *, list_depth: int = 0) -> str | None:
    if not isinstance(node, dict):
        return None
    t = node.get("type")
    if t == "paragraph":
        return _render_inline_seq(node.get("content") or [], sidecar)
    if t == "heading":
        level = max(1, min(6, int((node.get("attrs") or {}).get("level", 1))))
        return "#" * level + " " + _render_inline_seq(node.get("content") or [], sidecar)
    if t == "bulletList":
        return _render_list(node, sidecar, ordered=False, depth=list_depth)
    if t == "orderedList":
        return _render_list(node, sidecar, ordered=True, depth=list_depth)
    if t == "codeBlock":
        lang = (node.get("attrs") or {}).get("language") or ""
        text = "".join(c.get("text", "") for c in (node.get("content") or []) if isinstance(c, dict) and c.get("type") == "text")
        return f"```{lang}\n{text}\n```"
    if t == "blockquote":
        inner_lines: list[str] = []
        for child in node.get("content") or []:
            rendered = _render_block(child, sidecar, list_depth=list_depth)
            if rendered:
                inner_lines.append(rendered)
        joined = "\n\n".join(inner_lines)
        return "\n".join("> " + line if line else ">" for line in joined.splitlines())
    if t == "rule":
        return "---"
    if t == "table":
        return _render_table(node, sidecar)
    if t == "panel":
        ptype = ((node.get("attrs") or {}).get("panelType") or "info").lower()
        inner_lines: list[str] = []
        for child in node.get("content") or []:
            r = _render_block(child, sidecar, list_depth=list_depth)
            if r is not None:
                inner_lines.append(r)
        body = "\n\n".join(inner_lines)
        return f"::: panel {ptype}\n{body}\n:::"
    if t == "taskList":
        rendered_items: list[str] = []
        for item in node.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "taskItem":
                continue
            state = (item.get("attrs") or {}).get("state", "TODO")
            marker = "[x]" if str(state).upper() == "DONE" else "[ ]"
            body = _render_inline_seq(item.get("content") or [], sidecar).strip()
            rendered_items.append(f"- {marker} {body}")
        return "\n".join(rendered_items)
    # Anything else at the block level → opaque
    key = sidecar.add(node)
    return f"[[ADF:{key}]]"


def _render_list(node: dict[str, Any], sidecar: Sidecar, *, ordered: bool, depth: int) -> str:
    out_lines: list[str] = []
    items = node.get("content") or []
    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict) or item.get("type") != "listItem":
            continue
        marker = (f"{i}." if ordered else "-") + " "
        indent = "  " * depth
        sub_lines: list[str] = []
        for child in item.get("content") or []:
            rendered = _render_block(child, sidecar, list_depth=depth + 1)
            if rendered is None:
                continue
            sub_lines.append(rendered)
        if not sub_lines:
            out_lines.append(indent + marker)
            continue
        first, *rest = "\n\n".join(sub_lines).splitlines()
        out_lines.append(indent + marker + first)
        # Subsequent lines of the listItem body are continuation-indented.
        for line in rest:
            out_lines.append(indent + "  " + line)
    return "\n".join(out_lines)


def _render_table(node: dict[str, Any], sidecar: Sidecar) -> str:
    rows = node.get("content") or []
    if not rows:
        return ""
    # Each cell's content is rendered inline by joining its block renderings
    # with a single space (markdown tables don't support multi-line cells).
    def cell_text(cell: dict[str, Any]) -> str:
        parts: list[str] = []
        for child in cell.get("content") or []:
            rendered = _render_block(child, sidecar)
            if rendered:
                parts.append(rendered.replace("\n", " "))
        return " ".join(parts).strip() or " "

    out_lines: list[str] = []
    header_emitted = False
    for r_idx, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("type") != "tableRow":
            continue
        cells = row.get("content") or []
        rendered_cells = [cell_text(c) for c in cells if isinstance(c, dict) and c.get("type") in ("tableCell", "tableHeader")]
        if not rendered_cells:
            continue
        out_lines.append("| " + " | ".join(rendered_cells) + " |")
        # If row 0 contains tableHeader cells, treat as header — emit separator.
        if r_idx == 0 and any(isinstance(c, dict) and c.get("type") == "tableHeader" for c in cells):
            out_lines.append("|" + "|".join(["---"] * len(rendered_cells)) + "|")
            header_emitted = True
    # GFM tables require a header separator; if none seen, synthesize one
    # below row 0 so the table renders.
    if out_lines and not header_emitted and len(out_lines) >= 1:
        first = out_lines[0]
        col_count = first.count("|") - 1
        out_lines.insert(1, "|" + "|".join(["---"] * col_count) + "|")
    return "\n".join(out_lines)


# ────────────────────────────────────────────────────────────────────────────
# Inline-level renderers


def _render_inline_seq(nodes: list[Any], sidecar: Sidecar) -> str:
    return "".join(_render_inline(n, sidecar) for n in nodes)


def _render_inline(node: Any, sidecar: Sidecar) -> str:
    if not isinstance(node, dict):
        return ""
    t = node.get("type")
    if t == "text":
        text = node.get("text") or ""
        marks = node.get("marks") or []
        return _apply_marks(text, marks)
    if t == "hardBreak":
        return "  \n"  # markdown hard-break (two spaces + newline)
    if t == "status":
        attrs = node.get("attrs") or {}
        label = str(attrs.get("text") or "")
        color = str(attrs.get("color") or "neutral")
        return f"{{status:{label}|{color}}}"
    # Opaque inline node (inlineCard, mention, emoji, mediaInline, ...)
    key = sidecar.add(node)
    return f"[[ADF:{key}]]"


def _apply_marks(text: str, marks: list[Any]) -> str:
    """Apply marks innermost-first so they nest cleanly."""
    # `link` is treated as the outermost mark.
    link_url: str | None = None
    code = False
    strong = False
    em = False
    strike = False
    for mark in marks or []:
        if not isinstance(mark, dict):
            continue
        mt = mark.get("type")
        if mt == "link":
            url = (mark.get("attrs") or {}).get("href")
            if isinstance(url, str):
                link_url = url
        elif mt == "code":
            code = True
        elif mt == "strong":
            strong = True
        elif mt == "em":
            em = True
        elif mt == "strike":
            strike = True
        # subsup, underline, textColor, backgroundColor — we drop the mark
        # but preserve the text. (Round-trip caveat: agents can lose these.)

    out = text
    if code:
        out = f"`{out}`"
    if strong:
        out = f"**{out}**"
    if em:
        out = f"*{out}*"
    if strike:
        out = f"~~{out}~~"
    if link_url:
        # Escape closing bracket in link text.
        safe = out.replace("]", "\\]")
        out = f"[{safe}]({link_url})"
    return out

"""ADF ↔ markdown round-trip + targeted unit tests.

The round-trip property: for the supported subset, the structural shape
of ADF→md→ADF must match the original. For opaque nodes (inlineCard,
mediaInline, panel-with-rich-children, ...), they must be preserved
byte-identically via the sidecar.
"""
import json

import pytest

from adf import adf_to_markdown, markdown_to_adf, SmartLinkPolicy
from adf.types import Sidecar
from adf.from_md import _parse_inline


def _doc(*content):
    return {"type": "doc", "version": 1, "content": list(content)}


def _para(*text_parts):
    return {"type": "paragraph", "content": list(text_parts)}


def _t(text, marks=None):
    n = {"type": "text", "text": text}
    if marks:
        n["marks"] = marks
    return n


# ────────────────────────────────────────────────────────────────────────────
# adf_to_markdown — block-level
# ────────────────────────────────────────────────────────────────────────────


class TestADFToMD:
    def test_paragraph(self):
        md, _ = adf_to_markdown(_doc(_para(_t("hello world"))))
        assert md.strip() == "hello world"

    def test_heading_levels(self):
        for lv in range(1, 7):
            md, _ = adf_to_markdown(_doc({"type": "heading", "attrs": {"level": lv}, "content": [_t("h")]}))
            assert md.strip() == "#" * lv + " h"

    def test_strong_em_code_strike(self):
        md, _ = adf_to_markdown(_doc(_para(
            _t("bold", [{"type": "strong"}]),
            _t(" "),
            _t("em", [{"type": "em"}]),
            _t(" "),
            _t("code", [{"type": "code"}]),
            _t(" "),
            _t("dead", [{"type": "strike"}]),
        )))
        assert "**bold**" in md
        assert "*em*" in md
        assert "`code`" in md
        assert "~~dead~~" in md

    def test_link(self):
        md, _ = adf_to_markdown(_doc(_para(
            _t("docs", [{"type": "link", "attrs": {"href": "https://example.com"}}]),
        )))
        assert "[docs](https://example.com)" in md

    def test_bullet_list(self):
        md, _ = adf_to_markdown(_doc({
            "type": "bulletList",
            "content": [
                {"type": "listItem", "content": [_para(_t("a"))]},
                {"type": "listItem", "content": [_para(_t("b"))]},
            ],
        }))
        assert md.strip() == "- a\n- b"

    def test_ordered_list(self):
        md, _ = adf_to_markdown(_doc({
            "type": "orderedList",
            "content": [
                {"type": "listItem", "content": [_para(_t("a"))]},
                {"type": "listItem", "content": [_para(_t("b"))]},
            ],
        }))
        assert md.strip() == "1. a\n2. b"

    def test_code_block_with_language(self):
        md, _ = adf_to_markdown(_doc({
            "type": "codeBlock",
            "attrs": {"language": "python"},
            "content": [_t("print(1)")],
        }))
        assert md.strip() == "```python\nprint(1)\n```"

    def test_blockquote(self):
        md, _ = adf_to_markdown(_doc({
            "type": "blockquote",
            "content": [_para(_t("hi"))],
        }))
        assert md.strip() == "> hi"

    def test_hr(self):
        md, _ = adf_to_markdown(_doc({"type": "rule"}))
        assert md.strip() == "---"

    def test_table_with_header(self):
        md, _ = adf_to_markdown(_doc({
            "type": "table",
            "content": [
                {"type": "tableRow", "content": [
                    {"type": "tableHeader", "content": [_para(_t("A"))]},
                    {"type": "tableHeader", "content": [_para(_t("B"))]},
                ]},
                {"type": "tableRow", "content": [
                    {"type": "tableCell", "content": [_para(_t("1"))]},
                    {"type": "tableCell", "content": [_para(_t("2"))]},
                ]},
            ],
        }))
        assert "| A | B |" in md
        assert "|---|---|" in md
        assert "| 1 | 2 |" in md

    def test_opaque_inline_card_becomes_token(self):
        md, sidecar = adf_to_markdown(_doc(_para(
            {"type": "inlineCard", "attrs": {"url": "https://x.atlassian.net/browse/LLM-1"}},
        )))
        assert "[[ADF:a0]]" in md
        assert sidecar.nodes["a0"]["type"] == "inlineCard"


# ────────────────────────────────────────────────────────────────────────────
# markdown_to_adf — block-level
# ────────────────────────────────────────────────────────────────────────────


class TestMDToADF:
    def test_paragraph(self):
        adf = markdown_to_adf("hello world")
        assert adf["content"] == [_para(_t("hello world"))]

    def test_heading(self):
        adf = markdown_to_adf("### header")
        assert adf["content"][0]["type"] == "heading"
        assert adf["content"][0]["attrs"]["level"] == 3

    def test_inline_marks(self):
        adf = markdown_to_adf("**bold** and *em* and `code` and ~~dead~~")
        nodes = adf["content"][0]["content"]
        kinds = [(n.get("text"), tuple(m["type"] for m in n.get("marks") or [])) for n in nodes]
        assert ("bold", ("strong",)) in kinds
        assert ("em", ("em",)) in kinds
        assert ("code", ("code",)) in kinds
        assert ("dead", ("strike",)) in kinds

    def test_fenced_code_with_language(self):
        adf = markdown_to_adf("```python\nprint(1)\n```")
        block = adf["content"][0]
        assert block["type"] == "codeBlock"
        assert block["attrs"]["language"] == "python"
        assert block["content"][0]["text"] == "print(1)"

    def test_bullet_list(self):
        adf = markdown_to_adf("- a\n- b")
        block = adf["content"][0]
        assert block["type"] == "bulletList"
        assert [li["content"][0]["content"][0]["text"] for li in block["content"]] == ["a", "b"]

    def test_ordered_list(self):
        adf = markdown_to_adf("1. a\n2. b")
        block = adf["content"][0]
        assert block["type"] == "orderedList"
        assert block["attrs"] == {"order": 1}

    def test_blockquote(self):
        adf = markdown_to_adf("> hi")
        assert adf["content"][0]["type"] == "blockquote"

    def test_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        adf = markdown_to_adf(md)
        table = adf["content"][0]
        assert table["type"] == "table"
        rows = table["content"]
        assert rows[0]["content"][0]["type"] == "tableHeader"
        assert rows[1]["content"][0]["type"] == "tableCell"

    def test_link(self):
        adf = markdown_to_adf("[docs](https://example.com)")
        node = adf["content"][0]["content"][0]
        assert node["type"] == "text"
        assert node["text"] == "docs"
        assert node["marks"] == [{"type": "link", "attrs": {"href": "https://example.com"}}]

    def test_hard_break(self):
        adf = markdown_to_adf("line1  \nline2")
        para = adf["content"][0]
        kinds = [c["type"] for c in para["content"]]
        assert "hardBreak" in kinds


# ────────────────────────────────────────────────────────────────────────────
# Smart links
# ────────────────────────────────────────────────────────────────────────────


class TestSmartLinks:
    def test_jira_browse_becomes_inline_card(self):
        adf = markdown_to_adf("[X](https://x.atlassian.net/browse/LLM-1)")
        node = adf["content"][0]["content"][0]
        assert node == {"type": "inlineCard", "attrs": {"url": "https://x.atlassian.net/browse/LLM-1"}}

    def test_confluence_wiki_becomes_inline_card(self):
        adf = markdown_to_adf("[doc](https://x.atlassian.net/wiki/spaces/X/pages/42)")
        node = adf["content"][0]["content"][0]
        assert node["type"] == "inlineCard"

    def test_github_pr_becomes_inline_card(self):
        adf = markdown_to_adf("[pr](https://github.com/org/repo/pull/123)")
        node = adf["content"][0]["content"][0]
        assert node["type"] == "inlineCard"

    def test_unknown_url_stays_plain_link(self):
        adf = markdown_to_adf("[docs](https://example.com/path)")
        node = adf["content"][0]["content"][0]
        assert node["type"] == "text"
        assert node["marks"][0]["type"] == "link"

    def test_disabled_smart_links(self):
        adf = markdown_to_adf(
            "[X](https://x.atlassian.net/browse/LLM-1)",
            smart_links=SmartLinkPolicy(enabled=False),
        )
        node = adf["content"][0]["content"][0]
        assert node["type"] == "text"
        assert node["marks"][0]["type"] == "link"


# ────────────────────────────────────────────────────────────────────────────
# Sidecar round-trip (opaque preservation)
# ────────────────────────────────────────────────────────────────────────────


class TestSidecarRoundTrip:
    def test_opaque_inline_card_preserved_byte_identical(self):
        original = _doc(_para(
            _t("see "),
            {"type": "inlineCard", "attrs": {"url": "https://x.atlassian.net/browse/LLM-1"}},
            _t(" for context"),
        ))
        md, sidecar = adf_to_markdown(original)
        # Round-trip
        back = markdown_to_adf(md, sidecar=sidecar)
        para = back["content"][0]
        card = next(c for c in para["content"] if c.get("type") == "inlineCard")
        assert card == {"type": "inlineCard", "attrs": {"url": "https://x.atlassian.net/browse/LLM-1"}}

    def test_opaque_panel_block_preserved(self):
        panel = {
            "type": "panel",
            "attrs": {"panelType": "info"},
            "content": [_para(_t("note"))],
        }
        original = _doc(panel, _para(_t("after")))
        md, sidecar = adf_to_markdown(original)
        back = markdown_to_adf(md, sidecar=sidecar)
        assert back["content"][0] == panel

    def test_token_with_missing_sidecar_renders_as_literal(self):
        # If the agent invents a token that we don't have a sidecar for,
        # render as plain text instead of silently dropping it.
        adf = markdown_to_adf("here is [[ADF:bogus]] inline")
        text = "".join(n.get("text", "") for n in adf["content"][0]["content"] if n.get("type") == "text")
        assert "[[ADF:bogus]]" in text


# ────────────────────────────────────────────────────────────────────────────
# End-to-end round-trip on a realistic RFC-shaped doc
# ────────────────────────────────────────────────────────────────────────────


def test_rfc_shaped_doc_round_trips():
    original = _doc(
        {"type": "heading", "attrs": {"level": 1}, "content": [_t("[LLM-42] RFC: Sharded cache")]},
        _para(
            _t("Status: "),
            _t("Draft", [{"type": "strong"}]),
        ),
        _para(_t("Ticket: "), {"type": "inlineCard", "attrs": {"url": "https://x.atlassian.net/browse/LLM-42"}}),
        {"type": "heading", "attrs": {"level": 2}, "content": [_t("Problem")]},
        _para(_t("Single-shard cache pegs CPU at 90% under peak load.")),
        {"type": "heading", "attrs": {"level": 2}, "content": [_t("Decision")]},
        {
            "type": "table",
            "content": [
                {"type": "tableRow", "content": [
                    {"type": "tableHeader", "content": [_para(_t("Option"))]},
                    {"type": "tableHeader", "content": [_para(_t("Chosen"))]},
                    {"type": "tableHeader", "content": [_para(_t("Reason"))]},
                ]},
                {"type": "tableRow", "content": [
                    {"type": "tableCell", "content": [_para(_t("Sharded"))]},
                    {"type": "tableCell", "content": [_para(_t("yes"))]},
                    {"type": "tableCell", "content": [_para(_t("scales horizontally"))]},
                ]},
            ],
        },
        {"type": "heading", "attrs": {"level": 2}, "content": [_t("Open questions")]},
        {
            "type": "orderedList",
            "content": [
                {"type": "listItem", "content": [_para(_t("How many shards?"))]},
                {"type": "listItem", "content": [_para(_t("Re-balance strategy?"))]},
            ],
        },
    )

    md, sidecar = adf_to_markdown(original)
    # Markdown should contain the inline-card placeholder
    assert "[[ADF:" in md
    # And typical headings / table rows / list items
    assert "# [LLM-42] RFC: Sharded cache" in md
    assert "| Option | Chosen | Reason |" in md
    assert "1. How many shards?" in md

    back = markdown_to_adf(md, sidecar=sidecar)
    # Structural checks
    assert back["type"] == "doc"
    block_types = [b["type"] for b in back["content"]]
    assert block_types[0] == "heading"
    assert "table" in block_types
    assert "orderedList" in block_types
    # Inline card came back byte-identical
    para_with_card = next(b for b in back["content"] if b["type"] == "paragraph" and any(c.get("type") == "inlineCard" for c in b["content"]))
    card = next(c for c in para_with_card["content"] if c["type"] == "inlineCard")
    assert card == {"type": "inlineCard", "attrs": {"url": "https://x.atlassian.net/browse/LLM-42"}}


def test_table_without_header_separator_synthesizes_one_in_md():
    # Some ADF tables don't have tableHeader cells. The MD renderer
    # synthesizes a separator so the output is still valid GFM.
    adf = _doc({
        "type": "table",
        "content": [
            {"type": "tableRow", "content": [
                {"type": "tableCell", "content": [_para(_t("A"))]},
                {"type": "tableCell", "content": [_para(_t("B"))]},
            ]},
            {"type": "tableRow", "content": [
                {"type": "tableCell", "content": [_para(_t("1"))]},
                {"type": "tableCell", "content": [_para(_t("2"))]},
            ]},
        ],
    })
    md, _ = adf_to_markdown(adf)
    lines = [l for l in md.splitlines() if l.strip()]
    assert lines[0] == "| A | B |"
    assert lines[1] == "|---|---|"


class TestPanels:
    def test_md_to_adf_info_panel(self):
        adf = markdown_to_adf("::: panel info\nHeads up!\n:::")
        block = adf["content"][0]
        assert block["type"] == "panel"
        assert block["attrs"]["panelType"] == "info"
        # Inner content is a paragraph with the text
        para = block["content"][0]
        assert para["type"] == "paragraph"
        assert para["content"][0]["text"] == "Heads up!"

    def test_panel_round_trips(self):
        original = _doc({
            "type": "panel",
            "attrs": {"panelType": "warning"},
            "content": [_para(_t("Be careful"))],
        })
        md, sidecar = adf_to_markdown(original)
        assert "::: panel warning" in md and ":::" in md
        back = markdown_to_adf(md, sidecar=sidecar)
        assert back["content"][0]["type"] == "panel"
        assert back["content"][0]["attrs"]["panelType"] == "warning"

    def test_panel_with_richer_content(self):
        md = "::: panel note\n## Heading inside\n\nA list:\n\n- one\n- two\n:::"
        adf = markdown_to_adf(md)
        panel = adf["content"][0]
        assert panel["type"] == "panel"
        block_types = [b["type"] for b in panel["content"]]
        assert "heading" in block_types
        assert "bulletList" in block_types

    def test_unknown_panel_type_rejects(self):
        # The opener regex restricts panel types; unknown ones aren't matched
        # and the line falls through to be parsed as a paragraph instead.
        adf = markdown_to_adf("::: panel bogus\ntext\n:::")
        assert adf["content"][0]["type"] != "panel"


class TestStatusBadges:
    def test_status_inline(self):
        adf = markdown_to_adf("Current: {status:Draft|purple}")
        nodes = adf["content"][0]["content"]
        status = next(n for n in nodes if n.get("type") == "status")
        assert status["attrs"]["text"] == "Draft"
        assert status["attrs"]["color"] == "purple"

    def test_status_color_defaults_to_neutral(self):
        adf = markdown_to_adf("{status:Open}")
        status = adf["content"][0]["content"][0]
        assert status["attrs"]["color"] == "neutral"

    def test_status_unknown_color_falls_back_to_neutral(self):
        adf = markdown_to_adf("{status:Open|chartreuse}")
        status = adf["content"][0]["content"][0]
        assert status["attrs"]["color"] == "neutral"

    def test_status_round_trips(self):
        original = _doc(_para(_t("X "), {"type": "status", "attrs": {"text": "In Review", "color": "blue"}}))
        md, sidecar = adf_to_markdown(original)
        assert "{status:In Review|blue}" in md
        back = markdown_to_adf(md, sidecar=sidecar)
        para = back["content"][0]
        status = next(n for n in para["content"] if n.get("type") == "status")
        assert status["attrs"] == {"text": "In Review", "color": "blue"}


class TestTaskLists:
    def test_task_list_from_md(self):
        adf = markdown_to_adf("- [ ] open one\n- [x] done one\n- [ ] open two")
        block = adf["content"][0]
        assert block["type"] == "taskList"
        items = block["content"]
        assert [i["attrs"]["state"] for i in items] == ["TODO", "DONE", "TODO"]
        assert items[0]["content"][0]["text"] == "open one"
        assert items[1]["content"][0]["text"] == "done one"

    def test_plain_bullet_list_unaffected(self):
        adf = markdown_to_adf("- alpha\n- beta")
        block = adf["content"][0]
        assert block["type"] == "bulletList"

    def test_task_list_round_trips(self):
        original = _doc({
            "type": "taskList",
            "attrs": {"localId": "tasks"},
            "content": [
                {"type": "taskItem", "attrs": {"localId": "t0", "state": "TODO"}, "content": [_t("buy milk")]},
                {"type": "taskItem", "attrs": {"localId": "t1", "state": "DONE"}, "content": [_t("ship rfc")]},
            ],
        })
        md, sidecar = adf_to_markdown(original)
        assert "- [ ] buy milk" in md
        assert "- [x] ship rfc" in md
        back = markdown_to_adf(md, sidecar=sidecar)
        assert back["content"][0]["type"] == "taskList"
        states = [i["attrs"]["state"] for i in back["content"][0]["content"]]
        assert states == ["TODO", "DONE"]


def test_empty_doc_returns_empty_string():
    md, _ = adf_to_markdown(_doc())
    assert md == ""


def test_non_doc_returns_empty():
    md, _ = adf_to_markdown({"type": "paragraph"})
    assert md == ""

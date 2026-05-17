"""Atlas Doc Format ↔ markdown bidirectional translator.

We translate the subset our templates need:
  paragraph, heading 1-6, bulletList, orderedList, listItem,
  codeBlock (with language), inlineCode, link, hardBreak,
  blockquote, rule, table/row/cell/header.

Everything else (inlineCard, media*, panel with rich children, expand,
extension, ...) is preserved as an opaque "[[ADF:<id>]]" token backed by
a sidecar dict. The agent edits the markdown without knowing those nodes
exist; on the way back to ADF they get re-inserted verbatim.

Additionally: bare markdown links to known smart-link URL patterns
(*.atlassian.net/browse/.*, *.atlassian.net/wiki/.*, github.com/.../pull/.*)
are auto-rewritten to inlineCard nodes on the way INTO ADF, so the agent
can write `[text](url)` and get a smart link for free.
"""

from .to_md import adf_to_markdown
from .from_md import markdown_to_adf, SmartLinkPolicy
from .types import Sidecar

__all__ = ["adf_to_markdown", "markdown_to_adf", "Sidecar", "SmartLinkPolicy"]

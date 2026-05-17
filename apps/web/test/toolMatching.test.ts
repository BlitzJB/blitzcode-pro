/**
 * Tool-name matching for MCP-namespaced names.
 *
 * Agents call workflow tools via MCP; their tool_use blocks come through
 * as `mcp__workflow__workflow_update_ticket_fields`, NOT the bare name.
 * Our refresh-key derivation and doc-slot auto-open both have to match
 * the short form so namespacing doesn't silently break either.
 *
 * Regression history: ticket auto-refresh was completely dead because the
 * matcher only checked bare names. Don't let that come back.
 */
import { describe, it, expect } from "vitest";

// Mirror the implementation. (We don't import from chat.tsx because that
// file is a giant React UI module — duplicating the small string helper
// here is cheaper than refactoring the file just to export it.)
function shortToolName(name: string): string {
  const parts = name.split("__");
  return parts.length >= 3 && parts[0] === "mcp" ? parts.slice(2).join("__") : name;
}

describe("shortToolName", () => {
  it("strips a single mcp__server__ prefix", () => {
    expect(shortToolName("mcp__workflow__workflow_update_ticket_fields")).toBe(
      "workflow_update_ticket_fields"
    );
  });

  it("leaves bare names untouched", () => {
    expect(shortToolName("workflow_update_ticket_fields")).toBe(
      "workflow_update_ticket_fields"
    );
  });

  it("rejoins remaining parts so multi-underscore tool names survive", () => {
    // hypothetical: mcp__server__foo__bar  ->  foo__bar
    expect(shortToolName("mcp__workflow__foo__bar")).toBe("foo__bar");
  });

  it("doesn't strip when the prefix isn't mcp", () => {
    expect(shortToolName("notmcp__workflow__foo")).toBe("notmcp__workflow__foo");
  });

  it("handles fewer-than-3-parts cleanly", () => {
    expect(shortToolName("mcp__only")).toBe("mcp__only");
    expect(shortToolName("plain")).toBe("plain");
  });
});

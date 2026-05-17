/**
 * DocSlot — the right-rail's single "wide" panel is mutually exclusive across
 * plan / ticket / RFC / debrief. One slot, one state machine. Pure (no
 * React) so it's trivially testable in isolation.
 *
 * Three categories of drivers update the slot:
 *
 *   1. Plan-mode transitions  — when the agent's permission_mode becomes
 *      "plan" and a plan exists, the slot auto-opens to {kind:"plan"}.
 *      When the agent's pre-plan mode is restored, the slot returns to
 *      hidden (unless the user had manually toggled something else).
 *
 *   2. Tool-use activity      — when the agent's last tool call was
 *      workflow_write_rfc, workflow_write_debrief, or
 *      workflow_update_ticket_fields, auto-open to the matching doc.
 *      Latest tool wins.
 *
 *   3. Explicit user clicks   — clicking a doc tile toggles that doc;
 *      clicking the same tile or Esc hides.
 *
 * "Latest event wins" — every driver call replaces the current slot state.
 * Workspace-scoped: changing workspace resets the slot to hidden.
 */

export type DocSlotState =
  | { kind: "hidden" }
  | { kind: "plan"; reason: "plan_mode" | "manual" }
  | { kind: "ticket"; ticketKey: string }
  | { kind: "rfc"; workspaceId: string }
  | { kind: "debrief"; workspaceId: string };

export type DocSlotAction =
  | { type: "plan_mode_entered" }            // agent is now in plan mode AND we have a plan
  | { type: "plan_mode_exited" }             // agent left plan mode (and there's no manual override)
  | { type: "tool_use_doc_write"; doc: "ticket" | "rfc" | "debrief"; workspaceId: string; ticketKey: string }
  | { type: "user_toggle"; target: { kind: "plan" } | { kind: "ticket"; ticketKey: string } | { kind: "rfc"; workspaceId: string } | { kind: "debrief"; workspaceId: string } }
  | { type: "hide" }
  | { type: "workspace_changed" };           // reset on workspace switch

export const initialDocSlot: DocSlotState = { kind: "hidden" };

export function reduceDocSlot(state: DocSlotState, action: DocSlotAction): DocSlotState {
  switch (action.type) {
    case "workspace_changed":
      return { kind: "hidden" };

    case "plan_mode_entered":
      // While in plan mode the slot is FORCED to plan — no other doc can
      // be visible. This is the only auto-open we treat as authoritative.
      return { kind: "plan", reason: "plan_mode" };

    case "plan_mode_exited":
      // Only retract if we're currently showing plan-from-plan-mode. If
      // the user had manually opened plan or any other doc, leave it.
      if (state.kind === "plan" && state.reason === "plan_mode") {
        return { kind: "hidden" };
      }
      return state;

    case "tool_use_doc_write": {
      // If we're in plan mode (forced), the agent shouldn't be writing
      // docs anyway. But to be safe, plan-from-plan-mode wins over a
      // stray doc write — UI hint to user that we're still planning.
      if (state.kind === "plan" && state.reason === "plan_mode") {
        return state;
      }
      switch (action.doc) {
        case "ticket":
          return { kind: "ticket", ticketKey: action.ticketKey };
        case "rfc":
          return { kind: "rfc", workspaceId: action.workspaceId };
        case "debrief":
          return { kind: "debrief", workspaceId: action.workspaceId };
      }
    }

    case "user_toggle": {
      // Plan mode is sticky — user cannot manually dismiss plan view while
      // still in plan mode. They'd have to flip the mode first.
      if (state.kind === "plan" && state.reason === "plan_mode" && action.target.kind !== "plan") {
        return state;
      }
      // Clicking the same target hides; clicking a different one opens it.
      if (isSameSlot(state, action.target)) {
        return { kind: "hidden" };
      }
      switch (action.target.kind) {
        case "plan":
          return { kind: "plan", reason: "manual" };
        case "ticket":
          return { kind: "ticket", ticketKey: action.target.ticketKey };
        case "rfc":
          return { kind: "rfc", workspaceId: action.target.workspaceId };
        case "debrief":
          return { kind: "debrief", workspaceId: action.target.workspaceId };
      }
    }

    case "hide":
      // Same plan-mode stickiness as user_toggle.
      if (state.kind === "plan" && state.reason === "plan_mode") return state;
      return { kind: "hidden" };
  }
}

function isSameSlot(
  state: DocSlotState,
  target: { kind: "plan" } | { kind: "ticket"; ticketKey: string } | { kind: "rfc"; workspaceId: string } | { kind: "debrief"; workspaceId: string },
): boolean {
  if (state.kind !== target.kind) return false;
  switch (target.kind) {
    case "plan": return true;
    case "ticket": return state.kind === "ticket" && state.ticketKey === target.ticketKey;
    case "rfc": return state.kind === "rfc" && state.workspaceId === target.workspaceId;
    case "debrief": return state.kind === "debrief" && state.workspaceId === target.workspaceId;
  }
}

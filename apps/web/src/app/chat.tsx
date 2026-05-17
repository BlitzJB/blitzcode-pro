"use client";

import { useAgentMux, useActiveSession, type AgentMux, type SessionState } from "@agent-webkit/react";
import type { SessionListEntry } from "@agent-webkit/core";
import { AnimatePresence, motion, type Variants } from "framer-motion";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type StoredSession = SessionListEntry;

function deriveLabel(s: StoredSession): string {
  if (s.cwd) {
    const parts = s.cwd.split("/").filter(Boolean);
    if (parts.length === 0) return s.cwd;
    if (parts.length === 1) return parts[0]!;
    return parts.slice(-2).join("/");
  }
  return `Untitled · ${s.id.slice(0, 6)}`;
}

function relativeTime(ts: number): string {
  const diffMs = Date.now() - ts * 1000;
  const sec = Math.round(diffMs / 1000);
  if (sec < 60) return "just now";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  return `${day}d ago`;
}

// ────────────────────────────────────────────────────────────────────────────
// View model
//
// Reducer state is a flat list of {user | assistant | tool_result} messages.
// For display we collapse it into a sequence of "turns": each tool_use block
// inside an assistant message gets paired with its tool_result (matched by
// tool_use_id) and rendered as one combined card. Assistant text and tool
// calls interleave in their natural order within a turn.
// ────────────────────────────────────────────────────────────────────────────

type ContentBlock = { type?: string; text?: string; id?: string; name?: string; input?: unknown };

type ToolCallItem = {
  kind: "tool_call";
  id: string;
  name: string;
  input: unknown;
  status: "running" | "done" | "error";
  result?: { output: unknown; is_error: boolean };
};

type AssistantTextItem = {
  kind: "assistant_text";
  id: string;
  text: string;
  streaming: boolean;
};

type UserItem = { kind: "user"; id: string; content: string };

type ToolGroupItem = {
  kind: "tool_group";
  id: string;
  items: ToolCallItem[];
};

type ChatItem = UserItem | AssistantTextItem | ToolCallItem | ToolGroupItem;

type AnySessionMessage = ReturnType<typeof useActiveSession>["messages"][number];

function buildChatItems(messages: AnySessionMessage[]): ChatItem[] {
  // Map tool_use_id → tool_result so we can pair them as we walk.
  const results = new Map<string, { output: unknown; is_error: boolean }>();
  for (const m of messages) {
    if (m.kind === "tool_result") {
      results.set(m.tool_use_id, { output: m.output, is_error: Boolean(m.is_error) });
    }
  }

  // Track which tool_use ids we've already emitted. The same tool_use can
  // appear in two assistant messages when history-replay's message_id
  // differs from the live message_complete's id — both messages survive
  // the history_loaded dedupe (which keys on message_id) and each
  // re-emits the same block. Keep the LAST occurrence (live state is
  // fresher than transcript replay) by walking right-to-left.
  const seenToolUse = new Set<string>();
  const keepBlock = new Map<string, Set<number>>(); // message id → block indices to keep

  for (let mi = messages.length - 1; mi >= 0; mi--) {
    const m = messages[mi]!;
    if (m.kind !== "assistant") continue;
    const blocks = (m.content as ContentBlock[]) ?? [];
    const keep = new Set<number>();
    blocks.forEach((block, i) => {
      if (block?.type === "tool_use") {
        const tid = block.id ?? `${m.id}-tu-${i}`;
        if (seenToolUse.has(tid)) return;
        seenToolUse.add(tid);
      }
      keep.add(i);
    });
    keepBlock.set(m.id, keep);
  }

  const items: ChatItem[] = [];
  for (const m of messages) {
    if (m.kind === "user") {
      const content = typeof m.content === "string"
        ? m.content
        : Array.isArray(m.content)
          ? m.content.filter((b: ContentBlock) => typeof b.text === "string").map((b: ContentBlock) => b.text).join("")
          : "";
      items.push({ kind: "user", id: m.id, content });
      continue;
    }
    if (m.kind === "assistant") {
      const blocks = (m.content as ContentBlock[]) ?? [];
      const allowed = keepBlock.get(m.id);
      blocks.forEach((block, i) => {
        if (allowed && !allowed.has(i)) return;
        if (block?.type === "tool_use") {
          const tid = block.id ?? `${m.id}-tu-${i}`;
          const result = results.get(tid);
          items.push({
            kind: "tool_call",
            id: tid,
            name: block.name ?? "tool",
            input: block.input ?? {},
            status: result ? (result.is_error ? "error" : "done") : "running",
            result,
          });
          return;
        }
        const text = typeof block?.text === "string" ? block.text : "";
        if (!text) return;
        items.push({
          kind: "assistant_text",
          id: `${m.id}-t-${i}`,
          text,
          // The reducer flags the whole assistant message as streaming; we
          // inherit it on every text fragment within it.
          streaming: Boolean((m as { streaming?: boolean }).streaming),
        });
      });
      continue;
    }
    // tool_result was consumed by the lookup above; nothing to emit.
  }
  return collapseToolRuns(items);
}

// Walk the items and fold consecutive tool_call items into a single
// tool_group. Single isolated tool calls (one tool surrounded by text)
// keep their own card — only runs of 2+ collapse, since a one-off doesn't
// need a summary line.
function collapseToolRuns(items: ChatItem[]): ChatItem[] {
  const out: ChatItem[] = [];
  let run: ToolCallItem[] = [];
  const flush = () => {
    if (run.length === 0) return;
    if (run.length === 1) {
      out.push(run[0]!);
    } else {
      out.push({
        kind: "tool_group",
        id: `grp-${run[0]!.id}-${run[run.length - 1]!.id}-${run.length}`,
        items: run,
      });
    }
    run = [];
  };
  for (const it of items) {
    if (it.kind === "tool_call") {
      run.push(it);
      continue;
    }
    flush();
    out.push(it);
  }
  flush();
  return out;
}

// ────────────────────────────────────────────────────────────────────────────
// Motion presets
// ────────────────────────────────────────────────────────────────────────────

const EASE_OUT = [0.23, 1, 0.32, 1] as const;

const itemVariants: Variants = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.32, ease: EASE_OUT } },
};

// ────────────────────────────────────────────────────────────────────────────
// App-layer "needs input" tracking — NOT part of agent-webkit.
// Server: apps/server/acks.py persists per-session timestamps.
// Client: this hook syncs them and derives needs_input.
// ────────────────────────────────────────────────────────────────────────────

interface AckRecord {
  last_completion_at: number;
  last_ack_at: number;
}

export interface AppAcks {
  /** Per-session ack state (timestamps in epoch seconds). */
  map: Record<string, AckRecord>;
  /** Optimistic: bump local completion time (called when /stream emits a `result`). */
  markCompletionLocal: (sid: string) => void;
  /** Optimistic POST: mark this session as acknowledged (last_ack_at = now). */
  acknowledge: (sid: string) => Promise<void>;
}

function useAppAcks(baseUrl: string): AppAcks {
  const [map, setMap] = useState<Record<string, AckRecord>>({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${baseUrl}/app/state`);
        if (!r.ok) return;
        const data = (await r.json()) as { acks?: Record<string, AckRecord> };
        if (cancelled) return;
        setMap(data.acks ?? {});
      } catch {
        /* offline-ish fallback: leave map empty, treat all as idle */
      }
    })();
    return () => { cancelled = true; };
  }, [baseUrl]);

  const markCompletionLocal = useCallback((sid: string) => {
    setMap((prev) => {
      const cur = prev[sid] ?? { last_completion_at: 0, last_ack_at: 0 };
      // Server tracks the same in-process; we mirror it client-side so the
      // UI updates without a refetch. Use ms→s for consistency with server.
      return { ...prev, [sid]: { ...cur, last_completion_at: Date.now() / 1000 } };
    });
  }, []);

  const acknowledge = useCallback(
    async (sid: string) => {
      const now = Date.now() / 1000;
      setMap((prev) => {
        const cur = prev[sid] ?? { last_completion_at: 0, last_ack_at: 0 };
        return { ...prev, [sid]: { ...cur, last_ack_at: now } };
      });
      try {
        const r = await fetch(`${baseUrl}/app/sessions/${encodeURIComponent(sid)}/acknowledge`, {
          method: "POST",
        });
        if (!r.ok) throw new Error(`ack failed: ${r.status}`);
        const data = (await r.json()) as { last_completion_at: number; last_ack_at: number };
        setMap((prev) => ({ ...prev, [sid]: { last_completion_at: data.last_completion_at, last_ack_at: data.last_ack_at } }));
      } catch {
        /* leave the optimistic value — next /app/state refresh will reconcile */
      }
    },
    [baseUrl]
  );

  return { map, markCompletionLocal, acknowledge };
}

// ────────────────────────────────────────────────────────────────────────────
// Completions (slash commands, skills, agents) — NOT wire protocol.
// Server: GET /app/sessions/{sid}/completions reads {cwd}/.claude/* and
// ~/.claude/*. Cached client-side by sid so palette opens are instant.
// ────────────────────────────────────────────────────────────────────────────

export interface CompletionItem {
  name: string;
  description?: string;
  source: "project" | "user" | "builtin";
  kind: "command" | "skill" | "agent";
  argument_hint?: string;
  tools?: string[];
}

interface CompletionsData {
  commands: CompletionItem[];
  skills: CompletionItem[];
  agents: CompletionItem[];
}

function useCompletions(baseUrl: string): {
  get: (sid: string) => CompletionsData | null;
  refresh: (sid: string) => Promise<void>;
} {
  // Cached per session. Map identity is stable; we trigger renders via the
  // version counter so React re-runs anything reading from `get`.
  const cacheRef = useRef<Map<string, CompletionsData>>(new Map());
  const inflightRef = useRef<Map<string, Promise<void>>>(new Map());
  const [, bump] = useReducer((n: number) => n + 1, 0);

  const refresh = useCallback(
    async (sid: string): Promise<void> => {
      const existing = inflightRef.current.get(sid);
      if (existing) return existing;
      const p = (async () => {
        try {
          const r = await fetch(`${baseUrl}/app/sessions/${encodeURIComponent(sid)}/completions`);
          if (!r.ok) return;
          const data = (await r.json()) as CompletionsData;
          cacheRef.current.set(sid, {
            commands: data.commands ?? [],
            skills: data.skills ?? [],
            agents: data.agents ?? [],
          });
          bump();
        } catch {
          /* leave cache unset; palette will show empty */
        } finally {
          inflightRef.current.delete(sid);
        }
      })();
      inflightRef.current.set(sid, p);
      return p;
    },
    [baseUrl]
  );

  const get = useCallback((sid: string): CompletionsData | null => {
    return cacheRef.current.get(sid) ?? null;
  }, []);

  // Memoize the returned object so its identity is stable across renders.
  // Without this, the `bump` triggered after refresh would produce a new
  // object every render, and any effect listing `completions` in its deps
  // (e.g. the eager refresh in ChatView) would loop.
  return useMemo(() => ({ get, refresh }), [get, refresh]);
}

type SessionGroup = "working" | "needs_input" | "idle";

function categorizeSession(
  sessionState: SessionState | undefined,
  ack: AckRecord | undefined,
): SessionGroup {
  const status = sessionState?.status ?? "idle";
  if (status === "streaming" || status === "awaiting_hook") return "working";
  if (status === "awaiting_permission" || status === "awaiting_question") return "needs_input";
  if (ack && ack.last_completion_at > ack.last_ack_at) return "needs_input";
  return "idle";
}

// ────────────────────────────────────────────────────────────────────────────
// Top-level Chat
// ────────────────────────────────────────────────────────────────────────────

export default function Chat({ baseUrl }: { baseUrl: string }) {
  const acks = useAppAcks(baseUrl);
  const completions = useCompletions(baseUrl);
  // One persistent multiplexed stream for the whole app. Switching between
  // sessions is now pure render (no network) — past messages come from
  // mux.loadHistory(sid), future events are already flowing in via /stream.
  const mux = useAgentMux({
    baseUrl,
    onEvent: (ev) => {
      // App-layer "needs input" signal: an agent turn has completed.
      // The server persists this too via its in-process event subscription,
      // but bumping locally avoids a refetch round-trip.
      if (ev.event === "result" && ev.session_id) {
        acks.markCompletionLocal(ev.session_id);
      }
    },
  });
  const [activeId, setActiveId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  // Track which sessions we've already loaded history for so we don't refetch
  // the snapshot every time the user switches back to one.
  const historyLoaded = useRef<Set<string>>(new Set());

  // Once the initial /sessions list is in, default to the most recent.
  useEffect(() => {
    if (!mux.hydrated || activeId !== null) return;
    const first = mux.sessionList[0];
    if (first) setActiveId(first.id);
  }, [mux.hydrated, mux.sessionList, activeId]);

  // Whenever the active session changes, kick off the history fetch once.
  useEffect(() => {
    if (!activeId) return;
    if (historyLoaded.current.has(activeId)) return;
    historyLoaded.current.add(activeId);
    void mux.loadHistory(activeId);
  }, [activeId, mux]);

  const createNew = useCallback(
    async (cwd?: string) => {
      setCreating(true);
      try {
        const sid = await mux.createSession({
          include_partial_messages: true,
          // Default to acceptEdits: most sessions want auto-approved edits;
          // plan and bypassPermissions are opt-in via the header menu.
          permission_mode: "acceptEdits",
          ...(cwd ? { cwd } : {}),
        });
        setActiveId(sid);
      } finally {
        setCreating(false);
      }
    },
    [mux]
  );

  const deleteSession = useCallback(
    async (id: string) => {
      await mux.deleteSession(id);
      historyLoaded.current.delete(id);
      if (activeId === id) {
        const next = mux.sessionList.find((s) => s.id !== id);
        setActiveId(next?.id ?? null);
      }
    },
    [activeId, mux]
  );

  return (
    <div className="flex h-screen bg-canvas text-ink">
      <Sidebar
        baseUrl={baseUrl}
        sessions={mux.sessionList}
        sessionStates={mux.sessions}
        acks={acks}
        activeId={activeId}
        creating={creating}
        onSelect={setActiveId}
        onCreate={createNew}
        onDelete={deleteSession}
      />
      <div className="flex-1 min-w-0 flex flex-col">
        {!mux.hydrated ? (
          <LoadingState />
        ) : activeId ? (
          <ChatView mux={mux} sessionId={activeId} acks={acks} completions={completions} />
        ) : (
          <NoActiveSession onCreate={() => createNew()} creating={creating} />
        )}
      </div>
    </div>
  );
}

// TodoWrite is a normal SDK tool, not part of agent-webkit's wire protocol.
// Each call replaces the full list, so the latest tool_use with
// name === "TodoWrite" is authoritative. Derived purely from the per-session
// messages slot — no extra state, survives session switches.
interface TodoItem {
  content: string;
  status: "pending" | "in_progress" | "completed";
  activeForm?: string;
}

/** Latest ExitPlanMode tool_use's `input.plan`, or null if none. The agent
 *  may call ExitPlanMode multiple times as it refines — last one wins. */
function derivePlan(messages: AnySessionMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (!m || m.kind !== "assistant") continue;
    const blocks = (m.content as ContentBlock[]) ?? [];
    for (let j = blocks.length - 1; j >= 0; j--) {
      const b = blocks[j];
      if (b?.type === "tool_use" && b.name === "ExitPlanMode") {
        const input = (b.input ?? {}) as { plan?: unknown };
        return typeof input.plan === "string" ? input.plan : null;
      }
    }
  }
  return null;
}

function deriveTodos(messages: AnySessionMessage[]): TodoItem[] | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (!m || m.kind !== "assistant") continue;
    const blocks = (m.content as ContentBlock[]) ?? [];
    for (let j = blocks.length - 1; j >= 0; j--) {
      const b = blocks[j];
      if (b?.type === "tool_use" && b.name === "TodoWrite") {
        const input = (b.input ?? {}) as { todos?: unknown };
        if (!Array.isArray(input.todos)) return null;
        const out: TodoItem[] = [];
        for (const raw of input.todos) {
          if (!raw || typeof raw !== "object") continue;
          const obj = raw as Record<string, unknown>;
          if (typeof obj.content !== "string" || typeof obj.status !== "string") continue;
          const status = obj.status as TodoItem["status"];
          if (status !== "pending" && status !== "in_progress" && status !== "completed") continue;
          out.push({
            content: obj.content,
            status,
            ...(typeof obj.activeForm === "string" ? { activeForm: obj.activeForm } : {}),
          });
        }
        return out;
      }
    }
  }
  return null;
}

function ChatView({
  mux,
  sessionId,
  acks,
  completions,
}: {
  mux: AgentMux;
  sessionId: string;
  acks: AppAcks;
  completions: ReturnType<typeof useCompletions>;
}) {
  // Refresh the completions cache for this session on switch. The hook
  // dedupes inflight fetches, so this is safe to call eagerly.
  useEffect(() => {
    void completions.refresh(sessionId);
  }, [sessionId, completions]);
  const completionsForSession = completions.get(sessionId);
  const session = useActiveSession(mux, sessionId);
  const ack = acks.map[sessionId];
  // Only show Acknowledge when there's *something* to acknowledge AND no modal
  // owns the attention flow (permission/question have their own buttons).
  const needsAck =
    !!ack &&
    ack.last_completion_at > ack.last_ack_at &&
    !session.pendingPermission &&
    !session.pendingQuestion;
  const onAcknowledge = useCallback(() => acks.acknowledge(sessionId), [acks, sessionId]);
  const [input, setInput] = useState("");
  const scrollerRef = useRef<HTMLDivElement | null>(null);
  const items = useMemo(() => buildChatItems(session.messages), [session.messages]);
  const todos = useMemo(() => deriveTodos(session.messages), [session.messages]);
  // Sidebar visible only when there's an active plan — auto-retract once
  // everything is done.
  const showTodos = !!todos && todos.length > 0 && todos.some((t) => t.status !== "completed");

  const planText = useMemo(() => derivePlan(session.messages), [session.messages]);
  const currentMode =
    session.permissionMode ??
    mux.sessionList.find((s) => s.id === sessionId)?.permission_mode ??
    null;
  const inPlanMode = currentMode === "plan";
  const pendingPlanApproval =
    session.pendingPermission?.tool_name === "ExitPlanMode"
      ? session.pendingPermission
      : null;
  // Manual toggle so the user can re-open the plan view AFTER approving
  // (and being switched out of plan mode). The sidebar is forced open
  // whenever we're in plan mode or there's a pending approval; otherwise
  // it follows the user's last toggle. Reset on session switch.
  const [planSidebarManualOpen, setPlanSidebarManualOpen] = useState(false);
  useEffect(() => { setPlanSidebarManualOpen(false); }, [sessionId]);
  const showPlan =
    planText !== null &&
    (inPlanMode || pendingPlanApproval !== null || planSidebarManualOpen);

  // Snap to bottom on session switch — drop the flag, then the next render
  // (which has the new session's items) does an instant jump. Async history
  // loads might come in after the switch too; we keep the flag armed until
  // we actually have something to scroll past.
  const pendingSnapRef = useRef(true);
  useEffect(() => {
    pendingSnapRef.current = true;
  }, [sessionId]);

  // Auto-scroll on new items / status change. On session switch, snap
  // (instant, no smooth) to the bottom regardless of where we were. After
  // the snap, only follow new content if we're already near the bottom —
  // so reading old messages mid-stream doesn't yank the user.
  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    if (pendingSnapRef.current && items.length > 0) {
      el.scrollTo({ top: el.scrollHeight, behavior: "auto" });
      pendingSnapRef.current = false;
      return;
    }
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 200;
    if (nearBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [items, session.status]);

  const canSend =
    (session.status === "idle" || session.status === "error") && input.trim().length > 0;

  const onSend = () => {
    if (!canSend) return;
    session.send(input.trim());
    setInput("");
  };

  return (
    <div className="flex flex-1 min-h-0 bg-canvas text-ink">
      <div className="flex flex-col flex-1 min-w-0">
      <Header
        status={session.status}
        sessionId={sessionId}
        permissionMode={
          session.permissionMode ??
          mux.sessionList.find((s) => s.id === sessionId)?.permission_mode ??
          null
        }
        onChangeMode={(next) => {
          void mux.setPermissionMode(sessionId, next);
        }}
      />

      <div
        ref={scrollerRef}
        className="scroll-quiet flex-1 overflow-y-auto"
      >
        <div className="mx-auto w-full max-w-2xl px-6 pt-10 pb-40">
          {items.length === 0 && session.status === "idle" && <EmptyState />}

          <ol className="space-y-7">
            <AnimatePresence initial={false}>
              {items.map((item) => (
                <motion.li
                  key={item.id}
                  variants={itemVariants}
                  initial="initial"
                  animate="animate"
                  layout="position"
                >
                  {item.kind === "user" && <UserMessage content={item.content} />}
                  {item.kind === "assistant_text" && (
                    <AssistantText text={item.text} streaming={item.streaming} />
                  )}
                  {item.kind === "tool_call" && <ToolCall item={item} />}
                  {item.kind === "tool_group" && <ToolGroup group={item} />}
                </motion.li>
              ))}
            </AnimatePresence>

            {/* Streaming indicator when no message yet but we know one's coming */}
            {session.status === "streaming" &&
              !items.some(
                (i) => i.kind === "assistant_text" && i.streaming
              ) && (
                <li>
                  <ThinkingIndicator />
                </li>
              )}
          </ol>
        </div>
      </div>

      <AnimatePresence>
        {/* ExitPlanMode is handled inline in the PlanSidebar, not as a modal,
            so the user can keep the plan visible while reviewing it. Every
            other permission_request uses the standard modal. */}
        {session.pendingPermission && session.pendingPermission.tool_name !== "ExitPlanMode" && (
          <Modal key="perm">
            <PermissionCard
              req={session.pendingPermission}
              onApprove={(updatedPermissions) =>
                session.approve(session.pendingPermission!.correlation_id, {
                  ...(updatedPermissions ? { updatedPermissions } : {}),
                })
              }
              onDeny={() => session.deny(session.pendingPermission!.correlation_id, {})}
            />
          </Modal>
        )}
        {session.pendingQuestion && (
          <Modal key="q">
            <QuestionCard
              req={session.pendingQuestion}
              onAnswer={(answers) =>
                session.answer(session.pendingQuestion!.correlation_id, answers)
              }
            />
          </Modal>
        )}
        {session.lastError && (
          <Modal key="err">
            <ErrorCard
              code={session.lastError.code}
              message={session.lastError.message}
              onRestart={() => window.location.reload()}
            />
          </Modal>
        )}
      </AnimatePresence>

      <Composer
        value={input}
        onChange={setInput}
        onSubmit={onSend}
        disabled={!canSend}
        status={session.status}
        needsAck={needsAck}
        onAcknowledge={onAcknowledge}
        hasPlan={planText !== null}
        planSidebarOpen={showPlan}
        planSidebarLocked={inPlanMode || pendingPlanApproval !== null}
        onTogglePlanSidebar={() => setPlanSidebarManualOpen((v) => !v)}
        completions={completionsForSession}
      />
      </div>
      <AnimatePresence initial={false}>
        {showPlan && (
          <PlanSidebar
            key="plan-sidebar"
            plan={planText}
            pendingApproval={pendingPlanApproval}
            onApprove={() =>
              pendingPlanApproval && session.approve(pendingPlanApproval.correlation_id, {})
            }
            onDeny={() =>
              pendingPlanApproval &&
              session.deny(pendingPlanApproval.correlation_id, {
                // Claude's API rejects an is_error=true tool_result with
                // empty content (HTTP 400). Always send a non-empty deny
                // message — also gives the agent a hint to keep iterating.
                message: "User declined to approve this plan. Keep refining.",
              })
            }
          />
        )}
        {showTodos && todos && <TodoSidebar key="todo-sidebar" todos={todos} />}
      </AnimatePresence>
    </div>
  );
}

function PlanSidebar({
  plan,
  pendingApproval,
  onApprove,
  onDeny,
}: {
  plan: string | null;
  pendingApproval: { correlation_id: string } | null;
  onApprove: () => void;
  onDeny: () => void;
}) {
  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 630, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.24, ease: EASE_OUT }}
      className="shrink-0 border-l border-line bg-canvas overflow-hidden flex flex-col min-h-0"
    >
      <div className="px-4 py-3 border-b border-line flex items-baseline justify-between shrink-0">
        <span className="text-[11px] uppercase tracking-[0.14em] font-mono" style={{ color: "var(--accent-cool)" }}>
          Plan
        </span>
        <span className="text-[10px] font-mono text-ink-faint">
          {pendingApproval ? "awaiting approval" : "drafting"}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto scroll-quiet px-4 py-3">
        {plan === null ? (
          <div className="text-[12px] text-ink-faint italic">No plan yet…</div>
        ) : (
          <div
            className="prose prose-sm max-w-none prose-headings:font-serif prose-headings:text-ink
              prose-p:text-ink prose-li:text-ink prose-strong:text-ink
              prose-code:text-ink prose-code:bg-surface-sunk prose-code:px-1 prose-code:py-0.5 prose-code:rounded
              prose-code:before:hidden prose-code:after:hidden
              prose-a:text-[color:var(--accent-cool)] prose-a:no-underline hover:prose-a:underline
              prose-pre:bg-surface-sunk prose-pre:border prose-pre:border-line"
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{plan}</ReactMarkdown>
          </div>
        )}
      </div>
      {pendingApproval && (
        <div className="px-4 py-3 border-t border-line shrink-0 flex items-center justify-end gap-2 bg-surface">
          <motion.button
            type="button"
            onClick={onDeny}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.12, ease: EASE_OUT }}
            className="px-3 py-1.5 rounded-md text-[12px] text-ink-soft hover:text-ink hover:bg-surface-sunk transition-colors"
          >
            Keep planning
          </motion.button>
          <motion.button
            type="button"
            onClick={onApprove}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.12, ease: EASE_OUT }}
            className="px-3 py-1.5 rounded-md text-[12px] font-medium text-white transition-opacity"
            style={{ background: "var(--ink)" }}
          >
            Approve & run
          </motion.button>
        </div>
      )}
    </motion.aside>
  );
}

function TodoSidebar({ todos }: { todos: TodoItem[] }) {
  const done = todos.filter((t) => t.status === "completed").length;
  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 280, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.24, ease: EASE_OUT }}
      className="shrink-0 border-l border-line bg-canvas overflow-hidden flex flex-col min-h-0"
    >
      <div className="px-4 py-3 border-b border-line flex items-baseline justify-between shrink-0">
        <span className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-faint">
          To-do
        </span>
        <span className="text-[10px] font-mono text-ink-faint">
          {done}/{todos.length}
        </span>
      </div>
      <ul className="flex-1 overflow-y-auto scroll-quiet py-2 px-3 space-y-1">
        {todos.map((t, i) => (
          <TodoRow key={`${i}-${t.content}`} todo={t} />
        ))}
      </ul>
    </motion.aside>
  );
}

function TodoRow({ todo }: { todo: TodoItem }) {
  const display = todo.status === "in_progress" && todo.activeForm ? todo.activeForm : todo.content;
  return (
    <li className="flex items-start gap-2 px-1 py-1">
      <TodoStatusIcon status={todo.status} />
      <span
        className={`text-[12.5px] leading-snug ${
          todo.status === "completed"
            ? "text-ink-faint line-through"
            : todo.status === "in_progress"
            ? "text-ink font-medium"
            : "text-ink-soft"
        }`}
      >
        {display}
      </span>
    </li>
  );
}

function TodoStatusIcon({ status }: { status: TodoItem["status"] }) {
  if (status === "completed") {
    return (
      <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden className="mt-[3px] shrink-0" style={{ color: "var(--accent-ok)" }}>
        <circle cx="6.5" cy="6.5" r="5.5" stroke="currentColor" strokeWidth="1.2" />
        <path d="M3.8 6.7L5.6 8.5L9.2 4.9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (status === "in_progress") {
    return (
      <span className="mt-[5px] shrink-0 relative inline-block size-2">
        <motion.span
          className="absolute inset-0 rounded-full"
          style={{ background: "var(--accent-warm)" }}
          animate={{ scale: [0.7, 1, 0.7], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
        />
      </span>
    );
  }
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden className="mt-[3px] shrink-0 text-ink-faint">
      <circle cx="6.5" cy="6.5" r="5.5" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Sidebar — session list + create/delete
// ────────────────────────────────────────────────────────────────────────────

function Sidebar({
  baseUrl,
  sessions,
  sessionStates,
  acks,
  activeId,
  creating,
  onSelect,
  onCreate,
  onDelete,
}: {
  baseUrl: string;
  sessions: StoredSession[];
  sessionStates: AgentMux["sessions"];
  acks: AppAcks;
  activeId: string | null;
  creating: boolean;
  onSelect: (id: string) => void;
  onCreate: (cwd?: string) => void;
  onDelete: (id: string) => void;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);

  return (
    <aside className="w-[260px] shrink-0 border-r border-line bg-canvas flex flex-col">
      <div className="px-4 py-4 border-b border-line">
        <div className="flex items-baseline gap-2 mb-3">
          <span className="font-serif text-xl leading-none italic tracking-tight">
            blitzcode
          </span>
        </div>
        <motion.button
          type="button"
          onClick={() => setPickerOpen(true)}
          disabled={creating}
          whileTap={{ scale: 0.97 }}
          transition={{ duration: 0.12, ease: EASE_OUT }}
          className="w-full px-3 py-1.5 rounded-md text-[13px] font-medium text-white disabled:opacity-40 transition-opacity flex items-center justify-center gap-1.5"
          style={{ background: "var(--ink)" }}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden>
            <path d="M5 1V9M1 5H9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          {creating ? "Creating…" : "New session"}
        </motion.button>
        <AnimatePresence>
          {pickerOpen && (
            <Modal>
              <FolderPicker
                baseUrl={baseUrl}
                onPick={(p) => {
                  setPickerOpen(false);
                  onCreate(p);
                }}
                onCancel={() => setPickerOpen(false)}
                confirmLabel="Create session"
              />
            </Modal>
          )}
        </AnimatePresence>
      </div>

      <div className="flex-1 overflow-y-auto scroll-quiet py-2">
        {sessions.length === 0 ? (
          <div className="px-4 py-6 text-xs text-ink-faint">
            No sessions yet. Create one to start.
          </div>
        ) : (
          (() => {
            const groups: Record<SessionGroup, StoredSession[]> = {
              working: [],
              needs_input: [],
              idle: [],
            };
            for (const s of sessions) {
              groups[categorizeSession(sessionStates[s.id], acks.map[s.id])].push(s);
            }
            const sections: { key: SessionGroup; label: string; list: StoredSession[] }[] = [
              { key: "working", label: "Working", list: groups.working },
              { key: "needs_input", label: "Needs input", list: groups.needs_input },
              { key: "idle", label: "Idle", list: groups.idle },
            ];
            return (
              <div className="space-y-2">
                {sections.map((sec) =>
                  sec.list.length === 0 ? null : (
                    <SidebarSection key={sec.key} label={sec.label} count={sec.list.length}>
                      <ul className="space-y-px">
                        {sec.list.map((s) => (
                          <SessionRow
                            key={s.id}
                            session={s}
                            status={sessionStates[s.id]?.status ?? "idle"}
                            active={s.id === activeId}
                            onSelect={() => onSelect(s.id)}
                            onDelete={() => onDelete(s.id)}
                          />
                        ))}
                      </ul>
                    </SidebarSection>
                  )
                )}
              </div>
            );
          })()
        )}
      </div>
    </aside>
  );
}

function SidebarSection({
  label,
  count,
  children,
}: {
  label: string;
  count: number;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="px-4 pt-2 pb-1 flex items-baseline gap-2">
        <span className="text-[10px] uppercase tracking-[0.14em] font-mono text-ink-faint">
          {label}
        </span>
        <span className="text-[10px] font-mono text-ink-faint">{count}</span>
      </div>
      {children}
    </section>
  );
}

function SessionRow({
  session,
  status,
  active,
  onSelect,
  onDelete,
}: {
  session: StoredSession;
  status: SessionState["status"];
  active: boolean;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const [hover, setHover] = useState(false);
  return (
    <li
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className={`relative group mx-2 rounded-md transition-colors ${
        active ? "bg-surface-tinted" : "hover:bg-surface-sunk"
      }`}
    >
      <button
        type="button"
        onClick={onSelect}
        className="w-full text-left px-2.5 py-2 pr-9"
      >
        <div className="text-[13px] text-ink leading-tight truncate">
          {deriveLabel(session)}
        </div>
        <div className="flex items-center gap-1.5 mt-0.5">
          <SessionStatusDot status={status} hasTurns={!!session.sdk_session_id} />
          <div className="text-[10px] font-mono text-ink-faint truncate">
            {relativeTime(session.last_seen_at)}
          </div>
        </div>
      </button>
      {(hover || active) && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            if (confirm("Delete this session permanently?")) onDelete();
          }}
          className="absolute right-1.5 top-1/2 -translate-y-1/2 size-6 rounded flex items-center justify-center text-ink-faint hover:text-ink-soft hover:bg-canvas transition-colors"
          aria-label="Delete session"
          title="Delete session"
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
            <path
              d="M3 3L9 9M9 3L3 9"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
            />
          </svg>
        </button>
      )}
    </li>
  );
}

// Per-session activity indicator shown in the sidebar so cross-session work
// is visible even when the user is viewing a different session.
//   streaming                       → pulsing accent dot (live tokens)
//   awaiting_permission / question  → solid amber (needs your attention)
//   awaiting_hook                   → solid amber dim
//   error                           → solid red
//   idle + no turns yet             → small warn dot (existing affordance)
//   idle + has turns                → nothing
function SessionStatusDot({
  status,
  hasTurns,
}: {
  status: SessionState["status"];
  hasTurns: boolean;
}) {
  if (status === "streaming") {
    return (
      <motion.span
        className="inline-block size-1.5 rounded-full"
        style={{ background: "var(--accent-warm)" }}
        animate={{ opacity: [0.35, 1, 0.35], scale: [0.85, 1, 0.85] }}
        transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
        title="Streaming"
        aria-label="Streaming"
      />
    );
  }
  if (status === "awaiting_permission" || status === "awaiting_question") {
    return (
      <motion.span
        className="inline-block size-1.5 rounded-full"
        style={{ background: "var(--accent-warn)" }}
        animate={{ opacity: [0.55, 1, 0.55] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        title={status === "awaiting_permission" ? "Awaiting permission" : "Awaiting your answer"}
        aria-label="Needs your attention"
      />
    );
  }
  if (status === "awaiting_hook") {
    return (
      <span
        className="inline-block size-1 rounded-full opacity-60"
        style={{ background: "var(--accent-warn)" }}
        title="Awaiting hook"
        aria-hidden
      />
    );
  }
  if (status === "error") {
    return (
      <span
        className="inline-block size-1.5 rounded-full"
        style={{ background: "var(--accent-err)" }}
        title="Error"
        aria-label="Error"
      />
    );
  }
  if (!hasTurns) {
    return (
      <span
        className="inline-block size-1 rounded-full"
        style={{ background: "var(--accent-warn)" }}
        title="No turns yet"
        aria-hidden
      />
    );
  }
  return null;
}

function LoadingState() {
  return (
    <div className="flex flex-1 items-center justify-center">
      <div className="flex gap-1.5 items-center" style={{ height: "1.625em" }}>
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="size-1.5 rounded-full"
            style={{ background: "var(--ink-faint)" }}
            animate={{ opacity: [0.25, 1, 0.25] }}
            transition={{
              duration: 1.1,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.15,
            }}
          />
        ))}
      </div>
    </div>
  );
}

function NoActiveSession({
  onCreate,
  creating,
}: {
  onCreate: () => void;
  creating: boolean;
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
      <h1 className="font-serif text-4xl leading-[1.05] tracking-tight text-ink">
        Pick a session or start a new one.
      </h1>
      <p className="mt-3 text-ink-muted text-[15px] max-w-md leading-relaxed">
        Sessions persist across restarts. Each can run in a different working
        directory.
      </p>
      <motion.button
        type="button"
        onClick={onCreate}
        disabled={creating}
        whileTap={{ scale: 0.97 }}
        transition={{ duration: 0.12, ease: EASE_OUT }}
        className="mt-6 px-4 py-2 rounded-md text-sm font-medium text-white disabled:opacity-40"
        style={{ background: "var(--ink)" }}
      >
        {creating ? "Creating…" : "New session"}
      </motion.button>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Header
// ────────────────────────────────────────────────────────────────────────────

function Header({
  status,
  sessionId,
  permissionMode,
  onChangeMode,
}: {
  status: string;
  sessionId?: string;
  permissionMode?: string | null;
  onChangeMode?: (mode: string) => void;
}) {
  return (
    <header className="border-b border-line bg-canvas/80 backdrop-blur-sm">
      <div className="mx-auto w-full max-w-2xl px-6 h-14 flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          {sessionId && (
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
              {sessionId.slice(0, 8)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {onChangeMode && (
            <PermissionModeMenu mode={permissionMode ?? "default"} onChange={onChangeMode} />
          )}
          <StatusBadge status={status} />
        </div>
      </div>
    </header>
  );
}

const MODE_LABELS: Record<string, string> = {
  default: "default",
  acceptEdits: "auto-edit",
  plan: "plan",
  bypassPermissions: "yolo",
};

const MODE_DOT_COLOR: Record<string, string> = {
  default: "var(--ink-faint)",
  acceptEdits: "var(--accent-ok)",
  plan: "var(--accent-cool)",
  bypassPermissions: "var(--accent-err)",
};

function PermissionModeMenu({
  mode,
  onChange,
}: {
  mode: string;
  onChange: (mode: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const label = MODE_LABELS[mode] ?? mode;
  const dot = MODE_DOT_COLOR[mode] ?? "var(--ink-faint)";
  const isAttention = mode === "plan" || mode === "bypassPermissions";

  return (
    <div ref={ref} className="relative">
      <motion.button
        type="button"
        onClick={() => setOpen((v) => !v)}
        whileTap={{ scale: 0.96 }}
        transition={{ duration: 0.12, ease: EASE_OUT }}
        className={`flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.18em] px-2 py-0.5 rounded border transition-colors ${
          isAttention
            ? "border-transparent text-white"
            : "border-line text-ink-soft hover:border-line-strong"
        }`}
        style={isAttention ? { background: dot } : undefined}
        title="Permission mode"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {!isAttention && (
          <span className="inline-block size-1.5 rounded-full" style={{ background: dot }} aria-hidden />
        )}
        {label}
        <svg width="7" height="7" viewBox="0 0 7 7" fill="none" aria-hidden>
          <path d="M1 2L3.5 5L6 2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </motion.button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.14, ease: EASE_OUT }}
            className="absolute right-0 mt-1 w-[180px] z-50 rounded-md border border-line bg-canvas shadow-xl overflow-hidden"
            role="menu"
          >
            {(["default", "acceptEdits", "plan", "bypassPermissions"] as const).map((m) => (
              <button
                key={m}
                type="button"
                role="menuitem"
                onClick={() => { onChange(m); setOpen(false); }}
                className={`w-full text-left px-3 py-2 text-[12px] flex items-center gap-2 hover:bg-surface-sunk transition-colors ${
                  m === mode ? "bg-surface-tinted" : ""
                }`}
              >
                <span className="inline-block size-1.5 rounded-full shrink-0" style={{ background: MODE_DOT_COLOR[m] }} aria-hidden />
                <span className="flex-1 font-mono text-[10px] uppercase tracking-[0.14em] text-ink">{MODE_LABELS[m]}</span>
                {m === mode && (
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden className="text-ink-faint">
                    <path d="M2 5.5L4 7.5L8 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
              </button>
            ))}
            <div className="px-3 py-2 border-t border-line text-[10px] leading-relaxed text-ink-faint">
              {mode === "bypassPermissions" && "All tools auto-approved. Use with care."}
              {mode === "plan" && "Agent will plan first, won't execute until approved."}
              {mode === "acceptEdits" && "File edits auto-approved; other tools still prompt."}
              {mode === "default" && "All sensitive tools prompt for permission."}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; color: string; pulse: boolean }> = {
    idle: { label: "ready", color: "var(--accent-ok)", pulse: false },
    streaming: { label: "thinking", color: "var(--accent-warm)", pulse: true },
    awaiting_permission: { label: "needs you", color: "var(--accent-cool)", pulse: true },
    awaiting_question: { label: "needs you", color: "var(--accent-purple)", pulse: true },
    awaiting_hook: { label: "hook", color: "var(--accent-warn)", pulse: true },
    error: { label: "error", color: "var(--accent-err)", pulse: false },
  };
  const s = map[status] ?? { label: status, color: "var(--ink-faint)", pulse: false };
  return (
    <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-muted">
      <span
        className={`inline-block size-1.5 rounded-full ${s.pulse ? "dot-pulse" : ""}`}
        style={{ background: s.color }}
        aria-hidden
      />
      {s.label}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Empty state
// ────────────────────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="flex flex-col items-start py-16">
      <h1 className="font-serif text-5xl leading-[1.05] tracking-tight text-ink">
        Ask anything.
      </h1>
      <p className="mt-3 text-ink-muted text-[15px] max-w-md leading-relaxed">
        This is a working agent surface. Type below to start a turn. Tools and
        results group together as the agent works.
      </p>
      <div className="mt-8 flex flex-wrap gap-2">
        {[
          "Read package.json and summarize it",
          "List the files in this directory",
          "Write a haiku about debugging",
        ].map((s) => (
          <button
            key={s}
            type="button"
            className="text-xs px-3 py-1.5 rounded-md border border-line bg-surface text-ink-soft hover:border-line-strong transition-colors"
            onClick={() => {
              const ev = new CustomEvent("blitz:suggest", { detail: s });
              window.dispatchEvent(ev);
            }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// User message
// ────────────────────────────────────────────────────────────────────────────

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-br-md bg-surface-tinted text-ink text-[15px] leading-relaxed whitespace-pre-wrap">
        {content}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Assistant text
// ────────────────────────────────────────────────────────────────────────────

function AssistantText({ text, streaming }: { text: string; streaming: boolean }) {
  return (
    <div className="flex gap-3 items-start">
      <TurnMarker lineHeight="1.625em" />
      <div
        className={`flex-1 min-w-0 text-[15px] leading-relaxed text-ink prose prose-sm max-w-none
          prose-p:my-2 prose-p:leading-relaxed prose-p:text-ink
          prose-headings:font-serif prose-headings:tracking-tight prose-headings:text-ink
          prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg
          prose-strong:text-ink prose-strong:font-semibold
          prose-em:italic
          prose-a:text-[color:var(--accent-cool)] prose-a:no-underline hover:prose-a:underline
          prose-code:font-mono prose-code:text-[0.9em] prose-code:bg-[color:var(--surface-sunk)] prose-code:text-ink-soft prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none
          prose-pre:bg-[color:var(--surface-sunk)] prose-pre:text-ink-soft prose-pre:border prose-pre:border-[color:var(--line)] prose-pre:rounded-lg prose-pre:p-3.5 prose-pre:text-[12px]
          prose-blockquote:border-l-2 prose-blockquote:border-[color:var(--line-strong)] prose-blockquote:not-italic prose-blockquote:text-ink-soft
          prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5 prose-li:marker:text-ink-faint
          prose-hr:border-[color:var(--line)]
          prose-table:border-collapse prose-th:border prose-th:border-[color:var(--line)] prose-th:px-3 prose-th:py-1.5 prose-th:bg-[color:var(--surface-sunk)] prose-th:text-left
          prose-td:border prose-td:border-[color:var(--line)] prose-td:px-3 prose-td:py-1.5
        `}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        {streaming && <span className="stream-caret" aria-hidden />}
      </div>
    </div>
  );
}

// Vertically centers a small dot against the first line of body text. The
// 6.5px top offset accounts for the prose plugin's first-paragraph margin
// (prose-p:my-2 = 8px top, partially collapsed by the flex container) so the
// dot sits exactly on the visual midline of the first text line.
function TurnMarker({ lineHeight = "1.625em" }: { lineHeight?: string }) {
  return (
    <span
      aria-hidden
      className="shrink-0 flex items-center justify-center"
      style={{ height: lineHeight, width: "0.375rem", marginTop: "6.5px" }}
    >
      <span
        className="size-1.5 rounded-full block"
        style={{ background: "var(--accent-warm)" }}
      />
    </span>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Tool call (use + result paired)
// ────────────────────────────────────────────────────────────────────────────

function summarizeInput(name: string, input: unknown): string {
  if (!input || typeof input !== "object") return "";
  const obj = input as Record<string, unknown>;
  // Friendly one-liners for the most common tools
  if (typeof obj.command === "string") return String(obj.command);
  if (typeof obj.file_path === "string") return String(obj.file_path);
  if (typeof obj.path === "string") return String(obj.path);
  if (typeof obj.url === "string") return String(obj.url);
  if (typeof obj.pattern === "string") return String(obj.pattern);
  if (typeof obj.query === "string") return String(obj.query);
  // Fallback: first string value, truncated
  for (const v of Object.values(obj)) {
    if (typeof v === "string") return v.length > 80 ? v.slice(0, 80) + "…" : v;
  }
  return "";
}

function renderOutput(output: unknown): string {
  if (output == null) return "";
  if (typeof output === "string") return output;
  if (Array.isArray(output)) {
    // Anthropic tool_result content is often [{type:'text', text:'...'}]
    const parts = output
      .map((b: { text?: string }) => (typeof b?.text === "string" ? b.text : null))
      .filter(Boolean) as string[];
    if (parts.length) return parts.join("\n");
  }
  try {
    return JSON.stringify(output, null, 2);
  } catch {
    return String(output);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Tool-call grouping — collapses runs of consecutive tool calls into one
// summary row with a count per category. Click to expand → reveals the
// individual ToolCall cards (each still independently expandable).
// ────────────────────────────────────────────────────────────────────────────

type ToolCategory = "read" | "write" | "bash" | "search" | "web" | "task" | "other";

function categorize(name: string): ToolCategory {
  const lc = name.toLowerCase();
  if (lc === "read" || lc === "notebookread") return "read";
  if (lc === "write" || lc === "edit" || lc === "multiedit" || lc === "notebookedit") return "write";
  if (lc === "bash" || lc === "killbash" || lc === "bashoutput") return "bash";
  if (lc === "grep" || lc === "glob") return "search";
  if (lc.startsWith("webfetch") || lc.startsWith("websearch") || lc === "fetch") return "web";
  if (lc === "task" || lc === "todowrite") return "task";
  return "other";
}

function phraseFor(cat: ToolCategory, n: number): string {
  switch (cat) {
    case "read": return `Read ${n} file${n === 1 ? "" : "s"}`;
    case "write": return `Wrote ${n} file${n === 1 ? "" : "s"}`;
    case "bash": return `Ran ${n} command${n === 1 ? "" : "s"}`;
    case "search": return `Searched ${n} time${n === 1 ? "" : "s"}`;
    case "web": return `Made ${n} web request${n === 1 ? "" : "s"}`;
    case "task": return `Planned ${n} task${n === 1 ? "" : "s"}`;
    case "other": return `Ran ${n} tool${n === 1 ? "" : "s"}`;
  }
}

function summarizeGroup(items: ToolCallItem[]): string {
  // Preserve first-seen order so the summary reads in the order they ran:
  // "Ran 2 commands · Read 3 files" matches the visual stack below it.
  const order: ToolCategory[] = [];
  const counts: Record<string, number> = {};
  for (const it of items) {
    const c = categorize(it.name);
    if (counts[c] === undefined) order.push(c);
    counts[c] = (counts[c] ?? 0) + 1;
  }
  return order.map((c) => phraseFor(c, counts[c]!)).join(" · ");
}

function ToolGroup({ group }: { group: ToolGroupItem }) {
  const [open, setOpen] = useState(false);
  const summary = useMemo(() => summarizeGroup(group.items), [group.items]);
  // Bubble status: error > running > done (so the row dot reflects the
  // worst-case state of the contained calls).
  const anyError = group.items.some((i) => i.status === "error");
  const anyRunning = group.items.some((i) => i.status === "running");
  const tone = {
    color: anyError ? "var(--accent-err)" : anyRunning ? "var(--accent-warn)" : "var(--accent-ok)",
    pulse: anyRunning && !anyError,
  };

  return (
    // items-stretch so the gutter column grows to the full group height —
    // that's what lets the vertical line span the expanded body.
    <div className="flex gap-3 items-stretch">
      <div
        className="shrink-0 flex flex-col items-center"
        style={{ width: "0.375rem" }}
        aria-hidden
      >
        {/* Marker dot, aligned to the trigger row's visual center. Tiny
            top nudge compensates for the button text's optical center
            sitting slightly below the geometric center of its line box. */}
        <span
          className="shrink-0 flex items-center justify-center"
          style={{ height: "36px", width: "0.375rem", marginTop: "2px" }}
        >
          <span
            className="size-1.5 rounded-full block"
            style={{ background: "var(--ink-faint)" }}
          />
        </span>
        {/* When expanded, a single hairline replaces the per-card dots —
            anchored under the marker, running through every nested card. */}
        <AnimatePresence>
          {open && (
            <motion.div
              key="line"
              initial={{ opacity: 0, scaleY: 0 }}
              animate={{ opacity: 1, scaleY: 1 }}
              exit={{ opacity: 0, scaleY: 0 }}
              transition={{ duration: 0.22, ease: EASE_OUT }}
              className="flex-1 w-px origin-top"
              style={{ background: "var(--line)" }}
            />
          )}
        </AnimatePresence>
      </div>
      <div className="flex-1 min-w-0">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center gap-3 rounded-md px-3 py-2 text-left hover:bg-surface-sunk transition-colors"
          aria-expanded={open}
        >
          <span
            className={`inline-block size-1.5 rounded-full shrink-0 ${tone.pulse ? "dot-pulse" : ""}`}
            style={{ background: tone.color }}
            aria-hidden
          />
          <span className="text-[13px] text-ink-soft truncate">{summary}</span>
          <span className="ml-auto flex items-center gap-2 shrink-0">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
              {group.items.length} call{group.items.length === 1 ? "" : "s"}
            </span>
            <motion.span
              animate={{ rotate: open ? 90 : 0 }}
              transition={{ duration: 0.18, ease: EASE_OUT }}
              className="text-ink-faint"
              aria-hidden
            >
              ›
            </motion.span>
          </span>
        </button>

        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              key="children"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: EASE_OUT }}
              className="overflow-hidden"
            >
              <ol className="space-y-3 pt-2 pb-1">
                {group.items.map((it) => (
                  <li key={it.id}>
                    <ToolCall item={it} bare />
                  </li>
                ))}
              </ol>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function ToolCall({ item, bare = false }: { item: ToolCallItem; bare?: boolean }) {
  const [open, setOpen] = useState(false);
  const preview = summarizeInput(item.name, item.input);

  const tone = {
    running: { ring: "var(--line)", dot: "var(--accent-warn)", label: "running", pulse: true },
    done: { ring: "var(--line)", dot: "var(--accent-ok)", label: "done", pulse: false },
    error: { ring: "var(--accent-err)", dot: "var(--accent-err)", label: "error", pulse: false },
  }[item.status];

  // The card itself — same in both modes.
  const card = (
    <div
      className="flex-1 min-w-0 rounded-lg border bg-surface overflow-hidden transition-colors"
      style={{ borderColor: tone.ring }}
    >
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center gap-3 px-3.5 py-2.5 text-left hover:bg-surface-sunk transition-colors"
          aria-expanded={open}
        >
          <span
            className={`inline-block size-1.5 rounded-full shrink-0 ${tone.pulse ? "dot-pulse" : ""}`}
            style={{ background: tone.dot }}
            aria-hidden
          />
          <span className="font-mono text-[12px] text-ink shrink-0">{item.name}</span>
          {preview && (
            <span className="font-mono text-[12px] text-ink-muted truncate">{preview}</span>
          )}
          <span className="ml-auto flex items-center gap-2 shrink-0">
            <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
              {tone.label}
            </span>
            <motion.span
              animate={{ rotate: open ? 90 : 0 }}
              transition={{ duration: 0.18, ease: EASE_OUT }}
              className="text-ink-faint"
              aria-hidden
            >
              ›
            </motion.span>
          </span>
        </button>

        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              key="body"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: EASE_OUT }}
              className="overflow-hidden"
            >
              <div className="border-t border-line">
                <Section label="Input">
                  <CodeBlock>{JSON.stringify(item.input, null, 2)}</CodeBlock>
                </Section>
                {item.result ? (
                  <Section
                    label={item.result.is_error ? "Error" : "Output"}
                    tone={item.result.is_error ? "err" : undefined}
                  >
                    <CodeBlock tone={item.result.is_error ? "err" : "ok"}>
                      {renderOutput(item.result.output)}
                    </CodeBlock>
                  </Section>
                ) : (
                  <Section label="Output">
                    <div className="px-3.5 py-3 text-ink-muted text-xs font-mono flex items-center gap-2">
                      <span
                        className="inline-block size-1.5 rounded-full dot-pulse"
                        style={{ background: "var(--accent-warn)" }}
                        aria-hidden
                      />
                      waiting for result…
                    </div>
                  </Section>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
    </div>
  );

  // Inside a ToolGroup the gutter is owned by the group's vertical line —
  // skip our own dot + flex wrapper.
  if (bare) return card;
  return (
    <div className="flex gap-3 items-start">
      <span
        aria-hidden
        className="shrink-0 flex items-center justify-center"
        style={{ height: "44px", width: "0.375rem" }}
      >
        <span
          className="size-1.5 rounded-full block"
          style={{ background: "var(--ink-faint)" }}
        />
      </span>
      {card}
    </div>
  );
}

function Section({
  label,
  tone,
  children,
}: {
  label: string;
  tone?: "ok" | "err";
  children: React.ReactNode;
}) {
  const color =
    tone === "err"
      ? "var(--accent-err)"
      : tone === "ok"
        ? "var(--accent-ok)"
        : "var(--ink-faint)";
  return (
    <div>
      <div
        className="px-3.5 pt-2.5 pb-1 font-mono text-[10px] uppercase tracking-[0.18em]"
        style={{ color }}
      >
        {label}
      </div>
      {children}
    </div>
  );
}

function CodeBlock({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone?: "ok" | "err";
}) {
  const bg =
    tone === "err"
      ? "var(--tint-err)"
      : tone === "ok"
        ? "var(--surface-sunk)"
        : "var(--surface-sunk)";
  const fg = tone === "err" ? "var(--accent-err)" : "var(--ink-soft)";
  return (
    <pre
      className="px-3.5 py-3 m-0 font-mono text-[12px] leading-relaxed overflow-x-auto whitespace-pre-wrap break-words"
      style={{ background: bg, color: fg }}
    >
      {children}
    </pre>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Thinking indicator (between user message and first assistant token)
// ────────────────────────────────────────────────────────────────────────────

function ThinkingIndicator() {
  return (
    <div className="flex gap-1.5 items-center" style={{ height: "1.625em" }}>
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="size-1.5 rounded-full"
          style={{ background: "var(--ink-faint)" }}
          animate={{ opacity: [0.25, 1, 0.25] }}
          transition={{
            duration: 1.1,
            repeat: Infinity,
            ease: "easeInOut",
            delay: i * 0.15,
          }}
        />
      ))}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Modal shell (used for permission / question / error)
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
// Folder picker — Finder-style column view. Each click adds a column to the
// right with that folder's contents; columns truncate when you re-select
// shallower. Fixed modal dimensions; columns scroll vertically, rail scrolls
// horizontally. The deepest selected folder is the pick.
// ────────────────────────────────────────────────────────────────────────────

interface FsListing {
  path: string;
  parent: string | null;
  home: string;
  entries: { name: string }[];
}

interface PickerColumn {
  /** Stable id so async resolves can patch the right column even if the
   *  server-resolved path differs from what we sent (symlinks, normalization). */
  id: number;
  path: string;
  loading: boolean;
  err: string | null;
  entries: { name: string }[];
  /** Which entry name in THIS column is currently selected (drives the next column). */
  selected: string | null;
}

let _pickerColId = 0;
const nextPickerColId = () => ++_pickerColId;

const joinPath = (base: string, name: string): string =>
  base === "/" ? `/${name}` : `${base}/${name}`;

function FolderPicker({
  baseUrl,
  initialPath,
  onPick,
  onCancel,
  confirmLabel = "Use this folder",
}: {
  baseUrl: string;
  initialPath?: string | null;
  onPick: (path: string) => void;
  onCancel: () => void;
  confirmLabel?: string;
}) {
  const [columns, setColumns] = useState<PickerColumn[]>([]);
  const [homePath, setHomePath] = useState<string | null>(null);
  const railRef = useRef<HTMLDivElement | null>(null);

  const fetchListing = useCallback(
    async (path?: string): Promise<FsListing> => {
      const url = new URL(`${baseUrl}/app/fs/list`);
      if (path) url.searchParams.set("path", path);
      const r = await fetch(url.toString());
      if (!r.ok) {
        const text = await r.text().catch(() => "");
        throw new Error(text || `HTTP ${r.status}`);
      }
      return (await r.json()) as FsListing;
    },
    [baseUrl]
  );

  // Root column on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const root = await fetchListing(initialPath ?? undefined);
        if (cancelled) return;
        setHomePath(root.home);
        setColumns([
          { id: nextPickerColId(), path: root.path, loading: false, err: null, entries: root.entries, selected: null },
        ]);
      } catch (e) {
        if (cancelled) return;
        setColumns([
          {
            id: nextPickerColId(),
            path: initialPath ?? "/",
            loading: false,
            err: e instanceof Error ? e.message : String(e),
            entries: [],
            selected: null,
          },
        ]);
      }
    })();
    return () => { cancelled = true; };
  }, [fetchListing, initialPath]);

  // Auto-scroll the rail to keep the rightmost column visible.
  useEffect(() => {
    const el = railRef.current;
    if (!el) return;
    el.scrollTo({ left: el.scrollWidth, behavior: "smooth" });
  }, [columns.length]);

  const openChild = useCallback(
    async (colIdx: number, name: string) => {
      const parent = columns[colIdx];
      if (!parent) return;
      const childPath = joinPath(parent.path, name);
      const newColId = nextPickerColId();
      // Truncate to colIdx + 1, mark selection, append a loading column.
      setColumns((prev) => {
        const trimmed = prev.slice(0, colIdx + 1);
        trimmed[colIdx] = { ...trimmed[colIdx]!, selected: name };
        return [
          ...trimmed,
          { id: newColId, path: childPath, loading: true, err: null, entries: [], selected: null },
        ];
      });
      try {
        const data = await fetchListing(childPath);
        setColumns((prev) => {
          const next = [...prev];
          const target = next.findIndex((c) => c.id === newColId);
          if (target === -1) return prev;
          next[target] = { id: newColId, path: data.path, loading: false, err: null, entries: data.entries, selected: null };
          return next;
        });
      } catch (e) {
        setColumns((prev) => {
          const next = [...prev];
          const target = next.findIndex((c) => c.id === newColId);
          if (target === -1) return prev;
          next[target] = { ...next[target]!, loading: false, err: e instanceof Error ? e.message : String(e) };
          return next;
        });
      }
    },
    [columns, fetchListing]
  );

  const goHome = useCallback(async () => {
    if (!homePath) return;
    try {
      const root = await fetchListing(homePath);
      setColumns([{ id: nextPickerColId(), path: root.path, loading: false, err: null, entries: root.entries, selected: null }]);
    } catch {
      /* ignore */
    }
  }, [homePath, fetchListing]);

  // The picked path is the deepest column's path (since clicking always
  // opens the child as a new column, the rightmost column's `path` is
  // exactly "what the user is currently inside").
  const pickedPath = columns.length > 0 ? columns[columns.length - 1]!.path : null;

  return (
    <div
      className="bg-canvas border border-line rounded-xl shadow-xl overflow-hidden flex flex-col"
      style={{ width: "min(820px, 92vw)", height: "min(520px, 80vh)" }}
    >
      <div className="px-4 py-3 border-b border-line flex items-center justify-between gap-3 shrink-0">
        <div className="text-[13px] font-medium text-ink">Choose a folder</div>
        <button
          type="button"
          onClick={goHome}
          disabled={!homePath}
          className="text-[11px] px-2 py-0.5 rounded border border-line text-ink-soft hover:text-ink hover:bg-surface-sunk transition-colors disabled:opacity-40"
          title="Home"
        >
          ~
        </button>
      </div>

      <div
        ref={railRef}
        className="flex-1 min-h-0 flex overflow-x-auto overflow-y-hidden scroll-quiet"
      >
        {columns.map((col, idx) => (
          <PickerColumnView
            key={col.id}
            col={col}
            onSelect={(name) => void openChild(idx, name)}
          />
        ))}
      </div>

      <div className="px-4 py-3 border-t border-line flex items-center justify-end gap-2 bg-surface shrink-0">
        <button
          type="button"
          onClick={onCancel}
          className="px-3 py-1.5 rounded-md text-[12px] text-ink-soft hover:text-ink hover:bg-surface-sunk transition-colors"
        >
          Cancel
        </button>
        <motion.button
          type="button"
          disabled={!pickedPath}
          onClick={() => pickedPath && onPick(pickedPath)}
          whileTap={{ scale: 0.97 }}
          transition={{ duration: 0.12, ease: EASE_OUT }}
          className="px-3 py-1.5 rounded-md text-[12px] font-medium text-white disabled:opacity-40 transition-opacity"
          style={{ background: "var(--ink)" }}
        >
          {confirmLabel}
        </motion.button>
      </div>
    </div>
  );
}

function PickerColumnView({
  col,
  onSelect,
}: {
  col: PickerColumn;
  onSelect: (name: string) => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.18, ease: EASE_OUT }}
      className="w-[220px] shrink-0 h-full overflow-y-auto scroll-quiet border-r border-line bg-canvas"
    >
      {col.err ? (
        <div className="px-3 py-4 text-[11px] font-mono text-ink-muted">{col.err}</div>
      ) : col.loading ? (
        <div className="px-3 py-4 text-[11px] text-ink-faint">Loading…</div>
      ) : col.entries.length === 0 ? (
        <div className="px-3 py-4 text-[11px] text-ink-faint">Empty</div>
      ) : (
        <ul className="py-1">
          {col.entries.map((e) => {
            const isSelected = col.selected === e.name;
            return (
              <li key={e.name}>
                <button
                  type="button"
                  onClick={() => onSelect(e.name)}
                  className={`w-full text-left px-3 py-1.5 text-[13px] transition-colors flex items-center gap-2 ${
                    isSelected
                      ? "bg-surface-tinted text-ink"
                      : "text-ink hover:bg-surface-sunk"
                  }`}
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden className="shrink-0 text-ink-faint">
                    <path d="M1 3.5C1 2.67 1.67 2 2.5 2H4.5L5.5 3H9.5C10.33 3 11 3.67 11 4.5V8.5C11 9.33 10.33 10 9.5 10H2.5C1.67 10 1 9.33 1 8.5V3.5Z" stroke="currentColor" strokeWidth="1" />
                  </svg>
                  <span className="truncate flex-1">{e.name}</span>
                  <span className="text-ink-faint text-[10px]">›</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </motion.div>
  );
}

function Modal({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      className="fixed inset-0 z-40 flex items-end sm:items-center justify-center p-4 sm:p-8 bg-black/8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18, ease: EASE_OUT }}
    >
      <motion.div
        initial={{ opacity: 0, y: 12, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 8, scale: 0.98 }}
        transition={{ duration: 0.26, ease: EASE_OUT }}
        className="w-full max-w-lg origin-bottom sm:origin-center"
      >
        {children}
      </motion.div>
    </motion.div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Permission card
// ────────────────────────────────────────────────────────────────────────────

// Matches the SDK's PermissionUpdate.to_dict() control-protocol shape —
// rule keys are camelCase (toolName / ruleContent) so they round-trip cleanly
// through PermissionUpdate.from_dict on the server when we send the suggestion
// back as updated_permissions.
type Suggestion = {
  type?: string;
  mode?: string | null;
  behavior?: string | null;
  rules?: Array<{ toolName?: string; ruleContent?: string | null }> | null;
  directories?: string[] | null;
  destination?: string | null;
};

function summarizeSuggestion(s: Suggestion): string {
  if (s.type === "setMode" && s.mode) return `Switch mode → ${s.mode}`;
  if (s.type === "addRules" && s.rules?.length) {
    const rule = s.rules[0]!;
    const target = rule.ruleContent ? ` ${rule.ruleContent}` : "";
    const scope = s.destination ? ` · ${s.destination}` : "";
    const verb = s.behavior === "deny" ? "Always deny" : "Always allow";
    return `${verb} ${rule.toolName ?? "tool"}${target}${scope}`;
  }
  if (s.type === "addDirectories" && s.directories?.length) {
    return `Add dir: ${s.directories.join(", ")}`;
  }
  return s.type ?? "Apply";
}

// ────────────────────────────────────────────────────────────────────────────
// Tool-input renderers — turn the raw can_use_tool input dict into a
// human-readable preview when we know the tool's shape. Each renderer
// validates the shape it expects; on mismatch it returns null and the
// caller falls back to the generic JSON dump. The model may produce
// custom variants for the same tool, so never trust the shape blindly.
// ────────────────────────────────────────────────────────────────────────────

function isStr(v: unknown): v is string {
  return typeof v === "string";
}

function renderToolInput(toolName: string, raw: unknown): React.ReactNode | null {
  if (!raw || typeof raw !== "object") return null;
  const input = raw as Record<string, unknown>;
  switch (toolName) {
    case "Bash":
      return renderBashInput(input);
    case "Read":
      return renderReadInput(input);
    case "Write":
      return renderWriteInput(input);
    case "Edit":
      return renderEditInput(input);
    case "Grep":
      return renderGrepInput(input);
    default:
      return null;
  }
}

function ToolInputFrame({
  meta,
  body,
  bodyTone,
}: {
  meta?: React.ReactNode;
  body: React.ReactNode;
  bodyTone?: "ok" | "err";
}) {
  return (
    <div>
      {meta && (
        <div className="px-5 pt-3 pb-2 text-[11px] text-ink-muted flex flex-wrap items-center gap-x-3 gap-y-1">
          {meta}
        </div>
      )}
      <CodeBlock tone={bodyTone}>{body}</CodeBlock>
    </div>
  );
}

function renderBashInput(input: Record<string, unknown>): React.ReactNode | null {
  if (!isStr(input.command)) return null;
  const description = isStr(input.description) ? input.description : null;
  const bg = input.run_in_background === true;
  const timeout = typeof input.timeout === "number" ? input.timeout : null;
  return (
    <ToolInputFrame
      meta={
        <>
          {description && <span className="text-ink-soft">{description}</span>}
          {bg && <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">bg</span>}
          {timeout !== null && <span className="font-mono text-[10px] text-ink-faint">timeout {timeout}ms</span>}
        </>
      }
      body={
        <>
          <span style={{ color: "var(--ink-faint)" }}>$ </span>
          {input.command}
        </>
      }
    />
  );
}

function renderReadInput(input: Record<string, unknown>): React.ReactNode | null {
  if (!isStr(input.file_path)) return null;
  const offset = typeof input.offset === "number" ? input.offset : null;
  const limit = typeof input.limit === "number" ? input.limit : null;
  const range =
    offset !== null && limit !== null
      ? `lines ${offset}–${offset + limit}`
      : offset !== null
      ? `from line ${offset}`
      : limit !== null
      ? `first ${limit} lines`
      : "full file";
  return (
    <ToolInputFrame
      meta={<span className="font-mono text-ink-soft">{range}</span>}
      body={input.file_path}
    />
  );
}

function renderWriteInput(input: Record<string, unknown>): React.ReactNode | null {
  if (!isStr(input.file_path) || !isStr(input.content)) return null;
  const lineCount = input.content.split("\n").length;
  const charCount = input.content.length;
  return (
    <ToolInputFrame
      meta={
        <>
          <span className="font-mono text-ink">{input.file_path}</span>
          <span className="font-mono text-[10px] text-ink-faint">
            {lineCount} line{lineCount === 1 ? "" : "s"} · {charCount} chars
          </span>
        </>
      }
      body={input.content}
    />
  );
}

function renderEditInput(input: Record<string, unknown>): React.ReactNode | null {
  if (!isStr(input.file_path) || !isStr(input.old_string) || !isStr(input.new_string)) return null;
  const replaceAll = input.replace_all === true;
  return (
    <div>
      <div className="px-5 pt-3 pb-2 text-[11px] text-ink-muted flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="font-mono text-ink">{input.file_path}</span>
        {replaceAll && <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">replace all</span>}
      </div>
      <div className="border-t border-line">
        <div className="px-5 py-1.5 text-[10px] font-mono uppercase tracking-[0.14em] text-ink-faint bg-surface-sunk">− removed</div>
        <CodeBlock tone="err">{input.old_string}</CodeBlock>
      </div>
      <div className="border-t border-line">
        <div className="px-5 py-1.5 text-[10px] font-mono uppercase tracking-[0.14em] text-ink-faint bg-surface-sunk">+ added</div>
        <CodeBlock tone="ok">{input.new_string}</CodeBlock>
      </div>
    </div>
  );
}

function renderGrepInput(input: Record<string, unknown>): React.ReactNode | null {
  if (!isStr(input.pattern)) return null;
  const path = isStr(input.path) ? input.path : null;
  const glob = isStr(input.glob) ? input.glob : null;
  const type = isStr(input.type) ? input.type : null;
  return (
    <ToolInputFrame
      meta={
        <>
          {path && <span className="font-mono text-ink-soft">{path}</span>}
          {glob && <span className="font-mono text-[10px] text-ink-faint">glob {glob}</span>}
          {type && <span className="font-mono text-[10px] text-ink-faint">type {type}</span>}
        </>
      }
      body={input.pattern}
    />
  );
}

function PermissionCard({
  req,
  onApprove,
  onDeny,
}: {
  req: {
    tool_name: string;
    input: unknown;
    context?: { suggestions?: Suggestion[] };
  };
  onApprove: (updatedPermissions?: unknown[]) => void;
  onDeny: () => void;
}) {
  const suggestions = req.context?.suggestions ?? [];
  return (
    <div
      className="rounded-xl border bg-surface overflow-hidden flex flex-col"
      style={{ borderColor: "var(--accent-cool)", maxHeight: "min(720px, 85vh)" }}
    >
      {/* Scrollable region: header + body + suggestions. Action buttons stay
          pinned at the bottom so they remain reachable when tool input is
          huge (e.g. a big multi-paragraph Write). */}
      <div className="flex-1 min-h-0 overflow-y-auto scroll-quiet">
        <div className="px-5 pt-5 pb-4">
          <div className="flex items-baseline gap-2 mb-1">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em]"
                  style={{ color: "var(--accent-cool)" }}>
              Permission
            </span>
          </div>
          <h2 className="font-serif text-2xl leading-tight tracking-tight text-ink">
            Allow{" "}
            <span className="font-mono text-[0.85em] tracking-tight">{req.tool_name}</span>
            ?
          </h2>
          <p className="mt-1 text-sm text-ink-muted">
            The agent wants to run this tool with the following input.
          </p>
        </div>

        <div className="border-t border-line">
          {renderToolInput(req.tool_name, req.input) ?? (
            <CodeBlock>{JSON.stringify(req.input, null, 2)}</CodeBlock>
          )}
        </div>

        {suggestions.length > 0 && (
          <div className="px-5 py-4 border-t border-line">
            <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint mb-2">
              Or apply a rule
            </div>
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => onApprove([s])}
                  title={JSON.stringify(s)}
                  className="text-xs px-2.5 py-1.5 rounded-md border border-line bg-surface hover:bg-surface-sunk hover:border-line-strong text-ink-soft transition-colors"
                >
                  {summarizeSuggestion(s)}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="px-5 py-4 flex flex-wrap gap-2 border-t border-line bg-canvas shrink-0">
        <PrimaryButton onClick={() => onApprove()} tone="ok">
          Allow once
        </PrimaryButton>
        <GhostButton onClick={onDeny} tone="err">
          Deny
        </GhostButton>
      </div>

      <KbdHint hints={[["enter", "allow once"], ["esc", "deny"]]} />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Question card (AskUserQuestion)
// ────────────────────────────────────────────────────────────────────────────

type QuestionShape = {
  question: string;
  header?: string;
  multiSelect?: boolean;
  options: { label: string; description?: string }[];
};

function QuestionCard({
  req,
  onAnswer,
}: {
  req: { questions: { questions: QuestionShape[] } };
  onAnswer: (answers: unknown) => void;
}) {
  const questions = req.questions?.questions ?? [];
  const [picks, setPicks] = useState<Record<number, string[]>>({});
  const [custom, setCustom] = useState<Record<number, string>>({});

  const togglePick = (qIdx: number, label: string, multi: boolean) => {
    setPicks((prev) => {
      const cur = prev[qIdx] ?? [];
      if (!multi) return { ...prev, [qIdx]: [label] };
      const next = cur.includes(label) ? cur.filter((l) => l !== label) : [...cur, label];
      return { ...prev, [qIdx]: next };
    });
    // Single-select: typing into "custom" deselects pre-defined; clicking a
    // pre-defined clears custom. Multi-select: both can coexist independently.
    if (!multi) {
      setCustom((prev) => ({ ...prev, [qIdx]: "" }));
    }
  };

  const setCustomText = (qIdx: number, value: string, multi: boolean) => {
    setCustom((prev) => ({ ...prev, [qIdx]: value }));
    if (!multi && value.trim()) {
      // Single-select: typing makes custom the active answer.
      setPicks((prev) => ({ ...prev, [qIdx]: [] }));
    }
  };

  // Effective answer per question.
  const answerFor = (i: number, q: QuestionShape): string => {
    const text = (custom[i] ?? "").trim();
    if (q.multiSelect) {
      const list = [...(picks[i] ?? [])];
      if (text) list.push(text);
      return list.join(", ");
    }
    if (text) return text;
    return picks[i]?.[0] ?? "";
  };

  const canSubmit = questions.every((q, i) => answerFor(i, q).length > 0);

  const submit = () => {
    if (!canSubmit) return;
    // Schema: answers = Record<questionText, answerString>.
    // Multi-select answers are comma-separated.
    const answers: Record<string, string> = {};
    questions.forEach((q, i) => {
      answers[q.question] = answerFor(i, q);
    });
    onAnswer(answers);
  };

  return (
    <div className="rounded-xl border bg-surface overflow-hidden"
         style={{ borderColor: "var(--accent-purple)" }}>
      <div className="px-5 pt-5 pb-3">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] mb-1"
             style={{ color: "var(--accent-purple)" }}>
          Question{questions.length > 1 ? `s · ${questions.length}` : ""}
        </div>
        <h2 className="font-serif text-2xl leading-tight tracking-tight text-ink">
          The agent needs an answer.
        </h2>
      </div>

      <div className="border-t border-line max-h-[55vh] overflow-y-auto scroll-quiet">
        {questions.map((q, i) => {
          const multi = Boolean(q.multiSelect);
          const text = custom[i] ?? "";
          // Single-select: custom is active when text is non-empty OR no
          // predefined option has been picked yet (default selection).
          const customActive = multi
            ? text.trim().length > 0
            : text.trim().length > 0 || (picks[i]?.length ?? 0) === 0;

          return (
            <div key={i} className={i > 0 ? "border-t border-line" : ""}>
              <div className="px-5 py-4">
                {q.header && (
                  <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint mb-1">
                    {q.header}
                  </div>
                )}
                <p className="text-[15px] text-ink leading-relaxed mb-3">
                  {q.question}
                  {multi && (
                    <span className="ml-2 text-[10px] uppercase tracking-[0.18em] text-ink-faint">
                      pick multiple
                    </span>
                  )}
                </p>
                <div className="space-y-1.5">
                  {q.options.map((opt) => {
                    const selected = (picks[i] ?? []).includes(opt.label);
                    return (
                      <button
                        key={opt.label}
                        type="button"
                        onClick={() => togglePick(i, opt.label, multi)}
                        className="w-full text-left px-3 py-2.5 rounded-md border transition-all"
                        style={{
                          borderColor: selected ? "var(--accent-purple)" : "var(--line)",
                          background: selected ? "var(--tint-purple)" : "var(--surface)",
                        }}
                      >
                        <div className="flex items-center gap-2.5">
                          <SelectionDot selected={selected} multi={multi} />
                          <div className="min-w-0">
                            <div className="text-[14px] text-ink leading-tight">
                              {opt.label}
                            </div>
                            {opt.description && (
                              <div className="text-[12px] text-ink-muted mt-0.5 leading-snug">
                                {opt.description}
                              </div>
                            )}
                          </div>
                        </div>
                      </button>
                    );
                  })}

                  {/* Custom answer row */}
                  <div
                    className="rounded-md border transition-all"
                    style={{
                      borderColor: customActive ? "var(--accent-purple)" : "var(--line)",
                      background: customActive ? "var(--tint-purple)" : "var(--surface)",
                    }}
                  >
                    <label className="flex items-center gap-2.5 px-3 py-2.5">
                      <SelectionDot selected={customActive} multi={multi} />
                      <input
                        type="text"
                        value={text}
                        onChange={(e) => setCustomText(i, e.target.value, multi)}
                        placeholder={
                          multi ? "Or add your own answer" : "Or write your own answer"
                        }
                        className="flex-1 min-w-0 bg-transparent text-[14px] text-ink placeholder:text-ink-faint focus:outline-none"
                      />
                    </label>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="border-t border-line px-5 py-4 flex items-center justify-between">
        <div className="text-xs text-ink-muted">
          {canSubmit ? "Ready to submit." : "Answer every question to continue."}
        </div>
        <PrimaryButton onClick={submit} tone="ok" disabled={!canSubmit}>
          Submit
        </PrimaryButton>
      </div>
    </div>
  );
}

function SelectionDot({ selected, multi }: { selected: boolean; multi: boolean }) {
  // Radio for single-select, checkbox for multi-select. Both use the same
  // selected/unselected color treatment.
  if (multi) {
    return (
      <span
        className="inline-flex items-center justify-center size-3.5 rounded-[3px] border shrink-0 transition-colors"
        style={{
          borderColor: selected ? "var(--accent-purple)" : "var(--line-strong)",
          background: selected ? "var(--accent-purple)" : "transparent",
        }}
        aria-hidden
      >
        {selected && (
          <svg width="9" height="9" viewBox="0 0 9 9" fill="none" aria-hidden>
            <path
              d="M1.5 4.5L3.5 6.5L7.5 2.5"
              stroke="white"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        )}
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center justify-center size-3.5 rounded-full border shrink-0 transition-colors"
      style={{
        borderColor: selected ? "var(--accent-purple)" : "var(--line-strong)",
        background: selected ? "var(--accent-purple)" : "transparent",
      }}
      aria-hidden
    >
      {selected && (
        <span className="size-1.5 rounded-full bg-white" />
      )}
    </span>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Error card (transient banner-as-modal)
// ────────────────────────────────────────────────────────────────────────────

function ErrorCard({
  code,
  message,
  onRestart,
}: {
  code: string;
  message: string;
  onRestart: () => void;
}) {
  // Map known L2 error codes to human-readable framing. Falls through to the
  // raw message for anything we don't recognize.
  const ui = (() => {
    switch (code) {
      case "session_not_found":
        return {
          label: "Session expired",
          title: "This session has ended.",
          body:
            "The server reaped this session after a period of inactivity. Your chat history is still visible, but the agent's context is gone. Start a new session to continue.",
          cta: "Start new session",
        };
      case "session_evicted":
        return {
          label: "Stream lost",
          title: "Couldn't pick up where we left off.",
          body:
            "Too many events accumulated while disconnected for the server to replay. The session is still alive but in-flight state is unreliable — best to start fresh.",
          cta: "Start new session",
        };
      case "unauthorized":
        return {
          label: "Unauthorized",
          title: "The server rejected the request.",
          body: message,
          cta: "Reload",
        };
      default:
        return {
          label: "Error",
          title: "Something went wrong.",
          body: message,
          cta: "Reload",
        };
    }
  })();

  return (
    <div
      className="rounded-xl border bg-surface overflow-hidden"
      style={{ borderColor: "var(--accent-err)" }}
    >
      <div className="px-5 pt-5 pb-3">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-1"
          style={{ color: "var(--accent-err)" }}
        >
          {ui.label}
        </div>
        <h2 className="font-serif text-2xl leading-tight tracking-tight text-ink">
          {ui.title}
        </h2>
        <p className="mt-2 text-[14px] text-ink-soft leading-relaxed">{ui.body}</p>
      </div>
      <div className="border-t border-line px-5 py-4 flex items-center justify-end gap-2">
        <PrimaryButton onClick={onRestart}>{ui.cta}</PrimaryButton>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Buttons
// ────────────────────────────────────────────────────────────────────────────

function PrimaryButton({
  children,
  onClick,
  tone = "ok",
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  tone?: "ok" | "warm" | "cool";
  disabled?: boolean;
}) {
  const bg = tone === "warm" ? "var(--accent-warm)" : tone === "cool" ? "var(--accent-cool)" : "var(--ink)";
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={disabled}
      whileTap={{ scale: 0.97 }}
      transition={{ duration: 0.12, ease: EASE_OUT }}
      className="px-3.5 py-2 rounded-md text-[13px] font-medium text-white disabled:opacity-40 transition-opacity"
      style={{ background: bg }}
    >
      {children}
    </motion.button>
  );
}

function GhostButton({
  children,
  onClick,
  tone,
}: {
  children: React.ReactNode;
  onClick: () => void;
  tone?: "err";
}) {
  const color = tone === "err" ? "var(--accent-err)" : "var(--ink)";
  return (
    <motion.button
      type="button"
      onClick={onClick}
      whileTap={{ scale: 0.97 }}
      transition={{ duration: 0.12, ease: EASE_OUT }}
      className="px-3.5 py-2 rounded-md text-[13px] font-medium border border-line bg-surface hover:bg-surface-sunk transition-colors"
      style={{ color }}
    >
      {children}
    </motion.button>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Composer (input bar)
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
// Completion palette helpers
// ────────────────────────────────────────────────────────────────────────────

interface TriggerMatch {
  kind: "/" | "@";
  /** Absolute index in `value` where the trigger char sits. */
  start: number;
  /** The full token including the trigger char (e.g. "/git:co"). */
  token: string;
  /** Just the part after the trigger char (used to filter). */
  query: string;
}

/** Find the active `/` or `@` token at the caret, or null if none.
 *  Valid only at start-of-input or after whitespace, so emails / file paths
 *  (foo@bar.com, ./script.sh) don't open the palette. */
function detectTrigger(value: string, caret: number): TriggerMatch | null {
  if (caret < 0 || caret > value.length) return null;
  // Walk left from caret until we hit whitespace or a non-name char.
  let i = caret;
  while (i > 0) {
    const ch = value[i - 1]!;
    if (/\s/.test(ch)) break;
    if (ch === "/" || ch === "@") {
      const start = i - 1;
      // Trigger must be at start-of-input or preceded by whitespace.
      if (start > 0 && !/\s/.test(value[start - 1]!)) return null;
      // Capture the rest of the token after the caret too, so the palette
      // doesn't disappear if the user moves the cursor backwards.
      let end = caret;
      while (end < value.length && !/\s/.test(value[end]!)) end++;
      const token = value.slice(start, end);
      return {
        kind: ch as "/" | "@",
        start,
        token,
        query: token.slice(1),
      };
    }
    // Name chars: letters, digits, `-`, `_`, `:`, `.` (we're permissive)
    if (!/[A-Za-z0-9_\-:.]/.test(ch)) return null;
    i--;
  }
  return null;
}

/** Subsequence + prefix fuzzy filter. Returns prefix matches first, then
 *  contiguous substring, then subsequence. Case-insensitive. Pure;
 *  preserves input order within each tier. */
function fuzzyFilter<T extends { name: string }>(items: T[], query: string): T[] {
  const q = query.toLowerCase();
  if (!q) return items;
  const prefix: T[] = [];
  const substring: T[] = [];
  const subseq: T[] = [];
  for (const it of items) {
    const n = it.name.toLowerCase();
    if (n.startsWith(q)) prefix.push(it);
    else if (n.includes(q)) substring.push(it);
    else if (isSubsequence(q, n)) subseq.push(it);
  }
  return [...prefix, ...substring, ...subseq];
}

function isSubsequence(needle: string, haystack: string): boolean {
  let i = 0;
  for (const ch of haystack) {
    if (ch === needle[i]) i++;
    if (i === needle.length) return true;
  }
  return false;
}

function CompletionPalette({
  items,
  index,
  onPick,
  onHover,
  trigger,
}: {
  items: CompletionItem[];
  index: number;
  onPick: (item: CompletionItem) => void;
  onHover: (i: number) => void;
  trigger: "/" | "@";
}) {
  // Scroll the active item into view as the user navigates.
  const listRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${index}"]`);
    if (el && "scrollIntoView" in el) {
      (el as HTMLElement).scrollIntoView({ block: "nearest" });
    }
  }, [index]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.14, ease: EASE_OUT }}
      className="absolute left-0 right-0 bottom-full mb-2 z-30 rounded-lg border border-line bg-canvas shadow-xl overflow-hidden"
    >
      <div className="px-3 py-2 border-b border-line flex items-baseline justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
          {trigger === "/" ? "Commands & skills" : "Agents"}
        </span>
        <span className="text-[10px] font-mono text-ink-faint">
          ↑↓ select · ⏎ insert · esc dismiss
        </span>
      </div>
      <div ref={listRef} className="max-h-[280px] overflow-y-auto scroll-quiet">
        {items.length === 0 ? (
          <div className="px-3 py-4 text-[12px] text-ink-faint">
            No {trigger === "/" ? "commands or skills" : "agents"} match.
            <div className="mt-1 text-[11px] text-ink-faint">
              {trigger === "/" ? (
                <>
                  Drop .md files in <span className="font-mono">.claude/commands/</span>{" "}
                  or <span className="font-mono">~/.claude/commands/</span> to add some.
                </>
              ) : (
                <>
                  Drop .md files in <span className="font-mono">.claude/agents/</span>{" "}
                  or <span className="font-mono">~/.claude/agents/</span> to add some.
                </>
              )}
            </div>
          </div>
        ) : (
          items.map((item, i) => (
            <CompletionRow
              key={`${item.kind}-${item.name}-${item.source}`}
              item={item}
              active={i === index}
              onPick={() => onPick(item)}
              onHover={() => onHover(i)}
              trigger={trigger}
              idx={i}
            />
          ))
        )}
      </div>
    </motion.div>
  );
}

function CompletionRow({
  item,
  active,
  onPick,
  onHover,
  trigger,
  idx,
}: {
  item: CompletionItem;
  active: boolean;
  onPick: () => void;
  onHover: () => void;
  trigger: "/" | "@";
  idx: number;
}) {
  const kindGlyph = item.kind === "agent" ? "◉" : item.kind === "skill" ? "✦" : "/";
  const scopeLabel =
    item.source === "project" ? "project" : item.source === "user" ? "user" : "built-in";
  const scopeColor =
    item.source === "project"
      ? "var(--accent-cool)"
      : item.source === "builtin"
      ? "var(--ink-faint)"
      : "var(--ink-soft)";
  return (
    <button
      type="button"
      data-idx={idx}
      onMouseEnter={onHover}
      // Prevent the textarea from losing focus before we get the click.
      onMouseDown={(e) => { e.preventDefault(); onPick(); }}
      className={`w-full text-left px-3 py-2 flex items-center gap-3 transition-colors ${
        active ? "bg-surface-tinted" : "hover:bg-surface-sunk"
      }`}
    >
      <span
        className="font-mono text-[11px] w-4 text-center shrink-0"
        style={{ color: item.kind === "agent" ? "var(--accent-warm)" : "var(--accent-cool)" }}
        aria-hidden
      >
        {kindGlyph}
      </span>
      <span className="font-mono text-[13px] text-ink shrink-0">
        {trigger}{item.name}
      </span>
      {item.argument_hint && (
        <span className="font-mono text-[11px] text-ink-faint shrink-0">
          {item.argument_hint}
        </span>
      )}
      {item.description && (
        <span className="text-[12px] text-ink-muted truncate flex-1">
          {item.description}
        </span>
      )}
      <span
        className="font-mono text-[9px] uppercase tracking-[0.14em] shrink-0"
        style={{ color: scopeColor }}
      >
        {scopeLabel}
      </span>
    </button>
  );
}

function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  status,
  needsAck,
  onAcknowledge,
  hasPlan,
  planSidebarOpen,
  planSidebarLocked,
  onTogglePlanSidebar,
  completions,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  status: string;
  needsAck: boolean;
  onAcknowledge: () => void;
  hasPlan: boolean;
  planSidebarOpen: boolean;
  planSidebarLocked: boolean;
  onTogglePlanSidebar: () => void;
  completions: CompletionsData | null;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [caret, setCaret] = useState(0);

  // Auto-grow textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [value]);

  // Suggestion chip click → populate input
  useEffect(() => {
    const onSuggest = (e: Event) => {
      const detail = (e as CustomEvent<string>).detail;
      onChange(detail);
      textareaRef.current?.focus();
    };
    window.addEventListener("blitz:suggest", onSuggest);
    return () => window.removeEventListener("blitz:suggest", onSuggest);
  }, [onChange]);

  // Detect an active `/` or `@` trigger token at the caret. The trigger
  // is valid at start-of-input or right after whitespace, so accidental
  // emails / paths like "foo@bar" don't pop the palette.
  const trigger = useMemo(() => detectTrigger(value, caret), [value, caret]);
  const palette: CompletionsData | null = useMemo(() => {
    if (!trigger || !completions) return null;
    if (trigger.kind === "/") {
      // Skills are slash-invocable too — same palette as commands.
      return {
        commands: completions.commands,
        skills: completions.skills,
        agents: [],
      };
    }
    return { commands: [], skills: [], agents: completions.agents };
  }, [trigger, completions]);
  const items = useMemo(() => {
    if (!palette || !trigger) return [] as CompletionItem[];
    const flat = [...palette.commands, ...palette.skills, ...palette.agents];
    return fuzzyFilter(flat, trigger.query);
  }, [palette, trigger]);
  const [paletteIndex, setPaletteIndex] = useState(0);
  useEffect(() => { setPaletteIndex(0); }, [trigger?.token, items.length]);
  // Open whenever there's an active trigger, even if items is empty —
  // empty palette renders a "no matches" hint so the user knows the
  // trigger fired. Closes naturally when the trigger is dismissed
  // (whitespace, Esc, leaving the token).
  const paletteOpen = trigger !== null;

  const insertCompletion = useCallback((item: CompletionItem) => {
    if (!trigger) return;
    const triggerChar = trigger.kind;
    const before = value.slice(0, trigger.start);
    const after = value.slice(trigger.start + trigger.token.length);
    const inserted = `${triggerChar}${item.name} `;
    const next = before + inserted + after;
    onChange(next);
    // Park the cursor right after the trailing space so the user can
    // continue typing args.
    const nextCaret = (before + inserted).length;
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(nextCaret, nextCaret);
      setCaret(nextCaret);
    });
  }, [trigger, value, onChange]);

  return (
    <div className="border-t border-line bg-canvas">
      <div className="mx-auto w-full max-w-2xl px-6 py-4">
        <div className="relative rounded-xl border border-line bg-surface focus-within:border-line-strong transition-colors shadow-[0_1px_0_rgba(0,0,0,0.02)]">
          <AnimatePresence>
            {paletteOpen && trigger && (
              <CompletionPalette
                items={items}
                index={paletteIndex}
                onPick={insertCompletion}
                onHover={setPaletteIndex}
                trigger={trigger.kind}
              />
            )}
          </AnimatePresence>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => {
              onChange(e.target.value);
              setCaret(e.target.selectionStart ?? 0);
            }}
            onKeyUp={(e) => setCaret(e.currentTarget.selectionStart ?? 0)}
            onClick={(e) => setCaret(e.currentTarget.selectionStart ?? 0)}
            onKeyDown={(e) => {
              if (paletteOpen) {
                if (e.key === "Escape") {
                  e.preventDefault();
                  setCaret(-1);
                  return;
                }
                // Nav / insert keys only steal focus when there are items —
                // otherwise Tab/Enter behave normally so the user isn't
                // trapped in an empty palette.
                if (items.length > 0) {
                  if (e.key === "ArrowDown") {
                    e.preventDefault();
                    setPaletteIndex((i) => Math.min(i + 1, items.length - 1));
                    return;
                  }
                  if (e.key === "ArrowUp") {
                    e.preventDefault();
                    setPaletteIndex((i) => Math.max(i - 1, 0));
                    return;
                  }
                  if (e.key === "Enter" || e.key === "Tab") {
                    e.preventDefault();
                    const item = items[paletteIndex];
                    if (item) insertCompletion(item);
                    return;
                  }
                }
              }
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSubmit();
              }
            }}
            rows={1}
            placeholder={
              status === "streaming"
                ? "Agent is working…"
                : status === "awaiting_permission"
                  ? "Resolve the permission prompt above…"
                  : status === "awaiting_question"
                    ? "Answer the question above…"
                    : "Ask anything. ⏎ to send, ⇧⏎ for newline."
            }
            className="w-full resize-none bg-transparent px-4 py-3 pr-14 text-[15px] leading-relaxed text-ink placeholder:text-ink-faint focus:outline-none"
          />
          <div className="absolute right-2 bottom-2">
            <SendButton onClick={onSubmit} disabled={disabled} />
          </div>
        </div>
        <div className="mt-2 flex items-center justify-between text-[11px] text-ink-faint">
          <span>
            <Kbd>⏎</Kbd> send · <Kbd>⇧⏎</Kbd> newline
          </span>
          <div className="flex items-center gap-1">
            <AnimatePresence initial={false}>
              {hasPlan && (
                <motion.button
                  key="plan-toggle"
                  type="button"
                  onClick={onTogglePlanSidebar}
                  disabled={planSidebarLocked}
                  initial={{ opacity: 0, y: 2 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 2 }}
                  transition={{ duration: 0.18, ease: EASE_OUT }}
                  whileTap={planSidebarLocked ? undefined : { scale: 0.97 }}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-colors flex items-center gap-1.5 ${
                    planSidebarOpen
                      ? "text-ink bg-surface-tinted"
                      : "text-ink-soft hover:text-ink hover:bg-surface-sunk"
                  } disabled:cursor-default disabled:opacity-80`}
                  title={
                    planSidebarLocked
                      ? "Plan is locked open while in plan mode"
                      : planSidebarOpen
                      ? "Hide plan"
                      : "Show plan"
                  }
                  aria-pressed={planSidebarOpen}
                >
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden style={{ color: "var(--accent-cool)" }}>
                    <path d="M2 1H6.5L8.5 3V9H2V1Z" stroke="currentColor" strokeWidth="1" />
                    <path d="M6.5 1V3H8.5" stroke="currentColor" strokeWidth="1" />
                  </svg>
                  plan.md
                </motion.button>
              )}
              {needsAck && (
                <motion.button
                  key="ack"
                  type="button"
                  onClick={onAcknowledge}
                  initial={{ opacity: 0, y: 2 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 2 }}
                  transition={{ duration: 0.18, ease: EASE_OUT }}
                  whileTap={{ scale: 0.97 }}
                  className="px-2.5 py-1 rounded-md text-[11px] text-ink-soft hover:text-ink hover:bg-surface-sunk transition-colors flex items-center gap-1.5"
                  title="Move this session out of 'Needs input'"
                >
                  <span
                    className="inline-block size-1.5 rounded-full"
                    style={{ background: "var(--accent-warm)" }}
                    aria-hidden
                  />
                  Acknowledge
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

function SendButton({ onClick, disabled }: { onClick: () => void; disabled: boolean }) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={disabled}
      whileTap={{ scale: 0.94 }}
      transition={{ duration: 0.12, ease: EASE_OUT }}
      className="size-9 rounded-lg flex items-center justify-center text-white disabled:opacity-30 transition-opacity"
      style={{ background: "var(--ink)" }}
      aria-label="Send"
    >
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
        <path
          d="M1 7H13M13 7L8 2M13 7L8 12"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </motion.button>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd
      className="inline-flex items-center px-1.5 py-0.5 rounded border border-line bg-surface-sunk text-[10px] font-mono text-ink-soft"
    >
      {children}
    </kbd>
  );
}

function KbdHint({ hints }: { hints: [string, string][] }) {
  return (
    <div className="border-t border-line px-5 py-2.5 flex items-center gap-4 bg-surface-sunk">
      {hints.map(([k, label]) => (
        <span key={k} className="text-[11px] text-ink-muted flex items-center gap-1.5">
          <Kbd>{k}</Kbd>
          <span>{label}</span>
        </span>
      ))}
    </div>
  );
}

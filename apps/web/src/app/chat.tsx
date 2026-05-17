"use client";

import { useAgentMux, useActiveSession, type AgentMux, type SessionState } from "@agent-webkit/react";
import type { SessionListEntry } from "@agent-webkit/core";
import { AnimatePresence, motion, type Variants } from "framer-motion";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { initialDocSlot, reduceDocSlot, type DocSlotState } from "./docSlot";

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

// ────────────────────────────────────────────────────────────────────────────
// Workspaces (blitzcode-pro top-level primitive)
// A workspace = ticket + dir with N git worktrees + N sessions. The agent-
// webkit core knows nothing about this; everything lives in apps/server
// (workspaces.json) and is exposed via /app/workspaces.
// ────────────────────────────────────────────────────────────────────────────

export interface WorkspaceRepoDTO {
  source_path: string;
  worktree_path: string;
  branch: string;
}

export interface WorkspaceDocRefDTO {
  page_id: string | null;
  version: number | null;
  title: string | null;
  url: string | null;
  last_synced_at: number | null;
}

export interface WorkspaceDTO {
  id: string;
  ticket_key: string;
  ticket_title: string | null;
  initiative_key: string | null;
  dir: string;
  repos: WorkspaceRepoDTO[];
  session_ids: string[];
  session_names: Record<string, string>;
  docs: Record<string, WorkspaceDocRefDTO>;
  created_at: number;
  archived_at: number | null;
}

export interface InitiativeDTO {
  key: string;
  display_name: string;
  epic_jira_key: string | null;
  confluence_root_page_id: string | null;
  repo_paths: string[];
}

interface CreateWorkspaceArgs {
  ticket_key: string;
  ticket_title?: string;
  initiative_key?: string;
  repos: { source_path: string; branch?: string }[];
  spawn_initial_session?: boolean;
  permission_mode?: string;
}

interface UseWorkspaces {
  list: WorkspaceDTO[];
  hydrated: boolean;
  refresh: () => Promise<void>;
  create: (args: CreateWorkspaceArgs) => Promise<{ workspace: WorkspaceDTO; first_session_id: string | null }>;
  archive: (id: string) => Promise<void>;
  unarchive: (id: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  spawnSession: (id: string, opts?: { permission_mode?: string }) => Promise<string>;
  addRepo: (id: string, args: { source_path: string; branch?: string }) => Promise<void>;
  renameSession: (workspaceId: string, sessionId: string, name: string | null) => Promise<void>;
  deleteSession: (workspaceId: string, sessionId: string) => Promise<void>;
}

function useWorkspaces(baseUrl: string): UseWorkspaces {
  const [list, setList] = useState<WorkspaceDTO[]>([]);
  const [hydrated, setHydrated] = useState(false);

  const refresh = useCallback(async () => {
    try {
      // Include archived workspaces — the UI filters to "live" by default
      // but needs the archived set for the restore modal.
      const r = await fetch(`${baseUrl}/app/workspaces?include_archived=1`);
      if (!r.ok) return;
      const data = (await r.json()) as { workspaces: WorkspaceDTO[] };
      setList(data.workspaces ?? []);
    } finally {
      setHydrated(true);
    }
  }, [baseUrl]);

  useEffect(() => { void refresh(); }, [refresh]);

  const create = useCallback(
    async (args: CreateWorkspaceArgs) => {
      const r = await fetch(`${baseUrl}/app/workspaces`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
      });
      if (!r.ok) {
        const text = await r.text().catch(() => "");
        throw new Error(text || `HTTP ${r.status}`);
      }
      const data = await r.json();
      await refresh();
      return data as { workspace: WorkspaceDTO; first_session_id: string | null };
    },
    [baseUrl, refresh]
  );

  const archive = useCallback(
    async (id: string) => {
      await fetch(`${baseUrl}/app/workspaces/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archived: true }),
      });
      await refresh();
    },
    [baseUrl, refresh]
  );

  const unarchive = useCallback(
    async (id: string) => {
      await fetch(`${baseUrl}/app/workspaces/${encodeURIComponent(id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archived: false }),
      });
      await refresh();
    },
    [baseUrl, refresh]
  );

  const remove = useCallback(
    async (id: string) => {
      await fetch(`${baseUrl}/app/workspaces/${encodeURIComponent(id)}`, { method: "DELETE" });
      await refresh();
    },
    [baseUrl, refresh]
  );

  const spawnSession = useCallback(
    async (id: string, opts?: { permission_mode?: string }) => {
      const r = await fetch(`${baseUrl}/app/workspaces/${encodeURIComponent(id)}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(opts ?? {}),
      });
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      const data = await r.json();
      await refresh();
      return data.session_id as string;
    },
    [baseUrl, refresh]
  );

  const addRepo = useCallback(
    async (id: string, args: { source_path: string; branch?: string }) => {
      const r = await fetch(`${baseUrl}/app/workspaces/${encodeURIComponent(id)}/repos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
      });
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      await refresh();
    },
    [baseUrl, refresh]
  );

  const renameSession = useCallback(
    async (workspaceId: string, sessionId: string, name: string | null) => {
      const r = await fetch(
        `${baseUrl}/app/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        }
      );
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      await refresh();
    },
    [baseUrl, refresh]
  );

  const deleteSession = useCallback(
    async (workspaceId: string, sessionId: string) => {
      const r = await fetch(
        `${baseUrl}/app/workspaces/${encodeURIComponent(workspaceId)}/sessions/${encodeURIComponent(sessionId)}`,
        { method: "DELETE" }
      );
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      await refresh();
    },
    [baseUrl, refresh]
  );

  return useMemo(
    () => ({
      list, hydrated, refresh, create, archive, unarchive, remove,
      spawnSession, addRepo, renameSession, deleteSession,
    }),
    [list, hydrated, refresh, create, archive, unarchive, remove,
     spawnSession, addRepo, renameSession, deleteSession]
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Atlassian credentials hook + typeahead
// ────────────────────────────────────────────────────────────────────────────

export interface AtlassianCredsMeta {
  has_creds: boolean;
  site_url: string | null;
  email: string | null;
}

interface UseAtlassianCreds {
  meta: AtlassianCredsMeta;
  refresh: () => Promise<void>;
  set: (args: { site_url: string; email: string; api_token: string }) => Promise<void>;
  clear: () => Promise<void>;
}

// ────────────────────────────────────────────────────────────────────────────
// User settings (theme, etc) — server-persisted, shallow-merged per category.
// Theme is the only setting for now; the store generalises so we can add
// more later (density, default permission_mode, font scale…) without a
// schema change. Persistence path: ~/.agent-webkit/blitzcode-pro/settings.json
// ────────────────────────────────────────────────────────────────────────────

export type ThemePreference = "light" | "dark" | "system";

export interface SettingsShape {
  appearance?: {
    theme?: ThemePreference;
  };
  // Future: density, default permission mode, ...
}

interface UseSettings {
  settings: SettingsShape;
  hydrated: boolean;
  refresh: () => Promise<void>;
  patch: (updates: Partial<SettingsShape>) => Promise<void>;
  theme: ThemePreference;
}

function useSettings(baseUrl: string): UseSettings {
  const [settings, setSettings] = useState<SettingsShape>({});
  const [hydrated, setHydrated] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${baseUrl}/app/settings`);
      if (!r.ok) return;
      const data = (await r.json()) as { settings: SettingsShape };
      setSettings(data.settings ?? {});
    } finally {
      setHydrated(true);
    }
  }, [baseUrl]);

  useEffect(() => { void refresh(); }, [refresh]);

  const patch = useCallback(
    async (updates: Partial<SettingsShape>) => {
      // Optimistic merge — UI doesn't wait for the round-trip to feel snappy.
      setSettings((prev) => mergeSettings(prev, updates));
      try {
        const r = await fetch(`${baseUrl}/app/settings`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updates),
        });
        if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
        const data = (await r.json()) as { settings: SettingsShape };
        setSettings(data.settings ?? {});
      } catch {
        // On failure, refetch to recover canonical state.
        void refresh();
      }
    },
    [baseUrl, refresh]
  );

  return useMemo(
    () => ({
      settings,
      hydrated,
      refresh,
      patch,
      theme: settings.appearance?.theme ?? "system",
    }),
    [settings, hydrated, refresh, patch]
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Theme application — resolves "light" | "dark" | "system" → the active
// scheme and writes [data-theme] on <html>. The FOUC script in layout.tsx
// reads the same localStorage mirror before first paint; this hook keeps
// it fresh when the user picks something different or the OS scheme
// flips while "system" is active. Mount once at the app root.
// ────────────────────────────────────────────────────────────────────────────
const THEME_STORAGE_KEY = "blitz.theme";

function useResolvedTheme(pref: ThemePreference): "light" | "dark" {
  // We need a stable initial value during SSR — defaulting to "light" is
  // fine because the FOUC script has already corrected <html> by the
  // time React paints. The first client render syncs us to reality.
  const [systemDark, setSystemDark] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemDark(mq.matches);
    if (pref !== "system") return; // only listen when it actually matters
    const onChange = (e: MediaQueryListEvent) => setSystemDark(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [pref]);
  if (pref === "dark") return "dark";
  if (pref === "light") return "light";
  return systemDark ? "dark" : "light";
}

function useApplyTheme(pref: ThemePreference, hydrated: boolean): void {
  const resolved = useResolvedTheme(pref);
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.setAttribute("data-theme", resolved);
    // Only mirror the user's *preference* (not the resolved value) so
    // the boot script can re-resolve "system" against the OS on next
    // load. Wait for settings hydration so a transient "system" default
    // doesn't overwrite a real saved choice.
    if (!hydrated) return;
    try {
      if (pref === "system") localStorage.removeItem(THEME_STORAGE_KEY);
      else localStorage.setItem(THEME_STORAGE_KEY, pref);
    } catch { /* localStorage may be blocked; FOUC is best-effort */ }
  }, [resolved, pref, hydrated]);
}

// ────────────────────────────────────────────────────────────────────────────
// Tauri shell integration — fullscreen detection only.
//
// Drag-region behavior lives in the Rust shell's initialization_script
// (must hook mousedown synchronously so AppKit can pick up the native
// drag gesture; a React listener fires too late). Here we just listen
// for resize events and stamp [data-fullscreen] on <html> so the
// `tauri-windowed:` variant retracts the traffic-light strip when the
// OS hides the traffic lights during native fullscreen.
// ────────────────────────────────────────────────────────────────────────────
function useTauriShell(): void {
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!("__TAURI_INTERNALS__" in window)) return;

    let unlistenResize: (() => void) | null = null;
    let cancelled = false;

    (async () => {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      if (cancelled) return;
      const w = getCurrentWindow();

      const syncFullscreen = async () => {
        try {
          const fs = await w.isFullscreen();
          document.documentElement.toggleAttribute("data-fullscreen", fs);
        } catch { /* ignore — Tauri may be tearing down */ }
      };
      await syncFullscreen();
      unlistenResize = await w.onResized(() => { void syncFullscreen(); });
    })();

    return () => {
      cancelled = true;
      if (unlistenResize) unlistenResize();
    };
  }, []);
}

function mergeSettings(prev: SettingsShape, updates: Partial<SettingsShape>): SettingsShape {
  const out: SettingsShape = { ...prev };
  for (const [cat, vals] of Object.entries(updates) as Array<[keyof SettingsShape, Record<string, unknown>]>) {
    if (!vals || typeof vals !== "object") continue;
    const merged: Record<string, unknown> = { ...(out[cat] ?? {}) };
    for (const [k, v] of Object.entries(vals)) {
      if (v === null) delete merged[k];
      else merged[k] = v;
    }
    out[cat] = merged as any;
  }
  return out;
}

function useAtlassianCreds(baseUrl: string): UseAtlassianCreds {
  const [meta, setMeta] = useState<AtlassianCredsMeta>({ has_creds: false, site_url: null, email: null });

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${baseUrl}/app/atlassian/has-creds`);
      if (!r.ok) return;
      setMeta(await r.json());
    } catch { /* ignore */ }
  }, [baseUrl]);

  useEffect(() => { void refresh(); }, [refresh]);

  const set = useCallback(
    async (args: { site_url: string; email: string; api_token: string }) => {
      const r = await fetch(`${baseUrl}/app/atlassian/creds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args),
      });
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      setMeta(await r.json());
    },
    [baseUrl]
  );

  const clear = useCallback(async () => {
    await fetch(`${baseUrl}/app/atlassian/creds`, { method: "DELETE" });
    await refresh();
  }, [baseUrl, refresh]);

  return useMemo(() => ({ meta, refresh, set, clear }), [meta, refresh, set, clear]);
}

export interface TicketSearchResult {
  key: string;
  title: string;
  status: string | null;
  issuetype: string | null;
}

async function searchTickets(baseUrl: string, q: string, signal?: AbortSignal): Promise<TicketSearchResult[]> {
  if (!q.trim()) return [];
  const base = baseUrl || "";
  const url = new URL(`${base}/app/workflow/search-tickets`, window.location.origin);
  url.searchParams.set("q", q);
  const r = await fetch(url.toString(), { signal });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    if (r.status === 400 && text.includes("requires_credentials")) {
      throw new RequiresCredsError();
    }
    throw new Error(text || `HTTP ${r.status}`);
  }
  const data = (await r.json()) as { results: TicketSearchResult[] };
  return data.results ?? [];
}

class RequiresCredsError extends Error {
  constructor() { super("Atlassian credentials required"); this.name = "RequiresCredsError"; }
}

interface UseInitiatives {
  list: InitiativeDTO[];
  refresh: () => Promise<void>;
  upsert: (i: Partial<InitiativeDTO> & { key: string; display_name: string }) => Promise<InitiativeDTO>;
  remove: (key: string) => Promise<void>;
}

function useInitiatives(baseUrl: string): UseInitiatives {
  const [list, setList] = useState<InitiativeDTO[]>([]);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(`${baseUrl}/app/initiatives`);
      if (!r.ok) return;
      const data = (await r.json()) as { initiatives: InitiativeDTO[] };
      setList(data.initiatives ?? []);
    } catch { /* ignore */ }
  }, [baseUrl]);

  useEffect(() => { void refresh(); }, [refresh]);

  const upsert = useCallback(
    async (i: Partial<InitiativeDTO> & { key: string; display_name: string }) => {
      const r = await fetch(`${baseUrl}/app/initiatives`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          key: i.key,
          display_name: i.display_name,
          epic_jira_key: i.epic_jira_key ?? null,
          confluence_root_page_id: i.confluence_root_page_id ?? null,
          repo_paths: i.repo_paths ?? [],
        }),
      });
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      const out = (await r.json()) as InitiativeDTO;
      await refresh();
      return out;
    },
    [baseUrl, refresh]
  );

  const remove = useCallback(
    async (key: string) => {
      await fetch(`${baseUrl}/app/initiatives/${encodeURIComponent(key)}`, { method: "DELETE" });
      await refresh();
    },
    [baseUrl, refresh]
  );

  return useMemo(() => ({ list, refresh, upsert, remove }), [list, refresh, upsert, remove]);
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
  const workspaces = useWorkspaces(baseUrl);
  const initiatives = useInitiatives(baseUrl);
  const atlassianCreds = useAtlassianCreds(baseUrl);
  const userSettings = useSettings(baseUrl);
  useApplyTheme(userSettings.theme, userSettings.hydrated);
  useTauriShell();

  // One persistent multiplexed stream for the whole app. Sessions are
  // spawned via /app/workspaces/{id}/sessions (server-side), but show up
  // in mux.sessionList via the standard /sessions list refresh.
  const mux = useAgentMux({
    baseUrl,
    onEvent: (ev) => {
      if (ev.event === "result" && ev.session_id) {
        acks.markCompletionLocal(ev.session_id);
      }
    },
  });

  // Primary nav state: the active workspace, then a session within it.
  // Both stored as raw ids so URL/route plumbing can pick them up later.
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  const [activeSessionByWs, setActiveSessionByWs] = useState<Record<string, string>>({});
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [initiativeModalOpen, setInitiativeModalOpen] = useState(false);
  const [credsModalOpen, setCredsModalOpen] = useState(false);
  const [archiveModalOpen, setArchiveModalOpen] = useState(false);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const historyLoaded = useRef<Set<string>>(new Set());

  // Pick a default workspace once both hydrations land.
  useEffect(() => {
    if (!workspaces.hydrated || activeWorkspaceId !== null) return;
    const first = workspaces.list[0];
    if (first) setActiveWorkspaceId(first.id);
  }, [workspaces.hydrated, workspaces.list, activeWorkspaceId]);

  // For each workspace we visit, pick its first session as active by default.
  const activeWorkspace = useMemo(
    () => workspaces.list.find((w) => w.id === activeWorkspaceId) ?? null,
    [workspaces.list, activeWorkspaceId]
  );
  useEffect(() => {
    if (!activeWorkspace) return;
    if (activeSessionByWs[activeWorkspace.id]) return;
    const first = activeWorkspace.session_ids[0];
    if (first) setActiveSessionByWs((prev) => ({ ...prev, [activeWorkspace.id]: first }));
  }, [activeWorkspace, activeSessionByWs]);

  const activeSessionId = activeWorkspace
    ? activeSessionByWs[activeWorkspace.id] ?? activeWorkspace.session_ids[0] ?? null
    : null;

  // Load history for whichever session is currently active.
  useEffect(() => {
    if (!activeSessionId) return;
    if (historyLoaded.current.has(activeSessionId)) return;
    historyLoaded.current.add(activeSessionId);
    void mux.loadHistory(activeSessionId);
  }, [activeSessionId, mux]);

  const createWorkspace = useCallback(
    async (args: CreateWorkspaceArgs) => {
      setCreatingWorkspace(true);
      try {
        const out = await workspaces.create(args);
        setActiveWorkspaceId(out.workspace.id);
        if (out.first_session_id) {
          setActiveSessionByWs((prev) => ({ ...prev, [out.workspace.id]: out.first_session_id! }));
        }
        // Make sure mux picks up the new session in its sessionList.
        void mux.refreshSessions();
      } finally {
        setCreatingWorkspace(false);
      }
    },
    [workspaces, mux]
  );

  const spawnSessionIntoWorkspace = useCallback(
    async (workspaceId: string) => {
      const sid = await workspaces.spawnSession(workspaceId);
      setActiveSessionByWs((prev) => ({ ...prev, [workspaceId]: sid }));
      await mux.refreshSessions();
      return sid;
    },
    [workspaces, mux]
  );

  // Deleting a session may be the active one for its workspace — repoint
  // to whichever sibling remains (or null) so the chat view doesn't crash
  // trying to read a session that no longer exists.
  const deleteSessionFromWorkspace = useCallback(
    async (workspaceId: string, sessionId: string) => {
      await workspaces.deleteSession(workspaceId, sessionId);
      await mux.refreshSessions();
      setActiveSessionByWs((prev) => {
        if (prev[workspaceId] !== sessionId) return prev;
        const ws = workspaces.list.find((w) => w.id === workspaceId);
        const next = (ws?.session_ids ?? []).find((s) => s !== sessionId) ?? null;
        const out = { ...prev };
        if (next) out[workspaceId] = next;
        else delete out[workspaceId];
        return out;
      });
    },
    [workspaces, mux]
  );

  return (
    <div className="flex h-screen bg-canvas text-ink">
      <TicketSidebar
        workspaces={workspaces.list}
        initiatives={initiatives.list}
        sessions={mux.sessionList}
        sessionStates={mux.sessions}
        acks={acks}
        activeWorkspaceId={activeWorkspaceId}
        atlassianMeta={atlassianCreds.meta}
        onSelect={(id) => setActiveWorkspaceId(id)}
        onOpenCreate={() => setCreateModalOpen(true)}
        onOpenInitiatives={() => setInitiativeModalOpen(true)}
        onOpenArchive={() => setArchiveModalOpen(true)}
        onOpenSettings={() => setSettingsModalOpen(true)}
        onArchive={workspaces.archive}
        onUnarchive={workspaces.unarchive}
        onDelete={workspaces.remove}
      />
      <div className="flex-1 min-w-0 flex flex-col">
        {!workspaces.hydrated || !mux.hydrated ? (
          <LoadingState />
        ) : activeWorkspace && activeSessionId ? (
          <ChatView
            baseUrl={baseUrl}
            mux={mux}
            sessionId={activeSessionId}
            workspace={activeWorkspace}
            workspaceSessions={activeWorkspace.session_ids}
            onPickSession={(sid) => setActiveSessionByWs((p) => ({ ...p, [activeWorkspace.id]: sid }))}
            onSpawnSession={() => spawnSessionIntoWorkspace(activeWorkspace.id)}
            onRenameSession={(sid, name) => workspaces.renameSession(activeWorkspace.id, sid, name)}
            onDeleteSession={(sid) => deleteSessionFromWorkspace(activeWorkspace.id, sid)}
            onWorkspaceMaybeChanged={() => { void workspaces.refresh(); }}
            acks={acks}
            completions={completions}
          />
        ) : (
          <NoActiveWorkspace
            onCreate={() => setCreateModalOpen(true)}
            creating={creatingWorkspace}
          />
        )}
      </div>
      <AnimatePresence>
        {createModalOpen && (
          <Modal key="ws-create">
            <WorkspaceCreateModal
              baseUrl={baseUrl}
              initiatives={initiatives.list}
              hasCreds={atlassianCreds.meta.has_creds}
              onRequestCreds={() => setCredsModalOpen(true)}
              creating={creatingWorkspace}
              onCancel={() => setCreateModalOpen(false)}
              onCreate={async (args) => {
                await createWorkspace(args);
                setCreateModalOpen(false);
              }}
            />
          </Modal>
        )}
        {initiativeModalOpen && (
          <Modal key="init-mgr">
            <InitiativeManager
              baseUrl={baseUrl}
              initiatives={initiatives}
              onClose={() => setInitiativeModalOpen(false)}
            />
          </Modal>
        )}
        {credsModalOpen && (
          <Modal key="creds">
            <CredsModal
              creds={atlassianCreds}
              onClose={() => setCredsModalOpen(false)}
            />
          </Modal>
        )}
        {archiveModalOpen && (
          <Modal key="archive">
            <ArchiveModal
              workspaces={workspaces.list.filter((w) => w.archived_at !== null)}
              onRestore={async (id) => { await workspaces.unarchive(id); }}
              onDelete={async (id) => { await workspaces.remove(id); }}
              onOpen={(id) => { setActiveWorkspaceId(id); setArchiveModalOpen(false); }}
              onClose={() => setArchiveModalOpen(false)}
            />
          </Modal>
        )}
        {settingsModalOpen && (
          <Modal key="settings">
            <SettingsModal
              settings={userSettings}
              creds={atlassianCreds}
              onClose={() => setSettingsModalOpen(false)}
            />
          </Modal>
        )}
      </AnimatePresence>
    </div>
  );
}

function NoActiveWorkspace({ onCreate, creating }: { onCreate: () => void; creating: boolean }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
      <h1 className="font-serif text-4xl leading-[1.05] tracking-tight text-ink">
        No active ticket.
      </h1>
      <p className="mt-3 text-ink-muted text-[15px] max-w-md leading-relaxed">
        Each ticket runs in its own workspace — a directory holding git
        worktrees of the repos involved. Create one to get started.
      </p>
      <motion.button
        type="button"
        onClick={onCreate}
        disabled={creating}
        whileTap={{ scale: 0.97 }}
        transition={{ duration: 0.12, ease: EASE_OUT }}
        className="mt-6 px-4 py-2 rounded-md text-[14px] font-medium text-canvas disabled:opacity-40 transition-opacity"
        style={{ background: "var(--ink)" }}
      >
        {creating ? "Creating…" : "Add ticket"}
      </motion.button>
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

/** Latest workflow_write_* tool call kind, or null. Used to auto-open the
 *  doc slot when the agent edits something. */
/** MCP tools come through with namespaced names like
 *  `mcp__workflow__workflow_update_ticket_fields`. We do all matching on
 *  the unqualified short name so the same set works whether the agent
 *  called the tool via MCP or the bare name. */
function shortToolName(name: string): string {
  // Pattern: mcp__<server>__<tool>  (double-underscore separators)
  const parts = name.split("__");
  return parts.length >= 3 && parts[0] === "mcp" ? parts.slice(2).join("__") : name;
}

/** Count how many tool_use blocks across the session COMPLETED (i.e. have
 *  a matching tool_result), filtered by `names`. The total is opaque —
 *  callers use it as a refresh key. Re-running a mutation increments the
 *  count, causing any viewer's useEffect listing it as a dep to re-fetch.
 *
 *  Why tool_result and not tool_use: tool_use appears the moment the model
 *  emits the call, but the actual side-effect (JIRA write, Confluence
 *  update, etc.) happens on tool execution. Refetching at tool_use time
 *  races the write — we'd see stale data and the user has to manually
 *  refresh. Counting tool_results ensures the upstream system has already
 *  committed by the time we refetch. */
function countToolUses(messages: AnySessionMessage[], names: ReadonlySet<string>): number {
  // First pass: build a tool_use_id → short tool name index from assistant
  // tool_use blocks. tool_results only carry the id, not the tool name.
  const idToName = new Map<string, string>();
  for (const m of messages) {
    if (!m || m.kind !== "assistant") continue;
    const blocks = (m.content as ContentBlock[]) ?? [];
    for (const b of blocks) {
      if (b?.type === "tool_use" && typeof b.id === "string" && typeof b.name === "string") {
        idToName.set(b.id, shortToolName(b.name));
      }
    }
  }
  // Second pass: count tool_results whose mapped tool name matches.
  let n = 0;
  for (const m of messages) {
    if (!m || m.kind !== "tool_result") continue;
    const name = idToName.get(m.tool_use_id);
    if (name && names.has(name)) n++;
  }
  return n;
}

const TICKET_MUTATION_TOOLS: ReadonlySet<string> = new Set([
  "workflow_update_ticket_fields",
  "workflow_set_status",
  "workflow_add_comment",
  "workflow_flag",
  "workflow_unflag",
]);
const RFC_MUTATION_TOOLS: ReadonlySet<string> = new Set(["workflow_write_rfc"]);
const DEBRIEF_MUTATION_TOOLS: ReadonlySet<string> = new Set(["workflow_write_debrief"]);

function deriveLatestDocWrite(messages: AnySessionMessage[]): "ticket" | "rfc" | "debrief" | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (!m || m.kind !== "assistant") continue;
    const blocks = (m.content as ContentBlock[]) ?? [];
    for (let j = blocks.length - 1; j >= 0; j--) {
      const b = blocks[j];
      if (b?.type !== "tool_use" || typeof b.name !== "string") continue;
      const short = shortToolName(b.name);
      if (short === "workflow_write_rfc") return "rfc";
      if (short === "workflow_write_debrief") return "debrief";
      if (short === "workflow_update_ticket_fields") return "ticket";
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
  baseUrl,
  mux,
  sessionId,
  workspace,
  workspaceSessions,
  onPickSession,
  onSpawnSession,
  onRenameSession,
  onDeleteSession,
  onWorkspaceMaybeChanged,
  acks,
  completions,
}: {
  baseUrl: string;
  mux: AgentMux;
  sessionId: string;
  workspace: WorkspaceDTO;
  workspaceSessions: string[];
  onPickSession: (sid: string) => void;
  onSpawnSession: () => Promise<string>;
  onRenameSession: (sid: string, name: string | null) => Promise<void>;
  onDeleteSession: (sid: string) => Promise<void>;
  /** Refetch workspaces; the docs.rfc/debrief refs change server-side when
   *  the agent writes those pages, and we need a fresh fetch to see them. */
  onWorkspaceMaybeChanged: () => void;
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

  // ── DocSlot — single shared right-rail wide panel across plan/ticket/RFC/debrief
  const [docSlot, dispatchDocSlot] = useReducer(reduceDocSlot, initialDocSlot);
  // Reset on workspace change.
  useEffect(() => {
    dispatchDocSlot({ type: "workspace_changed" });
  }, [workspace.id]);
  // Auto-drive plan-mode transitions.
  const planActive = (inPlanMode || pendingPlanApproval !== null) && planText !== null;
  useEffect(() => {
    if (planActive) {
      dispatchDocSlot({ type: "plan_mode_entered" });
    } else {
      dispatchDocSlot({ type: "plan_mode_exited" });
    }
  }, [planActive]);
  // Refresh keys — bump every time the agent invokes a mutation tool that
  // would have changed the corresponding remote doc. Open viewers re-fetch.
  const ticketRefreshKey = useMemo(
    () => countToolUses(session.messages, TICKET_MUTATION_TOOLS),
    [session.messages]
  );
  const rfcRefreshKey = useMemo(
    () => countToolUses(session.messages, RFC_MUTATION_TOOLS),
    [session.messages]
  );
  const debriefRefreshKey = useMemo(
    () => countToolUses(session.messages, DEBRIEF_MUTATION_TOOLS),
    [session.messages]
  );
  // Auto-open the doc slot ONLY on writes that happen after the
  // workspace+session were entered — not on history replay. We snapshot
  // the counts whenever the workspace OR session id changes, and only
  // dispatch when subsequent counts exceed that baseline. Without this,
  // switching to a session whose transcript already contains a write
  // yanks the slot open every time.
  const baselineRef = useRef({ ticket: 0, rfc: 0, debrief: 0 });
  useEffect(() => {
    baselineRef.current = {
      ticket: ticketRefreshKey,
      rfc: rfcRefreshKey,
      debrief: debriefRefreshKey,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace.id, sessionId]);
  useEffect(() => {
    // Snapshot what bumped BEFORE mutating the baseline. Two separate
    // effects sharing the same baseline would race — the first would
    // advance the counter and the second would see no bump.
    const b = baselineRef.current;
    const ticketBumped = ticketRefreshKey > b.ticket;
    const rfcBumped = rfcRefreshKey > b.rfc;
    const debriefBumped = debriefRefreshKey > b.debrief;

    if (ticketBumped) {
      b.ticket = ticketRefreshKey;
      dispatchDocSlot({
        type: "tool_use_doc_write", doc: "ticket",
        workspaceId: workspace.id, ticketKey: workspace.ticket_key,
      });
    }
    if (rfcBumped) {
      b.rfc = rfcRefreshKey;
      dispatchDocSlot({
        type: "tool_use_doc_write", doc: "rfc",
        workspaceId: workspace.id, ticketKey: workspace.ticket_key,
      });
    }
    if (debriefBumped) {
      b.debrief = debriefRefreshKey;
      dispatchDocSlot({
        type: "tool_use_doc_write", doc: "debrief",
        workspaceId: workspace.id, ticketKey: workspace.ticket_key,
      });
    }

    // Refetch the workspaces list when an RFC/Debrief write completes —
    // the server stores the new page_id/version onto Workspace.docs and
    // we need that fresh data for the tile (versions, sync-time) and
    // for the viewer's pageRef.
    if (rfcBumped || debriefBumped) {
      onWorkspaceMaybeChanged();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticketRefreshKey, rfcRefreshKey, debriefRefreshKey, workspace.id, workspace.ticket_key]);

  const planLocked = inPlanMode || pendingPlanApproval !== null;

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
        workspace={workspace}
        workspaceSessions={workspaceSessions}
        sessionStates={mux.sessions}
        sessionNames={workspace.session_names}
        sessionList={mux.sessionList}
        workspaceDir={workspace.dir}
        onPickSession={onPickSession}
        onSpawnSession={onSpawnSession}
        onRenameSession={onRenameSession}
        onDeleteSession={onDeleteSession}
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
        completions={completionsForSession}
      />
      </div>
      <AnimatePresence initial={false}>
        {docSlot.kind !== "hidden" && (
          <DocSlotSidebar
            key={`docslot-${docSlot.kind}`}
            slot={docSlot}
            workspace={workspace}
            baseUrl={baseUrl}
            ticketRefreshKey={ticketRefreshKey}
            rfcRefreshKey={rfcRefreshKey}
            debriefRefreshKey={debriefRefreshKey}
            planText={planText}
            pendingPlanApproval={pendingPlanApproval}
            onClose={() => dispatchDocSlot({ type: "hide" })}
            onApprovePlan={() =>
              pendingPlanApproval && session.approve(pendingPlanApproval.correlation_id, {})
            }
            onDenyPlan={() =>
              pendingPlanApproval &&
              session.deny(pendingPlanApproval.correlation_id, {
                message: "User declined to approve this plan. Keep refining.",
              })
            }
          />
        )}
        {(showTodos && todos) || workspace ? (
          <RightRail
            key="right-rail"
            todos={showTodos ? todos : null}
            workspace={workspace}
            docSlot={docSlot}
            hasPlan={planText !== null}
            planLocked={planLocked}
            onPickDoc={(target) =>
              dispatchDocSlot({ type: "user_toggle", target })
            }
          />
        ) : null}
      </AnimatePresence>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// DocSlotSidebar — single right-rail wide panel that hosts whichever doc is
// active per the DocSlot reducer. Internally dispatches to the appropriate
// renderer (plan / ticket / RFC / debrief).
// ────────────────────────────────────────────────────────────────────────────

function DocSlotSidebar({
  slot,
  workspace,
  baseUrl,
  ticketRefreshKey,
  rfcRefreshKey,
  debriefRefreshKey,
  planText,
  pendingPlanApproval,
  onClose,
  onApprovePlan,
  onDenyPlan,
}: {
  slot: DocSlotState;
  workspace: WorkspaceDTO;
  baseUrl: string;
  ticketRefreshKey: number;
  rfcRefreshKey: number;
  debriefRefreshKey: number;
  planText: string | null;
  pendingPlanApproval: { correlation_id: string } | null;
  onClose: () => void;
  onApprovePlan: () => void;
  onDenyPlan: () => void;
}) {
  if (slot.kind === "hidden") return null;
  if (slot.kind === "plan") {
    return (
      <PlanSidebar
        plan={planText}
        pendingApproval={pendingPlanApproval}
        onApprove={onApprovePlan}
        onDeny={onDenyPlan}
      />
    );
  }
  if (slot.kind === "ticket") {
    return (
      <TicketViewer
        baseUrl={baseUrl}
        ticketKey={slot.ticketKey}
        refreshKey={ticketRefreshKey}
        onClose={onClose}
      />
    );
  }
  // RFC / Debrief — fetch from the workspace's stored doc ref.
  const ref = workspace.docs?.[slot.kind];
  return (
    <ConfluencePageViewer
      baseUrl={baseUrl}
      kind={slot.kind}
      workspace={workspace}
      pageRef={ref}
      refreshKey={slot.kind === "rfc" ? rfcRefreshKey : debriefRefreshKey}
      onClose={onClose}
    />
  );
}

// ────────────────────────────────────────────────────────────────────────────
// DocChrome — shared aside frame so every viewer has the same geometry,
// motion, and header. Body content varies.
// ────────────────────────────────────────────────────────────────────────────

function DocChrome({
  kind,
  title,
  subtitle,
  meta,
  externalUrl,
  onClose,
  children,
}: {
  kind: "ticket" | "rfc" | "debrief";
  title: string;
  subtitle?: string | null;
  meta?: React.ReactNode;
  externalUrl?: string | null;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const accent =
    kind === "ticket" ? "var(--accent-warm)" : kind === "rfc" ? "var(--accent-cool)" : "var(--accent-ok)";
  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 630, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.24, ease: EASE_OUT }}
      className="shrink-0 border-l border-line bg-canvas overflow-hidden flex flex-col min-h-0"
    >
      <div
        className="px-4 py-3 border-b border-line flex items-start justify-between gap-3 shrink-0"
        data-tauri-drag-region
      >
        <div className="min-w-0" data-tauri-drag-region>
          <div className="flex items-baseline gap-2" data-tauri-drag-region>
            <span className="text-[10px] uppercase tracking-[0.14em] font-mono" style={{ color: accent }}>
              {kind}
            </span>
            {subtitle && <span className="text-[10px] font-mono text-ink-faint truncate">{subtitle}</span>}
          </div>
          <div className="text-[14px] text-ink mt-0.5 truncate" data-tauri-drag-region>{title}</div>
          {meta && <div className="mt-0.5 text-[10px] text-ink-faint flex items-center gap-2" data-tauri-drag-region>{meta}</div>}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {externalUrl && (
            <a
              href={externalUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-center size-7 rounded text-ink-faint hover:text-ink hover:bg-surface-sunk transition-colors duration-100 ease-out active:scale-[0.95]"
              title="Open in Atlassian"
              aria-label="Open in Atlassian"
            >
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                <path d="M5 3H3.5C3.22 3 3 3.22 3 3.5V9.5C3 9.78 3.22 10 3.5 10H9.5C9.78 10 10 9.78 10 9.5V8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                <path d="M7 3H10V6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M6 7L10 3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
            </a>
          )}
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center justify-center size-7 rounded text-ink-faint hover:text-ink hover:bg-surface-sunk transition-colors duration-100 ease-out active:scale-[0.95]"
            aria-label="Close"
            title="Close"
          >
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
              <path d="M2.5 2.5L8.5 8.5M8.5 2.5L2.5 8.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto scroll-quiet">
        {children}
      </div>
    </motion.aside>
  );
}

const PROSE_CLASSES =
  "prose prose-sm max-w-none px-5 py-4 prose-headings:font-serif prose-headings:text-ink " +
  "prose-p:text-ink prose-li:text-ink prose-strong:text-ink " +
  "prose-code:text-ink prose-code:bg-surface-sunk prose-code:px-1 prose-code:py-0.5 prose-code:rounded " +
  "prose-code:before:hidden prose-code:after:hidden " +
  "prose-a:text-[color:var(--accent-cool)] prose-a:no-underline hover:prose-a:underline " +
  "prose-pre:bg-surface-sunk prose-pre:border prose-pre:border-line " +
  "prose-th:text-ink prose-td:text-ink-soft prose-th:border-line prose-td:border-line " +
  "prose-blockquote:text-ink-soft prose-hr:border-line";

// ────────────────────────────────────────────────────────────────────────────
// TicketViewer — fetches /app/workflow/ticket-meta + the full /ticket
// endpoint. Renders title, status pill, description body (markdown when
// JIRA returns ADF, raw when not).
// ────────────────────────────────────────────────────────────────────────────

interface TicketBody {
  key: string;
  title: string;
  status: string | null;
  issuetype: string | null;
  description_adf: unknown;
  url: string | null;
}

function useTicketBody(baseUrl: string, key: string, externalRefreshKey: number): {
  state: ResolveState<TicketBody>;
  reload: () => void;
} {
  const [state, setState] = useState<ResolveState<TicketBody>>({ kind: "idle" });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!key) { setState({ kind: "idle" }); return; }
    setState({ kind: "loading" });
    const ctrl = new AbortController();
    (async () => {
      try {
        const r = await fetch(`${baseUrl}/app/workflow/ticket/${encodeURIComponent(key)}`, { signal: ctrl.signal });
        if (r.status === 400) {
          const text = await r.text().catch(() => "");
          if (text.includes("requires_credentials")) { setState({ kind: "missing_creds" }); return; }
          setState({ kind: "error", message: text || "Not found" });
          return;
        }
        if (!r.ok) { setState({ kind: "error", message: `HTTP ${r.status}` }); return; }
        const raw = await r.json();
        const f = raw?.fields ?? {};
        const meta = await fetch(`${baseUrl}/app/workflow/ticket-meta/${encodeURIComponent(key)}`).then((r) => r.ok ? r.json() : null).catch(() => null);
        setState({
          kind: "ok",
          value: {
            key: String(raw?.key ?? key),
            title: f.summary ?? "",
            status: typeof f.status?.name === "string" ? f.status.name : null,
            issuetype: typeof f.issuetype?.name === "string" ? f.issuetype.name : null,
            description_adf: f.description ?? null,
            url: meta?.url ?? null,
          },
        });
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        setState({ kind: "error", message: String(e?.message ?? e) });
      }
    })();
    return () => ctrl.abort();
  }, [baseUrl, key, tick, externalRefreshKey]);
  return { state, reload: () => setTick((t) => t + 1) };
}

function TicketViewer({ baseUrl, ticketKey, refreshKey, onClose }: { baseUrl: string; ticketKey: string; refreshKey: number; onClose: () => void }) {
  const { state, reload } = useTicketBody(baseUrl, ticketKey, refreshKey);
  const ok = state.kind === "ok" ? state.value : null;

  // Render the ADF description as best-effort plain markdown. The real
  // round-trip lives server-side in apps/server/adf — for now we just
  // walk the ADF and emit headings + paragraphs + lists so the user gets
  // something legible in the panel.
  const bodyMarkdown = useMemo(() => ok?.description_adf ? adfToPlainMarkdown(ok.description_adf) : "", [ok?.description_adf]);

  return (
    <DocChrome
      kind="ticket"
      title={ok?.title || ticketKey}
      subtitle={ticketKey}
      meta={
        ok && (
          <>
            {ok.status && (
              <span className="font-mono uppercase tracking-[0.14em]" style={{ color: "var(--accent-warm)" }}>
                {ok.status}
              </span>
            )}
            {ok.issuetype && <span className="text-ink-faint">· {ok.issuetype}</span>}
            <button type="button" onClick={reload} className="ml-1 text-ink-faint hover:text-ink underline">refresh</button>
          </>
        )
      }
      externalUrl={ok?.url ?? null}
      onClose={onClose}
    >
      <ViewerBody state={state}>
        {bodyMarkdown ? (
          <div className={PROSE_CLASSES}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{bodyMarkdown}</ReactMarkdown>
          </div>
        ) : (
          <div className="px-5 py-6 text-[13px] text-ink-faint italic">(no description on this ticket)</div>
        )}
      </ViewerBody>
    </DocChrome>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// ConfluencePageViewer — RFC + Debrief share this. Reads the workspace's
// stored doc ref (set by workflow_write_*) and fetches the page metadata.
// Until the markdown round-trip lands client-side, we surface a clear empty
// state for "not created yet" and a deep link to open the page in Confluence.
// ────────────────────────────────────────────────────────────────────────────

interface PageBody {
  id: string;
  title: string;
  version: number;
  url: string | null;
  body_adf: unknown;
}

function usePageBody(baseUrl: string, pageId: string | null, refreshKey: number): ResolveState<PageBody> {
  const [state, setState] = useState<ResolveState<PageBody>>({ kind: "idle" });
  useEffect(() => {
    if (!pageId) { setState({ kind: "idle" }); return; }
    setState({ kind: "loading" });
    const ctrl = new AbortController();
    (async () => {
      try {
        const r = await fetch(`${baseUrl}/app/workflow/page/${encodeURIComponent(pageId)}`, { signal: ctrl.signal });
        if (r.status === 400) {
          const text = await r.text().catch(() => "");
          if (text.includes("requires_credentials")) { setState({ kind: "missing_creds" }); return; }
          setState({ kind: "error", message: text || "Not found" });
          return;
        }
        if (!r.ok) { setState({ kind: "error", message: `HTTP ${r.status}` }); return; }
        const data = await r.json();
        setState({
          kind: "ok",
          value: {
            id: data.id,
            title: data.title,
            version: data.version,
            url: data.url,
            body_adf: data.body_adf ?? null,
          },
        });
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        setState({ kind: "error", message: String(e?.message ?? e) });
      }
    })();
    return () => ctrl.abort();
  }, [baseUrl, pageId, refreshKey]);
  return state;
}

function ConfluencePageViewer({
  baseUrl,
  kind,
  workspace,
  pageRef,
  refreshKey,
  onClose,
}: {
  baseUrl: string;
  kind: "rfc" | "debrief";
  workspace: WorkspaceDTO;
  pageRef: WorkspaceDocRefDTO | undefined;
  refreshKey: number;
  onClose: () => void;
}) {
  const pageId = pageRef?.page_id ?? null;
  const state = usePageBody(baseUrl, pageId, refreshKey);
  const ok = state.kind === "ok" ? state.value : null;
  const label = kind === "rfc" ? "RFC" : "Debrief";

  if (!pageId) {
    // Not created yet — clear empty state.
    return (
      <DocChrome
        kind={kind}
        title={`No ${label} yet`}
        subtitle={workspace.ticket_key}
        onClose={onClose}
      >
        <div className="px-5 py-6 text-[13px] text-ink-muted leading-relaxed">
          The agent will create the {label} as a child of this initiative's
          Confluence root page when you ask it to. Try:
          <div className="mt-3 px-3 py-2 rounded-md bg-surface-sunk font-mono text-[12px] text-ink">
            Draft {kind === "rfc" ? "an RFC" : "a debrief"} for {workspace.ticket_key}.
          </div>
        </div>
      </DocChrome>
    );
  }

  return (
    <DocChrome
      kind={kind}
      title={ok?.title || pageRef?.title || `${label} for ${workspace.ticket_key}`}
      subtitle={pageId}
      meta={
        ok && (
          <>
            <span className="font-mono">v{ok.version}</span>
            {pageRef?.last_synced_at && (
              <span className="text-ink-faint">· synced {relativeTime(pageRef.last_synced_at)}</span>
            )}
          </>
        )
      }
      externalUrl={ok?.url ?? pageRef?.url ?? null}
      onClose={onClose}
    >
      <ViewerBody state={state}>
        <ConfluencePageBodyRenderer body={ok?.body_adf} url={ok?.url ?? null} />
      </ViewerBody>
    </DocChrome>
  );
}

// Shared loading/error states for any viewer body.
function ConfluencePageBodyRenderer({ body, url }: { body: unknown; url: string | null }) {
  // Empty body — render a soft empty state instead of a blank pane.
  const isEmpty =
    !body ||
    typeof body !== "object" ||
    ((body as any).type === "doc" && !((body as any).content?.length));
  if (isEmpty) {
    return (
      <div className="px-5 py-6 text-[13px] text-ink-muted italic">
        (page exists but has no body content
        {url && (
          <>
            {" "}— <a className="not-italic underline" href={url} target="_blank" rel="noreferrer">open in Confluence</a>
          </>
        )}
        )
      </div>
    );
  }
  const md = useMemo(() => adfToPlainMarkdown(body), [body]);
  return (
    <div className={PROSE_CLASSES}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{md}</ReactMarkdown>
    </div>
  );
}

function ViewerBody<T>({ state, children }: { state: ResolveState<T>; children: React.ReactNode }) {
  if (state.kind === "loading") {
    return <div className="px-5 py-6 text-[13px] text-ink-faint italic">Loading…</div>;
  }
  if (state.kind === "missing_creds") {
    return <div className="px-5 py-6 text-[13px] text-[color:var(--accent-warn)]">Atlassian credentials not configured. Connect from the sidebar.</div>;
  }
  if (state.kind === "error") {
    return <div className="px-5 py-6 text-[13px] text-[color:var(--accent-err)] font-mono">{state.message}</div>;
  }
  if (state.kind === "idle") return null;
  return <>{children}</>;
}

// Cheap ADF → plain markdown so the ticket description renders something
// readable. Server already has a full round-trip; this is the client-side
// MVP and intentionally lossy — JIRA's inlineCards become bare links.
function adfToPlainMarkdown(node: unknown): string {
  if (!node || typeof node !== "object") return "";
  const n = node as Record<string, unknown>;
  if (n.type === "doc") {
    return arrayJoin((n.content as unknown[]) ?? [], "\n\n", adfToPlainMarkdown);
  }
  if (n.type === "paragraph") {
    return arrayJoin((n.content as unknown[]) ?? [], "", adfInline);
  }
  if (n.type === "heading") {
    const level = Math.max(1, Math.min(6, Number((n.attrs as any)?.level || 1)));
    return "#".repeat(level) + " " + arrayJoin((n.content as unknown[]) ?? [], "", adfInline);
  }
  if (n.type === "bulletList") {
    return ((n.content as unknown[]) ?? []).map((li) => "- " + adfPlainItem(li)).join("\n");
  }
  if (n.type === "orderedList") {
    return ((n.content as unknown[]) ?? []).map((li, i) => `${i + 1}. ` + adfPlainItem(li)).join("\n");
  }
  if (n.type === "codeBlock") {
    const lang = (n.attrs as any)?.language ?? "";
    const text = ((n.content as unknown[]) ?? []).map((c: any) => c?.text ?? "").join("");
    return "```" + lang + "\n" + text + "\n```";
  }
  if (n.type === "rule") return "---";
  if (n.type === "blockquote") {
    const inner = arrayJoin((n.content as unknown[]) ?? [], "\n", adfToPlainMarkdown);
    return inner.split("\n").map((l) => "> " + l).join("\n");
  }
  if (n.type === "panel") {
    // Render as a GFM blockquote prefixed with a small emoji / glyph so
    // ReactMarkdown shows it visually distinct from regular blockquotes.
    const ptype = String((n.attrs as any)?.panelType || "info").toLowerCase();
    const glyph = ({ info: "ℹ︎", note: "✎", warning: "⚠︎", success: "✓", error: "✗" } as Record<string, string>)[ptype] || "ℹ︎";
    const inner = arrayJoin((n.content as unknown[]) ?? [], "\n\n", adfToPlainMarkdown);
    return inner.split("\n").map((l, i) => i === 0 ? `> **${glyph} ${ptype.toUpperCase()}** — ${l}` : `> ${l}`).join("\n");
  }
  if (n.type === "taskList") {
    return ((n.content as unknown[]) ?? []).map((it: any) => {
      const state = String(it?.attrs?.state || "TODO").toUpperCase();
      const marker = state === "DONE" ? "[x]" : "[ ]";
      const body = arrayJoin((it?.content as unknown[]) ?? [], "", adfInline);
      return `- ${marker} ${body}`;
    }).join("\n");
  }
  if (n.type === "table") {
    const rows = ((n.content as unknown[]) ?? []) as any[];
    if (rows.length === 0) return "";
    const out: string[] = [];
    let headerEmitted = false;
    rows.forEach((row, ri) => {
      const cells = ((row?.content as unknown[]) ?? []) as any[];
      const rendered = cells.map((c) => {
        const inner = arrayJoin((c?.content as unknown[]) ?? [], " ", adfToPlainMarkdown);
        return inner.replace(/\n/g, " ").trim() || " ";
      });
      out.push("| " + rendered.join(" | ") + " |");
      if (ri === 0 && cells.some((c) => c?.type === "tableHeader")) {
        out.push("|" + rendered.map(() => "---").join("|") + "|");
        headerEmitted = true;
      }
    });
    if (!headerEmitted && out.length > 0) {
      const cols = out[0]!.split("|").length - 2;
      out.splice(1, 0, "|" + new Array(cols).fill("---").join("|") + "|");
    }
    return out.join("\n");
  }
  return "";
}

function adfInline(node: unknown): string {
  if (!node || typeof node !== "object") return "";
  const n = node as Record<string, unknown>;
  if (n.type === "text") {
    let text = String(n.text ?? "");
    const marks = ((n.marks as unknown[]) ?? []) as Array<{ type?: string; attrs?: any }>;
    for (const m of marks) {
      if (m.type === "code") text = `\`${text}\``;
      else if (m.type === "strong") text = `**${text}**`;
      else if (m.type === "em") text = `*${text}*`;
      else if (m.type === "link" && typeof m.attrs?.href === "string") text = `[${text}](${m.attrs.href})`;
    }
    return text;
  }
  if (n.type === "inlineCard" && typeof (n.attrs as any)?.url === "string") {
    return (n.attrs as any).url;
  }
  if (n.type === "status") {
    // Render as inline code with the status text — visually distinct without
    // needing a custom React component yet.
    const attrs = (n.attrs as any) ?? {};
    return `\`${attrs.text ?? "status"}\``;
  }
  if (n.type === "date" && (n.attrs as any)?.timestamp) {
    const ts = Number((n.attrs as any).timestamp);
    if (Number.isFinite(ts)) {
      return new Date(ts).toISOString().slice(0, 10);
    }
  }
  if (n.type === "mention" && typeof (n.attrs as any)?.text === "string") {
    return (n.attrs as any).text;
  }
  if (n.type === "emoji" && typeof (n.attrs as any)?.shortName === "string") {
    return (n.attrs as any).shortName;
  }
  if (n.type === "hardBreak") return "  \n";
  return "";
}

function adfPlainItem(li: unknown): string {
  if (!li || typeof li !== "object") return "";
  return arrayJoin((((li as any).content) ?? []) as unknown[], " ", adfToPlainMarkdown);
}

function arrayJoin<T>(arr: T[], sep: string, fn: (x: T) => string): string {
  return arr.map(fn).filter(Boolean).join(sep);
}

// ────────────────────────────────────────────────────────────────────────────
// RightRail — narrow column hosting TodoSidebar + DocsPanel (the three doc
// tiles). Always mounted when there's a workspace, regardless of doc-slot
// state. The wide DocSlotSidebar mounts SEPARATELY to its left.
// ────────────────────────────────────────────────────────────────────────────

type DocPickTarget =
  | { kind: "plan" }
  | { kind: "ticket"; ticketKey: string }
  | { kind: "rfc"; workspaceId: string }
  | { kind: "debrief"; workspaceId: string };

function RightRail({
  todos,
  workspace,
  docSlot,
  hasPlan,
  planLocked,
  onPickDoc,
}: {
  todos: TodoItem[] | null;
  workspace: WorkspaceDTO;
  docSlot: DocSlotState;
  hasPlan: boolean;
  planLocked: boolean;
  onPickDoc: (target: DocPickTarget) => void;
}) {
  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 280, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.24, ease: EASE_OUT }}
      className="shrink-0 border-l border-line bg-canvas overflow-hidden flex flex-col min-h-0"
    >
      {todos && todos.length > 0 && <TodoSidebarBody todos={todos} />}
      <DocsPanel
        workspace={workspace}
        docSlot={docSlot}
        hasPlan={hasPlan}
        planLocked={planLocked}
        onPick={onPickDoc}
      />
    </motion.aside>
  );
}

function DocsPanel({
  workspace,
  docSlot,
  hasPlan,
  planLocked,
  onPick,
}: {
  workspace: WorkspaceDTO;
  docSlot: DocSlotState;
  hasPlan: boolean;
  planLocked: boolean;
  onPick: (target: DocPickTarget) => void;
}) {
  const planActive = docSlot.kind === "plan";
  const ticketActive = docSlot.kind === "ticket" && docSlot.ticketKey === workspace.ticket_key;
  const rfcActive = docSlot.kind === "rfc" && docSlot.workspaceId === workspace.id;
  const debriefActive = docSlot.kind === "debrief" && docSlot.workspaceId === workspace.id;
  const rfcDoc = workspace.docs?.rfc;
  const debriefDoc = workspace.docs?.debrief;
  return (
    <div className="border-t border-line shrink-0">
      <div className="px-4 py-3 flex items-baseline justify-between">
        <span className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-faint">
          Documents
        </span>
      </div>
      <ul className="pb-3 space-y-px">
        {/* Plan lives at the top while it's relevant — disappears once
            the agent isn't planning and there's no plan history. The lock
            badge shows when plan-mode is forcing the slot open. */}
        {(hasPlan || planLocked) && (
          <DocTile
            glyph="◆"
            accent="var(--accent-cool)"
            label={planLocked ? "Plan (locked)" : "Plan"}
            subtitle={planLocked ? "Active while in plan mode" : "Latest agent plan"}
            active={planActive}
            onClick={() => onPick({ kind: "plan" })}
          />
        )}
        <DocTile
          glyph="★"
          accent="var(--accent-warm)"
          label="Ticket"
          subtitle={workspace.ticket_key}
          active={ticketActive}
          onClick={() => onPick({ kind: "ticket", ticketKey: workspace.ticket_key })}
        />
        <DocTile
          glyph="✦"
          accent="var(--accent-cool)"
          label="RFC"
          subtitle={rfcDoc?.page_id ? `v${rfcDoc.version ?? "?"}` : "— not created"}
          active={rfcActive}
          onClick={() => onPick({ kind: "rfc", workspaceId: workspace.id })}
        />
        <DocTile
          glyph="✦"
          accent="var(--accent-ok)"
          label="Debrief"
          subtitle={debriefDoc?.page_id ? `v${debriefDoc.version ?? "?"}` : "— not created"}
          active={debriefActive}
          onClick={() => onPick({ kind: "debrief", workspaceId: workspace.id })}
        />
      </ul>
    </div>
  );
}

function DocTile({
  glyph,
  accent,
  label,
  subtitle,
  active,
  onClick,
}: {
  glyph: string;
  accent: string;
  label: string;
  subtitle: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={`w-full text-left px-4 py-2 flex items-center gap-3 transition-colors ${
          active ? "bg-surface-tinted" : "hover:bg-surface-sunk"
        }`}
      >
        <span className="text-[12px] w-4 text-center shrink-0" style={{ color: accent }} aria-hidden>{glyph}</span>
        <span className="flex-1 min-w-0">
          <span className="block text-[12px] text-ink">{label}</span>
          <span className="block text-[10px] font-mono text-ink-faint truncate">{subtitle}</span>
        </span>
      </button>
    </li>
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
              prose-pre:bg-surface-sunk prose-pre:border prose-pre:border-line
              prose-th:text-ink prose-td:text-ink-soft prose-blockquote:text-ink-soft"
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
            className="px-3 py-1.5 rounded-md text-[12px] font-medium text-canvas transition-opacity"
            style={{ background: "var(--ink)" }}
          >
            Approve & run
          </motion.button>
        </div>
      )}
    </motion.aside>
  );
}

// Standalone TodoSidebar kept for legacy callers (none after the RightRail
// refactor — but harmless to keep around as a reference).
function TodoSidebar({ todos }: { todos: TodoItem[] }) {
  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 280, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.24, ease: EASE_OUT }}
      className="shrink-0 border-l border-line bg-canvas overflow-hidden flex flex-col min-h-0"
    >
      <TodoSidebarBody todos={todos} />
    </motion.aside>
  );
}

/** Inner body (no <motion.aside>) — usable inside another motion container. */
function TodoSidebarBody({ todos }: { todos: TodoItem[] }) {
  const done = todos.filter((t) => t.status === "completed").length;
  return (
    <>
      <div className="px-4 py-3 border-b border-line flex items-baseline justify-between shrink-0">
        <span className="text-[11px] uppercase tracking-[0.14em] font-mono text-ink-faint">
          To-do
        </span>
        <span className="text-[10px] font-mono text-ink-faint">
          {done}/{todos.length}
        </span>
      </div>
      <ul className="flex-1 overflow-y-auto scroll-quiet py-2 px-3 space-y-1 min-h-0">
        {todos.map((t, i) => (
          <TodoRow key={`${i}-${t.content}`} todo={t} />
        ))}
      </ul>
    </>
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
          className="w-full px-3 py-1.5 rounded-md text-[13px] font-medium text-canvas disabled:opacity-40 transition-opacity flex items-center justify-center gap-1.5"
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

// ────────────────────────────────────────────────────────────────────────────
// TicketSidebar — the new left rail. Lists workspaces (tickets) instead of
// raw sessions. Status pill rolls up across all sessions in the workspace.
// More-actions kebab replaces the old delete X-button.
// ────────────────────────────────────────────────────────────────────────────

function TicketSidebar({
  workspaces,
  initiatives,
  sessions,
  sessionStates,
  acks,
  activeWorkspaceId,
  atlassianMeta,
  onSelect,
  onOpenCreate,
  onOpenInitiatives,
  onOpenArchive,
  onOpenSettings,
  onArchive,
  onUnarchive,
  onDelete,
}: {
  workspaces: WorkspaceDTO[];
  initiatives: InitiativeDTO[];
  sessions: StoredSession[];
  sessionStates: AgentMux["sessions"];
  acks: AppAcks;
  activeWorkspaceId: string | null;
  atlassianMeta: AtlassianCredsMeta;
  onSelect: (id: string) => void;
  onOpenCreate: () => void;
  onOpenInitiatives: () => void;
  onOpenArchive: () => void;
  onOpenSettings: () => void;
  onArchive: (id: string) => Promise<void>;
  onUnarchive: (id: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const live = workspaces.filter((w) => w.archived_at === null);
  const archived = workspaces.filter((w) => w.archived_at !== null);

  // Roll session statuses up into a per-workspace category. Most-attention-
  // worthy session wins: needs_input > working > idle.
  const wsCategory = (w: WorkspaceDTO): SessionGroup => {
    let best: SessionGroup = "idle";
    for (const sid of w.session_ids) {
      const cat = categorizeSession(sessionStates[sid], acks.map[sid]);
      if (cat === "needs_input") return "needs_input";
      if (cat === "working") best = "working";
    }
    return best;
  };

  const groups: Record<SessionGroup, WorkspaceDTO[]> = {
    working: [],
    needs_input: [],
    idle: [],
  };
  for (const w of live) groups[wsCategory(w)].push(w);

  const sections: { key: SessionGroup; label: string; list: WorkspaceDTO[] }[] = [
    { key: "needs_input", label: "Needs input", list: groups.needs_input },
    { key: "working", label: "Working", list: groups.working },
    { key: "idle", label: "Idle", list: groups.idle },
  ];

  return (
    <aside className="w-[280px] shrink-0 border-r border-line bg-canvas flex flex-col">
      {/* Title-bar strip — only present in the Tauri shell when the
          window is not in fullscreen. Acts as the native title-bar
          handle: drag to move, double-click to zoom. The drag is
          wired by useTauriShell(); this attribute is just the marker. */}
      <div
        className="hidden tauri-windowed:block h-7 shrink-0"
        data-tauri-drag-region
      />
      <div className="px-4 py-4 border-b border-line">
        <div className="flex items-baseline gap-2 mb-3">
          <span className="font-serif text-xl leading-none italic tracking-tight">
            blitzcode
          </span>
          <span className="font-mono text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--accent-cool)" }}>
            pro
          </span>
        </div>
        <motion.button
          type="button"
          onClick={onOpenCreate}
          whileTap={{ scale: 0.97 }}
          transition={{ duration: 0.12, ease: EASE_OUT }}
          className="w-full px-3 py-1.5 rounded-md text-[13px] font-medium bg-ink text-canvas dark:bg-surface-tinted dark:text-ink-soft dark:hover:text-ink dark:border dark:border-line transition-colors flex items-center justify-center gap-1.5"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden>
            <path d="M5 1V9M1 5H9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          Add ticket
        </motion.button>
      </div>

      {/* Bottom-left utility rail: initiatives + archived + atlassian.
          Lives at the bottom of the aside (pushed by flex-1 list above)
          so it's a stable anchor regardless of how many tickets exist. */}

      <div className="flex-1 overflow-y-auto scroll-quiet py-2">
        {live.length === 0 ? (
          <div className="px-4 py-6 text-xs text-ink-faint">
            No tickets yet. Add one to start.
          </div>
        ) : (
          <div className="space-y-2">
            {sections.map((sec) =>
              sec.list.length === 0 ? null : (
                <SidebarSection key={sec.key} label={sec.label} count={sec.list.length}>
                  <ul className="space-y-px">
                    {sec.list.map((w) => (
                      <TicketRow
                        key={w.id}
                        workspace={w}
                        status={wsCategory(w)}
                        active={w.id === activeWorkspaceId}
                        onSelect={() => onSelect(w.id)}
                        onArchive={() => onArchive(w.id)}
                        onDelete={() => onDelete(w.id)}
                      />
                    ))}
                  </ul>
                </SidebarSection>
              )
            )}
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-line px-1.5 py-1.5 flex items-center justify-between gap-1 text-[11px]">
        <div className="flex items-center gap-0.5 min-w-0">
          <SidebarUtilButton onClick={onOpenInitiatives} label={`Initiatives (${initiatives.length})`} />
          <SidebarUtilButton
            onClick={onOpenArchive}
            label={`Archived (${archived.length})`}
            disabled={archived.length === 0}
            title={archived.length === 0 ? "Nothing archived yet" : `${archived.length} archived ticket${archived.length === 1 ? "" : "s"}`}
          />
        </div>
        <div className="flex items-center gap-0.5 shrink-0">
          <motion.button
            type="button"
            onClick={onOpenSettings}
            whileTap={{ scale: 0.94 }}
            transition={{ duration: 0.1, ease: EASE_OUT }}
            title={atlassianMeta.has_creds ? "Settings" : "Settings — Atlassian not connected"}
            aria-label="Open settings"
            className="relative inline-flex items-center justify-center size-7 rounded-md text-ink-faint hover:text-ink hover:bg-surface-sunk transition-[color,background-color] duration-100 ease-out"
          >
            <CogIcon />
            {!atlassianMeta.has_creds && (
              <span
                className="absolute top-0.5 right-0.5 size-1.5 rounded-full"
                style={{ background: "var(--accent-warn)" }}
                aria-hidden
              />
            )}
          </motion.button>
        </div>
      </div>
    </aside>
  );
}

function SidebarUtilButton({
  label,
  onClick,
  title,
  disabled,
  leadingDotColor,
  tone = "default",
}: {
  label: string;
  onClick: () => void;
  title?: string;
  disabled?: boolean;
  leadingDotColor?: string;
  tone?: "default" | "warn";
}) {
  const baseColor = tone === "warn" ? "text-[color:var(--accent-warn)]" : "text-ink-soft";
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={disabled}
      whileTap={disabled ? undefined : { scale: 0.97 }}
      transition={{ duration: 0.1, ease: EASE_OUT }}
      title={title}
      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] cursor-pointer ${baseColor} hover:text-ink hover:bg-surface-sunk transition-[color,background-color] duration-100 ease-out disabled:cursor-default disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:${baseColor.replace("text-", "text-")}`}
    >
      {leadingDotColor && (
        <span
          className="inline-block size-1.5 rounded-full shrink-0"
          style={{ background: leadingDotColor }}
          aria-hidden
        />
      )}
      <span>{label}</span>
    </motion.button>
  );
}

function TicketRow({
  workspace,
  status,
  active,
  archived,
  onSelect,
  onArchive,
  onDelete,
}: {
  workspace: WorkspaceDTO;
  status: SessionGroup;
  active: boolean;
  archived?: boolean;
  onSelect: () => void;
  onArchive: () => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [hover, setHover] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [menuOpen]);

  return (
    <li
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className={`relative group mx-2 rounded-md transition-colors ${
        active ? "bg-surface-tinted" : "hover:bg-surface-sunk"
      } ${archived ? "opacity-60" : ""}`}
    >
      <button type="button" onClick={onSelect} className="w-full text-left px-2.5 py-2 pr-9">
        <div className="flex items-baseline gap-1.5">
          <span className="font-mono text-[11px] text-ink-soft">{workspace.ticket_key}</span>
          {status !== "idle" && <StatusPipDot status={status} />}
        </div>
        <div className="text-[13px] text-ink leading-tight truncate mt-0.5">
          {workspace.ticket_title || (
            <span className="italic text-ink-faint">untitled</span>
          )}
        </div>
        <div className="text-[10px] font-mono text-ink-faint truncate mt-0.5">
          {workspace.repos.length} repo{workspace.repos.length === 1 ? "" : "s"} ·{" "}
          {workspace.session_ids.length} session{workspace.session_ids.length === 1 ? "" : "s"}
        </div>
      </button>
      <div ref={menuRef} className="absolute right-1.5 top-1.5">
        {(hover || active || menuOpen) && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              setMenuOpen((v) => !v);
            }}
            className="size-6 rounded flex items-center justify-center text-ink-faint hover:text-ink-soft hover:bg-canvas transition-colors"
            aria-label="More"
            title="More actions"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
              <circle cx="3" cy="7" r="1.2" fill="currentColor" />
              <circle cx="7" cy="7" r="1.2" fill="currentColor" />
              <circle cx="11" cy="7" r="1.2" fill="currentColor" />
            </svg>
          </button>
        )}
        <AnimatePresence>
          {menuOpen && (
            <motion.div
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.12, ease: EASE_OUT }}
              className="absolute right-0 mt-1 w-[180px] rounded-md border border-line bg-canvas shadow-xl z-30 overflow-hidden"
              role="menu"
            >
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  void navigator.clipboard.writeText(workspace.dir);
                  setMenuOpen(false);
                }}
                className="w-full text-left px-3 py-2 text-[12px] hover:bg-surface-sunk transition-colors"
              >
                Copy workspace path
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  void onArchive();
                  setMenuOpen(false);
                }}
                className="w-full text-left px-3 py-2 text-[12px] hover:bg-surface-sunk transition-colors"
              >
                {archived ? "Unarchive" : "Archive"}
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Delete workspace for ${workspace.ticket_key}? This removes worktrees and the workspace dir.`)) {
                    void onDelete();
                  }
                  setMenuOpen(false);
                }}
                className="w-full text-left px-3 py-2 text-[12px] hover:bg-surface-sunk transition-colors"
                style={{ color: "var(--accent-err)" }}
              >
                Delete workspace…
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </li>
  );
}

function StatusPipDot({ status }: { status: SessionGroup }) {
  if (status === "working") {
    return (
      <motion.span
        className="inline-block size-1.5 rounded-full"
        style={{ background: "var(--accent-warm)" }}
        animate={{ opacity: [0.35, 1, 0.35] }}
        transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
        title="Working"
        aria-label="Working"
      />
    );
  }
  if (status === "needs_input") {
    return (
      <motion.span
        className="inline-block size-1.5 rounded-full"
        style={{ background: "var(--accent-cool)" }}
        animate={{ opacity: [0.55, 1, 0.55] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        title="Needs input"
        aria-label="Needs input"
      />
    );
  }
  return null;
}

// ────────────────────────────────────────────────────────────────────────────
// WorkspaceCreateModal — ticket key + initiative + repos picker
// ────────────────────────────────────────────────────────────────────────────

function WorkspaceCreateModal({
  baseUrl,
  initiatives,
  hasCreds,
  onRequestCreds,
  creating,
  onCancel,
  onCreate,
}: {
  baseUrl: string;
  initiatives: InitiativeDTO[];
  hasCreds: boolean;
  onRequestCreds: () => void;
  creating: boolean;
  onCancel: () => void;
  onCreate: (args: CreateWorkspaceArgs) => Promise<void>;
}) {
  const [ticketKey, setTicketKey] = useState("");
  const [ticketTitle, setTicketTitle] = useState("");
  const [titleManuallyEdited, setTitleManuallyEdited] = useState(false);
  const [initiativeKey, setInitiativeKey] = useState<string>("");
  const [repos, setRepos] = useState<{ source_path: string; branch?: string }[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Live ticket resolution drives the picker's pill state. Once the user
  // picks (or types a valid key with creds), the picker collapses into a
  // rich pill — title/status/link inline, no second-row validation.
  const ticketResolve = useResolveTicket(baseUrl, ticketKey);

  // Auto-populate title from the resolved ticket if the user hasn't typed
  // their own. Cheap convenience — every ticket worth opening a workspace
  // for already has a real title in JIRA.
  useEffect(() => {
    if (titleManuallyEdited) return;
    if (ticketResolve.kind === "ok" && ticketResolve.value.title) {
      setTicketTitle(ticketResolve.value.title);
    }
  }, [ticketResolve, titleManuallyEdited]);

  // When initiative changes, pre-seed repos from its known list.
  useEffect(() => {
    if (!initiativeKey) return;
    const init = initiatives.find((i) => i.key === initiativeKey);
    if (!init) return;
    setRepos((prev) => {
      const seen = new Set(prev.map((r) => r.source_path));
      const out = [...prev];
      for (const p of init.repo_paths) {
        if (!seen.has(p)) out.push({ source_path: p });
      }
      return out;
    });
  }, [initiativeKey, initiatives]);

  const validKey = /^[A-Z][A-Z0-9]*-\d+$/.test(ticketKey);
  const canCreate = validKey && repos.length > 0 && !creating;

  const submit = async () => {
    setError(null);
    try {
      await onCreate({
        ticket_key: ticketKey.toUpperCase(),
        ticket_title: ticketTitle.trim() || undefined,
        initiative_key: initiativeKey || undefined,
        repos,
        spawn_initial_session: true,
        permission_mode: "acceptEdits",
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="bg-canvas border border-line rounded-xl shadow-xl overflow-hidden flex flex-col" style={{ width: "min(560px, 92vw)", maxHeight: "min(620px, 85vh)" }}>
      <div className="px-5 py-4 border-b border-line shrink-0">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">New ticket</div>
        <h2 className="font-serif text-xl leading-tight text-ink mt-0.5">Create workspace</h2>
      </div>
      <div className="flex-1 overflow-y-auto scroll-quiet px-5 py-4 space-y-4">
        <div>
          <div className="flex items-baseline justify-between mb-1">
            <span className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">Ticket</span>
            {!hasCreds && (
              <button type="button" onClick={onRequestCreds} className="text-[10px] text-[color:var(--accent-warn)] hover:text-ink transition-colors duration-100 ease-out">
                Connect Atlassian for typeahead
              </button>
            )}
          </div>
          <SearchPicker
            value={ticketKey}
            display={ticketResolve}
            onChange={(id, item) => {
              setTicketKey(id.toUpperCase());
              if (item && !titleManuallyEdited && item.label) {
                setTicketTitle(item.label);
              }
            }}
            fetcher={async (q, signal) => {
              const results = await searchTickets(baseUrl, q, signal);
              return results.map((r) => ({
                id: r.key,
                label: r.title,
                sublabel: r.status ?? undefined,
              }));
            }}
            placeholder={hasCreds ? "Search by ticket key or title…" : "LLM-1234"}
            emptyHint={hasCreds ? "Type to search, or paste an exact key." : "Enter the ticket key in PROJ-123 format."}
            monoValue
          />
        </div>
        <div>
          <span className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">Ticket title</span>
          <input
            type="text"
            value={ticketTitle}
            onChange={(e) => {
              setTicketTitle(e.target.value);
              setTitleManuallyEdited(true);
            }}
            placeholder={ticketResolve.kind === "loading" ? "Auto-filling…" : "Short summary (auto-filled if found)"}
            className="mt-1 w-full px-3 py-2 rounded-md border border-line bg-surface text-ink text-[14px] outline-none focus:border-line-strong transition-[border-color] duration-150 ease-out"
          />
        </div>
        <label className="block">
          <span className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">Initiative (optional)</span>
          <select
            value={initiativeKey}
            onChange={(e) => setInitiativeKey(e.target.value)}
            className="mt-1 w-full px-3 py-2 rounded-md border border-line bg-surface text-ink text-[14px] focus:outline-none focus:border-line-strong"
          >
            <option value="">— None —</option>
            {initiatives.map((i) => (
              <option key={i.key} value={i.key}>{i.display_name}</option>
            ))}
          </select>
        </label>

        <div>
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">Repos</span>
            <button
              type="button"
              onClick={() => setPickerOpen(true)}
              className="text-[11px] text-ink-soft hover:text-ink transition-colors"
            >
              + Add repo
            </button>
          </div>
          {repos.length === 0 ? (
            <div className="mt-1 text-[12px] text-ink-faint italic">Pick at least one repo to create worktrees from.</div>
          ) : (
            <>
              <div className="mt-1 text-[10px] text-ink-faint">
                Each repo gets a new git worktree on branch{" "}
                <span className="font-mono text-ink-soft">{ticketKey || "<ticket-key>"}</span>.
              </div>
              <ul className="mt-2 space-y-1">
                {repos.map((r, i) => (
                  <li key={r.source_path} className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-surface text-[12px]">
                    <span className="font-mono text-ink truncate flex-1" title={r.source_path}>{r.source_path}</span>
                    <button
                      type="button"
                      onClick={() => setRepos((prev) => prev.filter((_, j) => j !== i))}
                      className="text-ink-faint hover:text-ink transition-colors duration-100 ease-out active:scale-[0.95]"
                      aria-label="Remove"
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>

        {error && (
          <div className="text-[12px] text-[color:var(--accent-err)] font-mono">{error}</div>
        )}
      </div>
      <div className="px-5 py-3 border-t border-line flex items-center justify-end gap-2 bg-surface shrink-0">
        <button type="button" onClick={onCancel} className="px-3 py-1.5 rounded-md text-[12px] text-ink-soft hover:text-ink hover:bg-surface-sunk transition-colors">
          Cancel
        </button>
        <motion.button
          type="button"
          onClick={submit}
          disabled={!canCreate}
          whileTap={canCreate ? { scale: 0.97 } : undefined}
          transition={{ duration: 0.12, ease: EASE_OUT }}
          className="px-3 py-1.5 rounded-md text-[12px] font-medium text-canvas disabled:opacity-40 transition-opacity"
          style={{ background: "var(--ink)" }}
        >
          {creating ? "Creating…" : "Create workspace"}
        </motion.button>
      </div>
      <AnimatePresence>
        {pickerOpen && (
          <Modal>
            <FolderPicker
              baseUrl={baseUrl}
              onPick={(p) => {
                setRepos((prev) => {
                  if (prev.some((r) => r.source_path === p)) return prev;
                  return [...prev, { source_path: p }];
                });
                setPickerOpen(false);
              }}
              onCancel={() => setPickerOpen(false)}
              confirmLabel="Add repo"
            />
          </Modal>
        )}
      </AnimatePresence>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// InitiativeManager — minimal CRUD UI for the initiative list.
// ────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────
// CredsModal — Atlassian site + email + API token entry. Token is never
// rendered after submit; modal closes and meta refreshes.
// ────────────────────────────────────────────────────────────────────────────

function CredsModal({ creds, onClose }: { creds: UseAtlassianCreds; onClose: () => void }) {
  const [siteUrl, setSiteUrl] = useState(creds.meta.site_url ?? "");
  const [email, setEmail] = useState(creds.meta.email ?? "");
  const [token, setToken] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setSaving(true);
    setError(null);
    try {
      await creds.set({ site_url: siteUrl.trim(), email: email.trim(), api_token: token });
      setToken("");
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-canvas border border-line rounded-xl shadow-xl overflow-hidden flex flex-col" style={{ width: "min(520px, 92vw)" }}>
      <div className="px-5 py-4 border-b border-line shrink-0">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">Atlassian</div>
        <h2 className="font-serif text-xl leading-tight text-ink mt-0.5">
          {creds.meta.has_creds ? "Update credentials" : "Connect your Atlassian account"}
        </h2>
        <p className="mt-1 text-[12px] text-ink-muted">
          Enables ticket typeahead, JIRA reads/writes, and Confluence RFC/debrief writes. The token is stored locally (0600 perms) and never sent anywhere else.
        </p>
      </div>
      <div className="px-5 py-4 space-y-3">
        <label className="block">
          <span className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">Site URL</span>
          <input type="text" value={siteUrl} onChange={(e) => setSiteUrl(e.target.value)} placeholder="https://your-site.atlassian.net" className="mt-1 w-full px-3 py-2 rounded-md border border-line bg-surface text-ink font-mono text-[13px] focus:outline-none focus:border-line-strong" />
        </label>
        <label className="block">
          <span className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">Email</span>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" className="mt-1 w-full px-3 py-2 rounded-md border border-line bg-surface text-ink text-[13px] focus:outline-none focus:border-line-strong" />
        </label>
        <label className="block">
          <span className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">API token</span>
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder={creds.meta.has_creds ? "(unchanged — enter to replace)" : "atlassian API token"}
            className="mt-1 w-full px-3 py-2 rounded-md border border-line bg-surface text-ink font-mono text-[13px] focus:outline-none focus:border-line-strong"
          />
          <a
            href="https://id.atlassian.com/manage-profile/security/api-tokens"
            target="_blank"
            rel="noreferrer"
            className="mt-1 inline-block text-[11px] text-ink-soft hover:text-ink underline"
          >
            Generate one →
          </a>
        </label>
        {error && <div className="text-[11px] text-[color:var(--accent-err)] font-mono">{error}</div>}
      </div>
      <div className="px-5 py-3 border-t border-line flex items-center justify-between gap-2 bg-surface shrink-0">
        <div>
          {creds.meta.has_creds && (
            <button
              type="button"
              onClick={() => {
                if (confirm("Forget Atlassian credentials?")) {
                  void creds.clear().then(onClose);
                }
              }}
              className="text-[11px] text-[color:var(--accent-err)] hover:text-ink"
            >
              Forget creds
            </button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={onClose} className="px-3 py-1.5 rounded-md text-[12px] text-ink-soft hover:text-ink hover:bg-surface-sunk transition-colors">
            Cancel
          </button>
          <motion.button
            type="button"
            onClick={submit}
            disabled={saving || !siteUrl.trim() || !email.trim() || (!creds.meta.has_creds && !token.trim())}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.12, ease: EASE_OUT }}
            className="px-3 py-1.5 rounded-md text-[12px] font-medium text-canvas disabled:opacity-40 transition-opacity"
            style={{ background: "var(--ink)" }}
          >
            {saving ? "Saving…" : "Save"}
          </motion.button>
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// SearchPicker — generic typeahead combobox.
// One implementation, reused for JIRA-ticket search and Confluence-page
// search in the Initiative Manager. Caller provides a fetcher; component
// owns debounce, loading, error state, and keyboard nav.
// ────────────────────────────────────────────────────────────────────────────

export interface SearchPickerItem {
  id: string;        // the canonical value stored when picked (e.g. ticket key or page id)
  label: string;     // primary display line
  sublabel?: string; // muted secondary line (e.g. status, url)
  url?: string;      // optional "open ↗" link
}

function SearchPicker({
  value,
  display,
  onChange,
  fetcher,
  placeholder,
  emptyHint,
  monoValue,
  inputNormalize,
}: {
  value: string;
  /** Optional resolver state. When .kind !== "idle" and value is set, the
   *  picker renders an inline "pill" (the resolved record IS the validation).
   *  When undefined or idle, it stays in input mode. */
  display?: ResolveState<{ title: string; url?: string | null; status?: string | null }>;
  onChange: (id: string, item?: SearchPickerItem) => void;
  fetcher: (q: string, signal: AbortSignal) => Promise<SearchPickerItem[]>;
  placeholder: string;
  emptyHint: string;
  monoValue?: boolean;
  inputNormalize?: (raw: string) => string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState(value);
  const [results, setResults] = useState<SearchPickerItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [missingCreds, setMissingCreds] = useState(false);
  const [hoverIdx, setHoverIdx] = useState(0);
  // When the user explicitly enters edit mode on a resolved pill, we stay
  // in the input until they pick again or click outside.
  const [editing, setEditing] = useState<boolean>(!value);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Keep query in sync when caller resets value externally.
  useEffect(() => { setQuery(value); }, [value]);
  // Cleared value → drop back to edit mode.
  useEffect(() => { if (!value) setEditing(true); }, [value]);
  // Display newly resolved → drop out of editing (only if the user wasn't
  // mid-search; we don't yank focus away while they're picking).
  useEffect(() => {
    if (display?.kind === "ok" && value && !open) setEditing(false);
  }, [display?.kind, value, open]);

  // Debounced fetch.
  useEffect(() => {
    if (!open) return;
    const q = query.trim();
    if (!q) { setResults([]); setLoading(false); return; }
    setLoading(true);
    setMissingCreds(false);
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      try {
        const items = await fetcher(q, ctrl.signal);
        setResults(items);
        setHoverIdx(0);
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        if (e instanceof RequiresCredsError) {
          setMissingCreds(true);
          setResults([]);
        }
      } finally {
        setLoading(false);
      }
    }, 280);
    return () => { ctrl.abort(); clearTimeout(t); };
  }, [open, query, fetcher]);

  // Click outside → close + drop editing if the value resolved successfully.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
        if (display?.kind === "ok" && value) setEditing(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open, display, value]);

  const pick = (item: SearchPickerItem) => {
    onChange(item.id, item);
    setQuery(item.id);
    setOpen(false);
    setEditing(false);
  };

  const enterEdit = () => {
    setEditing(true);
    setOpen(true);
    requestAnimationFrame(() => {
      const el = inputRef.current;
      if (el) { el.focus(); el.select(); }
    });
  };

  // Pill mode: value is set AND we have a non-idle display AND the user
  // hasn't explicitly entered edit mode.
  const showPill = !editing && !!value && !!display && display.kind !== "idle";

  return (
    <div ref={wrapperRef} className="relative">
      <AnimatePresence mode="wait" initial={false}>
        {showPill ? (
          <motion.div
            key="pill"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.18, ease: EASE_OUT }}
          >
            <PickerPill
              value={value}
              display={display!}
              onEdit={enterEdit}
              onClear={() => { onChange(""); setEditing(true); }}
            />
          </motion.div>
        ) : (
          <motion.input
            key="input"
            ref={inputRef as any}
            type="text"
            value={query}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.18, ease: EASE_OUT }}
            onFocus={() => setOpen(true)}
            onChange={(e) => {
              const v = inputNormalize ? inputNormalize(e.target.value) : e.target.value;
              setQuery(v);
              onChange(v);
              setOpen(true);
            }}
            onKeyDown={(e) => {
              if (!open) return;
              if (e.key === "Escape") { e.preventDefault(); setOpen(false); return; }
              if (results.length === 0) return;
              if (e.key === "ArrowDown") { e.preventDefault(); setHoverIdx((i) => Math.min(i + 1, results.length - 1)); return; }
              if (e.key === "ArrowUp") { e.preventDefault(); setHoverIdx((i) => Math.max(i - 1, 0)); return; }
              if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); pick(results[hoverIdx]!); return; }
            }}
            placeholder={placeholder}
            className={`w-full px-3 py-2 rounded-md border border-line bg-surface text-ink text-[13px] outline-none focus:border-line-strong transition-[border-color,background-color] duration-150 ease-out ${monoValue ? "font-mono" : ""}`}
          />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {!showPill && open && (loading || results.length > 0 || missingCreds || query.trim()) && (
          <motion.div
            initial={{ opacity: 0, y: -2, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -2, scale: 0.98 }}
            transition={{ duration: 0.14, ease: EASE_OUT }}
            style={{ transformOrigin: "top center" }}
            className="absolute left-0 right-0 mt-1 z-20 rounded-md border border-line bg-canvas shadow-xl overflow-hidden max-h-[240px] overflow-y-auto scroll-quiet"
            role="listbox"
          >
            {missingCreds && (
              <div className="px-3 py-2 text-[11px] text-[color:var(--accent-warn)]">
                Connect Atlassian to search.
              </div>
            )}
            {loading && results.length === 0 && !missingCreds && (
              <div className="px-3 py-2 text-[11px] text-ink-faint italic">Searching…</div>
            )}
            {!loading && results.length === 0 && !missingCreds && query.trim() && (
              <div className="px-3 py-2 text-[11px] text-ink-faint">No matches.</div>
            )}
            {results.map((item, i) => (
              <button
                key={item.id}
                type="button"
                role="option"
                aria-selected={i === hoverIdx}
                onMouseDown={(e) => { e.preventDefault(); pick(item); }}
                onMouseEnter={() => setHoverIdx(i)}
                className={`w-full text-left px-3 py-2 transition-colors duration-100 ease-out flex flex-col gap-0.5 active:scale-[0.99] ${
                  i === hoverIdx ? "bg-surface-tinted" : "hover:bg-surface-sunk"
                }`}
              >
                <div className="flex items-baseline gap-2">
                  <span className="text-[13px] text-ink truncate flex-1">
                    {item.label || <span className="italic text-ink-faint">(untitled)</span>}
                  </span>
                  <span className="font-mono text-[10px] text-ink-faint shrink-0">{item.id}</span>
                </div>
                {item.sublabel && (
                  <div className="text-[10px] font-mono text-ink-faint truncate">{item.sublabel}</div>
                )}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
      {!showPill && !open && (
        <div className="text-[10px] text-ink-faint mt-1">{emptyHint}</div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// PickerPill — the resolved-state inline chip. Same vertical slot as the
// input (no reflow). The pill ITSELF communicates validation: status dot
// color carries the meaning, no separate row beneath. Click body → edit.
// ────────────────────────────────────────────────────────────────────────────

function PickerPill({
  value,
  display,
  onEdit,
  onClear,
}: {
  value: string;
  display: ResolveState<{ title: string; url?: string | null; status?: string | null }>;
  onEdit: () => void;
  onClear: () => void;
}) {
  // Visual state per resolver kind. Keep the geometry identical across
  // states — only the dot color + title text vary — so the pill never
  // jumps height during a loading→ok transition.
  const tone =
    display.kind === "ok"
      ? { dot: "var(--accent-ok)", pulse: false, border: "var(--line)", title: display.value.title || "(untitled)" }
      : display.kind === "loading"
      ? { dot: "var(--accent-warm)", pulse: true, border: "var(--line)", title: "Resolving…" }
      : display.kind === "missing_creds"
      ? { dot: "var(--accent-warn)", pulse: false, border: "var(--accent-warn)", title: "Connect Atlassian to verify" }
      : display.kind === "error"
      ? { dot: "var(--accent-err)", pulse: false, border: "var(--accent-err)", title: display.message || "Not found" }
      : { dot: "var(--ink-faint)", pulse: false, border: "var(--line)", title: value };

  const status = display.kind === "ok" ? display.value.status : null;
  const url = display.kind === "ok" ? display.value.url : null;

  return (
    <div
      className="w-full rounded-md bg-surface flex items-center gap-2.5 pl-3 pr-1 py-1.5 transition-[border-color,background-color] duration-150 ease-out"
      style={{
        border: `1px solid ${tone.border}`,
        boxShadow: display.kind === "ok"
          ? `0 0 0 2px color-mix(in oklch, var(--accent-ok) 8%, transparent), inset 0 1px 0 rgba(255,255,255,0.4)`
          : undefined,
      }}
    >
      {/* Status dot — color carries the validation state, pulses while loading. */}
      <span
        className="relative inline-flex items-center justify-center shrink-0"
        style={{ width: 8, height: 8 }}
        aria-hidden
      >
        {tone.pulse && (
          <motion.span
            className="absolute inset-0 rounded-full"
            style={{ background: tone.dot, opacity: 0.35 }}
            animate={{ scale: [1, 1.8, 1], opacity: [0.35, 0, 0.35] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
          />
        )}
        <span
          className="inline-block rounded-full"
          style={{ background: tone.dot, width: 6, height: 6 }}
        />
      </span>

      {/* Click body to edit. Generous hit target. */}
      <button
        type="button"
        onClick={onEdit}
        className="flex-1 min-w-0 text-left flex items-baseline gap-2 group active:scale-[0.995] transition-transform duration-100 ease-out"
        aria-label="Edit selection"
      >
        <span
          className={`truncate text-[13px] ${
            display.kind === "ok" ? "text-ink" : display.kind === "error" ? "text-[color:var(--accent-err)]" : "text-ink-soft"
          }`}
        >
          {tone.title}
        </span>
        {value && display.kind === "ok" && (
          <span className="font-mono text-[10px] text-ink-faint shrink-0">{value}</span>
        )}
        {status && (
          <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint shrink-0">{status}</span>
        )}
      </button>

      {/* Open in new tab — only when we have a destination. */}
      {url && (
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="shrink-0 inline-flex items-center justify-center size-7 rounded text-ink-faint hover:text-ink hover:bg-surface-sunk transition-colors duration-100 ease-out active:scale-[0.95]"
          aria-label="Open in new tab"
          title="Open"
        >
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M5 3H3.5C3.22 3 3 3.22 3 3.5V9.5C3 9.78 3.22 10 3.5 10H9.5C9.78 10 10 9.78 10 9.5V8" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            <path d="M7 3H10V6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M6 7L10 3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
        </a>
      )}

      {/* Clear */}
      <button
        type="button"
        onClick={onClear}
        className="shrink-0 inline-flex items-center justify-center size-7 rounded text-ink-faint hover:text-ink hover:bg-surface-sunk transition-colors duration-100 ease-out active:scale-[0.95]"
        aria-label="Clear selection"
        title="Clear"
      >
        <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
          <path d="M2.5 2.5L8.5 8.5M8.5 2.5L2.5 8.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}

// ── Confluence page search fetcher (parallels searchTickets) ──────────────

interface ConfluencePageResult { id: string; title: string; url: string | null }

async function searchConfluencePages(baseUrl: string, q: string, signal?: AbortSignal): Promise<ConfluencePageResult[]> {
  if (!q.trim()) return [];
  const base = baseUrl || "";
  const url = new URL(`${base}/app/workflow/search-pages`, window.location.origin);
  url.searchParams.set("q", q);
  const r = await fetch(url.toString(), { signal });
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    if (r.status === 400 && text.includes("requires_credentials")) throw new RequiresCredsError();
    throw new Error(text || `HTTP ${r.status}`);
  }
  const data = (await r.json()) as { results: ConfluencePageResult[] };
  return data.results ?? [];
}

// ── Live validation lookups for Initiative Manager inputs ──────────────────

type ResolveState<T> =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ok"; value: T }
  | { kind: "missing_creds" }
  | { kind: "error"; message: string };

function useResolveTicket(baseUrl: string, key: string): ResolveState<{ key: string; title: string; status: string | null; url: string | null }> {
  const [state, setState] = useState<ResolveState<{ key: string; title: string; status: string | null; url: string | null }>>({ kind: "idle" });
  useEffect(() => {
    const k = key.trim();
    if (!k) { setState({ kind: "idle" }); return; }
    if (!/^[A-Z][A-Z0-9]*-\d+$/.test(k)) { setState({ kind: "idle" }); return; }
    setState({ kind: "loading" });
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`${baseUrl}/app/workflow/ticket-meta/${encodeURIComponent(k)}`, { signal: ctrl.signal });
        if (r.status === 400) {
          const text = await r.text().catch(() => "");
          if (text.includes("requires_credentials")) { setState({ kind: "missing_creds" }); return; }
          setState({ kind: "error", message: "not found" });
          return;
        }
        if (!r.ok) { setState({ kind: "error", message: `HTTP ${r.status}` }); return; }
        const data = await r.json();
        setState({ kind: "ok", value: { key: data.key, title: data.title, status: data.status, url: data.url } });
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        setState({ kind: "error", message: String(e?.message ?? e) });
      }
    }, 320);
    return () => { ctrl.abort(); clearTimeout(t); };
  }, [baseUrl, key]);
  return state;
}

function useResolveConfluencePage(baseUrl: string, pageId: string): ResolveState<{ id: string; title: string; url: string | null }> {
  const [state, setState] = useState<ResolveState<{ id: string; title: string; url: string | null }>>({ kind: "idle" });
  useEffect(() => {
    const id = pageId.trim();
    if (!id) { setState({ kind: "idle" }); return; }
    if (!/^\d+$/.test(id)) { setState({ kind: "idle" }); return; }
    setState({ kind: "loading" });
    const ctrl = new AbortController();
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`${baseUrl}/app/workflow/page/${encodeURIComponent(id)}?include_body=0`, { signal: ctrl.signal });
        if (r.status === 400) {
          const text = await r.text().catch(() => "");
          if (text.includes("requires_credentials")) { setState({ kind: "missing_creds" }); return; }
          setState({ kind: "error", message: "not found" });
          return;
        }
        if (!r.ok) { setState({ kind: "error", message: `HTTP ${r.status}` }); return; }
        const data = await r.json();
        setState({ kind: "ok", value: { id: data.id, title: data.title, url: data.url } });
      } catch (e: any) {
        if (e?.name === "AbortError") return;
        setState({ kind: "error", message: String(e?.message ?? e) });
      }
    }, 320);
    return () => { ctrl.abort(); clearTimeout(t); };
  }, [baseUrl, pageId]);
  return state;
}

function ResolveRow({ state, emptyHint, onAutofill }: { state: ResolveState<{ title: string; url?: string | null; status?: string | null }>; emptyHint: string; onAutofill?: (title: string) => void }) {
  if (state.kind === "idle") {
    return <div className="text-[10px] text-ink-faint mt-0.5">{emptyHint}</div>;
  }
  if (state.kind === "loading") {
    return <div className="text-[10px] text-ink-faint mt-0.5 italic">checking…</div>;
  }
  if (state.kind === "missing_creds") {
    return <div className="text-[10px] text-[color:var(--accent-warn)] mt-0.5">Connect Atlassian to verify</div>;
  }
  if (state.kind === "error") {
    return <div className="text-[10px] text-[color:var(--accent-err)] mt-0.5 font-mono">✗ {state.message}</div>;
  }
  return (
    <div className="text-[10px] mt-0.5 flex items-center gap-2">
      <span style={{ color: "var(--accent-ok)" }}>✓</span>
      <span className="text-ink-soft truncate flex-1">{state.value.title || "(untitled)"}</span>
      {state.value.status && <span className="font-mono text-ink-faint">{state.value.status}</span>}
      {state.value.url && (
        <a href={state.value.url} target="_blank" rel="noreferrer" className="text-ink-faint hover:text-ink underline">open ↗</a>
      )}
      {onAutofill && state.value.title && (
        <button type="button" onClick={() => onAutofill(state.value.title!)} className="text-ink-faint hover:text-ink">use as name</button>
      )}
    </div>
  );
}

function normalizeInitiativeKey(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/_/g, "-")
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// ────────────────────────────────────────────────────────────────────────────
// ArchiveModal — search + restore archived workspaces. No "delete forever"
// here intentionally; that's a per-row gesture from the live sidebar's
// kebab menu so the same action lives in one place across UIs.
// ────────────────────────────────────────────────────────────────────────────

function ArchiveModal({
  workspaces,
  onRestore,
  onDelete,
  onOpen,
  onClose,
}: {
  workspaces: WorkspaceDTO[];
  onRestore: (id: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onOpen: (id: string) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  // Filter on the same fields the sidebar surfaces.
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return workspaces;
    return workspaces.filter((w) =>
      w.ticket_key.toLowerCase().includes(q) ||
      (w.ticket_title ?? "").toLowerCase().includes(q) ||
      (w.initiative_key ?? "").toLowerCase().includes(q)
    );
  }, [workspaces, query]);

  const wrap = (id: string, fn: () => Promise<void>) => async () => {
    setBusyId(id);
    try { await fn(); } finally { setBusyId(null); }
  };

  return (
    <div
      className="bg-canvas border border-line rounded-xl shadow-xl overflow-hidden flex flex-col"
      style={{ width: "min(560px, 92vw)", height: "min(560px, 80vh)" }}
    >
      <div className="px-5 py-4 border-b border-line shrink-0 flex items-baseline justify-between">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">Archived</div>
          <h2 className="font-serif text-xl leading-tight text-ink mt-0.5">Restore a ticket</h2>
        </div>
        <button type="button" onClick={onClose} className="text-ink-faint hover:text-ink transition-colors duration-100 ease-out" aria-label="Close">×</button>
      </div>

      <div className="px-5 py-3 border-b border-line shrink-0">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by ticket key, title, or initiative…"
          autoFocus
          className="w-full px-3 py-2 rounded-md border border-line bg-surface text-ink text-[13px] outline-none focus:border-line-strong transition-[border-color] duration-150 ease-out"
        />
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto scroll-quiet">
        {workspaces.length === 0 ? (
          <div className="px-5 py-10 text-center text-[13px] text-ink-faint italic">
            Nothing archived yet.
          </div>
        ) : filtered.length === 0 ? (
          <div className="px-5 py-10 text-center text-[13px] text-ink-faint italic">
            No matches.
          </div>
        ) : (
          <ul>
            {filtered.map((w) => (
              <li
                key={w.id}
                className="px-5 py-3 border-b border-line last:border-b-0 flex items-center gap-3"
              >
                <button
                  type="button"
                  onClick={() => onOpen(w.id)}
                  className="flex-1 min-w-0 text-left group"
                  title="Open this workspace (still archived — restore to put back in the live list)"
                >
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-[11px] text-ink-soft">{w.ticket_key}</span>
                    {w.initiative_key && (
                      <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-ink-faint">
                        {w.initiative_key}
                      </span>
                    )}
                  </div>
                  <div className="text-[13px] text-ink leading-tight truncate mt-0.5 group-hover:text-[color:var(--accent-cool)] transition-colors duration-100 ease-out">
                    {w.ticket_title || <span className="italic text-ink-faint">untitled</span>}
                  </div>
                  <div className="text-[10px] font-mono text-ink-faint truncate mt-0.5">
                    {w.repos.length} repo{w.repos.length === 1 ? "" : "s"} · archived {relativeTime(w.archived_at ?? 0)}
                  </div>
                </button>

                <button
                  type="button"
                  onClick={wrap(w.id, async () => onRestore(w.id))}
                  disabled={busyId === w.id}
                  className="shrink-0 px-2.5 py-1 rounded-md text-[11px] text-ink-soft hover:text-ink hover:bg-surface-sunk transition-colors duration-100 ease-out active:scale-[0.97] disabled:opacity-40"
                  title="Move back to the live sidebar"
                >
                  Restore
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (confirm(`Permanently delete workspace for ${w.ticket_key}? Worktrees and the workspace dir will be removed.`)) {
                      void wrap(w.id, async () => onDelete(w.id))();
                    }
                  }}
                  disabled={busyId === w.id}
                  className="shrink-0 px-2.5 py-1 rounded-md text-[11px] transition-colors duration-100 ease-out active:scale-[0.97] disabled:opacity-40"
                  style={{ color: "var(--accent-err)" }}
                  title="Delete forever — worktrees and disk"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// SettingsModal — two-pane: category list on the left, options on the right.
// Settings persist on disk via /app/settings; the hook optimistically merges
// before the round-trip so toggles feel instant.
// ────────────────────────────────────────────────────────────────────────────

type SettingsCategory = "appearance" | "atlassian";

const SETTINGS_CATEGORIES: { key: SettingsCategory; label: string; description: string }[] = [
  { key: "appearance", label: "Appearance", description: "Theme, density, typography" },
  { key: "atlassian", label: "Atlassian", description: "JIRA + Confluence credentials" },
  // Future: { key: "workspaces", label: "Workspaces", description: "Defaults for new tickets" },
  // Future: { key: "agent", label: "Agent", description: "Permission mode defaults, model" },
];

function SettingsModal({
  settings,
  creds,
  initialCategory,
  onClose,
}: {
  settings: UseSettings;
  creds: UseAtlassianCreds;
  initialCategory?: SettingsCategory;
  onClose: () => void;
}) {
  const [active, setActive] = useState<SettingsCategory>(initialCategory ?? "appearance");
  return (
    <div
      className="bg-canvas border border-line rounded-xl shadow-xl overflow-hidden flex flex-col"
      style={{ width: "min(800px, 94vw)", height: "min(560px, 82vh)" }}
    >
      <div className="px-5 py-3 border-b border-line shrink-0 flex items-baseline justify-between">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">Settings</span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-ink-faint hover:text-ink transition-colors duration-100 ease-out"
          aria-label="Close"
        >
          ×
        </button>
      </div>
      <div className="flex-1 min-h-0 flex">
        {/* Left pane: categories */}
        <nav className="w-[200px] shrink-0 border-r border-line bg-surface-sunk/40 overflow-y-auto scroll-quiet py-2">
          <ul className="space-y-px px-1.5">
            {SETTINGS_CATEGORIES.map((c) => {
              const dot =
                c.key === "atlassian"
                  ? creds.meta.has_creds
                    ? "var(--accent-ok)"
                    : "var(--accent-warn)"
                  : null;
              return (
                <li key={c.key}>
                  <button
                    type="button"
                    onClick={() => setActive(c.key)}
                    className={`w-full text-left px-3 py-2 rounded-md transition-colors duration-100 ease-out ${
                      active === c.key
                        ? "bg-surface-tinted text-ink"
                        : "text-ink-soft hover:text-ink hover:bg-surface-sunk"
                    }`}
                  >
                    <div className="text-[13px] flex items-center gap-1.5">
                      {dot && (
                        <span
                          className="inline-block size-1.5 rounded-full shrink-0"
                          style={{ background: dot }}
                          aria-hidden
                        />
                      )}
                      {c.label}
                    </div>
                    <div className="text-[10px] text-ink-faint mt-0.5 truncate">{c.description}</div>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
        {/* Right pane: content */}
        <div className="flex-1 min-w-0 overflow-y-auto scroll-quiet px-6 py-6">
          {active === "appearance" && (
            <AppearanceSettings settings={settings} />
          )}
          {active === "atlassian" && (
            <AtlassianSettings creds={creds} />
          )}
        </div>
      </div>
      <div className="px-5 py-2.5 border-t border-line shrink-0 text-[10px] text-ink-faint flex items-center justify-end">
        <span className="font-mono">~/.agent-webkit/blitzcode-pro/settings.json</span>
      </div>
    </div>
  );
}

function AppearanceSettings({ settings }: { settings: UseSettings }) {
  return (
    <section>
      <h2 className="font-serif text-2xl text-ink leading-tight tracking-tight">Appearance</h2>
      <p className="mt-1 text-[13px] text-ink-muted">
        How blitzcode-pro looks. Changes save automatically.
      </p>

      <div className="mt-6">
        <SettingRow
          label="Theme"
          hint="Match the system, or pin a fixed appearance."
        >
          <ThemeSegmented
            value={settings.theme}
            onChange={(theme) => void settings.patch({ appearance: { theme } })}
          />
        </SettingRow>
      </div>
    </section>
  );
}

function AtlassianSettings({ creds }: { creds: UseAtlassianCreds }) {
  const [siteUrl, setSiteUrl] = useState(creds.meta.site_url ?? "");
  const [email, setEmail] = useState(creds.meta.email ?? "");
  const [token, setToken] = useState("");
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [forgetting, setForgetting] = useState(false);

  // Re-sync when the hook's meta refreshes after a save/clear elsewhere.
  useEffect(() => {
    setSiteUrl(creds.meta.site_url ?? "");
    setEmail(creds.meta.email ?? "");
  }, [creds.meta.site_url, creds.meta.email]);

  const dirty =
    siteUrl.trim() !== (creds.meta.site_url ?? "") ||
    email.trim() !== (creds.meta.email ?? "") ||
    token.trim().length > 0;
  const canSave =
    !saving &&
    dirty &&
    siteUrl.trim().length > 0 &&
    email.trim().length > 0 &&
    (creds.meta.has_creds || token.trim().length > 0);

  const submit = async () => {
    setSaving(true);
    setError(null);
    try {
      await creds.set({ site_url: siteUrl.trim(), email: email.trim(), api_token: token });
      setToken("");
      setSavedAt(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const forget = async () => {
    setForgetting(true);
    try {
      await creds.clear();
      setSiteUrl("");
      setEmail("");
      setToken("");
    } finally {
      setForgetting(false);
    }
  };

  return (
    <section>
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <h2 className="font-serif text-2xl text-ink leading-tight tracking-tight">Atlassian</h2>
          <p className="mt-1 text-[13px] text-ink-muted">
            Powers ticket typeahead, JIRA reads/writes, and Confluence RFC/debrief writes.
          </p>
        </div>
        <ConnectionPill connected={creds.meta.has_creds} />
      </div>

      <div className="mt-6 space-y-4">
        <FieldRow label="Site URL">
          <input
            type="text"
            value={siteUrl}
            onChange={(e) => setSiteUrl(e.target.value)}
            placeholder="https://your-site.atlassian.net"
            className="w-full px-3 py-2 rounded-md border border-line bg-surface text-ink font-mono text-[13px] focus:outline-none focus:border-line-strong"
          />
        </FieldRow>
        <FieldRow label="Email">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="w-full px-3 py-2 rounded-md border border-line bg-surface text-ink text-[13px] focus:outline-none focus:border-line-strong"
          />
        </FieldRow>
        <FieldRow
          label="API token"
          hint={
            <>
              Stored locally (chmod 0600), never sent anywhere else.{" "}
              <a
                href="https://id.atlassian.com/manage-profile/security/api-tokens"
                target="_blank"
                rel="noreferrer"
                className="text-[color:var(--accent-cool)] hover:underline"
              >
                Generate one ↗
              </a>
            </>
          }
        >
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder={creds.meta.has_creds ? "(unchanged — type to replace)" : "atlassian API token"}
            className="w-full px-3 py-2 rounded-md border border-line bg-surface text-ink font-mono text-[13px] focus:outline-none focus:border-line-strong"
          />
        </FieldRow>
      </div>

      {error && (
        <div className="mt-4 px-3 py-2 rounded-md bg-[color:var(--tint-err)] text-[color:var(--accent-err)] text-[12px] font-mono">
          {error}
        </div>
      )}

      <div className="mt-6 flex items-center justify-between gap-2">
        <div className="text-[11px] text-ink-faint">
          {savedAt && !dirty ? "Saved." : creds.meta.has_creds ? "Connected." : "Not connected."}
        </div>
        <div className="flex items-center gap-2">
          {creds.meta.has_creds && (
            <button
              type="button"
              onClick={() => { if (confirm("Forget Atlassian credentials?")) void forget(); }}
              disabled={forgetting}
              className="px-3 py-1.5 rounded-md text-[12px] text-[color:var(--accent-err)] hover:bg-[color:var(--tint-err)] transition-colors disabled:opacity-40"
            >
              {forgetting ? "Forgetting…" : "Forget creds"}
            </button>
          )}
          <motion.button
            type="button"
            onClick={() => void submit()}
            disabled={!canSave}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.12, ease: EASE_OUT }}
            className="px-3 py-1.5 rounded-md text-[12px] font-medium bg-ink text-canvas dark:bg-surface-tinted dark:text-ink-soft dark:hover:text-ink dark:border dark:border-line transition-colors disabled:opacity-40"
          >
            {saving ? "Saving…" : "Save"}
          </motion.button>
        </div>
      </div>
    </section>
  );
}

function ConnectionPill({ connected }: { connected: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono uppercase tracking-[0.14em] ${
        connected
          ? "bg-[color:var(--tint-ok)] text-[color:var(--accent-ok)]"
          : "bg-[color:var(--tint-warn)] text-[color:var(--accent-warn)]"
      }`}
    >
      <span
        className="inline-block size-1.5 rounded-full"
        style={{ background: connected ? "var(--accent-ok)" : "var(--accent-warn)" }}
        aria-hidden
      />
      {connected ? "Connected" : "Not connected"}
    </span>
  );
}

function FieldRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="block text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint mb-1.5">
        {label}
      </span>
      {children}
      {hint && <span className="block mt-1.5 text-[11px] text-ink-faint leading-relaxed">{hint}</span>}
    </label>
  );
}

function SettingRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="py-4 border-b border-line first:border-t flex items-start justify-between gap-6">
      <div className="min-w-0 flex-1 max-w-[280px]">
        <div className="text-[13px] text-ink">{label}</div>
        {hint && <div className="text-[11px] text-ink-faint mt-1 leading-relaxed">{hint}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function ThemeSegmented({
  value,
  onChange,
}: {
  value: ThemePreference;
  onChange: (next: ThemePreference) => void;
}) {
  const options: { key: ThemePreference; label: string; icon: React.ReactNode }[] = [
    {
      key: "light",
      label: "Light",
      icon: (
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
          <circle cx="6" cy="6" r="2.2" stroke="currentColor" strokeWidth="1.2" />
          <path d="M6 1.4v1.4M6 9.2v1.4M1.4 6h1.4M9.2 6h1.4M2.7 2.7l1 1M8.3 8.3l1 1M9.3 2.7l-1 1M3.7 8.3l-1 1" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
        </svg>
      ),
    },
    {
      key: "dark",
      label: "Dark",
      icon: (
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
          <path d="M10 7.2A4.5 4.5 0 1 1 4.8 2A3.6 3.6 0 0 0 10 7.2Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
        </svg>
      ),
    },
    {
      key: "system",
      label: "System",
      icon: (
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
          <rect x="1.5" y="2" width="9" height="6" rx="1" stroke="currentColor" strokeWidth="1.2" />
          <path d="M4.5 10.5h3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
      ),
    },
  ];
  return (
    <div className="inline-flex items-center gap-0.5 p-0.5 rounded-md border border-line bg-surface">
      {options.map((opt) => {
        const active = value === opt.key;
        return (
          <motion.button
            key={opt.key}
            type="button"
            onClick={() => onChange(opt.key)}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.1, ease: EASE_OUT }}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded text-[11px] transition-[color,background-color] duration-120 ease-out ${
              active
                ? "bg-surface-tinted text-ink"
                : "text-ink-soft hover:text-ink hover:bg-surface-sunk"
            }`}
            aria-pressed={active}
          >
            {opt.icon}
            <span>{opt.label}</span>
          </motion.button>
        );
      })}
    </div>
  );
}

function InitiativeManager({
  baseUrl,
  initiatives,
  onClose,
}: {
  baseUrl: string;
  initiatives: UseInitiatives;
  onClose: () => void;
}) {
  const [displayName, setDisplayName] = useState("");
  const [keyOverride, setKeyOverride] = useState<string | null>(null);
  const [epicKey, setEpicKey] = useState("");
  const [rootPage, setRootPage] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Key auto-derives from display name unless the user has typed something
  // explicit into the key field. Live-normalized either way.
  const derivedKey = normalizeInitiativeKey(displayName);
  const key = keyOverride ?? derivedKey;

  // Live validation against Atlassian for the two fields that are easy to
  // mistype: epic JIRA key and Confluence root page id. Both debounce
  // ~320ms and degrade gracefully when no creds are configured.
  const epicState = useResolveTicket(baseUrl, epicKey);
  const pageState = useResolveConfluencePage(baseUrl, rootPage);

  const submit = async () => {
    setError(null);
    if (!key) {
      setError("Please enter a display name (or a key directly).");
      return;
    }
    try {
      await initiatives.upsert({
        key,
        display_name: displayName.trim() || key,
        epic_jira_key: epicKey.trim() || null,
        confluence_root_page_id: rootPage.trim() || null,
        repo_paths: [],
      });
      setDisplayName(""); setKeyOverride(null); setEpicKey(""); setRootPage("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="bg-canvas border border-line rounded-xl shadow-xl overflow-hidden flex flex-col" style={{ width: "min(520px, 92vw)", maxHeight: "min(620px, 85vh)" }}>
      <div className="px-5 py-4 border-b border-line shrink-0 flex items-baseline justify-between">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">Initiatives</div>
          <h2 className="font-serif text-xl leading-tight text-ink mt-0.5">Manage umbrellas</h2>
        </div>
        <button type="button" onClick={onClose} className="text-ink-faint hover:text-ink" aria-label="Close">×</button>
      </div>
      <div className="flex-1 overflow-y-auto scroll-quiet px-5 py-4 space-y-4">
        {initiatives.list.length === 0 ? (
          <div className="text-[12px] text-ink-faint italic">No initiatives yet. Add your first below.</div>
        ) : (
          <ul className="space-y-1.5">
            {initiatives.list.map((i) => (
              <li key={i.key} className="px-3 py-2 rounded-md bg-surface flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <div className="text-[13px] text-ink">{i.display_name}</div>
                  <div className="text-[10px] font-mono text-ink-faint">{i.key}{i.epic_jira_key ? ` · ${i.epic_jira_key}` : ""}{i.confluence_root_page_id ? ` · root ${i.confluence_root_page_id}` : ""}</div>
                  {i.repo_paths.length > 0 && (
                    <div className="text-[10px] font-mono text-ink-faint truncate">{i.repo_paths.length} repo{i.repo_paths.length === 1 ? "" : "s"}</div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => {
                    if (confirm(`Remove initiative "${i.display_name}"?`)) void initiatives.remove(i.key);
                  }}
                  className="text-ink-faint hover:text-[color:var(--accent-err)] text-[11px]"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="border-t border-line pt-4 space-y-2">
          <div className="text-[11px] font-mono uppercase tracking-[0.14em] text-ink-faint">Add / update</div>
          <input
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="Display name (e.g. Meowtorq)"
            className="w-full px-3 py-2 rounded-md border border-line bg-surface text-ink text-[13px] focus:outline-none focus:border-line-strong"
          />
          <div className="relative">
            <input
              type="text"
              value={key}
              onChange={(e) => setKeyOverride(normalizeInitiativeKey(e.target.value))}
              placeholder="key (auto from display name)"
              className="w-full px-3 py-2 rounded-md border border-line bg-surface text-ink font-mono text-[13px] focus:outline-none focus:border-line-strong"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-ink-faint pointer-events-none">
              {keyOverride === null ? "auto" : "edited"}
            </span>
          </div>
          <SearchPicker
            value={epicKey}
            display={epicState}
            onChange={(id) => setEpicKey(id)}
            fetcher={async (q, signal) => {
              const results = await searchTickets(baseUrl, q, signal);
              return results.map((r) => ({
                id: r.key,
                label: r.title,
                sublabel: r.status ?? undefined,
              }));
            }}
            placeholder="Search ticket — key or title…"
            emptyHint="Optional — the epic this initiative rolls up under."
            monoValue
          />
          <SearchPicker
            value={rootPage}
            display={pageState}
            onChange={(id) => setRootPage(id)}
            fetcher={async (q, signal) => {
              const results = await searchConfluencePages(baseUrl, q, signal);
              return results.map((r) => ({
                id: r.id,
                label: r.title,
                sublabel: r.url ?? undefined,
                url: r.url ?? undefined,
              }));
            }}
            placeholder="Search Confluence — title…"
            emptyHint="Optional — root page under which RFCs/Debriefs are created."
            monoValue
          />
          {error && <div className="text-[11px] text-[color:var(--accent-err)] font-mono">{error}</div>}
          <button type="button" onClick={submit} disabled={!key.trim()} className="px-3 py-1.5 rounded-md text-[12px] font-medium text-canvas disabled:opacity-40" style={{ background: "var(--ink)" }}>
            Save initiative
          </button>
        </div>
      </div>
    </div>
  );
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
        className="mt-6 px-4 py-2 rounded-md text-sm font-medium text-canvas disabled:opacity-40"
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
  workspace,
  workspaceSessions,
  sessionStates,
  sessionNames,
  sessionList,
  workspaceDir,
  onPickSession,
  onSpawnSession,
  onRenameSession,
  onDeleteSession,
}: {
  status: string;
  sessionId?: string;
  permissionMode?: string | null;
  onChangeMode?: (mode: string) => void;
  workspace?: WorkspaceDTO;
  workspaceSessions?: string[];
  sessionStates?: AgentMux["sessions"];
  sessionNames?: Record<string, string>;
  sessionList?: StoredSession[];
  workspaceDir?: string;
  onPickSession?: (sid: string) => void;
  onSpawnSession?: () => Promise<string>;
  onRenameSession?: (sid: string, name: string | null) => Promise<void>;
  onDeleteSession?: (sid: string) => Promise<void>;
}) {
  return (
    <header
      className="border-b border-line bg-canvas/80 backdrop-blur-sm"
      data-tauri-drag-region
    >
      <div
        className="mx-auto w-full max-w-2xl px-6 h-14 flex items-center justify-between gap-4"
        data-tauri-drag-region
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          {workspace && (
            <div className="flex items-baseline gap-2 shrink-0">
              <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-ink">
                {workspace.ticket_key}
              </span>
              {workspace.ticket_title && (
                <span className="text-[12px] text-ink-muted truncate max-w-[200px]" title={workspace.ticket_title}>
                  {workspace.ticket_title}
                </span>
              )}
            </div>
          )}
          {workspaceSessions && workspaceSessions.length > 0 && (
            <SessionTabs
              sessionIds={workspaceSessions}
              activeId={sessionId ?? null}
              sessionStates={sessionStates ?? {}}
              sessionNames={sessionNames ?? {}}
              sessionList={sessionList ?? []}
              workspaceDir={workspaceDir ?? ""}
              onPick={onPickSession ?? (() => {})}
              onSpawn={onSpawnSession}
              onRename={onRenameSession}
              onDelete={onDeleteSession}
            />
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {onChangeMode && (
            <PermissionModeMenu mode={permissionMode ?? "default"} onChange={onChangeMode} />
          )}
          <StatusBadge status={status} />
        </div>
      </div>
    </header>
  );
}

function SessionTabs({
  sessionIds,
  activeId,
  sessionStates,
  sessionNames,
  sessionList,
  workspaceDir,
  onPick,
  onSpawn,
  onRename,
  onDelete,
}: {
  sessionIds: string[];
  activeId: string | null;
  sessionStates: AgentMux["sessions"];
  sessionNames: Record<string, string>;
  sessionList: StoredSession[];
  workspaceDir: string;
  onPick: (sid: string) => void;
  onSpawn?: () => Promise<string>;
  onRename?: (sid: string, name: string | null) => Promise<void>;
  onDelete?: (sid: string) => Promise<void>;
}) {
  const [menuFor, setMenuFor] = useState<string | null>(null);
  const [renameTarget, setRenameTarget] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const stored = useMemo(() => {
    const m = new Map<string, StoredSession>();
    for (const s of sessionList) m.set(s.id, s);
    return m;
  }, [sessionList]);

  return (
    <>
      <div className="flex items-center gap-1 min-w-0 overflow-x-auto scroll-quiet">
        {sessionIds.map((sid, i) => {
          const state = sessionStates[sid];
          const status = state?.status ?? "idle";
          const dot =
            status === "streaming" || status === "awaiting_hook"
              ? "var(--accent-warm)"
              : status === "awaiting_permission" || status === "awaiting_question"
                ? "var(--accent-cool)"
                : status === "error"
                  ? "var(--accent-err)"
                  : "var(--ink-faint)";
          const isActive = sid === activeId;
          const label = sessionNames[sid]?.trim() || `S${i + 1}`;
          const menuOpen = menuFor === sid;
          const sdkSessionId = stored.get(sid)?.sdk_session_id ?? null;
          return (
            <SessionTab
              key={sid}
              sid={sid}
              label={label}
              dot={dot}
              status={status}
              isActive={isActive}
              menuOpen={menuOpen}
              onPick={() => onPick(sid)}
              onOpenMenu={() => setMenuFor(menuOpen ? null : sid)}
              onCloseMenu={() => setMenuFor(null)}
              onRename={onRename ? () => { setMenuFor(null); setRenameTarget(sid); } : undefined}
              onDelete={onDelete ? () => { setMenuFor(null); setDeleteTarget(sid); } : undefined}
              onLocal={
                workspaceDir && sdkSessionId
                  ? () => {
                      const cmd = `cd "${workspaceDir}" && claude --resume ${sdkSessionId} --dangerously-skip-permissions`;
                      void navigator.clipboard.writeText(cmd);
                      setMenuFor(null);
                    }
                  : undefined
              }
            />
          );
        })}
        {onSpawn && (
          <motion.button
            type="button"
            onClick={() => void onSpawn()}
            whileTap={{ scale: 0.94 }}
            transition={{ duration: 0.12, ease: EASE_OUT }}
            className="px-2 py-1 rounded-md text-[11px] text-ink-faint hover:text-ink hover:bg-surface-sunk transition-colors shrink-0"
            title="Spawn a new session in this workspace"
            aria-label="New session"
          >
            +
          </motion.button>
        )}
      </div>

      <AnimatePresence>
        {renameTarget && onRename && (
          <Modal key="rename-session">
            <SessionRenameModal
              currentName={sessionNames[renameTarget] ?? ""}
              fallbackLabel={(() => {
                const i = sessionIds.indexOf(renameTarget);
                return i >= 0 ? `S${i + 1}` : renameTarget;
              })()}
              onClose={() => setRenameTarget(null)}
              onSubmit={async (name) => {
                await onRename(renameTarget, name);
                setRenameTarget(null);
              }}
            />
          </Modal>
        )}
        {deleteTarget && onDelete && (
          <Modal key="delete-session">
            <SessionDeleteConfirm
              label={(() => {
                const name = sessionNames[deleteTarget];
                if (name?.trim()) return name.trim();
                const i = sessionIds.indexOf(deleteTarget);
                return i >= 0 ? `S${i + 1}` : deleteTarget;
              })()}
              onClose={() => setDeleteTarget(null)}
              onConfirm={async () => {
                await onDelete(deleteTarget);
                setDeleteTarget(null);
              }}
            />
          </Modal>
        )}
      </AnimatePresence>
    </>
  );
}

function SessionTab({
  sid,
  label,
  dot,
  status,
  isActive,
  menuOpen,
  onPick,
  onOpenMenu,
  onCloseMenu,
  onRename,
  onDelete,
  onLocal,
}: {
  sid: string;
  label: string;
  dot: string;
  status: string;
  isActive: boolean;
  menuOpen: boolean;
  onPick: () => void;
  onOpenMenu: () => void;
  onCloseMenu: () => void;
  onRename?: () => void;
  onDelete?: () => void;
  onLocal?: () => void;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const kebabRef = useRef<HTMLButtonElement | null>(null);
  const [hover, setHover] = useState(false);
  // Outside-click + Escape closes the menu. The menu portals to <body>,
  // so we have to bound the "outside" check by both the tab and the
  // (portaled) menu root.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      const tgt = e.target as Node;
      if (wrapRef.current?.contains(tgt)) return;
      const menu = document.getElementById(`session-menu-${sid}`);
      if (menu?.contains(tgt)) return;
      onCloseMenu();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCloseMenu(); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen, onCloseMenu, sid]);

  const showKebab = hover || menuOpen;

  return (
    <div
      ref={wrapRef}
      className="relative shrink-0"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div
        className={`group flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-md text-[11px] font-mono transition-colors ${
          isActive ? "bg-surface-tinted text-ink" : "text-ink-soft hover:bg-surface-sunk"
        }`}
        title={sid}
      >
        <button
          type="button"
          onClick={onPick}
          className="flex items-center gap-1.5 min-w-0"
        >
          <span
            className={`inline-block size-1.5 rounded-full ${status === "streaming" ? "dot-pulse" : ""}`}
            style={{ background: dot }}
            aria-hidden
          />
          <span className="truncate max-w-[140px]">{label}</span>
        </button>
        {/* Kebab slot expands from 0 → 18px on hover so the tab "makes room"
            instead of leaving a permanent gap. Animating max-width keeps the
            layout shift smooth without invalidating the parent. */}
        <div
          className="overflow-hidden transition-[max-width,opacity,margin] duration-150 ease-out"
          style={{
            maxWidth: showKebab ? 22 : 0,
            opacity: showKebab ? 1 : 0,
            marginLeft: showKebab ? 2 : 0,
          }}
        >
          <button
            ref={kebabRef}
            type="button"
            onClick={(e) => { e.stopPropagation(); onOpenMenu(); }}
            aria-label="Session options"
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            className="grid place-items-center size-[18px] rounded text-ink-faint hover:text-ink hover:bg-canvas/60 transition-colors"
            tabIndex={showKebab ? 0 : -1}
          >
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden>
              <circle cx="5" cy="2" r="0.9" fill="currentColor" />
              <circle cx="5" cy="5" r="0.9" fill="currentColor" />
              <circle cx="5" cy="8" r="0.9" fill="currentColor" />
            </svg>
          </button>
        </div>
      </div>

      <SessionTabMenu
        sid={sid}
        open={menuOpen}
        anchorRef={kebabRef}
        onRename={onRename}
        onDelete={onDelete}
        onLocal={onLocal}
      />
    </div>
  );
}

function SessionTabMenu({
  sid,
  open,
  anchorRef,
  onRename,
  onDelete,
  onLocal,
}: {
  sid: string;
  open: boolean;
  anchorRef: React.RefObject<HTMLButtonElement | null>;
  onRename?: () => void;
  onDelete?: () => void;
  onLocal?: () => void;
}) {
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  useEffect(() => {
    if (!open) return;
    const place = () => {
      const r = anchorRef.current?.getBoundingClientRect();
      if (!r) return;
      const width = 200;
      const margin = 8;
      let left = r.right - width;
      if (left < margin) left = margin;
      const maxLeft = window.innerWidth - width - margin;
      if (left > maxLeft) left = maxLeft;
      setPos({ top: r.bottom + 6, left });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, anchorRef]);

  if (typeof document === "undefined") return null;
  return createPortal(
    <AnimatePresence>
      {open && pos && (
        <motion.div
          id={`session-menu-${sid}`}
          initial={{ opacity: 0, y: -4, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -2, scale: 0.98 }}
          transition={{ duration: 0.14, ease: EASE_OUT }}
          style={{ position: "fixed", top: pos.top, left: pos.left, width: 200 }}
          className="z-[60] rounded-md border border-line bg-canvas shadow-xl overflow-hidden origin-top-right"
          role="menu"
        >
          {onRename && (
            <MenuItem onSelect={onRename} icon={<IconPencil />} label="Rename" />
          )}
          {onLocal && (
            <MenuItem onSelect={onLocal} icon={<IconTerminal />} label="Locally" />
          )}
          {onDelete && (
            <>
              <div className="h-px bg-line/70" />
              <MenuItem onSelect={onDelete} icon={<IconTrash />} label="Delete" danger />
            </>
          )}
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}

function MenuItem({
  onSelect,
  icon,
  label,
  hint,
  danger,
}: {
  onSelect: () => void;
  icon: React.ReactNode;
  label: string;
  hint?: string;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onSelect}
      className={`group w-full text-left px-3 py-2 flex items-center gap-2 transition-colors ${
        danger
          ? "text-[color:var(--accent-err)] hover:bg-[color:var(--accent-err)]/8"
          : "text-ink hover:bg-surface-sunk"
      }`}
    >
      <span className={`size-3.5 grid place-items-center shrink-0 ${danger ? "text-[color:var(--accent-err)]" : "text-ink-faint group-hover:text-ink-soft"}`}>
        {icon}
      </span>
      <span className="text-[12px] flex-1">{label}</span>
      {hint && <span className="text-[10px] font-mono text-ink-faint">{hint}</span>}
    </button>
  );
}

function CogIcon() {
  // 8-tooth cog, axis-aligned + diagonal teeth, inner ring. Stroke-only
  // so it inherits currentColor and stays crisp at 14px.
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"
        stroke="currentColor"
        strokeWidth="1.4"
      />
      <path
        d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconPencil() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path d="M8.5 1.8l1.7 1.7L4.4 9.3l-2 .5.5-2L8.5 1.8z" stroke="currentColor" strokeWidth="1.1" strokeLinejoin="round" />
    </svg>
  );
}
function IconTerminal() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path d="M1.5 2.5h9v7h-9z" stroke="currentColor" strokeWidth="1.1" />
      <path d="M3.2 4.5l1.6 1.5-1.6 1.5M6 7.5h2.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
function IconTrash() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
      <path d="M2.5 3.2h7M4.7 3.2V2.2h2.6v1M3.4 3.2l.4 6.4h4.4l.4-6.4M5 5v3M7 5v3" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SessionRenameModal({
  currentName,
  fallbackLabel,
  onClose,
  onSubmit,
}: {
  currentName: string;
  fallbackLabel: string;
  onClose: () => void;
  onSubmit: (name: string | null) => Promise<void>;
}) {
  const [value, setValue] = useState(currentName);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => { inputRef.current?.focus(); inputRef.current?.select(); }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const submit = async (name: string | null) => {
    setBusy(true); setErr(null);
    try { await onSubmit(name); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div className="rounded-2xl bg-canvas border border-line shadow-2xl overflow-hidden">
      <form
        onSubmit={(e) => { e.preventDefault(); void submit(value.trim() || null); }}
        className="p-5"
      >
        <div className="text-[11px] uppercase tracking-[0.16em] font-mono text-ink-faint">Rename session</div>
        <div className="mt-1 text-[13px] text-ink-soft">
          Currently shown as <span className="font-mono text-ink">{currentName.trim() || fallbackLabel}</span>.
        </div>
        <label className="block mt-5">
          <span className="block text-[11px] uppercase tracking-[0.14em] font-mono text-ink-faint mb-2">Display name</span>
          <input
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={fallbackLabel}
            maxLength={48}
            className="w-full bg-surface-sunk border border-line rounded-md px-3 py-2 text-[13px] text-ink placeholder:text-ink-faint focus:outline-none focus:border-line-strong transition-colors"
          />
          <span className="block mt-1.5 text-[11px] text-ink-faint">
            Leave blank to restore the default tab label.
          </span>
        </label>
        {err && <div className="mt-3 text-[12px] text-[color:var(--accent-err)]">{err}</div>}
        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-md text-[12px] text-ink-soft hover:bg-surface-sunk transition-colors"
            disabled={busy}
          >
            Cancel
          </button>
          <motion.button
            type="submit"
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.12, ease: EASE_OUT }}
            className="px-3 py-1.5 rounded-md text-[12px] font-mono uppercase tracking-[0.14em] bg-ink text-canvas hover:opacity-90 transition-opacity disabled:opacity-50"
            disabled={busy}
          >
            {busy ? "Saving" : "Save"}
          </motion.button>
        </div>
      </form>
    </div>
  );
}

function SessionDeleteConfirm({
  label,
  onClose,
  onConfirm,
}: {
  label: string;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  const go = async () => {
    setBusy(true); setErr(null);
    try { await onConfirm(); }
    catch (e) { setErr(e instanceof Error ? e.message : String(e)); setBusy(false); }
  };

  return (
    <div className="rounded-2xl bg-canvas border border-line shadow-2xl overflow-hidden">
      <div className="p-5">
        <div className="text-[11px] uppercase tracking-[0.16em] font-mono text-ink-faint">Delete session</div>
        <div className="mt-2 text-[13px] text-ink-soft leading-relaxed">
          Permanently end <span className="font-mono text-ink">{label}</span>. The agent stops, the
          transcript is removed, and the tab disappears. The workspace, worktrees, and other
          sessions stay put.
        </div>
        {err && <div className="mt-3 text-[12px] text-[color:var(--accent-err)]">{err}</div>}
        <div className="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="px-3 py-1.5 rounded-md text-[12px] text-ink-soft hover:bg-surface-sunk transition-colors"
          >
            Cancel
          </button>
          <motion.button
            type="button"
            onClick={() => void go()}
            disabled={busy}
            whileTap={{ scale: 0.97 }}
            transition={{ duration: 0.12, ease: EASE_OUT }}
            className="px-3 py-1.5 rounded-md text-[12px] font-mono uppercase tracking-[0.14em] bg-[color:var(--accent-err)] text-canvas hover:opacity-90 transition-opacity disabled:opacity-50"
          >
            {busy ? "Deleting" : "Delete"}
          </motion.button>
        </div>
      </div>
    </div>
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
            ? "border-transparent text-[#fff]"
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
          prose-table:border-collapse prose-th:border prose-th:border-[color:var(--line)] prose-th:px-3 prose-th:py-1.5 prose-th:bg-[color:var(--surface-sunk)] prose-th:text-left prose-th:text-ink
          prose-td:border prose-td:border-[color:var(--line)] prose-td:px-3 prose-td:py-1.5 prose-td:text-ink-soft
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
      const base = baseUrl || "";
      const url = new URL(`${base}/app/fs/list`, window.location.origin);
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
          className="px-3 py-1.5 rounded-md text-[12px] font-medium text-canvas disabled:opacity-40 transition-opacity"
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
  // Portal into document.body so the fixed overlay escapes any ancestor
  // that has a transform / filter / will-change set. Modals-inside-modals
  // (e.g. the folder picker invoked from the workspace-create modal) were
  // getting centered inside the outer animated container instead of the
  // viewport because Framer Motion's transform breaks `position: fixed`.
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);
  const tree = (
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
  if (!mounted || typeof document === "undefined") return null;
  return createPortal(tree, document.body);
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
      className="px-3.5 py-2 rounded-md text-[13px] font-medium text-canvas disabled:opacity-40 transition-opacity"
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
  completions,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  status: string;
  needsAck: boolean;
  onAcknowledge: () => void;
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
      className="size-9 rounded-lg flex items-center justify-center text-canvas disabled:opacity-30 transition-opacity"
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

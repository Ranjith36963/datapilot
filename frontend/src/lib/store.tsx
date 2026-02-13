"use client";

import { useState, useEffect, useRef, type ReactNode } from "react";
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { getPreview } from "./api";
import type { ColumnInfo, AskResponse } from "./api";

// Types for chart history entries
export interface ChartHistoryEntry {
  id: string;
  chartType: string;
  x?: string;
  y?: string;
  hue?: string;
  title?: string;
  imageBase64?: string;
  plotlyJson?: Record<string, unknown>;
  insight?: string;
  createdAt: string;
}

// Chat message type (moved from explore page)
export interface ChatMessage {
  role: "user" | "assistant" | "error";
  content: string;
  data?: AskResponse;
  timestamp: string; // ISO string for serialization
}

interface SessionStore {
  // Session data
  sessionId: string | null;
  filename: string | null;
  shape: number[] | null;
  columns: ColumnInfo[];
  preview: Record<string, unknown>[];

  // Explore page state
  exploreMessages: ChatMessage[];

  // Visualize page state
  chartHistory: ChartHistoryEntry[];

  // Actions
  setSession: (data: {
    sessionId: string;
    filename: string;
    shape: number[];
    columns: ColumnInfo[];
    preview: Record<string, unknown>[];
  }) => void;
  clearSession: () => void;
  addExploreMessage: (msg: ChatMessage) => void;
  setExploreMessages: (msgs: ChatMessage[]) => void;
  clearExploreMessages: () => void;
  addChartEntry: (entry: ChartHistoryEntry) => void;
  clearChartHistory: () => void;
}

const useSessionStore = create<SessionStore>()(
  persist(
    (set) => ({
      // Initial state
      sessionId: null,
      filename: null,
      shape: null,
      columns: [],
      preview: [],
      exploreMessages: [],
      chartHistory: [],

      // Actions
      setSession: (data) =>
        set({
          sessionId: data.sessionId,
          filename: data.filename,
          shape: data.shape,
          columns: data.columns,
          preview: data.preview,
          exploreMessages: [],
          chartHistory: [],
        }),

      clearSession: () =>
        set({
          sessionId: null,
          filename: null,
          shape: null,
          columns: [],
          preview: [],
          exploreMessages: [],
          chartHistory: [],
        }),

      addExploreMessage: (msg) =>
        set((state) => ({
          exploreMessages: [...state.exploreMessages, msg],
        })),

      setExploreMessages: (msgs) => set({ exploreMessages: msgs }),

      clearExploreMessages: () => set({ exploreMessages: [] }),

      addChartEntry: (entry) =>
        set((state) => ({
          chartHistory: [...state.chartHistory, entry],
        })),

      clearChartHistory: () => set({ chartHistory: [] }),
    }),
    {
      name: "datapilot-session",
      storage: createJSONStorage(() => ({
        getItem: (name: string) => localStorage.getItem(name),
        setItem: (name: string, value: string) => {
          try {
            localStorage.setItem(name, value);
          } catch {
            // QuotaExceededError — silently drop; data survives in memory
          }
        },
        removeItem: (name: string) => localStorage.removeItem(name),
      })),
      partialize: (state) => ({
        sessionId: state.sessionId,
        filename: state.filename,
        columns: state.columns,
        shape: state.shape,
        chartHistory: state.chartHistory.slice(-5),
      }),
    }
  )
);

/** Backward-compatible hook — all existing pages import this. */
export function useSession() {
  return useSessionStore();
}

/**
 * Hook for pages that consume an existing session (Explore, Visualize, Export).
 * Reads localStorage directly (synchronous, no dependency on zustand persist
 * hydration timing) to restore sessions after page refresh. Validates restored
 * sessions with the backend via getPreview.
 */
export function useValidatedSession() {
  const sessionId = useSessionStore((s) => s.sessionId);
  const filename = useSessionStore((s) => s.filename);
  const shape = useSessionStore((s) => s.shape);
  const columns = useSessionStore((s) => s.columns);
  const preview = useSessionStore((s) => s.preview);
  const exploreMessages = useSessionStore((s) => s.exploreMessages);
  const chartHistory = useSessionStore((s) => s.chartHistory);
  const setSession = useSessionStore((s) => s.setSession);
  const clearSession = useSessionStore((s) => s.clearSession);
  const addExploreMessage = useSessionStore((s) => s.addExploreMessage);
  const setExploreMessages = useSessionStore((s) => s.setExploreMessages);
  const clearExploreMessages = useSessionStore((s) => s.clearExploreMessages);
  const addChartEntry = useSessionStore((s) => s.addChartEntry);
  const clearChartHistory = useSessionStore((s) => s.clearChartHistory);

  const [isReady, setIsReady] = useState(false);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;

    // 1. Fresh upload — preview is populated in-memory, skip validation
    if (sessionId && preview.length > 0) {
      setIsReady(true);
      return;
    }

    // 2. SSR or no window — ready with no session
    if (typeof window === "undefined") {
      setIsReady(true);
      return;
    }

    // 3. Determine which sessionId to validate (zustand hydration or raw localStorage)
    let idToValidate = sessionId;
    let storedState: { sessionId?: string; filename?: string; columns?: ColumnInfo[]; shape?: number[]; chartHistory?: ChartHistoryEntry[] } | undefined;

    if (!idToValidate) {
      // Check localStorage directly (synchronous, bypasses persist hydration)
      try {
        const raw = localStorage.getItem("datapilot-session");
        if (raw) {
          const parsed = JSON.parse(raw);
          storedState = parsed?.state;
          idToValidate = storedState?.sessionId ?? null;
        }
      } catch {
        // Corrupted localStorage — ignore
      }
    }

    if (!idToValidate) {
      // No session anywhere — show "upload first"
      setIsReady(true);
      return;
    }

    // 4. Restore session metadata from localStorage into zustand (if not already there)
    if (storedState && !sessionId) {
      useSessionStore.setState({
        sessionId: idToValidate,
        filename: storedState.filename ?? null,
        columns: storedState.columns ?? [],
        shape: storedState.shape ?? null,
        chartHistory: storedState.chartHistory ?? [],
      });
    }

    // 5. Always validate restored sessions with the backend
    const timeout = setTimeout(() => {
      setIsReady(true);
    }, 3000);

    getPreview(idToValidate, 1)
      .then(() => {
        if (!cancelledRef.current) setIsReady(true);
      })
      .catch(() => {
        // Session expired on backend — clear everything
        if (!cancelledRef.current) {
          clearSession();
          setIsReady(true);
        }
      })
      .finally(() => {
        clearTimeout(timeout);
      });

    return () => {
      cancelledRef.current = true;
      clearTimeout(timeout);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  return {
    sessionId, filename, shape, columns, preview,
    exploreMessages, chartHistory,
    setSession, clearSession,
    addExploreMessage, setExploreMessages, clearExploreMessages,
    addChartEntry, clearChartHistory,
    isReady,
  };
}

/** No-op wrapper for backward compatibility with providers.tsx. */
export function SessionProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

"use client";

import type { ReactNode } from "react";
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
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

      clearExploreMessages: () => set({ exploreMessages: [] }),

      addChartEntry: (entry) =>
        set((state) => ({
          chartHistory: [...state.chartHistory, entry],
        })),

      clearChartHistory: () => set({ chartHistory: [] }),
    }),
    {
      name: "datapilot-session",
      storage: createJSONStorage(() => sessionStorage),
    }
  )
);

/** Backward-compatible hook — all existing pages import this. */
export function useSession() {
  return useSessionStore();
}

/** No-op wrapper for backward compatibility with providers.tsx. */
export function SessionProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

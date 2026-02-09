/**
 * Simple client-side session store using React context.
 */

"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import type { ColumnInfo } from "./api";

export interface SessionState {
  sessionId: string | null;
  filename: string | null;
  shape: number[] | null;
  columns: ColumnInfo[];
  preview: Record<string, unknown>[];
}

interface SessionContextValue extends SessionState {
  setSession: (data: {
    sessionId: string;
    filename: string;
    shape: number[];
    columns: ColumnInfo[];
    preview: Record<string, unknown>[];
  }) => void;
  clearSession: () => void;
}

const initialState: SessionState = {
  sessionId: null,
  filename: null,
  shape: null,
  columns: [],
  preview: [],
};

const SessionContext = createContext<SessionContextValue>({
  ...initialState,
  setSession: () => {},
  clearSession: () => {},
});

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SessionState>(initialState);

  const setSession = useCallback(
    (data: {
      sessionId: string;
      filename: string;
      shape: number[];
      columns: ColumnInfo[];
      preview: Record<string, unknown>[];
    }) => {
      setState({
        sessionId: data.sessionId,
        filename: data.filename,
        shape: data.shape,
        columns: data.columns,
        preview: data.preview,
      });
    },
    []
  );

  const clearSession = useCallback(() => {
    setState(initialState);
  }, []);

  return (
    <SessionContext.Provider value={{ ...state, setSession, clearSession }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  return useContext(SessionContext);
}

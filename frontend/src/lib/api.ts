/**
 * DataPilot API client — communicates with the FastAPI backend.
 */

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ColumnInfo {
  name: string;
  dtype: string;
  semantic_type: string;
  n_unique: number;
  null_pct: number;
}

export interface UploadResponse {
  status: string;
  session_id: string;
  filename: string;
  shape: number[];
  columns: ColumnInfo[];
  preview: Record<string, unknown>[];
}

export interface AskResponse {
  status: string;
  question: string;
  skill: string;
  confidence: number;
  reasoning: string;
  route_method: string;
  result?: Record<string, unknown>;
  narrative?: string;
  key_points: string[];
  suggestions: string[];
  code_snippet?: string;
  columns_used: string[];
  elapsed_seconds: number;
  error?: string;
}

export interface AnalyzeResponse {
  status: string;
  skill: string;
  result?: Record<string, unknown>;
  elapsed_seconds: number;
  error?: string;
}

export interface ChartResponse {
  status: string;
  chart_type: string;
  image_base64?: string;
  plotly_json?: Record<string, unknown>;
  insight?: string;
  error?: string;
}

export interface ChartSuggestion {
  chart_type: string;
  x?: string;
  y?: string;
  hue?: string;
  title: string;
  reason?: string;
}

export interface SuggestChartResponse {
  suggestions: ChartSuggestion[];
}

export interface ExportResponse {
  status: string;
  format: string;
  filename: string;
  download_url: string;
}

export interface SessionInfo {
  session_id: string;
  shape?: number[];
  columns?: string[];
  llm_provider: string;
  history_count: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
  sessionId?: string
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (sessionId) {
    headers["x-session-id"] = sessionId;
  }

  // Don't set Content-Type for FormData (browser does it)
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.message || `HTTP ${res.status}`);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// API Methods
// ---------------------------------------------------------------------------

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch<UploadResponse>("/api/upload", {
    method: "POST",
    body: formData,
  });
}

export async function getPreview(
  sessionId: string,
  rows = 20
): Promise<{ status: string; shape: number[]; columns: string[]; data: Record<string, unknown>[] }> {
  return apiFetch(`/api/preview?rows=${rows}`, {}, sessionId);
}

export async function getProfile(
  sessionId: string
): Promise<{ status: string; profile: Record<string, unknown> }> {
  return apiFetch("/api/profile", {}, sessionId);
}

export interface ConversationEntry {
  question: string;
  summary: string;
}

export async function askQuestion(
  sessionId: string,
  question: string,
  narrate = true,
  conversationContext: ConversationEntry[] = []
): Promise<AskResponse> {
  return apiFetch<AskResponse>(
    "/api/ask",
    {
      method: "POST",
      body: JSON.stringify({
        question,
        narrate,
        conversation_context: conversationContext,
      }),
    },
    sessionId
  );
}

export interface HistoryEntry {
  question: string;
  skill: string;
  narrative?: string;
  key_points?: string[];
  confidence?: number;
  reasoning?: string;
  result?: Record<string, unknown>;
}

export async function getHistory(
  sessionId: string
): Promise<{ status: string; history: HistoryEntry[] }> {
  return apiFetch("/api/history", {}, sessionId);
}

export async function runAnalysis(
  sessionId: string,
  skill: string,
  params: Record<string, unknown> = {}
): Promise<AnalyzeResponse> {
  return apiFetch<AnalyzeResponse>(
    "/api/analyze",
    {
      method: "POST",
      body: JSON.stringify({ skill, params }),
    },
    sessionId
  );
}

export async function createChart(
  sessionId: string,
  options: {
    chart_type?: string;
    x?: string;
    y?: string;
    hue?: string;
    title?: string;
  }
): Promise<ChartResponse> {
  return apiFetch<ChartResponse>(
    "/api/chart/create",
    {
      method: "POST",
      body: JSON.stringify(options),
    },
    sessionId
  );
}

export async function suggestChart(
  sessionId: string
): Promise<SuggestChartResponse> {
  return apiFetch<SuggestChartResponse>("/api/chart/suggest", {}, sessionId);
}

export async function exportReport(
  sessionId: string,
  format: string,
  options: { title?: string; subtitle?: string; brand_name?: string } = {}
): Promise<ExportResponse> {
  return apiFetch<ExportResponse>(
    `/api/export/${format}`,
    {
      method: "POST",
      body: JSON.stringify({ include_history: true, ...options }),
    },
    sessionId
  );
}

export function getDownloadUrl(path: string): string {
  return `${API_URL}${path}`;
}

export async function listSessions(): Promise<{
  status: string;
  count: number;
  sessions: SessionInfo[];
}> {
  return apiFetch("/api/sessions");
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiFetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
}

export interface DomainFingerprintResponse {
  domain: string;
  domain_short: string;
  confidence: number;
  target_column?: string | null;
  target_type?: string | null;
  key_observations: string[];
  suggested_questions: string[];
  data_quality_notes: string[];
  provider_used?: string | null;
}

export async function getFingerprint(
  sessionId: string
): Promise<DomainFingerprintResponse> {
  return apiFetch<DomainFingerprintResponse>(
    `/api/fingerprint/${sessionId}`,
    { method: "POST" },
    sessionId
  );
}

export interface AutopilotStepResult {
  step: string;
  status: string;
  question?: string | null;
  narrative?: string | null;
  error?: string | null;
}

export interface AutopilotStatusResponse {
  status: "unavailable" | "planning" | "running" | "complete" | "failed";
  completed_steps?: number | null;
  total_steps?: number | null;
  results?: AutopilotStepResult[] | null;
  summary?: string | null;
  error?: string | null;
}

export async function getAutopilotStatus(
  sessionId: string
): Promise<AutopilotStatusResponse> {
  return apiFetch<AutopilotStatusResponse>(
    `/api/autopilot/${sessionId}`,
    {},
    sessionId
  );
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

export function connectChat(
  sessionId: string,
  onMessage: (msg: { type: string; content?: string; data?: Record<string, unknown> }) => void,
  onError?: (err: Event) => void
): {
  send: (question: string) => void;
  close: () => void;
} {
  const wsUrl = API_URL.replace(/^http/, "ws") + "/api/ws/chat";
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      onMessage(msg);
    } catch {
      onMessage({ type: "error", content: "Invalid server message" });
    }
  };

  ws.onerror = (event) => {
    onError?.(event);
  };

  return {
    send: (question: string) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "question",
            content: question,
            session_id: sessionId,
          })
        );
      }
    },
    close: () => ws.close(),
  };
}

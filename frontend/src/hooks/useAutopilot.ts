import { useEffect, useRef } from "react";
import { getAutopilotStatus, type AutopilotStatusResponse } from "@/lib/api";

const TERMINAL_STATES = new Set(["complete", "failed", "unavailable"]);
const POLL_INTERVAL_MS = 2000;

/**
 * Hook to poll autopilot status for a session.
 * Polls GET /api/autopilot/{session_id} every 2 seconds.
 * Auto-stops when status reaches a terminal state (complete, failed, unavailable).
 * The store (via onUpdate / setAutopilotStatus) is the source of truth.
 */
export function useAutopilot(
  sessionId: string | null,
  cachedStatus: AutopilotStatusResponse | null,
  onUpdate: (status: AutopilotStatusResponse) => void
) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    cancelledRef.current = false;

    // Clean up any previous interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // No session — nothing to poll
    if (!sessionId) return;

    // Already in terminal state — skip polling
    if (cachedStatus && TERMINAL_STATES.has(cachedStatus.status)) return;

    const poll = async () => {
      if (cancelledRef.current) return;

      try {
        const data = await getAutopilotStatus(sessionId);
        if (cancelledRef.current) return;

        onUpdate(data);

        // Stop polling on terminal state
        if (TERMINAL_STATES.has(data.status)) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
        }
      } catch {
        // Silently swallow network errors — next poll will retry
      }
    };

    // Fire immediately, then set interval
    poll();
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS);

    return () => {
      cancelledRef.current = true;
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Derive return values from cachedStatus (store is source of truth)
  const status = cachedStatus?.status ?? null;
  const results = cachedStatus?.results ?? null;
  const summary = cachedStatus?.summary ?? null;
  const completedSteps = cachedStatus?.completed_steps ?? 0;
  const totalSteps = cachedStatus?.total_steps ?? 0;
  const isLoading = status === "planning" || status === "running";

  return { status, results, summary, completedSteps, totalSteps, isLoading };
}

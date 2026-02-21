import { useEffect, useRef, useState } from "react";
import { getFingerprint, type DomainFingerprintResponse } from "@/lib/api";

export interface UseFingerprintResult {
  fingerprint: DomainFingerprintResponse | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * Hook to fetch domain fingerprint data for a session.
 * Automatically fetches when sessionId is provided.
 * Uses a ref guard to prevent duplicate POST requests (e.g., React strict mode).
 */
export function useFingerprint(
  sessionId: string | null,
  cachedFingerprint?: DomainFingerprintResponse | null
): UseFingerprintResult {
  const [fingerprint, setFingerprint] = useState<DomainFingerprintResponse | null>(
    cachedFingerprint || null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasFiredRef = useRef<string | null>(null);

  useEffect(() => {
    // No session ID — nothing to fetch
    if (!sessionId) {
      setFingerprint(null);
      setIsLoading(false);
      setError(null);
      hasFiredRef.current = null;
      return;
    }

    // Already have cached fingerprint for this session
    if (cachedFingerprint) {
      setFingerprint(cachedFingerprint);
      setIsLoading(false);
      setError(null);
      return;
    }

    // Prevent duplicate POST for the same session (React strict mode)
    if (hasFiredRef.current === sessionId) return;
    hasFiredRef.current = sessionId;

    // Fetch fingerprint data
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    getFingerprint(sessionId)
      .then((data) => {
        if (!cancelled) {
          setFingerprint(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message || "Failed to fetch domain fingerprint");
          setIsLoading(false);
          hasFiredRef.current = null; // Allow retry on error
        }
      });

    return () => {
      cancelled = true;
      hasFiredRef.current = null; // Allow re-fire on strict mode remount
    };
  }, [sessionId, cachedFingerprint]);

  return { fingerprint, isLoading, error };
}

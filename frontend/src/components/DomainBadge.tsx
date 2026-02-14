"use client";

import { useState } from "react";
import {
  Database,
  HelpCircle,
  Info,
  Target,
  Sparkles,
  AlertTriangle,
} from "lucide-react";

import type { DomainFingerprintResponse } from "@/lib/api";

interface DomainBadgeProps {
  fingerprint: DomainFingerprintResponse;
}

function getConfidenceColor(confidence: number): {
  bg: string;
  text: string;
  label: string;
} {
  if (confidence >= 0.8) {
    return {
      bg: "bg-emerald-50 dark:bg-emerald-950/30",
      text: "text-emerald-700 dark:text-emerald-400",
      label: "High confidence",
    };
  }
  if (confidence >= 0.5) {
    return {
      bg: "bg-amber-50 dark:bg-amber-950/30",
      text: "text-amber-700 dark:text-amber-400",
      label: "Moderate confidence",
    };
  }
  return {
    bg: "bg-red-50 dark:bg-red-950/30",
    text: "text-red-700 dark:text-red-400",
    label: "Low confidence",
  };
}

export function DomainBadge({ fingerprint }: DomainBadgeProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  const confidence = Math.round(fingerprint.confidence * 100);
  const colors = getConfidenceColor(fingerprint.confidence);

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {/* Domain Badge */}
      <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800">
        <Database className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
          {fingerprint.domain_short} Dataset
        </span>
      </div>

      {/* Confidence Indicator */}
      <div
        className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg ${colors.bg} border border-current/20`}
      >
        <span className={`text-sm font-medium ${colors.text}`}>
          {confidence}% {colors.label}
        </span>
      </div>

      {/* Details Tooltip Button */}
      <div className="relative">
        <button
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
          onClick={() => setShowTooltip((v) => !v)}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
          aria-label="Show dataset understanding details"
        >
          <Info className="h-3.5 w-3.5" />
          <span className="text-xs font-medium">Details</span>
        </button>

        {showTooltip && (
          <div className="absolute top-full left-0 mt-2 w-80 p-3 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-50">
            <div className="space-y-2">
              {/* Domain description */}
              <div>
                <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Domain
                </h4>
                <p className="text-xs text-slate-600 dark:text-slate-400 capitalize">
                  {fingerprint.domain}
                </p>
              </div>

              {/* Target column */}
              {fingerprint.target_column && (
                <div className="flex items-center gap-2">
                  <Target className="h-3.5 w-3.5 text-violet-500" />
                  <span className="text-xs text-slate-600 dark:text-slate-400">
                    Target:{" "}
                    <span className="font-mono px-1 py-0.5 rounded bg-violet-50 dark:bg-violet-950/30 text-violet-700 dark:text-violet-400">
                      {fingerprint.target_column}
                    </span>
                    {fingerprint.target_type && (
                      <span className="ml-1 text-slate-400 dark:text-slate-500">
                        ({fingerprint.target_type})
                      </span>
                    )}
                  </span>
                </div>
              )}

              {/* Key Observations */}
              {fingerprint.key_observations.length > 0 && (
                <div>
                  <div className="flex items-center gap-1 mb-1">
                    <Sparkles className="h-3 w-3 text-blue-500" />
                    <p className="text-xs font-medium text-slate-600 dark:text-slate-400">
                      Key Observations
                    </p>
                  </div>
                  <ul className="space-y-1">
                    {fingerprint.key_observations.map((obs, i) => (
                      <li
                        key={i}
                        className="text-xs text-slate-600 dark:text-slate-400 flex items-start gap-1.5"
                      >
                        <span className="text-blue-500 mt-0.5">&bull;</span>
                        <span>{obs}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Data Quality Notes */}
              {fingerprint.data_quality_notes.length > 0 && (
                <div>
                  <div className="flex items-center gap-1 mb-1">
                    <AlertTriangle className="h-3 w-3 text-amber-500" />
                    <p className="text-xs font-medium text-slate-600 dark:text-slate-400">
                      Data Quality
                    </p>
                  </div>
                  <ul className="space-y-1">
                    {fingerprint.data_quality_notes.map((note, i) => (
                      <li
                        key={i}
                        className="text-xs text-slate-600 dark:text-slate-400 flex items-start gap-1.5"
                      >
                        <span className="text-amber-500 mt-0.5">&bull;</span>
                        <span>{note}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Provider */}
              {fingerprint.provider_used && (
                <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
                  <p className="text-xs text-slate-400 dark:text-slate-500">
                    Analyzed by{" "}
                    <span className="font-medium capitalize">
                      {fingerprint.provider_used}
                    </span>
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function DomainBadgeSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <div className="h-8 w-32 bg-slate-200 dark:bg-slate-700 rounded-lg animate-pulse" />
      <div className="h-8 w-40 bg-slate-200 dark:bg-slate-700 rounded-lg animate-pulse" />
      <div className="h-8 w-20 bg-slate-200 dark:bg-slate-700 rounded-lg animate-pulse" />
    </div>
  );
}

export function DomainBadgeError({ error }: { error: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
      <HelpCircle className="h-4 w-4 text-slate-400 dark:text-slate-500" />
      <span className="text-sm text-slate-500 dark:text-slate-400">
        Domain detection unavailable
      </span>
    </div>
  );
}

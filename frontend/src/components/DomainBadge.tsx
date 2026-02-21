"use client";

import {
  HelpCircle,
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
  if (confidence >= 0.7) {
    return {
      bg: "bg-emerald-50 dark:bg-emerald-950/30",
      text: "text-emerald-700 dark:text-emerald-400",
      label: "High confidence",
    };
  }
  if (confidence >= 0.4) {
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

function getDomainIcon(domain: string, domainShort: string): string {
  const text = `${domain} ${domainShort}`.toLowerCase();
  if (/financ|banking|loan|credit|stock|invest/.test(text)) return "\uD83D\uDCB0";
  if (/health|medic|patient|clinic|hospital/.test(text)) return "\uD83C\uDFE5";
  if (/retail|shop|sale|ecommerce|customer|order/.test(text)) return "\uD83D\uDED2";
  if (/hr|human.?resource|employee|salary|hiring/.test(text)) return "\uD83D\uDC65";
  if (/market|advertis|campaign|seo/.test(text)) return "\uD83D\uDCC8";
  return "\uD83D\uDCCA";
}

export function DomainBadge({ fingerprint }: DomainBadgeProps) {
  const confidence = Math.round(fingerprint.confidence * 100);
  const colors = getConfidenceColor(fingerprint.confidence);

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {/* Domain Badge */}
      <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800">
        <span className="text-base leading-none" role="img" aria-label={fingerprint.domain_short}>
          {getDomainIcon(fingerprint.domain, fingerprint.domain_short)}
        </span>
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
    </div>
  );
}

export function DomainBadgeSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <div className="h-8 w-32 bg-slate-200 dark:bg-slate-700 rounded-lg animate-pulse" />
      <div className="h-8 w-40 bg-slate-200 dark:bg-slate-700 rounded-lg animate-pulse" />
    </div>
  );
}

export function DomainBadgeError({ error }: { error: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-600">
      <HelpCircle className="h-4 w-4 text-slate-400 dark:text-slate-500" />
      <span className="text-sm text-slate-500 dark:text-slate-400">
        {error || "Domain detection unavailable"}
      </span>
    </div>
  );
}

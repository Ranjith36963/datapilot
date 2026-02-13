"use client";

import { useState } from "react";
import {
  FileText,
  Download,
  Loader2,
  FileDown,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { exportReport, getDownloadUrl, type ExportResponse } from "@/lib/api";
import { useValidatedSession } from "@/lib/store";

const FORMATS = [
  {
    id: "pdf",
    label: "PDF",
    description: "Portable Document Format — best for sharing",
    icon: "pdf",
  },
  {
    id: "docx",
    label: "Word",
    description: "Microsoft Word — editable reports",
    icon: "docx",
  },
  {
    id: "pptx",
    label: "PowerPoint",
    description: "Presentation slides — for stakeholder meetings",
    icon: "pptx",
  },
];

export default function ExportPage() {
  const { sessionId, isReady } = useValidatedSession();

  const [format, setFormat] = useState("pdf");
  const [title, setTitle] = useState("");
  const [subtitle, setSubtitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!isReady) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3.5rem)]">
        <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
      </div>
    );
  }

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3.5rem)]">
        <div className="text-center">
          <FileText className="h-12 w-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
          <p className="text-slate-500 dark:text-slate-400">
            Upload a dataset and run some analyses first.
          </p>
        </div>
      </div>
    );
  }

  async function handleExport() {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await exportReport(sessionId, format, {
        title: title || undefined,
        subtitle: subtitle || undefined,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
        Export Report
      </h1>
      <p className="text-slate-500 dark:text-slate-400 mb-8">
        Generate a report from your analysis history.
      </p>

      <div className="space-y-6">
        {/* Format selector */}
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Format
          </label>
          <div className="grid grid-cols-3 gap-3">
            {FORMATS.map((f) => (
              <button
                key={f.id}
                onClick={() => setFormat(f.id)}
                className={`
                  p-4 rounded-xl border text-left transition-all
                  ${
                    format === f.id
                      ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30 ring-1 ring-blue-500"
                      : "border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 hover:border-slate-300 dark:hover:border-slate-700"
                  }
                `}
              >
                <p
                  className={`text-sm font-medium ${
                    format === f.id
                      ? "text-blue-700 dark:text-blue-300"
                      : "text-slate-900 dark:text-white"
                  }`}
                >
                  {f.label}
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {f.description}
                </p>
              </button>
            ))}
          </div>
        </div>

        {/* Title */}
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            Report Title
          </label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Data Analysis Report"
            className="w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Subtitle */}
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
            Subtitle
          </label>
          <input
            type="text"
            value={subtitle}
            onChange={(e) => setSubtitle(e.target.value)}
            placeholder="Prepared by DataPilot"
            className="w-full px-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Generate button */}
        <button
          onClick={handleExport}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-blue-600 text-white font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <FileDown className="h-4 w-4" />
          )}
          Generate {FORMATS.find((f) => f.id === format)?.label} Report
        </button>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-50 dark:bg-red-950/30 text-red-600 dark:text-red-400 text-sm">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Success */}
        {result && (
          <div className="rounded-xl border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/30 p-4">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
              <p className="text-sm font-medium text-emerald-700 dark:text-emerald-300">
                Report generated
              </p>
            </div>
            <a
              href={getDownloadUrl(result.download_url)}
              download={result.filename}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-700 transition-colors"
            >
              <Download className="h-4 w-4" />
              Download {result.filename}
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

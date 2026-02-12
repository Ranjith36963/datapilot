"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  BarChart3,
  Loader2,
  RefreshCw,
  Image as ImageIcon,
  AlertCircle,
  Download,
  Trash2,
  Copy,
  Check,
  Sparkles,
} from "lucide-react";
import {
  createChart,
  suggestChart,
  type ChartResponse,
  type ChartSuggestion,
} from "@/lib/api";
import { useSession, type ChartHistoryEntry } from "@/lib/store";

const CHART_TYPES = [
  "histogram",
  "bar",
  "scatter",
  "line",
  "box",
  "violin",
  "heatmap",
  "pie",
  "area",
  "strip",
];

const CHART_ICONS: Record<string, string> = {
  histogram: "\u{1F4CA}",
  bar: "\u{1F4CA}",
  scatter: "\u{1F535}",
  line: "\u{1F4C8}",
  box: "\u{1F4E6}",
  violin: "\u{1F3BB}",
  heatmap: "\u{1F7E5}",
  pie: "\u{1F967}",
  area: "\u{1F4C9}",
  strip: "\u{1F4CF}",
};

function downloadPng(base64: string, filename: string) {
  const link = document.createElement("a");
  link.href = `data:image/png;base64,${base64}`;
  link.download = filename;
  link.click();
}

function chartLabel(entry: ChartHistoryEntry): string {
  const parts = [entry.chartType];
  if (entry.x) parts.push(entry.x);
  if (entry.y) parts.push(`vs ${entry.y}`);
  if (entry.hue) parts.push(`colored by ${entry.hue}`);
  return parts.join(": ");
}

export default function VisualizePage() {
  const { sessionId, columns, chartHistory, addChartEntry, clearChartHistory } =
    useSession();

  const [chartType, setChartType] = useState("histogram");
  const [xCol, setXCol] = useState("");
  const [yCol, setYCol] = useState("");
  const [hueCol, setHueCol] = useState("");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestions, setSuggestions] = useState<ChartSuggestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [creatingIdx, setCreatingIdx] = useState<number | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const hasAutoSuggested = useRef(false);

  const colNames = columns.map((c) => c.name);

  // Auto-load suggestions on mount when session exists
  const loadSuggestions = useCallback(async () => {
    if (!sessionId || suggesting) return;
    setSuggesting(true);
    setError(null);
    try {
      const res = await suggestChart(sessionId);
      setSuggestions(res.suggestions || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "AI suggestion failed");
    } finally {
      setSuggesting(false);
    }
  }, [sessionId, suggesting]);

  useEffect(() => {
    if (sessionId && !hasAutoSuggested.current) {
      hasAutoSuggested.current = true;
      loadSuggestions();
    }
  }, [sessionId, loadSuggestions]);

  // Auto-scroll to bottom when new chart is added
  useEffect(() => {
    if (chartHistory.length > 0) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [chartHistory.length]);

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-3.5rem)]">
        <div className="text-center">
          <BarChart3 className="h-12 w-12 text-slate-300 dark:text-slate-600 mx-auto mb-3" />
          <p className="text-slate-500 dark:text-slate-400">
            Upload a dataset first to create charts.
          </p>
        </div>
      </div>
    );
  }

  async function handleCreate() {
    if (!sessionId) return;
    setLoading(true);
    setError(null);

    try {
      const res: ChartResponse = await createChart(sessionId, {
        chart_type: chartType,
        x: xCol || undefined,
        y: yCol || undefined,
        hue: hueCol || undefined,
        title: title || undefined,
      });
      if (res.status === "error") {
        setError(res.error || "Chart creation failed");
      } else {
        const entry: ChartHistoryEntry = {
          id: Date.now().toString(),
          chartType,
          x: xCol || undefined,
          y: yCol || undefined,
          hue: hueCol || undefined,
          title: title || undefined,
          imageBase64: res.image_base64,
          plotlyJson: res.plotly_json,
          insight: res.insight || undefined,
          createdAt: new Date().toISOString(),
        };
        addChartEntry(entry);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chart creation failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleSuggestionClick(s: ChartSuggestion, idx: number) {
    if (!sessionId || creatingIdx !== null) return;
    setCreatingIdx(idx);
    setError(null);

    try {
      const res: ChartResponse = await createChart(sessionId, {
        chart_type: s.chart_type || "histogram",
        x: s.x || undefined,
        y: s.y || undefined,
        hue: s.hue || undefined,
        title: s.title || undefined,
      });
      if (res.status === "error") {
        setError(res.error || "Chart creation failed");
      } else {
        const entry: ChartHistoryEntry = {
          id: Date.now().toString(),
          chartType: s.chart_type || "histogram",
          x: s.x || undefined,
          y: s.y || undefined,
          hue: s.hue || undefined,
          title: s.title || undefined,
          imageBase64: res.image_base64,
          plotlyJson: res.plotly_json,
          insight: res.insight || undefined,
          createdAt: new Date().toISOString(),
        };
        addChartEntry(entry);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chart creation failed");
    } finally {
      setCreatingIdx(null);
    }
  }

  function copyInsight(id: string, text: string) {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  }

  return (
    <div className="px-4 py-3 h-[calc(100vh-3.5rem)] flex flex-col">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1 min-h-0">
        {/* Left column: Controls */}
        <div className="lg:col-span-3 space-y-3 flex flex-col min-h-0">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-4 space-y-3 flex-1 overflow-y-auto">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-blue-500" />
                <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                  Chart Builder
                </p>
              </div>
              <button
                onClick={() => { setChartType("histogram"); setXCol(""); setYCol(""); setHueCol(""); setTitle(""); }}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 text-xs font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Reset
              </button>
            </div>
            {/* Chart type */}
            <div>
              <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Chart Type
              </label>
              <select
                value={chartType}
                onChange={(e) => setChartType(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {CHART_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            {/* X column */}
            <div>
              <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                X Axis
              </label>
              <select
                value={xCol}
                onChange={(e) => setXCol(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Auto / None</option>
                {colNames.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {/* Y column */}
            <div>
              <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Y Axis
              </label>
              <select
                value={yCol}
                onChange={(e) => setYCol(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Auto / None</option>
                {colNames.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {/* Hue */}
            <div>
              <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Color By
              </label>
              <select
                value={hueCol}
                onChange={(e) => setHueCol(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">None</option>
                {colNames.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>

            {/* Title */}
            <div>
              <label className="block text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">
                Title
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Optional chart title"
                className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Create button */}
            <button
              onClick={handleCreate}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <BarChart3 className="h-4 w-4" />
              )}
              Create Chart
            </button>

            {/* Clear All button */}
            {chartHistory.length > 0 && (
              <button
                onClick={clearChartHistory}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-red-200 dark:border-red-800 text-red-500 dark:text-red-400 text-sm font-medium hover:bg-red-50 dark:hover:bg-red-950/20 transition-colors"
              >
                <Trash2 className="h-4 w-4" />
                Clear All Charts
              </button>
            )}
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-500 text-sm rounded-xl border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/20 p-3">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* Center column: Chart feed */}
        <div className="lg:col-span-6 flex flex-col min-h-0">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 flex-1 overflow-y-auto">
            {chartHistory.length === 0 ? (
              <div className="flex items-center justify-center h-full min-h-[400px]">
                <div className="text-center text-slate-400 dark:text-slate-600">
                  <ImageIcon className="h-12 w-12 mx-auto mb-2" />
                  <p className="text-sm">
                    Create a chart using the form on the left, or click an AI
                    suggestion on the right.
                  </p>
                </div>
              </div>
            ) : (
              <div className="p-4 space-y-6">
                {chartHistory.map((entry) => (
                  <div
                    key={entry.id}
                    className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 overflow-hidden"
                  >
                    {/* Chart image */}
                    {entry.imageBase64 && (
                      <div className="flex justify-center bg-white dark:bg-slate-950 p-4">
                        <img
                          src={`data:image/png;base64,${entry.imageBase64}`}
                          alt={entry.title || "Chart"}
                          className="w-full max-h-[500px] object-contain"
                        />
                      </div>
                    )}

                    {/* Info bar */}
                    <div className="px-4 py-3 border-t border-slate-200 dark:border-slate-800">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                          <span>{CHART_ICONS[entry.chartType] || "\u{1F4CA}"}</span>
                          <span className="font-medium">{chartLabel(entry)}</span>
                          {entry.title && (
                            <span className="text-slate-400 dark:text-slate-500">
                              &mdash; {entry.title}
                            </span>
                          )}
                        </div>
                        {entry.imageBase64 && (
                          <button
                            onClick={() =>
                              downloadPng(
                                entry.imageBase64!,
                                `chart-${entry.chartType}-${entry.id}.png`
                              )
                            }
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                          >
                            <Download className="h-3 w-3" />
                            PNG
                          </button>
                        )}
                      </div>

                      {/* Insight */}
                      {entry.insight && (
                        <div className="mt-2 flex items-start gap-2">
                          <Sparkles className="h-3.5 w-3.5 text-amber-500 mt-0.5 shrink-0" />
                          <p className="text-sm text-slate-600 dark:text-slate-400 italic flex-1 leading-relaxed">
                            {entry.insight}
                          </p>
                          <button
                            onClick={() => copyInsight(entry.id, entry.insight!)}
                            className="flex items-center gap-1 px-2 py-1 rounded text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors shrink-0"
                            title="Copy insight"
                          >
                            {copiedId === entry.id ? (
                              <Check className="h-3 w-3 text-green-500" />
                            ) : (
                              <Copy className="h-3 w-3" />
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </div>

        {/* Right column: AI Suggestions */}
        <div className="lg:col-span-3 flex flex-col min-h-0">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 flex-1 overflow-y-auto">
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-blue-500" />
                  <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                    AI Suggestions
                  </p>
                </div>
                <button
                  onClick={loadSuggestions}
                  disabled={suggesting}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 text-xs font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors disabled:opacity-50"
                >
                  {suggesting ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5" />
                  )}
                  Refresh
                </button>
              </div>

              {suggesting && suggestions.length === 0 ? (
                <div className="flex items-center justify-center min-h-[200px]">
                  <div className="text-center text-slate-400 dark:text-slate-500">
                    <Loader2 className="h-6 w-6 mx-auto mb-2 animate-spin" />
                    <p className="text-xs">Analyzing your data...</p>
                  </div>
                </div>
              ) : suggestions.length > 0 ? (
                <div className="space-y-2">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => handleSuggestionClick(s, i)}
                      disabled={creatingIdx !== null}
                      className="w-full text-left rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-3 hover:border-blue-300 dark:hover:border-blue-700 hover:bg-blue-50 dark:hover:bg-blue-950/20 transition-all disabled:opacity-60"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-base">
                          {CHART_ICONS[s.chart_type] || "\u{1F4CA}"}
                        </span>
                        <span className="text-xs font-semibold text-slate-900 dark:text-white capitalize">
                          {s.chart_type}
                        </span>
                        {creatingIdx === i && (
                          <Loader2 className="h-3 w-3 animate-spin text-blue-500 ml-auto" />
                        )}
                      </div>
                      {s.title && (
                        <p className="text-xs text-slate-700 dark:text-slate-300 mb-0.5 leading-snug">
                          {s.title}
                        </p>
                      )}
                      {s.reason && (
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 italic leading-relaxed">
                          {s.reason}
                        </p>
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center min-h-[200px]">
                  <div className="text-center text-slate-400 dark:text-slate-500">
                    <Sparkles className="h-8 w-8 mx-auto mb-2" />
                    <p className="text-xs">Click Refresh for AI chart ideas</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

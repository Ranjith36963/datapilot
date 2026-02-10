"use client";

import { useState } from "react";
import {
  BarChart3,
  Loader2,
  Wand2,
  Image as ImageIcon,
  AlertCircle,
  Download,
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
  histogram: "📊",
  bar: "📊",
  scatter: "🔵",
  line: "📈",
  box: "📦",
  violin: "🎻",
  heatmap: "🟥",
  pie: "🥧",
  area: "📉",
  strip: "📏",
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
  const { sessionId, columns, chartHistory, addChartEntry } = useSession();

  const [chartType, setChartType] = useState("histogram");
  const [xCol, setXCol] = useState("");
  const [yCol, setYCol] = useState("");
  const [hueCol, setHueCol] = useState("");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestions, setSuggestions] = useState<ChartSuggestion[]>([]);
  const [error, setError] = useState<string | null>(null);

  const colNames = columns.map((c) => c.name);

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

  async function handleSuggest() {
    if (!sessionId) return;
    setSuggesting(true);
    setError(null);

    try {
      const res = await suggestChart(sessionId);
      setSuggestions(res.suggestions || []);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "AI suggestion failed"
      );
    } finally {
      setSuggesting(false);
    }
  }

  function applySuggestion(s: ChartSuggestion) {
    setChartType(s.chart_type || "histogram");
    setXCol(s.x || "");
    setYCol(s.y || "");
    setHueCol(s.hue || "");
    if (s.title) setTitle(s.title);
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 h-[calc(100vh-3.5rem)] flex flex-col">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
        Chart Builder
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
        {/* Left column: Controls */}
        <div className="space-y-4 overflow-y-auto">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-4 space-y-4">
            {/* AI Suggest */}
            <button
              onClick={handleSuggest}
              disabled={suggesting}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg border border-blue-200 dark:border-blue-800 text-blue-600 dark:text-blue-400 text-sm font-medium hover:bg-blue-50 dark:hover:bg-blue-950/30 transition-colors disabled:opacity-50"
            >
              {suggesting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Wand2 className="h-4 w-4" />
              )}
              AI Suggest
            </button>

            {/* Suggestion cards */}
            {suggestions.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                  Suggestions
                </p>
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => applySuggestion(s)}
                    className="w-full text-left rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-3 hover:border-blue-300 dark:hover:border-blue-700 hover:bg-blue-50 dark:hover:bg-blue-950/20 transition-colors"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-base">{CHART_ICONS[s.chart_type] || "📊"}</span>
                      <span className="text-sm font-medium text-slate-900 dark:text-white capitalize">
                        {s.chart_type}
                      </span>
                    </div>
                    {s.title && (
                      <p className="text-xs text-slate-700 dark:text-slate-300 mb-1">
                        {s.title}
                      </p>
                    )}
                    {s.reason && (
                      <p className="text-xs text-slate-500 dark:text-slate-400 italic">
                        {s.reason}
                      </p>
                    )}
                  </button>
                ))}
              </div>
            )}

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
          </div>

          {error && (
            <div className="flex items-center gap-2 text-red-500 text-sm rounded-xl border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950/20 p-3">
              <AlertCircle className="h-4 w-4 shrink-0" />
              {error}
            </div>
          )}
        </div>

        {/* Right column: Chart history */}
        <div className="lg:col-span-2 flex flex-col min-h-0">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 flex-1 overflow-y-auto">
            {chartHistory.length === 0 ? (
              <div className="flex items-center justify-center h-full min-h-[400px]">
                <div className="text-center text-slate-400 dark:text-slate-600">
                  <ImageIcon className="h-12 w-12 mx-auto mb-2" />
                  <p className="text-sm">
                    Configure and create a chart to see it here
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
                          className="max-w-full max-h-[500px]"
                        />
                      </div>
                    )}

                    {/* Info bar */}
                    <div className="flex items-center justify-between px-4 py-3 border-t border-slate-200 dark:border-slate-800">
                      <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                        <span>{CHART_ICONS[entry.chartType] || "📊"}</span>
                        <span className="font-medium">{chartLabel(entry)}</span>
                        {entry.title && (
                          <span className="text-slate-400 dark:text-slate-500">
                            — {entry.title}
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
                          Download PNG
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

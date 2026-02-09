"use client";

import { useState } from "react";
import {
  BarChart3,
  Loader2,
  Wand2,
  Image as ImageIcon,
  AlertCircle,
} from "lucide-react";
import {
  createChart,
  suggestChart,
  type ChartResponse,
  type ChartSuggestion,
} from "@/lib/api";
import { useSession } from "@/lib/store";

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

export default function VisualizePage() {
  const { sessionId, columns } = useSession();

  const [chartType, setChartType] = useState("histogram");
  const [xCol, setXCol] = useState("");
  const [yCol, setYCol] = useState("");
  const [hueCol, setHueCol] = useState("");
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [result, setResult] = useState<ChartResponse | null>(null);
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
      const res = await createChart(sessionId, {
        chart_type: chartType,
        x: xCol || undefined,
        y: yCol || undefined,
        hue: hueCol || undefined,
        title: title || undefined,
      });
      if (res.status === "error") {
        setError(res.error || "Chart creation failed");
        setResult(null);
      } else {
        setResult(res);
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

    try {
      const suggestion = await suggestChart(sessionId);
      setChartType(suggestion.chart_type || "histogram");
      if (suggestion.x) setXCol(suggestion.x);
      if (suggestion.y) setYCol(suggestion.y);
      if (suggestion.hue) setHueCol(suggestion.hue);
      if (suggestion.title) setTitle(suggestion.title);
    } catch {
      // Silently fail suggestion
    } finally {
      setSuggesting(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
        Chart Builder
      </h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="space-y-4">
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
        </div>

        {/* Preview */}
        <div className="lg:col-span-2">
          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 min-h-[400px] flex items-center justify-center">
            {error ? (
              <div className="flex items-center gap-2 text-red-500 text-sm p-4">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {error}
              </div>
            ) : result?.image_base64 ? (
              <img
                src={`data:image/png;base64,${result.image_base64}`}
                alt="Chart"
                className="max-w-full max-h-[600px] p-4"
              />
            ) : (
              <div className="text-center text-slate-400 dark:text-slate-600">
                <ImageIcon className="h-12 w-12 mx-auto mb-2" />
                <p className="text-sm">
                  Configure and create a chart to see the preview
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

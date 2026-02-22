"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Database,
  AlertTriangle,
  BarChart3,
} from "lucide-react";

interface ResultCardProps {
  result: Record<string, unknown>;
  skill: string;
}

// Metrics that represent 0-1 ratios and should be displayed as percentages
const PERCENT_METRICS = new Set([
  "accuracy", "precision", "recall", "f1", "auc_roc", "log_loss",
  "cv_accuracy_mean", "cv_accuracy_std", "cv_f1_mean", "cv_f1_std",
  "specificity", "sensitivity", "balanced_accuracy",
]);

function isPercentMetric(key: string): boolean {
  return PERCENT_METRICS.has(key);
}

function formatMetricValue(key: string, val: number): string {
  if (isPercentMetric(key)) {
    return (val * 100).toFixed(1) + "%";
  }
  if (Math.abs(val) >= 100) {
    return val.toLocaleString(undefined, { maximumFractionDigits: 1 });
  }
  return val.toFixed(3);
}

// Keys that get dedicated rendering or should be hidden from generic display
const HANDLED_KEYS = new Set([
  "status", "overview", "columns", "quality_score", "warnings",
  "recommendations", "correlations", "chart_base64", "image_base64",
  "chart_path", "chart_html_path", "chart_type", "chart_summary",
  "message", "correlation_matrix", "top_correlations",
  "multicollinearity_warning", "target_correlations",
  // Classification-specific
  "metrics", "confusion_matrix", "classification_report",
  "feature_importance", "cross_validation", "predictions",
  "features_used", "class_labels", "hyperparameters", "model_path",
  // Auto-classify comparison
  "auto_comparison", "comparison",
  // Clustering
  "cluster_profiles", "cluster_assignments", "output_path",
  "centroid", "distinguishing_features",
  // Hypothesis test
  "test", "statistic", "pvalue", "significant", "effect_size",
  "effect_interpretation", "conclusion", "details",
  // Outlier detection
  "total_rows", "outlier_count", "outlier_pct", "outlier_indices",
  "outlier_scores", "column_analysis",
]);

/**
 * Renders structured skill result data as a formatted card.
 * Handles common patterns: overview tables, quality scores,
 * warnings, correlations, and generic key-value data.
 */
export function ResultCard({ result }: Omit<ResultCardProps, "skill"> & { skill?: string }) {
  const [expanded, setExpanded] = useState(false);

  // Skip rendering if result only has "status"
  const keys = Object.keys(result).filter((k) => k !== "status");
  if (keys.length === 0) return null;

  const overview = result.overview as
    | Record<string, unknown>
    | undefined;
  const qualityScore = result.quality_score as number | undefined;
  const warnings = result.warnings as
    | { type: string; column: string; detail: string }[]
    | undefined;
  const correlations = result.correlations as
    | { high_correlations?: { col1: string; col2: string; correlation: number }[] }
    | undefined;
  const chartBase64 = result.chart_base64 as string | undefined;
  const imageBase64 = result.image_base64 as string | undefined;

  // Correlation-specific fields (from analyze_correlations skill)
  const topCorrelations = result.top_correlations as
    | { col1: string; col2: string; correlation: number; pvalue: number }[]
    | undefined;
  const multicollinearity = result.multicollinearity_warning as
    | { col1: string; col2: string; correlation: number }[]
    | undefined;
  const targetCorrelations = result.target_correlations as
    | { column: string; correlation: number; pvalue: number; significant: boolean }[]
    | undefined;

  // Classification-specific fields
  const metrics = result.metrics as Record<string, number | null> | undefined;
  const confusionMatrix = result.confusion_matrix as number[][] | undefined;
  const featureImportance = result.feature_importance as
    | { feature: string; importance: number }[]
    | undefined;
  const crossValidation = result.cross_validation as
    | { cv_folds: number; cv_accuracy_mean: number; cv_accuracy_std: number; cv_scores?: number[] }
    | undefined;
  const classLabels = result.class_labels as (string | number)[] | undefined;
  const comparison = (result.auto_comparison || result.comparison) as
    | { algorithm: string; cv_accuracy_mean: number; cv_accuracy_std: number; cv_f1_mean?: number }[]
    | undefined;

  // Clustering-specific fields
  const clusterProfiles = result.cluster_profiles as
    | {
        cluster: number;
        size: number;
        label_suggestion: string;
        distinguishing_features: { feature: string; cluster_mean: number; overall_mean: number; diff_pct: number }[];
      }[]
    | undefined;

  // For non-profiler skills: extract scalar stats to show
  const statsKeys = keys.filter(
    (k) => !HANDLED_KEYS.has(k) && typeof result[k] !== "object"
  );

  // Collect table-like arrays (e.g., numeric_summary, categorical_summary)
  const tableKeys = keys.filter(
    (k) =>
      !HANDLED_KEYS.has(k) &&
      Array.isArray(result[k]) &&
      (result[k] as unknown[]).length > 0
  );

  // Object-type entries for expandable raw data (includes correlation_matrix)
  const objectKeys = keys.filter(
    (k) =>
      !HANDLED_KEYS.has(k) &&
      typeof result[k] === "object" &&
      result[k] !== null &&
      !Array.isArray(result[k])
  );

  // Heavy objects only shown when expanded (like correlation_matrix)
  const heavyKeys = keys.filter(
    (k) =>
      k === "correlation_matrix" &&
      typeof result[k] === "object" &&
      result[k] !== null
  );

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden bg-slate-50 dark:bg-slate-900/50">
      {/* Header */}
      <div className="px-4 py-2.5 bg-slate-100 dark:bg-slate-800/50 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-300">
          <Database className="h-3.5 w-3.5" />
          Result data
        </div>
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 flex items-center gap-0.5"
        >
          {expanded ? "Collapse" : "Expand"}
          {expanded ? (
            <ChevronUp className="h-3 w-3" />
          ) : (
            <ChevronDown className="h-3 w-3" />
          )}
        </button>
      </div>

      <div className="px-4 py-3 space-y-3">
        {/* Overview stats grid */}
        {overview && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {Object.entries(overview).map(([key, val]) => (
              <div
                key={key}
                className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center"
              >
                <div className="text-lg font-semibold text-slate-900 dark:text-white">
                  {formatValue(val)}
                </div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                  {formatLabel(key)}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Quality score */}
        {qualityScore !== undefined && (
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  qualityScore >= 80
                    ? "bg-green-500"
                    : qualityScore >= 60
                    ? "bg-amber-500"
                    : "bg-red-500"
                }`}
                style={{ width: `${Math.min(qualityScore, 100)}%` }}
              />
            </div>
            <span className="text-xs font-medium text-slate-600 dark:text-slate-300 shrink-0">
              Quality: {qualityScore}%
            </span>
          </div>
        )}

        {/* Warnings */}
        {warnings && warnings.length > 0 && (
          <div className="space-y-1">
            {warnings.slice(0, expanded ? warnings.length : 3).map((w, i) => (
              <div
                key={i}
                className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400"
              >
                <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5" />
                <span>
                  <span className="font-medium">{w.column}</span>: {w.detail}
                </span>
              </div>
            ))}
            {!expanded && warnings.length > 3 && (
              <p className="text-[10px] text-slate-400">
                +{warnings.length - 3} more warnings
              </p>
            )}
          </div>
        )}

        {/* Recommendations */}
        {Array.isArray(result.recommendations) && (result.recommendations as string[]).length > 0 ? (
          <div className="space-y-1">
            <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400 uppercase tracking-wider">
              Recommendations
            </p>
            {(result.recommendations as string[]).slice(0, expanded ? undefined : 3).map((rec: string, i: number) => (
              <div key={i} className="flex items-start gap-1.5 text-xs text-slate-600 dark:text-slate-400">
                <span className="text-emerald-400 mt-0.5 shrink-0">&rarr;</span>
                <span>{rec}</span>
              </div>
            ))}
            {!expanded && (result.recommendations as string[]).length > 3 && (
              <p className="text-[10px] text-slate-400">
                +{(result.recommendations as string[]).length - 3} more recommendations
              </p>
            )}
          </div>
        ) : null}

        {/* High correlations (from profiler) */}
        {correlations?.high_correlations &&
          correlations.high_correlations.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400">
                <BarChart3 className="h-3 w-3" />
                High correlations
              </div>
              {correlations.high_correlations
                .slice(0, expanded ? correlations.high_correlations.length : 5)
                .map((c, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400 pl-4"
                  >
                    <span>
                      {c.col1} &harr; {c.col2}
                    </span>
                    <span
                      className={`font-mono ${
                        Math.abs(c.correlation) > 0.9
                          ? "text-red-500"
                          : "text-amber-500"
                      }`}
                    >
                      {c.correlation.toFixed(3)}
                    </span>
                  </div>
                ))}
            </div>
          )}

        {/* Top correlations (from analyze_correlations) */}
        {topCorrelations && topCorrelations.length > 0 && (
          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400">
              <BarChart3 className="h-3 w-3" />
              Top Correlations
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700">
                    <th className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Column 1</th>
                    <th className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Column 2</th>
                    <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">Correlation</th>
                    <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">p-value</th>
                  </tr>
                </thead>
                <tbody>
                  {topCorrelations
                    .slice(0, expanded ? topCorrelations.length : 10)
                    .map((c, i) => (
                    <tr key={i} className="border-b border-slate-100 dark:border-slate-800">
                      <td className="px-2 py-1 text-slate-600 dark:text-slate-300">{c.col1}</td>
                      <td className="px-2 py-1 text-slate-600 dark:text-slate-300">{c.col2}</td>
                      <td className={`px-2 py-1 text-right font-mono ${
                        Math.abs(c.correlation) > 0.9 ? "text-red-500" :
                        Math.abs(c.correlation) > 0.7 ? "text-amber-500" :
                        "text-slate-600 dark:text-slate-300"
                      }`}>
                        {c.correlation.toFixed(4)}
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-slate-500 dark:text-slate-400">
                        {c.pvalue < 0.001 ? "<0.001" : c.pvalue.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!expanded && topCorrelations.length > 10 && (
                <p className="text-[10px] text-slate-400 mt-1">
                  +{topCorrelations.length - 10} more pairs
                </p>
              )}
            </div>
          </div>
        )}

        {/* Multicollinearity warning (from analyze_correlations) */}
        {multicollinearity && multicollinearity.length > 0 ? (
          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-medium text-amber-600 dark:text-amber-400">
              <AlertTriangle className="h-3 w-3" />
              Multicollinearity Warning ({multicollinearity.length} pairs)
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700">
                    <th className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Column 1</th>
                    <th className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Column 2</th>
                    <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">|r|</th>
                  </tr>
                </thead>
                <tbody>
                  {multicollinearity.map((c, i) => (
                    <tr key={i} className="border-b border-slate-100 dark:border-slate-800">
                      <td className="px-2 py-1 text-slate-600 dark:text-slate-300">{c.col1}</td>
                      <td className="px-2 py-1 text-slate-600 dark:text-slate-300">{c.col2}</td>
                      <td className="px-2 py-1 text-right font-mono text-red-500">
                        {Math.abs(c.correlation).toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        {/* Target correlations (from analyze_correlations with target) */}
        {targetCorrelations && targetCorrelations.length > 0 ? (
          <div className="space-y-1">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Target Correlations
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700">
                    <th className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Column</th>
                    <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">Correlation</th>
                    <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">p-value</th>
                    <th className="px-2 py-1 text-center font-medium text-slate-500 dark:text-slate-400">Sig.</th>
                  </tr>
                </thead>
                <tbody>
                  {targetCorrelations.map((c, i) => (
                    <tr key={i} className="border-b border-slate-100 dark:border-slate-800">
                      <td className="px-2 py-1 text-slate-600 dark:text-slate-300">{c.column}</td>
                      <td className="px-2 py-1 text-right font-mono text-slate-600 dark:text-slate-300">
                        {c.correlation.toFixed(4)}
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-slate-500 dark:text-slate-400">
                        {c.pvalue < 0.001 ? "<0.001" : c.pvalue.toFixed(4)}
                      </td>
                      <td className="px-2 py-1 text-center">
                        {c.significant ? (
                          <span className="text-emerald-500">Yes</span>
                        ) : (
                          <span className="text-slate-400">No</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}

        {/* Classification: Metrics grid */}
        {metrics && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Model Performance
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
              {Object.entries(metrics)
                .filter(([, v]) => v !== null && v !== undefined)
                .map(([key, val]) => (
                  <div
                    key={key}
                    className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center"
                  >
                    <div className={`text-lg font-bold ${
                      key === "accuracy" || key === "f1" || key === "auc_roc"
                        ? (val as number) >= 0.8 ? "text-emerald-600 dark:text-emerald-400"
                          : (val as number) >= 0.6 ? "text-amber-600 dark:text-amber-400"
                          : "text-red-600 dark:text-red-400"
                        : "text-slate-900 dark:text-white"
                    }`}>
                      {typeof val === "number" ? formatMetricValue(key, val) : String(val)}
                    </div>
                    <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase tracking-wider">
                      {formatLabel(key)}
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Classification: Cross Validation */}
        {crossValidation && (
          <div className="flex items-center gap-4 text-xs text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800 rounded-lg px-4 py-2">
            <span className="text-slate-400 dark:text-slate-500">
              {crossValidation.cv_folds}-fold CV:
            </span>
            <span className="font-mono font-medium">
              {(crossValidation.cv_accuracy_mean * 100).toFixed(1)}%
              <span className="text-slate-400 dark:text-slate-500">
                {" "}± {(crossValidation.cv_accuracy_std * 100).toFixed(1)}%
              </span>
            </span>
          </div>
        )}

        {/* Classification: Feature Importance */}
        {featureImportance && featureImportance.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Feature Importance (top {Math.min(featureImportance.length, expanded ? 20 : 10)})
            </p>
            <div className="space-y-1">
              {featureImportance
                .slice(0, expanded ? 20 : 10)
                .map((fi, i) => {
                  const maxImp = featureImportance[0]?.importance || 1;
                  const pct = maxImp > 0 ? (fi.importance / maxImp) * 100 : 0;
                  return (
                    <div key={i} className="flex items-center gap-2">
                      <span className="text-xs text-slate-600 dark:text-slate-300 w-32 truncate shrink-0 text-right">
                        {fi.feature}
                      </span>
                      <div className="flex-1 h-4 rounded bg-slate-100 dark:bg-slate-800 overflow-hidden">
                        <div
                          className={`h-full rounded ${
                            i < 3
                              ? "bg-blue-500 dark:bg-blue-400"
                              : "bg-slate-300 dark:bg-slate-600"
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-slate-500 dark:text-slate-400 w-14 text-right shrink-0">
                        {fi.importance.toFixed(4)}
                      </span>
                    </div>
                  );
                })}
              {!expanded && featureImportance.length > 10 && (
                <p className="text-[10px] text-slate-400 mt-1">
                  +{featureImportance.length - 10} more features
                </p>
              )}
            </div>
          </div>
        )}

        {/* Classification: Confusion Matrix */}
        {confusionMatrix && confusionMatrix.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Confusion Matrix
            </p>
            <div className="overflow-x-auto">
              <table className="text-xs">
                <thead>
                  <tr>
                    <th className="px-2 py-1 text-slate-400 dark:text-slate-500"></th>
                    {(classLabels || confusionMatrix[0].map((_, i) => i)).map((label, i) => (
                      <th key={i} className="px-3 py-1 text-center font-medium text-slate-500 dark:text-slate-400">
                        Pred: {String(label)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {confusionMatrix.map((row, ri) => (
                    <tr key={ri}>
                      <td className="px-2 py-1 font-medium text-slate-500 dark:text-slate-400 text-right">
                        True: {String((classLabels || confusionMatrix.map((_, i) => i))[ri])}
                      </td>
                      {row.map((val, ci) => {
                        const isDiagonal = ri === ci;
                        return (
                          <td
                            key={ci}
                            className={`px-3 py-1.5 text-center font-mono ${
                              isDiagonal
                                ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 font-bold"
                                : val > 0
                                ? "bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400"
                                : "text-slate-400 dark:text-slate-600"
                            }`}
                          >
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Auto-classify: Algorithm Comparison */}
        {comparison && comparison.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Algorithm Comparison
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700">
                    <th className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Algorithm</th>
                    <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">CV Accuracy</th>
                    <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">± Std</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.map((c, i) => (
                    <tr key={i} className={`border-b border-slate-100 dark:border-slate-800 ${
                      i === 0 ? "bg-emerald-50/50 dark:bg-emerald-900/10" : ""
                    }`}>
                      <td className="px-2 py-1 text-slate-600 dark:text-slate-300">
                        {formatLabel(c.algorithm)}
                        {i === 0 && <span className="ml-1 text-emerald-600 dark:text-emerald-400 text-[10px]">best</span>}
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-slate-600 dark:text-slate-300">
                        {(c.cv_accuracy_mean * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-1 text-right font-mono text-slate-400">
                        {(c.cv_accuracy_std * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Clustering: Cluster Profiles */}
        {clusterProfiles && clusterProfiles.length > 0 && (
          <div className="space-y-3">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Cluster Profiles ({clusterProfiles.length} clusters)
            </p>
            {clusterProfiles
              .slice(0, expanded ? clusterProfiles.length : 4)
              .map((cp) => (
                <div
                  key={cp.cluster}
                  className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden"
                >
                  <div className="px-3 py-2 bg-slate-100 dark:bg-slate-800/50 flex items-center justify-between">
                    <span className="text-xs font-medium text-slate-700 dark:text-slate-300">
                      {cp.label_suggestion || `Cluster ${cp.cluster}`}
                    </span>
                    <span className="text-xs text-slate-400 dark:text-slate-500">
                      {cp.size.toLocaleString()} records
                    </span>
                  </div>
                  {cp.distinguishing_features && cp.distinguishing_features.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-200 dark:border-slate-700">
                            <th className="px-3 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Feature</th>
                            <th className="px-3 py-1 text-right font-medium text-slate-500 dark:text-slate-400">Cluster Avg</th>
                            <th className="px-3 py-1 text-right font-medium text-slate-500 dark:text-slate-400">Overall Avg</th>
                            <th className="px-3 py-1 text-right font-medium text-slate-500 dark:text-slate-400">Difference</th>
                          </tr>
                        </thead>
                        <tbody>
                          {cp.distinguishing_features.map((df, j) => (
                            <tr key={j} className="border-b border-slate-100 dark:border-slate-800">
                              <td className="px-3 py-1 text-slate-600 dark:text-slate-300">{df.feature}</td>
                              <td className="px-3 py-1 text-right font-mono text-slate-600 dark:text-slate-300">
                                {df.cluster_mean.toFixed(2)}
                              </td>
                              <td className="px-3 py-1 text-right font-mono text-slate-400 dark:text-slate-500">
                                {df.overall_mean.toFixed(2)}
                              </td>
                              <td className={`px-3 py-1 text-right font-mono font-medium ${
                                df.diff_pct > 20
                                  ? "text-emerald-600 dark:text-emerald-400"
                                  : df.diff_pct < -20
                                  ? "text-red-600 dark:text-red-400"
                                  : "text-slate-500 dark:text-slate-400"
                              }`}>
                                {df.diff_pct > 0 ? "+" : ""}{df.diff_pct.toFixed(1)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ))}
            {!expanded && clusterProfiles.length > 4 && (
              <p className="text-[10px] text-slate-400">
                +{clusterProfiles.length - 4} more clusters
              </p>
            )}
          </div>
        )}

        {/* Hypothesis Test Result */}
        {result.pvalue !== undefined && result.test ? (
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
                {String(result.test).replace(/_/g, " ").toUpperCase()}
              </span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                result.significant
                  ? "bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
              }`}>
                {result.significant ? "Significant" : "Not significant"}
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <div className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center">
                <div className="text-lg font-bold text-slate-900 dark:text-white">
                  {Number(result.pvalue) < 0.001 ? "<0.001" : Number(result.pvalue).toFixed(4)}
                </div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase">p-value</div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center">
                <div className="text-lg font-bold text-slate-900 dark:text-white">
                  {Number(result.statistic).toFixed(3)}
                </div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase">Test Statistic</div>
              </div>
              {result.effect_size != null ? (
                <div className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center">
                  <div className="text-lg font-bold text-slate-900 dark:text-white">
                    {Number(result.effect_size).toFixed(3)}
                  </div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase">Effect Size</div>
                </div>
              ) : null}
            </div>
            {result.conclusion ? (
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg px-3 py-2 text-sm text-slate-700 dark:text-slate-300">
                {String(result.conclusion)}
              </div>
            ) : null}
            {result.effect_interpretation ? (
              <p className="text-xs text-slate-500 dark:text-slate-400 italic">
                {String(result.effect_interpretation)}
              </p>
            ) : null}
          </div>
        ) : null}

        {/* Outlier Detection Result */}
        {result.outlier_count !== undefined ? (
          <div className="space-y-2">
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center">
                <div className="text-lg font-bold text-red-600 dark:text-red-400">
                  {Number(result.outlier_count).toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase">Outliers Found</div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center">
                <div className="text-lg font-bold text-slate-900 dark:text-white">
                  {Number(result.outlier_pct).toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase">Of Dataset</div>
              </div>
              <div className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center">
                <div className="text-lg font-bold text-slate-900 dark:text-white">
                  {String(result.method || "").replace(/_/g, " ")}
                </div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase">Method</div>
              </div>
            </div>
            {Array.isArray(result.column_analysis) && (result.column_analysis as Record<string, unknown>[]).length > 0 ? (
              <div className="space-y-1">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                  Column Analysis
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 dark:border-slate-700">
                        <th className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400">Column</th>
                        <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">Outliers</th>
                        <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">Low Threshold</th>
                        <th className="px-2 py-1 text-right font-medium text-slate-500 dark:text-slate-400">High Threshold</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(result.column_analysis as Record<string, unknown>[])
                        .slice(0, expanded ? undefined : 10)
                        .map((ca, i) => (
                          <tr key={i} className="border-b border-slate-100 dark:border-slate-800">
                            <td className="px-2 py-1 text-slate-600 dark:text-slate-300">
                              {String(ca.column || ca.feature || ca.name || "")}
                            </td>
                            <td className="px-2 py-1 text-right font-mono text-red-600 dark:text-red-400">
                              {String(ca.outlier_count ?? ca.n_outliers ?? "-")}
                            </td>
                            <td className="px-2 py-1 text-right font-mono text-slate-500 dark:text-slate-400">
                              {ca.threshold_low != null ? Number(ca.threshold_low).toFixed(2) : "-"}
                            </td>
                            <td className="px-2 py-1 text-right font-mono text-slate-500 dark:text-slate-400">
                              {ca.threshold_high != null ? Number(ca.threshold_high).toFixed(2) : "-"}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {/* Flat stats (non-object scalar values) */}
        {statsKeys.length > 0 && !overview && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {statsKeys.slice(0, expanded ? statsKeys.length : 9).map((key) => (
              <div
                key={key}
                className="bg-white dark:bg-slate-800 rounded-lg px-3 py-2 text-center"
              >
                <div className="text-sm font-semibold text-slate-900 dark:text-white truncate">
                  {formatValue(result[key])}
                </div>
                <div className="text-[10px] text-slate-400 dark:text-slate-500 uppercase tracking-wider truncate">
                  {formatLabel(key)}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Array tables (e.g., numeric_summary, categorical_summary) */}
        {tableKeys.slice(0, expanded ? tableKeys.length : 2).map((key) => {
          const arr = result[key] as Record<string, unknown>[];
          if (!arr[0] || typeof arr[0] !== "object") return null;
          const cols = Object.keys(arr[0]).slice(0, expanded ? undefined : 6);
          const rows = arr.slice(0, expanded ? arr.length : 5);
          return (
            <div key={key} className="space-y-1">
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                {formatLabel(key)}
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-200 dark:border-slate-700">
                      {cols.map((col) => (
                        <th
                          key={col}
                          className="px-2 py-1 text-left font-medium text-slate-500 dark:text-slate-400"
                        >
                          {formatLabel(col)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, ri) => (
                      <tr
                        key={ri}
                        className="border-b border-slate-100 dark:border-slate-800"
                      >
                        {cols.map((col) => (
                          <td
                            key={col}
                            className="px-2 py-1 text-slate-600 dark:text-slate-300 max-w-[200px]"
                          >
                            {formatCellValue(row[col])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!expanded && arr.length > 5 && (
                  <p className="text-[10px] text-slate-400 mt-1">
                    +{arr.length - 5} more rows
                  </p>
                )}
              </div>
            </div>
          );
        })}

        {/* Object entries (expandable, excludes heavy keys) */}
        {expanded &&
          objectKeys.map((key) => (
            <div key={key} className="space-y-1">
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                {formatLabel(key)}
              </p>
              <pre className="text-[11px] text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 rounded-lg p-2 overflow-x-auto max-h-48">
                {JSON.stringify(result[key], null, 2)}
              </pre>
            </div>
          ))}

        {/* Heavy data (correlation_matrix) — only when expanded */}
        {expanded &&
          heavyKeys.map((key) => (
            <div key={key} className="space-y-1">
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                {formatLabel(key)}
              </p>
              <pre className="text-[11px] text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 rounded-lg p-2 overflow-x-auto max-h-60">
                {JSON.stringify(result[key], null, 2)}
              </pre>
            </div>
          ))}

        {/* Chart image */}
        {(chartBase64 || imageBase64) && (
          <div className="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${chartBase64 || imageBase64}`}
              alt="Analysis chart"
              className="w-full"
            />
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) return "-";
  if (typeof val === "number") {
    if (Number.isInteger(val)) return val.toLocaleString();
    return val.toFixed(2);
  }
  if (typeof val === "boolean") return val ? "Yes" : "No";
  if (Array.isArray(val)) return `[${val.length} items]`;
  if (typeof val === "object") return summarizeObject(val as Record<string, unknown>);
  return String(val);
}

/**
 * Format a table cell value. Unlike formatValue, this handles nested
 * arrays-of-objects (like value_counts) by rendering them inline.
 */
function formatCellValue(val: unknown): string {
  if (val === null || val === undefined) return "-";
  if (typeof val === "number") {
    if (Number.isInteger(val)) return val.toLocaleString();
    return val.toFixed(2);
  }
  if (typeof val === "boolean") return val ? "Yes" : "No";

  // Handle nested arrays of {value, count, pct} (e.g., value_counts, top_values)
  if (Array.isArray(val)) {
    const items = val.slice(0, 5);
    const parts = items.map((item) => {
      if (typeof item === "object" && item !== null) {
        const obj = item as Record<string, unknown>;
        // value_counts format: {value, count, pct}
        if ("value" in obj && "count" in obj) {
          const pctStr = typeof obj.pct === "number" ? ` (${obj.pct}%)` : "";
          return `${obj.value}: ${obj.count}${pctStr}`;
        }
        // Generic object in array: show first meaningful field
        const entries = Object.entries(obj);
        if (entries.length > 0) {
          return entries.map(([k, v]) => `${k}=${v}`).join(", ");
        }
      }
      return String(item);
    });
    const suffix = val.length > 5 ? `, +${val.length - 5} more` : "";
    return parts.join("; ") + suffix;
  }

  // Handle plain objects (e.g., a nested dict)
  if (typeof val === "object") {
    return summarizeObject(val as Record<string, unknown>);
  }

  return String(val);
}

/** Summarize an object into a short readable string */
function summarizeObject(obj: Record<string, unknown>): string {
  // value_counts-style: {value, count, pct}
  if ("value" in obj && "count" in obj) {
    const pctStr = typeof obj.pct === "number" ? ` (${obj.pct}%)` : "";
    return `${obj.value}: ${obj.count}${pctStr}`;
  }
  const entries = Object.entries(obj).slice(0, 3);
  const parts = entries.map(([k, v]) => {
    if (typeof v === "number") return `${k}: ${Number.isInteger(v) ? v : (v as number).toFixed(2)}`;
    return `${k}: ${v}`;
  });
  const suffix = Object.keys(obj).length > 3 ? ", ..." : "";
  return parts.join(", ") + suffix;
}

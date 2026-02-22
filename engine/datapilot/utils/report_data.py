"""
Report data — format analysis results for LLM narrative generation.

Structures any analysis result into headline, key metrics, findings,
recommendations, and supporting data.
"""

from typing import Any

from .helpers import setup_logging
from .serializer import safe_json_serialize

logger = setup_logging("datapilot.report_data")


def format_for_narrative(results: dict, context: str | None = None) -> dict:
    """
    Structure any analysis result for narrative generation.

    Accepts raw output from any skill function and organises it into
    headline, key_metrics, key_findings, recommendations, supporting_data,
    and visualizations.
    """
    try:
        status = results.get("status", "unknown")
        if status == "error":
            return {
                "headline": "Analysis encountered an error",
                "key_metrics": [],
                "key_findings": [{"finding": results.get("message", "Unknown error"),
                                  "evidence": "", "importance": "high"}],
                "recommendations": [],
                "supporting_data": results,
                "visualizations": [],
            }

        headline = _generate_headline(results, context)
        metrics = _extract_metrics(results)
        findings = _extract_findings(results)
        recs = _extract_recommendations(results)
        viz = _extract_viz_paths(results)

        return safe_json_serialize({
            "headline": headline,
            "key_metrics": metrics,
            "key_findings": findings,
            "recommendations": recs,
            "supporting_data": results,
            "visualizations": viz,
        })

    except Exception as e:
        return {"headline": "Error formatting results", "error": str(e)}


def create_executive_summary_data(results: dict) -> dict:
    """Format for the executive summary section of a report."""
    try:
        formatted = format_for_narrative(results)
        return safe_json_serialize({
            "section": "executive_summary",
            "headline": formatted.get("headline", ""),
            "top_metrics": formatted.get("key_metrics", [])[:5],
            "top_findings": formatted.get("key_findings", [])[:3],
            "top_recommendations": formatted.get("recommendations", [])[:3],
        })
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_detailed_findings_data(results: dict) -> dict:
    """Format for the detailed findings section."""
    try:
        formatted = format_for_narrative(results)
        return safe_json_serialize({
            "section": "detailed_findings",
            "all_metrics": formatted.get("key_metrics", []),
            "all_findings": formatted.get("key_findings", []),
            "all_recommendations": formatted.get("recommendations", []),
            "visualizations": formatted.get("visualizations", []),
            "data": formatted.get("supporting_data", {}),
        })
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_headline(results: dict, context: str | None = None) -> str:
    """Generate a one-sentence summary."""
    if context:
        return context

    if "algorithm" in results:
        algo = results["algorithm"]
        target = results.get("target", "target")
        metrics = results.get("metrics", {})
        if "accuracy" in metrics:
            return f"{algo.title()} classifier achieved {metrics['accuracy']*100:.1f}% accuracy predicting {target}"
        if "r2" in metrics:
            return f"{algo.title()} regression achieved R²={metrics['r2']:.3f} predicting {target}"

    if "n_clusters" in results:
        return f"Found {results['n_clusters']} distinct clusters in the data"

    if "quality_score" in results:
        return f"Data quality score: {results['quality_score']}/100"

    if "thresholds" in results:
        n = len(results.get("thresholds", []))
        return f"Identified {n} significant threshold tipping points"

    if "trend" in results:
        direction = results["trend"].get("direction", "")
        return f"Time series shows {direction} trend"

    if "distribution" in results:
        return "Sentiment analysis complete"

    return "Analysis complete"


def _extract_metrics(results: dict) -> list[dict[str, Any]]:
    """Pull out key metrics from results."""
    metrics: list[dict[str, Any]] = []

    m = results.get("metrics", {})
    for key in ["accuracy", "precision", "recall", "f1", "auc_roc", "r2", "rmse", "mae"]:
        if key in m and m[key] is not None:
            metrics.append({
                "metric": key,
                "value": m[key],
                "change": None,
                "good_or_bad": "good" if key in ("accuracy", "r2", "auc_roc") and m[key] > 0.7 else "neutral",
            })

    overview = results.get("overview", {})
    if "rows" in overview:
        metrics.append({"metric": "rows", "value": overview["rows"], "change": None, "good_or_bad": "neutral"})
    if "quality_score" in results:
        qs = results["quality_score"]
        metrics.append({"metric": "quality_score", "value": qs,
                        "good_or_bad": "good" if qs > 80 else ("bad" if qs < 50 else "neutral")})

    sil = results.get("metrics", {}).get("silhouette_score")
    if sil is not None:
        metrics.append({"metric": "silhouette_score", "value": sil, "change": None,
                        "good_or_bad": "good" if sil > 0.5 else "neutral"})

    return metrics


def _extract_findings(results: dict) -> list[dict[str, Any]]:
    """Pull out key findings."""
    findings: list[dict[str, Any]] = []

    for w in results.get("warnings", []):
        findings.append({
            "finding": w.get("detail", ""),
            "evidence": f"Column: {w.get('column', '')}",
            "importance": "high" if w.get("type") == "high_missing" else "medium",
        })

    for t in results.get("thresholds", [])[:5]:
        findings.append({
            "finding": t.get("insight", ""),
            "evidence": f"lift={t.get('lift', 0)}x, p={t.get('pvalue', 'N/A')}",
            "importance": "high" if t.get("lift", 0) > 2 else "medium",
        })

    for f in results.get("feature_importance", [])[:5]:
        findings.append({
            "finding": f"Feature '{f['feature']}' is a top predictor (importance={f.get('importance', f.get('mean_shap', 0)):.4f})",
            "evidence": "Model feature importance",
            "importance": "medium",
        })

    for cp in results.get("cluster_profiles", [])[:3]:
        findings.append({
            "finding": f"Cluster {cp['cluster']}: {cp.get('label_suggestion', '')} ({cp['size']} records)",
            "evidence": "Clustering analysis",
            "importance": "medium",
        })

    return findings


def _extract_recommendations(results: dict) -> list[dict[str, Any]]:
    """Pull out recommendations."""
    recs: list[dict[str, Any]] = []

    for r in results.get("recommendations", []):
        if isinstance(r, str):
            recs.append({"action": r, "rationale": "", "priority": "medium"})
        elif isinstance(r, dict):
            recs.append(r)

    for t in results.get("thresholds", [])[:3]:
        feature = t.get("feature", "").replace("_", " ").title()
        threshold = t.get("threshold", 0)
        recs.append({
            "action": f"Monitor when {feature} reaches {threshold}",
            "rationale": t.get("insight", ""),
            "priority": "high" if t.get("lift", 0) > 2 else "medium",
        })

    return recs


def _extract_viz_paths(results: dict) -> list[str]:
    """Find any chart/plot paths in results."""
    paths = []
    for key in ["chart_path", "chart_html_path", "output_path",
                 "summary_plot_path", "force_plot_path",
                 "elbow_plot_path", "silhouette_plot_path",
                 "scree_plot_path", "forecast_plot_path",
                 "survival_plot_path", "decomposition_plot_path",
                 "topic_distribution_path"]:
        val = results.get(key)
        if val:
            paths.append(val)

    for chart in results.get("charts", []):
        if isinstance(chart, dict):
            for k in ["chart_path", "chart_html_path"]:
                if chart.get(k):
                    paths.append(chart[k])

    return paths

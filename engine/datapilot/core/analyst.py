"""
Analyst — the main public API for DataPilot.

Usage:
    from datapilot import Analyst

    analyst = Analyst("data.csv")
    result = analyst.ask("What are the key trends?")
    print(result.narrative)
"""

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..llm.provider import LLMProvider, NarrativeResult, RoutingResult
from ..utils.config import Config
from ..utils.helpers import load_data
from ..utils.serializer import safe_json_serialize
from .executor import ExecutionResult, Executor
from .router import Router, build_data_context

logger = logging.getLogger("datapilot.core.analyst")


@dataclass
class AnalystResult:
    """Result from Analyst.ask() — contains routing, execution, and narrative."""

    question: str
    routing: RoutingResult
    execution: ExecutionResult
    narrative: Optional[NarrativeResult] = None
    routing_ms: float = 0.0
    execution_ms: float = 0.0
    narration_ms: float = 0.0
    _narration_thread: Optional[threading.Thread] = field(
        default=None, repr=False, compare=False,
    )

    @property
    def skill_name(self) -> str:
        return self.routing.skill_name

    @property
    def status(self) -> str:
        return self.execution.status

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        return self.execution.result

    @property
    def route_method(self) -> str:
        return self.routing.route_method

    @property
    def code_snippet(self) -> Optional[str]:
        return self.execution.code_snippet

    @property
    def columns_used(self) -> Optional[list]:
        return self.execution.columns_used

    @property
    def narrative_pending(self) -> bool:
        """True if narration was requested but hasn't finished yet."""
        t = self._narration_thread
        return t is not None and t.is_alive()

    @property
    def text(self) -> str:
        if self.narrative:
            return self.narrative.text
        return "Analysis complete. See raw results for details."

    @property
    def key_points(self) -> List[str]:
        if self.narrative:
            return self.narrative.key_points
        return []

    @property
    def suggestions(self) -> List[str]:
        if self.narrative:
            return self.narrative.suggestions
        return []

    def wait_for_narrative(self, timeout: float = 30.0) -> Optional[NarrativeResult]:
        """Block until background narration finishes (or timeout)."""
        t = self._narration_thread
        if t is not None:
            t.join(timeout=timeout)
        return self.narrative

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "question": self.question,
            "skill": self.routing.skill_name,
            "confidence": self.routing.confidence,
            "reasoning": self.routing.reasoning,
            "route_method": self.routing.route_method,
            "status": self.execution.status,
            "elapsed_seconds": self.execution.elapsed_seconds,
            "routing_ms": round(self.routing_ms, 1),
            "execution_ms": round(self.execution_ms, 1),
            "narration_ms": round(self.narration_ms, 1),
        }
        if self.execution.result:
            d["result"] = self.execution.result
        if self.execution.code_snippet:
            d["code_snippet"] = self.execution.code_snippet
        if self.execution.columns_used:
            d["columns_used"] = self.execution.columns_used
        if self.narrative:
            d["narrative"] = self.narrative.text
            d["key_points"] = self.narrative.key_points
            d["suggestions"] = self.narrative.suggestions
        elif self.narrative_pending:
            d["narrative_pending"] = True
        if self.execution.error:
            d["error"] = self.execution.error
        return d

    def __repr__(self) -> str:
        return (
            f"AnalystResult(skill={self.skill_name!r}, "
            f"status={self.status!r}, "
            f"confidence={self.routing.confidence:.2f})"
        )


# Keys stripped from results before passing to LLM narration.
# These are large binary blobs or server paths with no analytical value.
_NARRATION_EXCLUDED_KEYS = {
    "chart_base64", "image_base64", "chart_path", "chart_html_path",
}


def _verify_narrative(
    narrative: NarrativeResult,
    analysis_result: Dict[str, Any],
) -> bool:
    """Verify that LLM narrative numbers actually appear in the analysis result.

    Extracts all numbers from the narrative text and checks what percentage
    appear in the result data. Rejects narratives where < 50% of numbers match.
    Also checks that suggestion column names match actual dataset columns.
    """
    import re as _re

    text = narrative.text
    if not text:
        return False

    # Extract all numbers from narrative text
    narrative_numbers = set()
    for match in _re.finditer(r'\d+\.?\d*', text):
        num_str = match.group()
        try:
            num = float(num_str)
            narrative_numbers.add(num)
            # Also add integer form if it's a whole number
            if num == int(num):
                narrative_numbers.add(int(num))
        except ValueError:
            continue

    if not narrative_numbers:
        # No numbers to verify — narrative is text-only, accept it
        return True

    # Flatten all numeric values from the analysis result (recursive)
    result_numbers = set()

    def _collect_numbers(obj):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            result_numbers.add(obj)
            if isinstance(obj, float) and obj == int(obj):
                result_numbers.add(int(obj))
            # Add common rounded forms
            if isinstance(obj, float):
                for decimals in (0, 1, 2, 3, 4):
                    result_numbers.add(round(obj, decimals))
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect_numbers(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect_numbers(item)
        elif isinstance(obj, str):
            # Extract numbers from string values too (e.g. "42.5%")
            for m in _re.finditer(r'\d+\.?\d*', obj):
                try:
                    result_numbers.add(float(m.group()))
                    if float(m.group()) == int(float(m.group())):
                        result_numbers.add(int(float(m.group())))
                except ValueError:
                    pass

    _collect_numbers(analysis_result)

    # Check what % of narrative numbers appear in result
    matched = sum(1 for n in narrative_numbers if n in result_numbers)
    match_pct = matched / len(narrative_numbers) if narrative_numbers else 1.0

    if match_pct < 0.5:
        logger.warning(
            f"Narrative verification failed: {matched}/{len(narrative_numbers)} "
            f"numbers matched ({match_pct:.0%})"
        )
        return False

    # Phase 2: Attribution check — are numbers attributed to the right columns?
    # Build number-to-context mapping from the result data
    number_context: Dict[float, set] = {}  # number -> set of context keys

    def _collect_with_context(obj, context_keys: frozenset = frozenset()):
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            for rounded in (obj,) + tuple(round(obj, d) for d in range(5)):
                number_context.setdefault(rounded, set()).update(context_keys)
                if isinstance(rounded, float) and rounded == int(rounded):
                    number_context.setdefault(int(rounded), set()).update(context_keys)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _collect_with_context(v, context_keys | {str(k).lower()})
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _collect_with_context(item, context_keys)

    _collect_with_context(analysis_result)

    # Find (number, nearby_column_name) pairs in narrative
    dataset_columns = analysis_result.get("_dataset_columns", [])
    if dataset_columns and number_context:
        col_names_lower = {c.lower() for c in dataset_columns}
        attributions_checked = 0
        attributions_correct = 0

        # For each number in the narrative, find nearby column references
        for match in _re.finditer(r'\d+\.?\d*', text):
            try:
                num = float(match.group())
            except ValueError:
                continue
            if num not in number_context:
                continue

            # Look in a window of ~60 chars around the number for column names
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            window = text[start:end].lower()

            nearby_cols = [c for c in col_names_lower if c in window]
            if not nearby_cols:
                continue  # number not near any column name — skip

            # Check if any nearby column matches the number's source context
            source_contexts = number_context.get(num, set())
            attributions_checked += 1
            if any(col in ctx or ctx in col
                   for col in nearby_cols for ctx in source_contexts):
                attributions_correct += 1

        # Allow 30% misattribution tolerance
        if attributions_checked >= 3:
            attr_pct = attributions_correct / attributions_checked
            if attr_pct < 0.3:
                logger.warning(
                    f"Narrative attribution check failed: {attributions_correct}/"
                    f"{attributions_checked} attributions correct ({attr_pct:.0%})"
                )
                return False

    # Verify suggestions reference actual column names
    if dataset_columns and narrative.suggestions:
        col_names_lower_set = {c.lower() for c in dataset_columns}
        for suggestion in narrative.suggestions:
            words = suggestion.lower().split()
            has_column_ref = any(w in col_names_lower_set for w in words)
            if not has_column_ref:
                continue

    return True


def _sanitize_for_narration(
    result: Dict[str, Any],
    data_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Strip large/binary keys and inject column info for narration.

    - Removes base64 blobs and server file paths (no analytical value, waste tokens).
    - Injects _dataset_columns so the LLM knows the real column names.
    """
    import copy
    cleaned = {k: copy.deepcopy(v) for k, v in result.items() if k not in _NARRATION_EXCLUDED_KEYS}
    if data_context:
        cleaned["_dataset_columns"] = [c["name"] for c in data_context.get("columns", [])]

    # When pre-computed summary stats are available (smart_query), replace raw rows
    # so the LLM narrates from accurate aggregates instead of truncated row data
    if "data_summary" in cleaned and "data" in cleaned:
        total = cleaned.get("total_rows", "?")
        cleaned["data"] = f"[{total} rows — see data_summary for accurate statistics]"

    return cleaned


def _fmt(v: Any) -> str:
    """Format a value for narrative display — round floats to 2dp."""
    if v is None:
        return "?"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _template_narrative(
    result: Dict[str, Any],
    skill_name: str,
    status: str,
) -> NarrativeResult:
    """Generate a template-based narrative when LLM narration fails.

    Extracts key numbers from the result dict and builds a human-readable summary.
    Uses actual column names from _dataset_columns when generating suggestions.
    """
    if status == "error":
        msg = result.get("message", "The analysis encountered an error.")
        return NarrativeResult(
            text=f"I ran into an issue: {msg}",
            key_points=[],
            suggestions=["Try rephrasing your question", "Check if your data has the required columns"],
        )

    # Gather actual column names for suggestions
    dataset_columns = result.get("_dataset_columns", [])

    def _col_suggestions() -> List[str]:
        """Generate follow-up suggestions using actual column names."""
        sug: List[str] = []
        if len(dataset_columns) >= 2:
            sug.append(f"Show me a chart of {dataset_columns[0]} vs {dataset_columns[1]}")
        if dataset_columns:
            sug.append(f"Describe the distribution of {dataset_columns[0]}")
        sug.append("What are the key correlations?")
        return sug[:3]

    text = ""
    key_points: List[str] = []
    suggestions: List[str] = []

    if skill_name == "profile_data":
        overview = result.get("overview", {})
        rows = overview.get("rows", overview.get("total_rows", "?"))
        cols = overview.get("columns", overview.get("total_columns", "?"))
        quality = result.get("quality_score", "?")
        dupes = overview.get("duplicate_rows", 0)
        missing = overview.get("missing_cells_pct", overview.get("total_missing", 0))
        # Compute missing % from column profiles if overview doesn't have it
        col_profiles = result.get("columns", result.get("column_profiles", []))
        if missing == 0 and isinstance(col_profiles, list) and col_profiles:
            total_missing = sum(cp.get("missing_count", 0) for cp in col_profiles if isinstance(cp, dict))
            total_cells = rows * cols if isinstance(rows, int) and isinstance(cols, int) else 0
            missing = round(total_missing / total_cells * 100, 1) if total_cells > 0 else 0
        text = (
            f"Your dataset contains {rows:,} records across {cols} features. "
            f"Data quality score is {quality}%."
        ) if isinstance(rows, int) else (
            f"Your dataset has {rows} rows and {cols} columns. Quality score: {quality}%."
        )
        # Build specific, data-driven key points
        if isinstance(rows, int):
            key_points.append(f"{rows:,} rows and {cols} columns")
        if dupes == 0:
            key_points.append("No duplicate rows detected")
        elif dupes:
            key_points.append(f"{dupes} duplicate rows found")
        if missing == 0 or missing == "0%":
            key_points.append("No missing values")
        elif missing:
            key_points.append(f"Missing values: {missing}")
        # Add column-level insights from the profile
        col_profiles = result.get("columns", result.get("column_profiles", []))
        if isinstance(col_profiles, list):
            for cp in col_profiles[:2]:
                cname = cp.get("name", cp.get("column", ""))
                nunique = cp.get("n_unique", cp.get("unique_count"))
                if cname and nunique is not None:
                    key_points.append(f"'{cname}' has {nunique} unique values")
        warnings = result.get("warnings", [])
        if warnings:
            key_points.append(f"{len(warnings)} data quality warnings detected")
        suggestions = ["What are the key correlations?", "Are there any outliers?"]
        if dataset_columns:
            suggestions.append(f"Describe the distribution of {dataset_columns[0]}")

    elif skill_name == "describe_data":
        ns = result.get("numeric_summary", [])
        cs = result.get("categorical_summary", [])
        text = f"Analyzed {len(ns)} numeric and {len(cs)} categorical columns."
        if ns:
            key_points = [f"{s.get('column', '?')}: mean={_fmt(s.get('mean'))}, std={_fmt(s.get('std'))}" for s in ns[:3]]
        suggestions = ["What are the correlations?", "Are there any outliers?"]

    elif skill_name == "classify":
        metrics = result.get("metrics", {})
        algo = result.get("algorithm", "?")
        target = result.get("target", "?")
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")
        fi = result.get("feature_importance", [])
        top_feat = ", ".join(f.get("feature", "?") for f in fi[:3]) if fi else "N/A"
        acc_str = f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else "?"
        f1_str = f"{f1*100:.1f}%" if isinstance(f1, (int, float)) else "?"
        text = (
            f"I trained a {algo} model to predict '{target}'. "
            f"It achieved {acc_str} accuracy and {f1_str} F1 score. "
            f"The top predictive features are: {top_feat}."
        )
        key_points = [f"Accuracy: {acc_str}", f"F1 Score: {f1_str}", f"Top features: {top_feat}"]
        auto_comp = result.get("auto_comparison", [])
        if auto_comp:
            key_points.append(f"Compared {len(auto_comp)} algorithms, {algo} performed best")
        suggestions = [f"Show me a chart of feature importance for {target}", f"What are the correlations with {target}?"]

    elif skill_name == "detect_outliers":
        n = result.get("outlier_count", result.get("n_outliers", "?"))
        pct = result.get("outlier_pct", "?")
        if isinstance(pct, float):
            pct = f"{pct:.2f}"
        method = result.get("method", "Isolation Forest")
        total = result.get("total_rows", "?")
        text = f"Found {n} outlier records ({pct}%) out of {total} rows using {method}."
        suggestions = _col_suggestions()

    elif skill_name == "analyze_correlations":
        top = result.get("top_correlations", [])
        if top:
            best = top[0]
            if len(top) >= 2:
                text = (
                    f"The strongest correlation is between {best.get('col1', '?')} and "
                    f"{best.get('col2', '?')} (r={best.get('correlation', 0):.3f}). "
                    f"The next strongest is {top[1].get('col1', '?')} and "
                    f"{top[1].get('col2', '?')} (r={top[1].get('correlation', 0):.3f})."
                )
            else:
                text = (
                    f"The strongest correlation is between {best.get('col1', '?')} and "
                    f"{best.get('col2', '?')} (r={best.get('correlation', 0):.3f})."
                )
            key_points = [
                f"{c.get('col1')} ↔ {c.get('col2')}: r={c.get('correlation', 0):.3f}"
                for c in top[:5]
            ]
        else:
            text = "Correlation analysis complete. No strong correlations found."
        suggestions = ["Are there any outliers?"]
        if dataset_columns:
            suggestions.append(f"Show me a scatter plot of {top[0].get('col1', dataset_columns[0])} vs {top[0].get('col2', dataset_columns[-1])}" if top else f"Describe {dataset_columns[0]}")

    elif skill_name == "find_clusters":
        n = result.get("n_clusters", "?")
        method = result.get("method", "auto")
        text = f"Identified {n} distinct clusters using {method} clustering."
        suggestions = _col_suggestions()

    elif skill_name == "forecast":
        method = result.get("method", "auto")
        n = result.get("n_periods", "?")
        text = f"Generated {n}-period forecast using {method}."
        suggestions = _col_suggestions()

    elif skill_name == "run_hypothesis_test":
        test = result.get("test", "?")
        pval = result.get("pvalue")
        sig = result.get("significant", False)
        conclusion = result.get("conclusion", "")
        if conclusion:
            text = conclusion
        else:
            pval_str = f"p={pval:.4f}" if isinstance(pval, (int, float)) else "p=?"
            text = f"Ran {test} test ({pval_str}). {'Significant difference found.' if sig else 'No significant difference.'}"
        suggestions = ["Compute effect sizes", "Try a different statistical test"]

    elif skill_name == "create_chart":
        chart_type = result.get("chart_type", "chart")
        summary = result.get("chart_summary", {})
        x_col = summary.get("x_column", "")
        y_col = summary.get("y_column", "")
        col_desc = f" of {x_col}" if x_col else ""
        if y_col:
            col_desc = f" of {x_col} vs {y_col}"
        text = f"Created a {chart_type} chart{col_desc}."
        # Extract key points from chart summary data
        chart_data = summary.get("data", [])
        if chart_data:
            for item in chart_data[:3]:
                vals = list(item.values())
                if len(vals) >= 2:
                    key_points.append(f"{vals[0]}: {_fmt(vals[1])}")
        suggestions = _col_suggestions()

    elif skill_name == "compare_groups":
        group_col = result.get("group_column", "?")
        value_col = result.get("value_column", "?")
        n_groups = result.get("n_groups", "?")
        overall_mean = result.get("overall_mean")
        groups = result.get("groups", [])
        text = f"Compared {value_col} across {n_groups} groups defined by {group_col}."
        if overall_mean is not None:
            text += f" The overall mean is {overall_mean:.2f}."
        for g in groups[:5]:
            name = g.get("group", "?")
            mean = g.get("mean")
            diff = g.get("diff_from_overall_pct")
            if mean is not None:
                point = f"{name}: mean={mean:.2f}"
                if diff is not None:
                    point += f" ({diff:+.1f}% vs overall)"
                key_points.append(point)
            else:
                count = g.get("count", "?")
                key_points.append(f"{name}: {count} records")
        suggestions = [f"Run a hypothesis test on {value_col} by {group_col}"]
        if dataset_columns:
            suggestions.append(f"Show me a box plot of {value_col} by {group_col}")
        suggestions.append("What are the key correlations?")

    elif skill_name == "query_data":
        total = result.get("total_rows", 0)
        desc = result.get("query_description", "custom filter")
        text = f"Found {total} rows matching your query ({desc})."
        data = result.get("data", [])
        if isinstance(data, list):
            for item in data[:3]:
                vals = list(item.values())
                if vals:
                    key_points.append(" | ".join(str(v) for v in vals[:4]))
        suggestions = _col_suggestions()

    elif skill_name == "pivot_table":
        idx = result.get("index_column", "?")
        vals_col = result.get("values_column", "?")
        aggfunc = result.get("aggfunc", "mean")
        text = f"Pivot table: {aggfunc} of {vals_col} grouped by {idx}."
        data = result.get("data", [])
        if isinstance(data, list):
            for item in data[:5]:
                vals = list(item.values())
                if len(vals) >= 2:
                    key_points.append(f"{vals[0]}: {_fmt(vals[1])}")
        suggestions = _col_suggestions()

    elif skill_name == "value_counts":
        column = result.get("column", "?")
        total = result.get("total_values", 0)
        text = f"Frequency distribution of {column} ({total} total values)."
        data = result.get("data", {})
        if isinstance(data, dict):
            for k, v in list(data.items())[:5]:
                key_points.append(f"{k}: {v}")
        suggestions = _col_suggestions()

    elif skill_name == "top_n":
        n = result.get("n", 10)
        column = result.get("column", "?")
        direction = result.get("direction", "top")
        text = f"{direction.title()} {n} records ranked by {column}."
        data = result.get("data", [])
        if isinstance(data, list):
            for item in data[:5]:
                vals = list(item.values())
                if vals:
                    key_points.append(" | ".join(str(v) for v in vals[:4]))
        suggestions = _col_suggestions()

    elif skill_name == "cross_tab":
        row_col = result.get("row_column", "?")
        col_col = result.get("col_column", "?")
        text = f"Cross-tabulation of {row_col} vs {col_col}."
        data = result.get("data", [])
        if isinstance(data, list):
            for item in data[:5]:
                vals = list(item.values())
                if len(vals) >= 2:
                    key_points.append(f"{vals[0]}: {', '.join(str(v) for v in vals[1:4])}")
        suggestions = _col_suggestions()

    elif skill_name == "smart_query":
        code = result.get("generated_code", "")
        total = result.get("total_rows", "?")
        summary = result.get("data_summary", {})
        asked = result.get("question_asked", "")

        if summary and isinstance(summary, dict) and summary.get("columns"):
            col_summaries = summary.get("columns", {})
            preamble = f"To answer '{asked}': " if asked else ""
            text = f"{preamble}Custom query returned {total} rows across {len(col_summaries)} columns."
            if code:
                key_points.append(f"Code: {code[:80]}...")
            for col_name, col_stats in list(col_summaries.items())[:3]:
                ctype = col_stats.get("type", "")
                if ctype == "categorical":
                    vc = col_stats.get("value_counts", {})
                    if vc:
                        top = list(vc.items())[:3]
                        parts = ", ".join(f"{k}: {v['count'] if isinstance(v, dict) else v}" for k, v in top)
                        key_points.append(f"{col_name}: {parts}")
                elif ctype == "numeric":
                    mean = col_stats.get("mean")
                    if mean is not None:
                        key_points.append(f"{col_name}: mean={_fmt(mean)}, min={_fmt(col_stats.get('min'))}, max={_fmt(col_stats.get('max'))}")
                elif ctype == "boolean":
                    key_points.append(f"{col_name}: True={col_stats.get('true_count', 0)}, False={col_stats.get('false_count', 0)}")
        else:
            preamble = f"To answer '{asked}': " if asked else ""
            text = f"{preamble}Custom query executed successfully ({total} rows in result)."
            if code:
                key_points.append(f"Code: {code[:80]}...")
            data = result.get("data", [])
            if isinstance(data, list):
                for item in data[:3]:
                    vals = list(item.values())
                    if vals:
                        key_points.append(" | ".join(str(v) for v in vals[:4]))
        suggestions = _col_suggestions()

    else:
        # Smart generic: extract numeric results for any unhandled skill
        readable = skill_name.replace("_", " ").title()
        parts: List[str] = []
        for k, v in result.items():
            if k.startswith("_") or k == "status":
                continue
            if isinstance(v, (int, float)):
                label = k.replace("_", " ")
                if isinstance(v, float):
                    parts.append(f"{label}: {v:.4g}")
                else:
                    parts.append(f"{label}: {v}")
            elif isinstance(v, str) and len(v) < 100 and k not in ("message",):
                parts.append(f"{k.replace('_', ' ')}: {v}")
        if parts:
            text = f"{readable} analysis complete. " + "; ".join(parts[:6]) + "."
        else:
            text = f"{readable} analysis complete. See detailed results below."
        # Extract key points from list-type results
        for k, v in result.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                for item in v[:3]:
                    vals = [str(val) for val in item.values() if val is not None]
                    if vals:
                        key_points.append(" | ".join(vals[:3]))
                break
        suggestions = _col_suggestions()

    return NarrativeResult(text=text, key_points=key_points, suggestions=suggestions)


def _create_provider(name: str, **kwargs) -> LLMProvider:
    """Factory: create an LLM provider by name."""
    name = name.lower().strip()

    if name == "ollama":
        from ..llm.ollama import OllamaProvider
        return OllamaProvider(**kwargs)
    elif name in ("claude", "anthropic"):
        from ..llm.claude import ClaudeProvider
        return ClaudeProvider(**kwargs)
    elif name in ("openai", "gpt"):
        from ..llm.openai import OpenAIProvider
        return OpenAIProvider(**kwargs)
    elif name == "groq":
        from ..llm.groq import GroqProvider
        return GroqProvider(**kwargs)
    elif name == "gemini":
        from ..llm.gemini import GeminiProvider
        return GeminiProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{name}'. "
            "Supported: 'ollama', 'claude', 'openai', 'groq', 'gemini'"
        )


class Analyst:
    """DataPilot Analyst — load data, ask questions, get insights.

    Args:
        data: Path to a data file (CSV, Excel, JSON, Parquet) or a DataFrame.
        llm: LLM provider name ('ollama', 'claude', 'openai', 'groq') or LLMProvider instance.
        auto_profile: Whether to auto-profile the dataset on load.

    Example:
        analyst = Analyst("sales.csv")
        result = analyst.ask("Which features predict churn?")
        print(result.text)
    """

    def __init__(
        self,
        data: Union[str, "Path", Any],
        llm: Union[str, LLMProvider] = None,
        auto_profile: bool = True,
    ):
        import pandas as pd

        # Load data
        if isinstance(data, (str, Path)):
            self.data_path = str(data)
            self.df = load_data(self.data_path)
        elif isinstance(data, pd.DataFrame):
            self.data_path = None
            self.df = data
        else:
            raise TypeError(
                f"Expected file path or DataFrame, got {type(data).__name__}"
            )

        # Set up LLM provider
        if isinstance(llm, LLMProvider):
            self.provider = llm
        else:
            self.provider = self._build_provider(llm)

        # Core components
        self.router = Router(self.provider)
        self.executor = Executor()
        self.data_context = build_data_context(self.df)

        # History
        self.history: List[AnalystResult] = []

        # Temp file path for skills that expect file_path
        self._temp_path: Optional[str] = None

        # Auto-profile
        self._profile_cache: Optional[Dict] = None
        if auto_profile:
            self._auto_profile()

    @staticmethod
    def _build_provider(llm_name=None):
        """Build the best available LLM provider.

        If multiple API keys are present, creates a FailoverProvider with
        task-aware routing. Otherwise, creates a single provider.
        Falls back gracefully if no API keys are set.
        """
        from ..llm.failover import FailoverProvider

        providers = {}

        # Try to create each provider (skip if API key missing)
        if Config.GROQ_API_KEY:
            try:
                from ..llm.groq import GroqProvider
                providers["groq"] = GroqProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")

        if Config.GEMINI_API_KEY:
            try:
                from ..llm.gemini import GeminiProvider
                providers["gemini"] = GeminiProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")

        # If multiple providers available, use failover
        if len(providers) > 1:
            logger.info(f"Multi-LLM mode: {list(providers.keys())}")
            return FailoverProvider(providers=providers)

        # If one provider available, use it directly
        if len(providers) == 1:
            name = list(providers.keys())[0]
            logger.info(f"Single-LLM mode: {name}")
            return providers[name]

        # If explicit provider name given, try to create it (legacy path)
        if llm_name:
            try:
                return _create_provider(llm_name)
            except Exception as e:
                logger.warning(f"Failed to create provider '{llm_name}': {e}")

        # No LLM available — return None (deterministic fallbacks handle the rest)
        logger.warning("No LLM providers available — running in deterministic-only mode")
        return None

    @property
    def _file_path(self) -> str:
        """Return a file path for skills that expect file_path instead of df.

        Uses the original data_path if available, otherwise saves df to a temp CSV.
        """
        if self.data_path:
            return self.data_path
        if self._temp_path is None:
            import tempfile
            tmp_dir = Path(tempfile.gettempdir()) / "datapilot"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_path = str(tmp_dir / f"_analyst_{id(self)}.csv")
            self.df.to_csv(self._temp_path, index=False)
        return self._temp_path

    def _auto_profile(self):
        """Run a quick profile on load (cached)."""
        try:
            from ..data.profiler import profile_data
            self._profile_cache = profile_data(self._file_path)
            logger.info("Auto-profile complete")
        except Exception as e:
            logger.warning(f"Auto-profile failed: {e}")

    # ------------------------------------------------------------------
    # Main API: ask a natural-language question
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        narrate: bool = True,
        conversation_context: Optional[str] = None,
    ) -> AnalystResult:
        """Ask a natural-language question about the data.

        Routes the question to the best skill, executes it,
        and generates a narrative (synchronously so the caller gets
        a complete answer in one call).

        Args:
            question: Natural language question.
            narrate: Whether to generate a narrative via LLM.
            conversation_context: Optional summary of previous Q&A pairs.

        Returns:
            AnalystResult with narrative included.
        """
        # Route
        t0 = time.perf_counter()
        routing = self.router.route(question, self.data_context)
        t1 = time.perf_counter()
        routing_ms = (t1 - t0) * 1000
        logger.info(f"Routing: {routing_ms:.0f}ms -> {routing.skill_name}")

        # Execute
        execution = self.executor.execute(
            skill_name=routing.skill_name,
            df=self.df,
            parameters=routing.parameters,
            question=question,
            llm_provider=self.provider,
        )
        t2 = time.perf_counter()
        execution_ms = (t2 - t1) * 1000
        logger.info(f"Execution: {execution_ms:.0f}ms ({execution.status})")

        result = AnalystResult(
            question=question,
            routing=routing,
            execution=execution,
            routing_ms=routing_ms,
            execution_ms=execution_ms,
        )
        self.history.append(result)

        # Generate narrative synchronously
        if narrate and execution.result:
            t3 = time.perf_counter()
            result.narrative = self._generate_narrative(
                execution.result, question, routing.skill_name, execution.status,
                conversation_context=conversation_context,
            )
            t4 = time.perf_counter()
            result.narration_ms = (t4 - t3) * 1000
            logger.info(f"Narration: {result.narration_ms:.0f}ms")

        return result

    def _generate_narrative(
        self,
        analysis_result: Dict[str, Any],
        question: str,
        skill_name: str,
        status: str,
        conversation_context: Optional[str] = None,
    ) -> NarrativeResult:
        """Generate narrative via LLM, falling back to templates on failure.

        Sanitizes the result (strips base64/paths, injects column names) before
        passing to the LLM provider.
        """
        sanitized = _sanitize_for_narration(analysis_result, self.data_context)

        # Try LLM narration first (Groq preferred for speed)
        try:
            narrative = self._try_llm_narrative(
                sanitized, question, skill_name,
                conversation_context=conversation_context,
            )
            if narrative and narrative.text:
                # Reject narratives with "?" placeholders or too short
                if "? " not in narrative.text and len(narrative.text) > 30:
                    if _verify_narrative(narrative, sanitized):
                        return narrative
                    else:
                        print(f"[DataPilot] LLM narrative rejected (numbers don't match result data), using template", file=sys.stderr)
                        logger.warning("Narrative rejected: numbers don't match result data")
                else:
                    print(f"[DataPilot] LLM narrative rejected (contains '?' or too short), using template", file=sys.stderr)
        except Exception as e:
            print(f"[DataPilot] LLM narrative failed: {e}", file=sys.stderr)
            logger.warning(f"LLM narrative failed: {e}")

        # Fall back to template-based narrative (also uses sanitized for column info)
        return _template_narrative(sanitized, skill_name, status)

    def _try_llm_narrative(
        self,
        analysis_result: Dict[str, Any],
        question: str,
        skill_name: str,
        conversation_context: Optional[str] = None,
    ) -> Optional[NarrativeResult]:
        """Try to generate narrative via the LLM provider.

        Expects analysis_result to already be sanitized (no base64 blobs).
        FailoverProvider handles multi-provider failover internally.
        """
        if self.provider is None:
            return None

        try:
            result = self.provider.generate_narrative(
                analysis_result=analysis_result,
                question=question,
                skill_name=skill_name,
                conversation_context=conversation_context,
            )
            if result and result.text:
                return result
        except Exception as e:
            logger.warning(f"LLM narrative failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Direct skill shortcuts
    # ------------------------------------------------------------------

    def profile(self) -> Dict[str, Any]:
        """Profile the dataset."""
        if self._profile_cache:
            return self._profile_cache
        from ..data.profiler import profile_data
        self._profile_cache = profile_data(self._file_path)
        return self._profile_cache

    def describe(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Describe numeric and categorical columns."""
        from ..analysis.descriptive import describe_data
        return describe_data(self._file_path, columns=columns)

    def correlations(self, target: Optional[str] = None) -> Dict[str, Any]:
        """Analyze correlations, optionally focused on a target column."""
        from ..analysis.correlation import analyze_correlations
        kwargs = {}
        if target:
            kwargs["target"] = target
        return analyze_correlations(self._file_path, **kwargs)

    def anomalies(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect outliers and anomalies."""
        from ..analysis.anomaly import detect_outliers
        kwargs = {}
        if columns:
            kwargs["columns"] = columns
        return detect_outliers(self._file_path, **kwargs)

    def classify(
        self,
        target: str,
        algorithm: str = "auto",
    ) -> Dict[str, Any]:
        """Train a classifier on the data."""
        from ..analysis.classification import classify
        return classify(self._file_path, target=target, algorithm=algorithm)

    def regress(
        self,
        target: str,
        algorithm: str = "auto",
    ) -> Dict[str, Any]:
        """Train a regressor on the data."""
        from ..analysis.regression import predict_numeric
        return predict_numeric(self._file_path, target=target, algorithm=algorithm)

    def cluster(
        self,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Find clusters in the data."""
        from ..analysis.clustering import find_clusters
        kwargs = {}
        if n_clusters:
            kwargs["n_clusters"] = n_clusters
        return find_clusters(self._file_path, **kwargs)

    def forecast(
        self,
        date_col: str,
        value_col: str,
        periods: int = 12,
    ) -> Dict[str, Any]:
        """Forecast a time series."""
        from ..analysis.timeseries import forecast as ts_forecast
        return ts_forecast(
            self._file_path,
            date_col=date_col,
            value_col=value_col,
            periods=periods,
        )

    def chart(
        self,
        chart_type: str = "auto",
        x: Optional[str] = None,
        y: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a chart."""
        from ..viz.charts import create_chart, auto_chart
        if chart_type == "auto":
            return auto_chart(self._file_path)
        return create_chart(self._file_path, chart_type=chart_type, x=x, y=y, **kwargs)

    def suggest_chart(self) -> Dict[str, Any]:
        """Ask the LLM to suggest the best chart for this data."""
        if self.provider is None:
            return {"suggestions": []}
        return self.provider.suggest_chart(self.data_context)

    def _build_report_data(
        self,
        analysis_results: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Build a structured report data dict from analysis history.

        Combines narratives, key points, metrics, and chart paths into a
        format the export functions can render as rich report content.
        """
        # Deduplicate history by skill_name (keep latest entry per skill)
        seen_skills: Dict[str, int] = {}
        for idx, ar in enumerate(self.history):
            seen_skills[ar.skill_name] = idx
        deduped_indices = sorted(seen_skills.values())
        deduped_history = [self.history[i] for i in deduped_indices]

        # Collect sections from deduplicated history
        sections: List[Dict[str, Any]] = []
        all_key_points: List[str] = []
        all_metrics: List[Dict[str, str]] = []
        chart_paths: Dict[str, str] = {}

        for ar in deduped_history:
            section: Dict[str, Any] = {
                "heading": f"Analysis: {ar.skill_name.replace('_', ' ').title()}",
                "question": ar.question,
                "skill": ar.skill_name,
            }

            # Add narrative text
            if ar.narrative and ar.narrative.text:
                section["narrative"] = ar.narrative.text
            else:
                section["narrative"] = ar.text

            # Add key points
            points = ar.key_points
            if points:
                section["key_points"] = points
                all_key_points.extend(points)

            # Extract metrics from execution result
            result = ar.execution.result or {}
            if ar.skill_name == "classify":
                metrics = result.get("metrics", {})
                if metrics.get("accuracy") is not None:
                    all_metrics.append({"label": "Accuracy", "value": f"{metrics['accuracy']*100:.1f}%"})
                if metrics.get("f1") is not None:
                    all_metrics.append({"label": "F1 Score", "value": f"{metrics['f1']*100:.1f}%"})
                algo = result.get("algorithm")
                if algo:
                    all_metrics.append({"label": "Algorithm", "value": str(algo)})
            elif ar.skill_name == "detect_outliers":
                n = result.get("n_outliers")
                pct = result.get("outlier_pct")
                if n is not None:
                    all_metrics.append({"label": "Outliers Found", "value": str(n)})
                if pct is not None:
                    all_metrics.append({"label": "Outlier %", "value": f"{pct}%"})
            elif ar.skill_name == "analyze_correlations":
                top = result.get("top_correlations", [])
                if top:
                    best = top[0]
                    all_metrics.append({
                        "label": "Strongest Correlation",
                        "value": f"{best.get('col1', '?')} \u2194 {best.get('col2', '?')} (r={best.get('correlation', 0):.3f})",
                    })
            elif ar.skill_name == "profile_data":
                overview = result.get("overview", {})
                rows = overview.get("total_rows")
                cols = overview.get("total_columns")
                quality = result.get("quality_score")
                if rows is not None:
                    all_metrics.append({"label": "Total Rows", "value": f"{rows:,}" if isinstance(rows, int) else str(rows)})
                if cols is not None:
                    all_metrics.append({"label": "Total Columns", "value": str(cols)})
                if quality is not None:
                    all_metrics.append({"label": "Data Quality", "value": f"{quality}%"})

            # Collect chart paths
            chart_path = result.get("chart_path") or result.get("path")
            if chart_path:
                chart_paths[ar.skill_name] = chart_path

            sections.append(section)

        # Add dataset shape metrics from data_context if available
        dc = getattr(self, 'data_context', None)
        if isinstance(dc, dict):
            shape_metrics = []
            if dc.get("n_rows") is not None:
                shape_metrics.append({"label": "Dataset Rows", "value": f"{dc['n_rows']:,}" if isinstance(dc['n_rows'], int) else str(dc['n_rows'])})
            if dc.get("n_cols") is not None:
                shape_metrics.append({"label": "Dataset Columns", "value": str(dc['n_cols'])})
            # Add analyses performed count
            if deduped_history:
                shape_metrics.append({"label": "Analyses Performed", "value": str(len(deduped_history))})
            # Add data quality from column null percentages
            dc_columns = dc.get("columns", [])
            if dc_columns and not any(m["label"] == "Data Quality" for m in all_metrics):
                null_pcts = [c.get("null_pct", 0) for c in dc_columns]
                avg_completeness = 100 - (sum(null_pcts) / len(null_pcts))
                shape_metrics.append({"label": "Data Quality", "value": f"{avg_completeness:.1f}%"})
            all_metrics = shape_metrics + all_metrics

        # Extra chart paths from analysis_results
        if isinstance(analysis_results, list):
            for ar_dict in analysis_results:
                if isinstance(ar_dict, dict):
                    for key in ("chart_path", "path"):
                        cp = ar_dict.get(key)
                        if cp and cp not in chart_paths.values():
                            name = ar_dict.get("skill", ar_dict.get("skill_name", f"chart_{len(chart_paths)}"))
                            chart_paths[name] = cp

        # Synthesize executive summary
        skill_names = [s.get("heading", "").replace("Analysis: ", "") for s in sections]
        top_points = all_key_points[:5]
        if sections:
            summary_parts = [
                f"This report presents findings from {len(sections)} analyses ({', '.join(skill_names)}).",
            ]
            if top_points:
                summary_parts.append("Key findings: " + "; ".join(top_points) + ".")
            summary = " ".join(summary_parts)
            # Cap at 300 words
            words = summary.split()
            if len(words) > 300:
                summary = " ".join(words[:300]) + "..."
        else:
            summary = "Multiple analyses were performed on the dataset. See detailed sections below."

        return {
            "summary": summary,
            "sections": sections,
            "key_points": all_key_points,
            "metrics": all_metrics,
            "chart_paths": chart_paths,
        }

    def export(
        self,
        path: str,
        analysis_results: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """Export results to PDF, DOCX, or PPTX (inferred from extension).

        Args:
            path: Output file path (extension determines format).
            analysis_results: List of result dicts to include.

        Returns:
            Path to the exported file.
        """
        ext = Path(path).suffix.lower()

        # Build structured report data from history
        report_data = self._build_report_data(analysis_results)

        # Pass metrics and chart paths to export functions
        export_kwargs = dict(kwargs)
        if report_data["metrics"] and "metrics" not in export_kwargs:
            export_kwargs["metrics"] = report_data["metrics"]
        vis_paths = report_data.get("chart_paths")
        if vis_paths and "visualisation_paths" not in export_kwargs:
            export_kwargs["visualisation_paths"] = vis_paths

        if ext == ".pdf":
            from ..export.pdf import export_to_pdf
            return export_to_pdf(output_path=path, analysis_results=report_data, **export_kwargs)
        elif ext == ".docx":
            from ..export.docx import export_to_docx
            return export_to_docx(output_path=path, analysis_results=report_data, **export_kwargs)
        elif ext == ".pptx":
            from ..export.pptx import export_to_pptx
            return export_to_pptx(output_path=path, analysis_results=report_data, **export_kwargs)
        else:
            raise ValueError(
                f"Unsupported export format: '{ext}'. Use .pdf, .docx, or .pptx"
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple:
        return self.df.shape

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns)

    @property
    def dtypes(self) -> Dict[str, str]:
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def __repr__(self) -> str:
        src = self.data_path or "DataFrame"
        return (
            f"Analyst(data={src!r}, "
            f"shape={self.df.shape}, "
            f"llm={type(self.provider).__name__})"
        )

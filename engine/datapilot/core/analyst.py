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
    return cleaned


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
        rows = overview.get("total_rows", "?")
        cols = overview.get("total_columns", "?")
        quality = result.get("quality_score", "?")
        text = (
            f"Your dataset contains {rows:,} records across {cols} features. "
            f"Data quality score is {quality}%."
        ) if isinstance(rows, int) else (
            f"Your dataset has {rows} rows and {cols} columns. Quality score: {quality}%."
        )
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
            key_points = [f"{s.get('column', '?')}: mean={s.get('mean', '?')}, std={s.get('std', '?')}" for s in ns[:3]]
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
        n = result.get("n_outliers", "?")
        pct = result.get("outlier_pct", "?")
        method = result.get("method", "Isolation Forest")
        text = f"Found {n} outlier records ({pct}%) using {method}."
        suggestions = _col_suggestions()

    elif skill_name == "analyze_correlations":
        top = result.get("top_correlations", [])
        if top:
            best = top[0]
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
                    key_points.append(f"{vals[0]}: {vals[1]}")
        suggestions = _col_suggestions()

    else:
        text = f"Analysis complete using {skill_name}."
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
    else:
        raise ValueError(
            f"Unknown LLM provider: '{name}'. "
            "Supported: 'ollama', 'claude', 'openai', 'groq'"
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
        llm = llm or Config.LLM_PROVIDER
        if isinstance(llm, str):
            self.provider = _create_provider(llm)
        elif isinstance(llm, LLMProvider):
            self.provider = llm
        else:
            raise TypeError(f"Expected str or LLMProvider, got {type(llm).__name__}")

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
    ) -> AnalystResult:
        """Ask a natural-language question about the data.

        Routes the question to the best skill, executes it,
        and generates a narrative (synchronously so the caller gets
        a complete answer in one call).

        Args:
            question: Natural language question.
            narrate: Whether to generate a narrative via LLM.

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
    ) -> NarrativeResult:
        """Generate narrative via LLM, falling back to templates on failure.

        Sanitizes the result (strips base64/paths, injects column names) before
        passing to the LLM provider.
        """
        sanitized = _sanitize_for_narration(analysis_result, self.data_context)

        # Try LLM narration first (Groq preferred for speed)
        try:
            narrative = self._try_llm_narrative(sanitized, question, skill_name)
            if narrative and narrative.text:
                return narrative
        except Exception as e:
            logger.warning(f"LLM narrative failed: {e}")

        # Fall back to template-based narrative (also uses sanitized for column info)
        return _template_narrative(sanitized, skill_name, status)

    def _try_llm_narrative(
        self,
        analysis_result: Dict[str, Any],
        question: str,
        skill_name: str,
    ) -> Optional[NarrativeResult]:
        """Try to generate narrative via the primary LLM provider, then Groq fallback.

        Expects analysis_result to already be sanitized (no base64 blobs).
        """
        # Try primary provider
        try:
            result = self.provider.generate_narrative(
                analysis_result=analysis_result,
                question=question,
                skill_name=skill_name,
            )
            if result and result.text:
                return result
        except Exception as e:
            logger.warning(f"Primary narrative failed: {e}")

        # Try Groq fallback if primary isn't already Groq
        from ..llm.groq import GroqProvider
        if not isinstance(self.provider, GroqProvider):
            try:
                fallback = GroqProvider()
                result = fallback.generate_narrative(
                    analysis_result=analysis_result,
                    question=question,
                    skill_name=skill_name,
                )
                if result and result.text:
                    return result
            except Exception as e:
                logger.warning(f"Groq narrative fallback failed: {e}")

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
        return self.provider.suggest_chart(self.data_context)

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
        result_list = analysis_results or [
            r.execution.result for r in self.history
            if r.execution.result
        ]
        # Export functions expect a single dict; merge list into one
        if isinstance(result_list, list):
            merged: Dict[str, Any] = {}
            for r in result_list:
                if isinstance(r, dict):
                    merged.update(r)
            results = merged
        else:
            results = result_list

        if ext == ".pdf":
            from ..export.pdf import export_to_pdf
            return export_to_pdf(output_path=path, analysis_results=results, **kwargs)
        elif ext == ".docx":
            from ..export.docx import export_to_docx
            return export_to_docx(output_path=path, analysis_results=results, **kwargs)
        elif ext == ".pptx":
            from ..export.pptx import export_to_pptx
            return export_to_pptx(output_path=path, analysis_results=results, **kwargs)
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

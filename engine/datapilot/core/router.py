"""
Router — keyword + LLM-powered question-to-skill mapping.

First attempts fast keyword matching (85-95% confidence).
Falls back to LLM routing for ambiguous questions.

Priority order:
  1. Chart/visualization keywords (always checked first)
  2. Standard keyword routing table
  3. LLM provider (FailoverProvider handles multi-provider failover)
  4. profile_data (last resort)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..llm.provider import LLMProvider, RoutingResult
from ..llm.prompts import build_skill_catalog

logger = logging.getLogger("datapilot.core.router")


# ---------------------------------------------------------------------------
# Chart priority routing (checked FIRST, before all other keywords)
# ---------------------------------------------------------------------------

_CHART_PRIORITY_PATTERNS = [
    r"\bchart\b", r"\bplot\b", r"\bvisuali[sz]", r"\bgraph\b",
    r"\bdraw\b", r"\bshow me a\b",
]

_CHART_TYPE_MAP = {
    "histogram": "histogram",
    "hist": "histogram",
    "bar": "bar",
    "bar chart": "bar",
    "scatter": "scatter",
    "scatterplot": "scatter",
    "line": "line",
    "line chart": "line",
    "box": "box",
    "boxplot": "box",
    "heatmap": "heatmap",
    "pie": "pie",
    "violin": "violin",
    "density": "kde",
    "kde": "kde",
    "pair": "pairplot",
    "pairplot": "pairplot",
}


def _try_chart_priority(question: str, data_context: Dict[str, Any]) -> Optional[RoutingResult]:
    """Check if question is asking for a chart/visualization. Takes priority over all other keywords."""
    q = question.lower().strip()

    # Must match at least one chart priority pattern
    matched = None
    for pattern in _CHART_PRIORITY_PATTERNS:
        m = re.search(pattern, q)
        if m:
            matched = m.group(0)
            break

    if matched is None:
        return None

    params: Dict[str, Any] = {}

    # Extract chart type
    for keyword, chart_type in _CHART_TYPE_MAP.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", q):
            params["chart_type"] = chart_type
            break

    # If "distribution" is mentioned, default to histogram
    if "chart_type" not in params and re.search(r"\bdistribution\b", q):
        params["chart_type"] = "histogram"

    # Extract x, y from question matched against column names
    col_names_lower = {c["name"].lower(): c["name"] for c in data_context.get("columns", [])}

    # Pattern: "X vs Y", "X versus Y", "X against Y", "X and Y"
    vs_match = re.search(r"(?:of|for)?\s*(\w[\w\s]*?)\s+(?:vs\.?|versus|against|and)\s+(\w[\w\s]*?)(?:\s|$|\?)", q)
    if vs_match:
        x_hint = vs_match.group(1).strip()
        y_hint = vs_match.group(2).strip()
        x_col = _match_column(x_hint, col_names_lower)
        y_col = _match_column(y_hint, col_names_lower)
        if x_col:
            params["x"] = x_col
        if y_col:
            params["y"] = y_col

    # Pattern: "distribution of X", "chart of X", "plot X"
    if "x" not in params:
        of_match = re.search(r"(?:distribution|chart|plot|graph|histogram|boxplot|box plot)\s+(?:of|for)\s+(\w[\w\s]*?)(?:\s*(?:vs|and|by|\?|$))", q)
        if of_match:
            hint = of_match.group(1).strip()
            col = _match_column(hint, col_names_lower)
            if col:
                params["x"] = col

    # Pattern: "by group_col" → hue
    by_match = re.search(r"\bby\s+(\w[\w\s]*?)(?:\s*(?:\?|$))", q)
    if by_match and "hue" not in params:
        hint = by_match.group(1).strip()
        col = _match_column(hint, col_names_lower)
        if col:
            params["hue"] = col

    # Fall back: find column names mentioned anywhere in the question
    if "x" not in params:
        mentioned = _find_mentioned_columns(q, col_names_lower)
        if len(mentioned) >= 2:
            params["x"] = mentioned[0]
            params["y"] = mentioned[1]
        elif len(mentioned) == 1:
            params["x"] = mentioned[0]

    # Infer chart type from column data types if not explicitly specified
    if "chart_type" not in params:
        params["chart_type"] = _infer_chart_type(params, data_context)

    return RoutingResult(
        skill_name="create_chart",
        parameters=params,
        confidence=0.92,
        reasoning=f"Matched: '{matched}' -> create_chart (priority)",
        route_method="keyword",
    )


def _get_col_semantic_type(col_name: str, data_context: Dict[str, Any]) -> str:
    """Look up the semantic type for a column from data_context (case-insensitive)."""
    col_lower = col_name.lower()
    for c in data_context.get("columns", []):
        if c["name"].lower() == col_lower:
            return c.get("semantic_type", "text")
    return "text"


def _get_col_nunique(col_name: str, data_context: Dict[str, Any]) -> int:
    """Look up n_unique for a column from data_context (case-insensitive)."""
    col_lower = col_name.lower()
    for c in data_context.get("columns", []):
        if c["name"].lower() == col_lower:
            return c.get("n_unique", 999)
    return 999


def _infer_chart_type(params: Dict[str, Any], data_context: Dict[str, Any]) -> str:
    """Infer the best chart type from column types and cardinality.

    Uses n_unique to detect low-cardinality numerics (treated as categorical
    for charting) and binary columns (best shown as bar/proportion charts).
    """
    x_col = params.get("x")
    y_col = params.get("y")
    x_type = _get_col_semantic_type(x_col, data_context) if x_col else None
    y_type = _get_col_semantic_type(y_col, data_context) if y_col else None
    x_nunique = _get_col_nunique(x_col, data_context) if x_col else 0
    y_nunique = _get_col_nunique(y_col, data_context) if y_col else 0

    # Treat low-cardinality numerics as categorical for charting purposes
    x_discrete = x_type in ("categorical", "boolean") or (x_type == "numeric" and x_nunique <= 15)
    y_binary = y_nunique == 2

    if x_col and y_col:
        if y_binary:
            return "bar"      # rate/proportion chart
        if x_discrete:
            return "bar"      # grouped bar
        if x_type == "numeric" and y_type == "numeric":
            return "scatter"
        return "bar"          # safe default
    elif x_col:
        return "histogram" if (x_type == "numeric" and not x_discrete) else "count"
    else:
        return "histogram"


def _match_column(hint: str, col_names_lower: Dict[str, str]) -> Optional[str]:
    """Match a text hint against known column names."""
    hint = hint.lower().strip()
    # Exact match
    if hint in col_names_lower:
        return col_names_lower[hint]
    # Without spaces (e.g., "customer service calls" → "customer_service_calls")
    hint_no_space = hint.replace(" ", "_")
    if hint_no_space in col_names_lower:
        return col_names_lower[hint_no_space]
    # Partial match
    for col_lower, col_orig in col_names_lower.items():
        if hint in col_lower or col_lower in hint:
            return col_orig
        # Try without underscores
        col_clean = col_lower.replace("_", " ")
        if hint in col_clean or col_clean in hint:
            return col_orig
    return None


def _find_mentioned_columns(question: str, col_names_lower: Dict[str, str]) -> List[str]:
    """Find all column names mentioned in the question, in order of appearance."""
    found = []
    q = question.lower()
    for col_lower, col_orig in col_names_lower.items():
        # Check both original and space-separated versions
        if col_lower in q or col_lower.replace("_", " ") in q:
            pos = q.find(col_lower)
            if pos == -1:
                pos = q.find(col_lower.replace("_", " "))
            found.append((pos, col_orig))
    found.sort(key=lambda x: x[0])
    return [col for _, col in found]


# Patterns that suggest a specific column for describe_data
_DESCRIBE_COL_PATTERNS = [
    r"(?:distribution|describe|spread|range|stats?|statistics?)\s+(?:of|for)\s+(\w[\w\s]*?)(?:\s*(?:\?|$|by|vs|and))",
    r"how\s+is\s+(\w[\w\s]*?)\s+distributed",
    r"(\w[\w\s]*?)\s+distribution",
]


def _extract_describe_columns(
    question: str,
    data_context: Dict[str, Any],
) -> Optional[List[str]]:
    """Extract specific column names from a describe_data question."""
    q = question.lower().strip()
    col_names_lower = {c["name"].lower(): c["name"] for c in data_context.get("columns", [])}

    for pattern in _DESCRIBE_COL_PATTERNS:
        match = re.search(pattern, q)
        if match:
            hint = match.group(1).strip()
            col = _match_column(hint, col_names_lower)
            if col:
                return [col]

    # Fallback: check if any column name appears in the question
    mentioned = _find_mentioned_columns(q, col_names_lower)
    if len(mentioned) == 1:
        return mentioned
    if len(mentioned) > 1:
        return mentioned[:3]  # Limit to 3 columns

    return None


# ---------------------------------------------------------------------------
# Keyword routing table
# ---------------------------------------------------------------------------
# Each entry: (patterns, skill_name, default_params, confidence, reasoning_template)

_KEYWORD_ROUTES: List[Tuple[List[str], str, Dict[str, Any], float, str]] = [
    # Data querying — filter/select rows
    (
        [r"\bfilter\b", r"\bwhere\b", r"\bshow\b.*\brows?\b",
         r"\bselect\b.*\bwhere\b", r"\brows?\b.*\bwhere\b",
         r"\bquery\b(?!.*\bsmart\b)"],
        "query_data", {}, 0.88,
        "Matched: '{matched}' -> query_data",
    ),
    # Pivot / aggregation by group
    (
        [r"\bpivot\b", r"\baverage\b.*\bby\b", r"\bmean\b.*\bby\b",
         r"\bsum\b.*\bby\b", r"\baggregate\b.*\bby\b",
         r"\bgroup\b.*\b(?:average|mean|sum|count)\b",
         r"\btotal\b.*\bby\b"],
        "pivot_table", {}, 0.88,
        "Matched: '{matched}' -> pivot_table",
    ),
    # Value counts / frequency
    (
        [r"\bhow many\b.*\bper\b", r"\bfrequency\b", r"\bvalue.?counts?\b",
         r"\bcount\b.*\bper\b", r"\bcount\b.*\beach\b",
         r"\bnumber of\b.*\bper\b", r"\bnumber of\b.*\beach\b"],
        "value_counts", {}, 0.90,
        "Matched: '{matched}' -> value_counts",
    ),
    # Top N / ranking
    (
        [r"\btop\s+\d+\b", r"\bbottom\s+\d+\b", r"\bhighest\b",
         r"\blowest\b", r"\branking?\b", r"\bbest\b.*\d+",
         r"\bworst\b.*\d+", r"\blargest\b", r"\bsmallest\b"],
        "top_n", {}, 0.90,
        "Matched: '{matched}' -> top_n",
    ),
    # Cross-tabulation
    (
        [r"\bcross.?tab", r"\bcrosstab", r"\bcontingency\b",
         r"\b(?:breakdown|split)\b.*\bby\b.*\band\b"],
        "cross_tab", {}, 0.88,
        "Matched: '{matched}' -> cross_tab",
    ),
    # Profiling / overview / general insights
    (
        [r"\boverview\b", r"\bprofile\b", r"\bsummar(?:y|ize)\b", r"\bdescribe the data\b",
         r"\btell me about\b.*\bdata\b", r"\bwhat.s in\b.*\bdata",
         r"\bpatterns?\b", r"\binsights?\b", r"\bwhat do you see\b",
         r"\btell me about\b", r"\bwhat can you tell\b"],
        "profile_data", {}, 0.85,
        "Matched: '{matched}' -> profile_data",
    ),
    # Descriptive statistics (+ distribution keywords)
    (
        [r"\bdescri(?:be|ptive)\b.*\b(?:column|numeric|statistic|stats)\b",
         r"\bbasic stat", r"\bmean.*median", r"\bsummary stat",
         r"\bdistribution\b", r"\bspread\b", r"\brange\b"],
        "describe_data", {}, 0.90,
        "Matched: '{matched}' -> describe_data",
    ),
    # Correlation
    (
        [r"\bcorrelat", r"\brelationship between\b", r"\bhow.*\brelat",
         r"\bassociation\b", r"\brelate\b"],
        "analyze_correlations", {}, 0.92,
        "Matched: '{matched}' -> analyze_correlations",
    ),
    # Outliers / anomalies
    (
        [r"\boutlier", r"\banomaly\b", r"\banomalies\b", r"\babnormal\b",
         r"\bunusual\b.*\bvalue"],
        "detect_outliers", {}, 0.90,
        "Matched: '{matched}' -> detect_outliers",
    ),
    # Hypothesis testing — BEFORE classification (so "difference between X and churn"
    # doesn't get swallowed by the churn -> classify pattern)
    (
        [r"\bhypothesis\b", r"\bstatistical test\b", r"\bt-?test\b",
         r"\bchi.?square\b", r"\banova\b", r"\bp-?value\b",
         r"\bsignifican(?:t|ce)\b.*\bdifference\b",
         r"\bdifference between\b", r"\bcompare\b.*\bgroup",
         r"\bsignificant\b.*\bbetween\b"],
        "run_hypothesis_test", {}, 0.90,
        "Matched: '{matched}' -> run_hypothesis_test",
    ),
    # Classification (must come before regression so "predict churn" doesn't
    # match regression's "predict value")
    (
        [r"\bclassif(?:y|ication)\b", r"\bpredict\b(?!.*\b(?:number|value|price|amount|score)\b)",
         r"\bpredict\b.*\bchurn\b", r"\bwhat\b.*\bpredict",
         r"\bwhich features predict\b", r"\bfeature importance\b",
         r"\bchurn\b.*\b(?:model|predict|analys|classify|rate|risk)\b",
         r"\b(?:model|predict|analys|classify)\b.*\bchurn\b",
         r"\bchurn\b"],
        "classify", {}, 0.90,
        "Matched: '{matched}' -> classify",
    ),
    # Regression
    (
        [r"\bregress", r"\bpredict\b.*\b(?:number|value|price|amount|score)\b",
         r"\bforecast\b.*\b(?:value|number)"],
        "predict_numeric", {}, 0.88,
        "Matched: '{matched}' -> predict_numeric",
    ),
    # Clustering
    (
        [r"\bcluster", r"\bsegment", r"\bgroup(?:ing)?\b.*\bsimilar\b"],
        "find_clusters", {}, 0.90,
        "Matched: '{matched}' -> find_clusters",
    ),
    # Time series
    (
        [r"\btime.?series\b", r"\bforecast\b", r"\btrend\b.*\bover time\b",
         r"\bseasonal", r"\bover time\b", r"\bpredict future\b",
         r"\btrend\b"],
        "forecast", {}, 0.88,
        "Matched: '{matched}' -> forecast",
    ),
    # Feature importance / selection
    (
        [r"\bfeature\b.*\bimportan", r"\bwhich.*\bfeature.*\bmatter\b",
         r"\bvariable\b.*\bselect", r"\bselect\b.*\bfeature"],
        "select_features", {}, 0.88,
        "Matched: '{matched}' -> select_features",
    ),
    # Effect size
    (
        [r"\beffect.?size\b", r"\bcohen"],
        "compute_effect_sizes", {}, 0.90,
        "Matched: '{matched}' -> compute_effect_sizes",
    ),
    # Dimensionality reduction
    (
        [r"\bpca\b", r"\bdimensionality\b", r"\breduc.*dimension"],
        "reduce_dimensions", {}, 0.90,
        "Matched: '{matched}' -> reduce_dimensions",
    ),
    # Sentiment
    (
        [r"\bsentiment\b", r"\bopinion\b.*\banalys"],
        "analyze_sentiment", {}, 0.92,
        "Matched: '{matched}' -> analyze_sentiment",
    ),
    # Entities
    (
        [r"\bentit(?:y|ies)\b", r"\bnamed entity\b", r"\bner\b"],
        "extract_entities", {}, 0.90,
        "Matched: '{matched}' -> extract_entities",
    ),
    # Topics
    (
        [r"\btopic\b", r"\btheme"],
        "discover_topics", {}, 0.88,
        "Matched: '{matched}' -> discover_topics",
    ),
    # Data quality / validation
    (
        [r"\bdata quality\b", r"\bvalidat(?:e|ion)\b.*\bdata\b",
         r"\bclean.*\bdata\b", r"\bmissing\b.*\bvalue"],
        "validate_data", {}, 0.88,
        "Matched: '{matched}' -> validate_data",
    ),
    # Chart / visualization (fallback — lower priority than the chart priority check)
    (
        [r"\bhistogram\b", r"\bscatter\b", r"\bbar\b.*\bchart\b"],
        "create_chart", {}, 0.85,
        "Matched: '{matched}' -> create_chart",
    ),
    # Survival analysis
    (
        [r"\bsurvival\b", r"\bkaplan", r"\bhazard\b"],
        "survival_analysis", {}, 0.90,
        "Matched: '{matched}' -> survival_analysis",
    ),
    # Threshold finder
    (
        [r"\bthreshold\b", r"\bcutoff\b", r"\boptimal.*\bsplit\b"],
        "find_thresholds", {}, 0.88,
        "Matched: '{matched}' -> find_thresholds",
    ),
    # Explainability
    (
        [r"\bexplain\b.*\bmodel\b", r"\bshap\b", r"\bfeature.*\bcontribut"],
        "explain_model", {}, 0.88,
        "Matched: '{matched}' -> explain_model",
    ),
]


def _try_keyword_route(question: str) -> Optional[RoutingResult]:
    """Attempt to route via keyword matching. Returns None if no match."""
    q = question.lower().strip()
    for patterns, skill, params, confidence, reasoning_tpl in _KEYWORD_ROUTES:
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                matched_text = match.group(0)
                return RoutingResult(
                    skill_name=skill,
                    parameters=dict(params),
                    confidence=confidence,
                    reasoning=reasoning_tpl.format(matched=matched_text),
                    route_method="keyword",
                )
    return None


# ---------------------------------------------------------------------------
# Parameter enrichment — extract missing required params from the question
# ---------------------------------------------------------------------------

# Skills that require a `target` column parameter
_SKILLS_NEEDING_TARGET = {
    "classify", "predict_numeric", "select_features", "find_thresholds",
    "compute_effect_sizes", "explain_model",
}

# Regex patterns to extract target hints from questions
_TARGET_HINT_PATTERNS = [
    r"\bpredict(?:s|ing)?\s+(\w+)",
    r"\bclassif(?:y|ication)\b.*?\b(?:by|using|for|on)\s+(\w+)",
    r"\bimportan\w*\s+(?:for|of)\s+(\w+)",
    r"\bmodel\b.*?\b(?:for\s+)?(\w+)\s*\??$",
    r"\bthreshold\w*\s+(?:for|of|on)\s+(\w+)",
    r"\beffect.?size\w*\s+(?:for|of|on)\s+(\w+)",
]

# Words to ignore as target hints (too generic)
_TARGET_STOPWORDS = {
    "the", "a", "an", "my", "our", "this", "that", "it", "data", "dataset",
    "column", "columns", "variable", "variables", "feature", "features",
    "model", "result", "results", "value", "values", "number", "outcome",
    "best", "most", "important", "here", "there", "what", "which", "how",
    "customer", "customers", "user", "users", "all", "each", "every",
    "me", "us", "about", "between", "from", "with", "some", "any",
    "and", "or", "but", "not", "can", "will", "would", "should",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had",
    "time", "series", "future",
}


def _extract_target(
    question: str,
    data_context: Dict[str, Any],
) -> Optional[str]:
    """Try to extract a target column name from the question."""
    q = question.lower().strip()
    columns = data_context.get("columns", [])
    col_names = [c["name"] for c in columns]
    col_names_lower = {c.lower(): c for c in col_names}

    # Strategy 1: Regex hints matched against real column names
    for pattern in _TARGET_HINT_PATTERNS:
        match = re.search(pattern, q)
        if match:
            hint = match.group(1).lower()
            if hint in _TARGET_STOPWORDS:
                continue
            if hint in col_names_lower:
                return col_names_lower[hint]
            for col_lower, col_orig in col_names_lower.items():
                if hint in col_lower or col_lower in hint:
                    return col_orig

    # Strategy 2: Any word in the question matches a column name
    words = re.findall(r"\b\w+\b", q)
    matched_cols = []
    for word in words:
        if word in _TARGET_STOPWORDS:
            continue
        if word in col_names_lower:
            matched_cols.append(col_names_lower[word])

    if matched_cols:
        col_info = {c["name"]: c for c in columns}
        for col in matched_cols:
            info = col_info.get(col, {})
            if info.get("n_unique", 999) <= 10:
                return col
        return matched_cols[0]

    # Strategy 3: Guess best target from data context
    binary_cols = []
    categorical_cols = []
    for c in columns:
        if c["semantic_type"] == "categorical":
            if c["n_unique"] == 2:
                binary_cols.append(c["name"])
            elif c["n_unique"] <= 10:
                categorical_cols.append(c["name"])

    if binary_cols:
        logger.info(f"Auto-detected binary target: {binary_cols[0]}")
        return binary_cols[0]
    if categorical_cols:
        logger.info(f"Auto-detected categorical target: {categorical_cols[0]}")
        return categorical_cols[0]

    return None


def _enrich_forecast_params(
    result: RoutingResult,
    data_context: Dict[str, Any],
) -> RoutingResult:
    """Auto-detect date_column and value_column for forecast skill."""
    if result.skill_name != "forecast":
        return result

    columns = data_context.get("columns", [])

    if "date_column" not in result.parameters:
        for c in columns:
            if c["semantic_type"] == "datetime":
                result.parameters["date_column"] = c["name"]
                break
        else:
            # Look for column names that hint at dates
            for c in columns:
                name_l = c["name"].lower()
                if any(d in name_l for d in ("date", "time", "day", "month", "year", "timestamp")):
                    result.parameters["date_column"] = c["name"]
                    break

    if "value_column" not in result.parameters:
        for c in columns:
            if c["semantic_type"] == "numeric" and c["name"] != result.parameters.get("date_column"):
                result.parameters["value_column"] = c["name"]
                break

    return result


def _enrich_hypothesis_params(
    result: RoutingResult,
    question: str,
    data_context: Dict[str, Any],
) -> RoutingResult:
    """Extract test params from question for run_hypothesis_test."""
    if result.skill_name != "run_hypothesis_test":
        return result

    q = question.lower().strip()
    columns = data_context.get("columns", [])
    col_names_lower = {c["name"].lower(): c["name"] for c in columns}
    col_info = {c["name"]: c for c in columns}

    # Default to t_test
    if "test_type" not in result.parameters:
        if re.search(r"\bchi.?square\b", q):
            result.parameters["test_type"] = "chi_square"
        elif re.search(r"\banova\b", q):
            result.parameters["test_type"] = "anova"
        elif re.search(r"\bmann.?whitney\b", q):
            result.parameters["test_type"] = "mann_whitney"
        elif re.search(r"\bnormal(?:ity)?\b", q):
            result.parameters["test_type"] = "normality"
        else:
            result.parameters["test_type"] = "t_test"

    # Extract columns: "difference in X between Y"
    diff_match = re.search(r"(?:difference|compare)\s+(?:in\s+)?(\w[\w\s]*?)\s+(?:between|across|by)\s+(\w[\w\s]*?)(?:\s|$|\?)", q)
    if diff_match:
        val_hint = diff_match.group(1).strip()
        grp_hint = diff_match.group(2).strip()
        val_col = _match_column(val_hint, col_names_lower)
        grp_col = _match_column(grp_hint, col_names_lower)
        if val_col:
            result.parameters["value_col"] = val_col
        if grp_col:
            result.parameters["group_col"] = grp_col

    # If still missing, try to auto-detect from data context
    if "group_col" not in result.parameters or "value_col" not in result.parameters:
        mentioned = _find_mentioned_columns(q, col_names_lower)
        categoricals = [c["name"] for c in columns if c["semantic_type"] == "categorical" and c["n_unique"] <= 10]
        numerics = [c["name"] for c in columns if c["semantic_type"] == "numeric"]

        if "group_col" not in result.parameters:
            for col in mentioned:
                if col in categoricals:
                    result.parameters["group_col"] = col
                    break
            else:
                if categoricals:
                    result.parameters["group_col"] = categoricals[0]

        if "value_col" not in result.parameters:
            for col in mentioned:
                info = col_info.get(col, {})
                if info.get("semantic_type") == "numeric":
                    result.parameters["value_col"] = col
                    break
            else:
                if numerics:
                    result.parameters["value_col"] = numerics[0]

    return result


def _enrich_parameters(
    result: RoutingResult,
    question: str,
    data_context: Dict[str, Any],
) -> RoutingResult:
    """Enrich routing parameters for skills that need them."""
    # Target extraction for classification/regression/etc.
    if result.skill_name in _SKILLS_NEEDING_TARGET and "target" not in result.parameters:
        target = _extract_target(question, data_context)
        if target:
            result.parameters["target"] = target
            result.reasoning += f" (target='{target}')"
            logger.info(f"Enriched {result.skill_name} with target='{target}'")
        else:
            logger.warning(
                f"Skill '{result.skill_name}' needs a target column but none "
                f"could be extracted from question or data context"
            )

    # Forecast auto-detection
    result = _enrich_forecast_params(result, data_context)

    # Hypothesis test params
    result = _enrich_hypothesis_params(result, question, data_context)

    # Chart params (for chart routes that came through keyword table, not priority)
    if result.skill_name == "create_chart" and "chart_type" not in result.parameters:
        result.parameters["chart_type"] = _infer_chart_type(result.parameters, data_context)

    # describe_data column extraction
    if result.skill_name == "describe_data" and "columns" not in result.parameters:
        cols = _extract_describe_columns(question, data_context)
        if cols:
            result.parameters["columns"] = cols
            result.reasoning += f" (columns={cols})"
            logger.info(f"Enriched describe_data with columns={cols}")

    return result


# ---------------------------------------------------------------------------
# Data context builder
# ---------------------------------------------------------------------------

def build_data_context(df) -> Dict[str, Any]:
    """Build a data context dict from a DataFrame for LLM routing."""
    import pandas as pd

    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        # Check bool BEFORE numeric (numpy treats bool as numeric subtype)
        if pd.api.types.is_bool_dtype(df[col]):
            semantic = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            semantic = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            semantic = "datetime"
        elif df[col].nunique() < min(20, len(df) * 0.05):
            semantic = "categorical"
        else:
            semantic = "text"

        columns.append({
            "name": col,
            "dtype": dtype,
            "semantic_type": semantic,
            "n_unique": int(df[col].nunique()),
            "null_pct": round(float(df[col].isnull().mean() * 100), 1),
        })

    return {
        "shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "columns": columns,
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
    }


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------

class Router:
    """Routes user questions to the best analysis skill.

    Priority: chart keywords -> keyword table -> LLM (FailoverProvider) -> profile_data.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._catalog: Optional[str] = None

    @property
    def skill_catalog(self) -> str:
        if self._catalog is None:
            self._catalog = build_skill_catalog()
        return self._catalog

    def route(
        self,
        question: str,
        data_context: Dict[str, Any],
    ) -> RoutingResult:
        """Route a question to the best skill."""
        logger.info(f"Routing question: {question[:80]}...")

        # 0. Chart priority — always check first
        chart_result = _try_chart_priority(question, data_context)
        if chart_result is not None:
            logger.info(
                f"Chart priority routed to 'create_chart' "
                f"(params={chart_result.parameters})"
            )
            return chart_result

        # 1. Standard keyword routing
        keyword_result = _try_keyword_route(question)
        if keyword_result is not None:
            keyword_result = _enrich_parameters(keyword_result, question, data_context)
            logger.info(
                f"Keyword routed to '{keyword_result.skill_name}' "
                f"(confidence={keyword_result.confidence:.2f}, "
                f"params={keyword_result.parameters})"
            )
            return keyword_result

        # 2. Try LLM provider (FailoverProvider handles multi-provider failover)
        if self.provider is not None:
            result = self._try_llm_route(self.provider, question, data_context)
            if result is not None:
                result = _enrich_parameters(result, question, data_context)
                return result

        # 3. Last resort: profile_data
        logger.warning("All LLM routing failed, falling back to profile_data")
        return RoutingResult(
            skill_name="profile_data",
            parameters={},
            confidence=0.3,
            reasoning="Fallback: LLM routing unavailable, defaulting to data profile",
            route_method="fallback",
        )

    def _try_llm_route(
        self,
        provider: LLMProvider,
        question: str,
        data_context: Dict[str, Any],
    ) -> Optional[RoutingResult]:
        """Attempt LLM routing with a provider. Returns None on failure."""
        try:
            result = provider.route_question(
                question=question,
                data_context=data_context,
                skill_catalog=self.skill_catalog,
            )
            if result.confidence <= 0.1 and result.skill_name == "profile_data":
                logger.warning(
                    f"{type(provider).__name__} returned fallback result"
                )
                return None

            result.route_method = "llm"
            reasoning_prefix = "AI selected: "
            if not result.reasoning.startswith(reasoning_prefix):
                result.reasoning = reasoning_prefix + result.reasoning

            logger.info(
                f"LLM routed to '{result.skill_name}' "
                f"(confidence={result.confidence:.2f}) "
                f"via {type(provider).__name__}"
            )
            return result
        except Exception as e:
            logger.warning(f"{type(provider).__name__} routing failed: {e}")
            return None

"""
Router — semantic embedding question-to-skill mapping.

Uses sentence-transformers (all-MiniLM-L6-v2) to match user questions
to the best analytical skill by cosine similarity. Falls back to
smart_query (LLM-generated pandas) when no skill matches, or
profile_data when no LLM is available.

Priority:
  1. Semantic embedding match (local, fast, meaning-based)
  2. smart_query (LLM-generated pandas — flexible fallback)
  3. profile_data (last resort)
"""

import logging
import re
import threading
from typing import Any

from ..llm.provider import LLMProvider, RoutingResult

logger = logging.getLogger("datapilot.core.router")


# ---------------------------------------------------------------------------
# Keyword overrides — high-signal phrases that unambiguously map to a skill.
# Checked BEFORE semantic embedding so that statistical terms aren't confused
# with generic comparison skills by the embedding model.
# ---------------------------------------------------------------------------

_KEYWORD_OVERRIDES = [
    (re.compile(r"\bsignificant(?:ly)?\s+differ", re.IGNORECASE), "run_hypothesis_test"),
    (re.compile(r"\bp[- ]?value\b", re.IGNORECASE), "run_hypothesis_test"),
    (re.compile(r"\bt[- ]?test\b", re.IGNORECASE), "run_hypothesis_test"),
    (re.compile(r"\bchi[- ]?square\b", re.IGNORECASE), "run_hypothesis_test"),
    (re.compile(r"\banova\b", re.IGNORECASE), "run_hypothesis_test"),
    (re.compile(r"\bstatistical(?:ly)?\s+significant\b", re.IGNORECASE), "run_hypothesis_test"),
    (re.compile(r"\bhypothesis\s+test", re.IGNORECASE), "run_hypothesis_test"),
    (re.compile(r"\bmann[- ]?whitney\b", re.IGNORECASE), "run_hypothesis_test"),
]


# ---------------------------------------------------------------------------
# Chart type map — used by _enrich_parameters() for chart type extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Column / chart helpers
# ---------------------------------------------------------------------------

def _get_col_semantic_type(col_name: str, data_context: dict[str, Any]) -> str:
    """Look up the semantic type for a column from data_context (case-insensitive)."""
    col_lower = col_name.lower()
    for c in data_context.get("columns", []):
        if c["name"].lower() == col_lower:
            return c.get("semantic_type", "text")
    return "text"


def _get_col_nunique(col_name: str, data_context: dict[str, Any]) -> int:
    """Look up n_unique for a column from data_context (case-insensitive)."""
    col_lower = col_name.lower()
    for c in data_context.get("columns", []):
        if c["name"].lower() == col_lower:
            return c.get("n_unique", 999)
    return 999


def _infer_chart_type(params: dict[str, Any], data_context: dict[str, Any]) -> str:
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


def _match_column(hint: str, col_names_lower: dict[str, str]) -> str | None:
    """Scored column matching — avoids greedy substring ("age" != "passage").

    Tiered: exact (100) > word boundary (80) > stem (60) > partial (40, guarded).
    """
    hint = hint.lower().strip()
    if not hint:
        return None

    best_score = 0
    best_col = None

    for col_lower, col_orig in col_names_lower.items():
        score = _score_column(hint, col_lower)
        if score > best_score:
            best_score = score
            best_col = col_orig

    return best_col if best_score >= 40 else None


def _score_column(hint: str, col: str) -> int:
    """Score how well a hint matches a column name (0-100)."""
    # Exact
    if hint == col:
        return 100
    # Without spaces/underscores
    hint_clean = hint.replace(" ", "_")
    if hint_clean == col:
        return 95
    col_clean = col.replace("_", " ")
    if hint == col_clean:
        return 95
    # Word boundary: "age" matches "passenger_age" but NOT "passage"
    boundary_re = re.compile(r'(?:^|[_\s])' + re.escape(hint) + r'(?:$|[_\s])')
    if boundary_re.search(col) or boundary_re.search(col_clean):
        return 80
    # Stem match
    hint_stem = _stem_simple(hint)
    col_stem = _stem_simple(col)
    if hint_stem and col_stem and hint_stem == col_stem:
        return 60
    # Partial — guarded: min 4 chars, reject if column is 3x longer
    if len(hint) >= 4 and hint in col:
        if len(col) <= len(hint) * 3:
            return 40
    if len(col) >= 4 and col in hint:
        if len(hint) <= len(col) * 3:
            return 40
    return 0


def _find_mentioned_columns(question: str, col_names_lower: dict[str, str]) -> list[str]:
    """Find all column names mentioned in the question, using word boundary matching."""
    found = []
    q = question.lower()
    for col_lower, col_orig in col_names_lower.items():
        # Word boundary regex to avoid "age" matching inside "passage"
        pattern = re.compile(r'(?:^|[\s,;.!?()])' + re.escape(col_lower) + r'(?:$|[\s,;.!?()])')
        col_space = col_lower.replace("_", " ")
        pattern_space = re.compile(r'(?:^|[\s,;.!?()])' + re.escape(col_space) + r'(?:$|[\s,;.!?()])')
        m = pattern.search(q) or pattern_space.search(q)
        if m:
            found.append((m.start(), col_orig))
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
    data_context: dict[str, Any],
) -> list[str] | None:
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
# Parameter enrichment — extract missing required params from the question
# ---------------------------------------------------------------------------

# Skills that require a `target` column parameter
_SKILLS_NEEDING_TARGET = {
    "classify", "predict_numeric", "select_features", "find_thresholds",
    "calculate_effect_size", "explain_model",
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


def _stem_simple(word: str) -> str:
    """Strip common English suffixes for fuzzy column matching.

    Handles: survival→surviv, survived→surviv, churning→churn, etc.
    Only strips if the remaining stem is >= 3 chars to avoid over-stemming.
    """
    for suffix in ("ival", "ation", "tion", "ment", "ness", "ing", "ed", "er", "al", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _extract_target(
    question: str,
    data_context: dict[str, Any],
) -> str | None:
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
            # Exact match
            if hint in col_names_lower:
                return col_names_lower[hint]
            # Substring match
            for col_lower, col_orig in col_names_lower.items():
                if hint in col_lower or col_lower in hint:
                    return col_orig
            # Stem match: "survival" -> "surviv" matches "survived" -> "surviv"
            hint_stem = _stem_simple(hint)
            for col_lower, col_orig in col_names_lower.items():
                col_stem = _stem_simple(col_lower)
                if hint_stem == col_stem or hint_stem in col_stem or col_stem in hint_stem:
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
    # Check both categorical AND binary numeric columns (e.g. Survived: int 0/1)
    binary_cols = []
    categorical_cols = []
    for c in columns:
        n_unique = c.get("n_unique", 999)
        sem_type = c["semantic_type"]
        if n_unique == 2 and sem_type in ("categorical", "numeric", "boolean"):
            binary_cols.append(c["name"])
        elif sem_type == "categorical" and n_unique <= 10:
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
    data_context: dict[str, Any],
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
    data_context: dict[str, Any],
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


def _enrich_chart_params(
    result: RoutingResult,
    question: str,
    data_context: dict[str, Any],
) -> RoutingResult:
    """Extract chart type and column parameters for create_chart skill."""
    if result.skill_name != "create_chart":
        return result

    q = question.lower()
    col_names_lower = {c["name"].lower(): c["name"] for c in data_context.get("columns", [])}

    # Extract chart type from question
    if "chart_type" not in result.parameters:
        for keyword, chart_type in _CHART_TYPE_MAP.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", q):
                result.parameters["chart_type"] = chart_type
                break

        # Check for "distribution" -> histogram
        if "chart_type" not in result.parameters and "distribution" in q:
            result.parameters["chart_type"] = "histogram"

    # Extract x, y columns: "X vs Y", "X versus Y", "X against Y", "X and Y"
    if "x" not in result.parameters:
        vs_match = re.search(
            r"(?:of|for)?\s*(\w[\w\s]*?)\s+(?:vs\.?|versus|against|and)\s+(\w[\w\s]*?)(?:\s|$|\?)", q
        )
        if vs_match:
            x_hint = vs_match.group(1).strip()
            y_hint = vs_match.group(2).strip()
            x_col = _match_column(x_hint, col_names_lower)
            y_col = _match_column(y_hint, col_names_lower)
            if x_col:
                result.parameters["x"] = x_col
            if y_col:
                result.parameters["y"] = y_col

    # Pattern: "distribution of X", "chart of X", "plot X"
    if "x" not in result.parameters:
        of_match = re.search(
            r"(?:distribution|chart|plot|graph|histogram|boxplot|box plot)\s+(?:of|for)\s+(\w[\w\s]*?)(?:\s*(?:vs|and|by|\?|$))", q
        )
        if of_match:
            hint = of_match.group(1).strip()
            col = _match_column(hint, col_names_lower)
            if col:
                result.parameters["x"] = col

    # Pattern: "by group_col" -> hue
    by_match = re.search(r"\bby\s+(\w[\w\s]*?)(?:\s*(?:\?|$))", q)
    if by_match and "hue" not in result.parameters:
        hint = by_match.group(1).strip()
        col = _match_column(hint, col_names_lower)
        if col:
            result.parameters["hue"] = col

    # Fallback: find column names mentioned anywhere in the question
    if "x" not in result.parameters:
        mentioned = _find_mentioned_columns(q, col_names_lower)
        if len(mentioned) >= 2:
            result.parameters["x"] = mentioned[0]
            result.parameters["y"] = mentioned[1]
        elif len(mentioned) == 1:
            result.parameters["x"] = mentioned[0]

    # Infer chart type from column data types if not explicitly specified
    if "chart_type" not in result.parameters:
        result.parameters["chart_type"] = _infer_chart_type(result.parameters, data_context)

    return result


def _enrich_parameters(
    result: RoutingResult,
    question: str,
    data_context: dict[str, Any],
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

    # Chart params — full extraction (chart type, x, y, hue columns)
    result = _enrich_chart_params(result, question, data_context)

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

def build_data_context(df) -> dict[str, Any]:
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

    Uses semantic embeddings (sentence-transformers) as the primary routing
    mechanism. Falls back to smart_query (LLM-generated pandas) when no
    embedding match is found, or profile_data when no LLM is available.

    Priority:
      1. Semantic embedding match (local, fast, meaning-based)
      2. smart_query (LLM-generated pandas — when provider is available)
      3. profile_data (last resort)
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._semantic_matcher = None  # lazy-loaded on first use
        self._semantic_attempted = False  # avoid repeated import failures
        self._semantic_lock = threading.Lock()  # prevent concurrent model loads

    def route(
        self,
        question: str,
        data_context: dict[str, Any],
    ) -> RoutingResult:
        """Route a question to the best skill.

        Semantic-first: embedding match is tried first, then smart_query
        handles anything the embeddings don't catch.
        """
        logger.info(f"Routing question: {question[:80]}...")

        # 0. Keyword overrides — unambiguous statistical/analytical terms
        for pattern, skill_name in _KEYWORD_OVERRIDES:
            if pattern.search(question):
                logger.info(f"Keyword override: '{question[:50]}...' -> {skill_name}")
                result = RoutingResult(
                    skill_name=skill_name,
                    parameters={},
                    confidence=0.90,
                    reasoning=f"Keyword override matched: {pattern.pattern}",
                    route_method="keyword_override",
                )
                result = _enrich_parameters(result, question, data_context)
                return result

        # 1. Semantic embedding match (primary)
        result = self._try_semantic_embedding(question, data_context)
        if result is not None:
            result = _enrich_parameters(result, question, data_context)
            logger.info(
                f"Semantic embedding routed to '{result.skill_name}' "
                f"(confidence={result.confidence:.2f}, "
                f"params={result.parameters})"
            )
            return result

        # 2. smart_query (LLM fallback)
        if self.provider is not None:
            logger.info("No embedding match — routing to smart_query")
            return RoutingResult(
                skill_name="smart_query",
                parameters={},
                confidence=0.85,
                reasoning="LLM-generated pandas query",
                route_method="smart_first",
            )

        # 3. profile_data (last resort)
        logger.warning("No embedding match, no LLM — falling back to profile_data")
        return RoutingResult(
            skill_name="profile_data",
            parameters={},
            confidence=0.30,
            reasoning="No embedding match, no LLM — showing data profile",
            route_method="fallback",
        )

    def _try_semantic_embedding(
        self,
        question: str,
        data_context: dict[str, Any],
    ) -> RoutingResult | None:
        """Try semantic embedding match using sentence-transformers.

        Lazy-loads the model on first call. Silently skips if
        sentence-transformers is not installed.
        """
        if self._semantic_attempted and self._semantic_matcher is None:
            return None

        if self._semantic_matcher is None:
            with self._semantic_lock:
                # Double-check after acquiring lock (another thread may have loaded it)
                if self._semantic_matcher is not None:
                    pass  # already loaded by another thread
                elif self._semantic_attempted:
                    return None  # another thread tried and failed
                else:
                    try:
                        from .semantic_router import get_semantic_matcher
                        self._semantic_matcher = get_semantic_matcher()
                    except Exception as e:
                        logger.info(f"Semantic embedding unavailable: {e}")
                    finally:
                        # Set flag AFTER load attempt completes (success or failure)
                        self._semantic_attempted = True

        if self._semantic_matcher is None:
            return None

        result = self._semantic_matcher.match(question, threshold=0.35)
        if result is None:
            return None

        skill_name, score = result

        return RoutingResult(
            skill_name=skill_name,
            parameters={},
            confidence=round(min(score, 0.95), 2),
            reasoning=f"Semantic embedding: '{question[:50]}...' -> {skill_name} (score={score:.3f})",
            route_method="semantic_embedding",
        )

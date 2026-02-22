"""
Query skills — interactive data querying via natural language.

5 fixed skills for common patterns + 1 smart_query LLM fallback.
"""

import ast
import ctypes
import difflib
import json
import logging
import re
import threading
import time as _time

import numpy as np
import pandas as pd

from ..utils.serializer import safe_json_serialize

logger = logging.getLogger("datapilot.analysis.query")


# ---------------------------------------------------------------------------
# Smart query sandbox constants
# ---------------------------------------------------------------------------

_SMART_QUERY_TIMEOUT = 5
_SMART_QUERY_MAX_LINES = 10
_SMART_QUERY_MAX_ROWS = 500
_SMART_QUERY_MAX_TIMEOUTS = 3

_FORBIDDEN_NAMES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "open", "eval", "exec", "compile", "__import__",
    "globals", "locals", "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit",
})

_FORBIDDEN_ATTR_CALLS = frozenset({
    "eval", "exec", "compile", "system", "popen",
    "read_csv", "read_html", "read_sql", "read_excel",
    "read_json", "read_parquet", "read_pickle",
    "to_csv", "to_excel", "to_sql", "to_json",
    "to_parquet", "to_pickle", "to_html", "to_clipboard", "to_latex",
})

# ---------------------------------------------------------------------------
# Safe proxy objects for sandbox (Fix #1)
# ---------------------------------------------------------------------------

_ALLOWED_PD_METHODS = {
    "DataFrame", "Series", "to_numeric", "to_datetime", "cut", "qcut",
    "concat", "merge", "get_dummies", "crosstab", "pivot_table",
    "isna", "notna", "NA", "NaT",
}

_ALLOWED_NP_METHODS = {
    "mean", "median", "std", "var", "sum", "min", "max", "abs",
    "sqrt", "log", "log2", "log10", "exp", "round", "ceil", "floor",
    "where", "select", "clip", "nan", "inf", "array", "zeros", "ones",
    "arange", "linspace", "percentile", "quantile", "unique", "sort",
    "histogram", "corrcoef",
}


class _SafePandas:
    """Proxy that only exposes whitelisted pandas methods."""

    def __getattr__(self, name: str):
        if name in _ALLOWED_PD_METHODS:
            return getattr(pd, name)
        raise AttributeError(f"pd.{name} is not allowed in smart_query")


class _SafeNumpy:
    """Proxy that only exposes whitelisted numpy methods."""

    def __getattr__(self, name: str):
        if name in _ALLOWED_NP_METHODS:
            return getattr(np, name)
        raise AttributeError(f"np.{name} is not allowed in smart_query")


# Safe builtins whitelist for sandbox execution.
# smart_query is the primary path for data questions, so the sandbox needs
# common Python builtins like len(), round(), sorted(), etc.
_SAFE_BUILTINS = {
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "reversed": reversed, "list": list, "dict": dict,
    "tuple": tuple, "set": set, "frozenset": frozenset,
    "int": int, "float": float, "str": str, "bool": bool,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "isinstance": isinstance, "print": lambda *a, **k: None,
    "True": True, "False": False, "None": None,
}

# Time-windowed timeout tracking (recovers after window expires)
_timeout_timestamps: list = []
_TIMEOUT_WINDOW = 60  # seconds


def _is_smart_query_available() -> bool:
    """Check if smart_query is available (fewer than 3 timeouts in the last 60s)."""
    now = _time.time()
    _timeout_timestamps[:] = [t for t in _timeout_timestamps if now - t < _TIMEOUT_WINDOW]
    return len(_timeout_timestamps) < _SMART_QUERY_MAX_TIMEOUTS


def _record_timeout() -> None:
    """Record a timeout event."""
    _timeout_timestamps.append(_time.time())


# ---------------------------------------------------------------------------
# LLM parameter extraction helper
# ---------------------------------------------------------------------------

def _extract_params_via_llm(
    question: str,
    llm_provider,
    skill_name: str,
    param_schema: dict,
    df_columns: list,
) -> dict | None:
    """Extract skill parameters from natural language via LLM.

    Returns parsed dict or None if LLM fails / returns invalid JSON.
    """
    if llm_provider is None:
        return None
    try:
        schema_desc = "\n".join(f"  {k}: {v}" for k, v in param_schema.items())
        prompt = (
            f"Dataset columns: {', '.join(str(c) for c in df_columns)}\n"
            f"User question: {question}\n\n"
            f"Extract parameters for '{skill_name}':\n{schema_desc}\n\n"
            "Respond ONLY with valid JSON matching the schema above."
        )
        response = llm_provider.generate_plan(prompt)
        if response is None:
            return None
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM param extraction failed for {skill_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Backtick helper for AST validation
# ---------------------------------------------------------------------------

def _resolve_column(hint: str, df_columns: list) -> str | None:
    """Scored column name resolution — avoids greedy substring matches.

    Tiered scoring: exact (100) > word boundary (80) > stem (60) > partial (40).
    Rejects partial matches where the column is 3x longer than the hint,
    and requires minimum 4 chars for partial matching.
    Returns the actual column name or None.
    """
    if hint is None:
        return None
    hint_lower = hint.lower().strip()
    if not hint_lower:
        return None
    col_map = {c.lower(): c for c in df_columns}

    best_score = 0
    best_col = None

    for col_lower, col_orig in col_map.items():
        score = _score_column_match(hint_lower, col_lower)
        if score > best_score:
            best_score = score
            best_col = col_orig

    return best_col if best_score >= 40 else None


def _stem_simple_local(word: str) -> str:
    """Strip common English suffixes for fuzzy column matching."""
    for suffix in ("ival", "ation", "tion", "ment", "ness", "ing", "ed", "er", "al", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _score_column_match(hint: str, col: str) -> int:
    """Score how well a hint matches a column name (0-100)."""
    # Exact match
    if hint == col:
        return 100

    # Without spaces/underscores exact match
    hint_clean = hint.replace("_", "").replace(" ", "")
    col_clean = col.replace("_", "").replace(" ", "")
    if hint_clean == col_clean:
        return 95

    # Word boundary match: "age" matches "passenger_age" but not "passage"
    boundary_re = re.compile(r'(?:^|[_\s])' + re.escape(hint) + r'(?:$|[_\s])')
    if boundary_re.search(col) or boundary_re.search(col.replace("_", " ")):
        return 80

    # Stem match
    hint_stem = _stem_simple_local(hint)
    col_stem = _stem_simple_local(col)
    if hint_stem and col_stem and (hint_stem == col_stem):
        return 60

    # Partial match — guarded: min 4 chars, reject if column is 3x longer
    if len(hint) >= 4 and hint in col:
        if len(col) <= len(hint) * 3:
            return 40
    if len(col) >= 4 and col in hint:
        if len(hint) <= len(col) * 3:
            return 40

    return 0


def _column_not_found_message(col_name: str, df_columns: list) -> str:
    """Build a helpful 'column not found' message with fuzzy suggestions."""
    suggestions = difflib.get_close_matches(
        col_name.lower(),
        [c.lower() for c in df_columns],
        n=3,
        cutoff=0.5,
    )
    if suggestions:
        # Map back to original case
        col_map = {c.lower(): c for c in df_columns}
        originals = [col_map[s] for s in suggestions]
        return f"Column '{col_name}' not found. Did you mean: {', '.join(originals)}?"
    return f"Column '{col_name}' not found. Available columns: {', '.join(df_columns[:10])}"


def _sanitize_backticks_for_ast(expr: str) -> str:
    """Replace `backtick quoted names` with safe placeholders for AST validation.

    pandas df.query() supports backtick-quoted column names (e.g. `col name` == 0),
    but ast.parse() doesn't.  We swap them out before validation only.
    """
    return re.sub(r'`[^`]+`', '_col', expr)


# ---------------------------------------------------------------------------
# Skill 1: query_data
# ---------------------------------------------------------------------------

def query_data(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Filter and select rows from the dataset based on natural language conditions.

    Use this when the user asks to filter, show specific rows, or count rows matching criteria.
    Examples: "Show me rows where churn = Yes", "How many customers have >3 service calls"
    """
    try:
        params = _extract_params_via_llm(
            question, llm_provider, "query_data",
            {"filter_expression": "pandas query string (use backticks for column names with spaces, e.g. `col name` == value)", "columns": "list of column names or null"},
            list(df.columns),
        )

        filter_expression = None
        columns = None

        if params:
            filter_expression = params.get("filter_expression")
            columns = params.get("columns")
        else:
            # Heuristic fallback: multiple pattern matching
            cols = list(df.columns)

            # Pattern 1: "where COL = VAL" / "where COL is VAL"
            match = re.search(
                r"where\s+(\w[\w\s]*?)\s*(?:=|==|is)\s*(\w+)",
                question, re.IGNORECASE,
            )
            if match:
                col_hint, val = match.group(1).strip(), match.group(2)
                col = _resolve_column(col_hint, cols)
                if col:
                    filter_expression = f"`{col}` == '{val}'"

            # Pattern 2: "COL above/below/greater/less N"
            if not filter_expression:
                cmp_match = re.search(
                    r"(\w[\w\s]*?)\s+(?:above|greater than|more than|over|>)\s+(\d+(?:\.\d+)?)",
                    question, re.IGNORECASE,
                )
                if cmp_match:
                    col = _resolve_column(cmp_match.group(1).strip(), cols)
                    if col:
                        filter_expression = f"`{col}` > {cmp_match.group(2)}"
                else:
                    cmp_match = re.search(
                        r"(\w[\w\s]*?)\s+(?:below|less than|under|<)\s+(\d+(?:\.\d+)?)",
                        question, re.IGNORECASE,
                    )
                    if cmp_match:
                        col = _resolve_column(cmp_match.group(1).strip(), cols)
                        if col:
                            filter_expression = f"`{col}` < {cmp_match.group(2)}"

            # Pattern 3: "only CATEGORY" / "show me CATEGORY" — match against values
            if not filter_expression:
                only_match = re.search(
                    r"(?:only|just|show me|display)\s+(?:the\s+)?(\w+)",
                    question, re.IGNORECASE,
                )
                if only_match:
                    val_hint = only_match.group(1).lower()
                    # Search categorical columns for matching values
                    for c in cols:
                        if df[c].dtype == object or df[c].nunique() <= 20:
                            unique_vals = {str(v).lower(): str(v) for v in df[c].dropna().unique()}
                            if val_hint in unique_vals:
                                filter_expression = f"`{c}` == '{unique_vals[val_hint]}'"
                                break
                            # Partial match (e.g., "female" in "Female")
                            for uv_lower, uv_orig in unique_vals.items():
                                if val_hint in uv_lower or uv_lower in val_hint:
                                    filter_expression = f"`{c}` == '{uv_orig}'"
                                    break
                            if filter_expression:
                                break

        filtered_df = df

        # If no filter could be extracted, return informative error
        if not filter_expression and not columns:
            logger.warning(f"query_data: no filter extracted from '{question[:60]}...'")
            return {
                "status": "error",
                "message": (
                    "Could not extract a filter condition from your question. "
                    "Try being more specific, e.g., 'Show rows where Age > 30' "
                    f"or 'Filter by {df.columns[0]} = <value>'."
                ),
            }

        if filter_expression:
            # Validate: reject function calls in filter expressions
            # Sanitize backtick-quoted column names before AST check
            # (backticks are valid in df.query() but not in Python AST)
            try:
                sanitized = _sanitize_backticks_for_ast(filter_expression)
                filter_tree = ast.parse(sanitized, mode="eval")
                for node in ast.walk(filter_tree):
                    if isinstance(node, ast.Call):
                        return {"status": "error", "message": "Function calls are not allowed in filter expressions"}
            except SyntaxError:
                return {"status": "error", "message": "Invalid filter expression syntax"}

            try:
                filtered_df = df.query(filter_expression)
            except Exception as e:
                return {"status": "error", "message": f"Invalid filter expression: {e}"}

        if columns:
            try:
                filtered_df = filtered_df[columns]
            except KeyError as e:
                return {"status": "error", "message": f"Column not found: {e}"}

        records = safe_json_serialize(filtered_df.head(100).to_dict("records"))
        result = {
            "status": "success",
            "data": records,
            "total_rows": len(filtered_df),
            "query_description": filter_expression or "all rows",
        }
        # Add aggregate stats for LLM narration (prevents hallucination)
        if len(filtered_df) > 0:
            result["data_summary"] = _summarize_dataframe(filtered_df)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Skill 2: pivot_table
# ---------------------------------------------------------------------------

def pivot_table(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Aggregate data by groups using pivot tables.

    Use this for questions about averages, sums, or counts grouped by categories.
    Examples: "Average monthly charges by contract type", "Total revenue by state"
    """
    try:
        params = _extract_params_via_llm(
            question, llm_provider, "pivot_table",
            {"values": "str (column to aggregate)", "index": "str (column to group by)", "aggfunc": "str or list (mean, sum, count, etc.)"},
            list(df.columns),
        )

        # Validate LLM params have actual values (not None)
        if params and not params.get("values"):
            params = None

        if not params:
            # Heuristic fallback: "average/sum/count X by Y"
            match = re.search(
                r"(average|mean|sum|count|max|min)\s+(?:of\s+)?(\w+)\s+by\s+(\w+)",
                question, re.IGNORECASE,
            )
            if match:
                aggfunc_map = {"average": "mean", "mean": "mean", "sum": "sum",
                               "count": "count", "max": "max", "min": "min"}
                aggfunc = aggfunc_map.get(match.group(1).lower(), "mean")
                values_hint = match.group(2)
                index_hint = match.group(3)
                values_col = _resolve_column(values_hint, list(df.columns))
                index_col = _resolve_column(index_hint, list(df.columns))
                if values_col and index_col:
                    params = {"values": values_col, "index": index_col, "aggfunc": aggfunc}

        if not params:
            return {"status": "error", "message": "Could not extract pivot parameters"}

        values = params.get("values")
        index = params.get("index")
        aggfunc = params.get("aggfunc", "mean")

        # Resolve column names case-insensitively
        values = _resolve_column(values, list(df.columns)) or values
        index = _resolve_column(index, list(df.columns)) or index

        if values not in df.columns:
            return {"status": "error", "message": f"Column '{values}' not found in dataset"}
        if index not in df.columns:
            return {"status": "error", "message": f"Column '{index}' not found in dataset"}

        result_df = pd.pivot_table(df, values=values, index=index, aggfunc=aggfunc)
        result_df = result_df.reset_index()
        # Flatten MultiIndex columns from multi-aggfunc pivots
        if isinstance(result_df.columns, pd.MultiIndex):
            result_df.columns = ["_".join(str(c) for c in col).strip("_") for col in result_df.columns]
        records = safe_json_serialize(result_df.to_dict("records"))

        return {
            "status": "success",
            "data": records,
            "index_column": index,
            "values_column": values,
            "aggfunc": aggfunc,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Skill 3: value_counts
# ---------------------------------------------------------------------------

def value_counts(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Compute frequency distribution of a categorical column.

    Use this when the user asks about counts per category or frequency distributions.
    Examples: "How many customers per state?", "Distribution of contract types"
    """
    try:
        params = _extract_params_via_llm(
            question, llm_provider, "value_counts",
            {"column": "str (column to count)", "top_n": "int (optional, limit results)"},
            list(df.columns),
        )

        if params and not params.get("column"):
            params = None

        if not params:
            return {"status": "error", "message": "Could not extract column for value_counts"}

        column = _resolve_column(params.get("column"), list(df.columns)) or params.get("column")
        top_n_val = params.get("top_n")

        if column not in df.columns:
            return {"status": "error", "message": f"Column '{column}' not found in dataset"}

        counts = df[column].value_counts()
        if top_n_val:
            counts = counts.head(int(top_n_val))

        return {
            "status": "success",
            "data": safe_json_serialize(counts.to_dict()),
            "column": column,
            "total_values": int(df[column].count()),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Skill 4: top_n
# ---------------------------------------------------------------------------

def top_n(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Rank and return the top or bottom N records by a column.

    Use this for ranking questions like top/bottom/highest/lowest.
    Examples: "Top 10 customers by monthly charges", "Bottom 5 states by churn rate"
    """
    try:
        params = _extract_params_via_llm(
            question, llm_provider, "top_n",
            {"column": "str (column to rank by)", "n": "int (default 10)", "ascending": "bool (default False, True for bottom N)"},
            list(df.columns),
        )

        if params and not params.get("column"):
            params = None

        if not params:
            # Deterministic fallback: "top/bottom N by/in COLUMN" patterns
            q_lower = question.lower()
            cols = list(df.columns)
            col_map = {c.lower(): c for c in cols}

            n_val = 10
            ascending = False
            # Extract N
            n_match = re.search(r'\b(?:top|bottom|best|worst|highest|lowest)\s+(\d+)', q_lower)
            if n_match:
                n_val = int(n_match.group(1))
            ascending = bool(re.search(r'\b(?:bottom|worst|lowest|smallest|cheapest|least)\b', q_lower))

            # Find column: "by COLUMN" or "in COLUMN"
            by_match = re.search(r'\b(?:by|in|ranked by|sorted by)\s+(\w[\w\s]*?)(?:\s*(?:\?|$|for|from))', q_lower)
            col_hint = by_match.group(1).strip() if by_match else None
            resolved_col = _resolve_column(col_hint, cols) if col_hint else None

            if not resolved_col:
                # Try finding any column mentioned in the question
                for col_lower, col_orig in col_map.items():
                    alt = col_lower.replace("_", " ")
                    if col_lower in q_lower or alt in q_lower:
                        if pd.api.types.is_numeric_dtype(df[col_orig]):
                            resolved_col = col_orig
                            break

            if not resolved_col:
                # Last resort: first numeric column
                for c in cols:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        resolved_col = c
                        break

            if resolved_col:
                params = {"column": resolved_col, "n": n_val, "ascending": ascending}

        if not params:
            return {"status": "error", "message": "Could not extract top_n parameters"}

        column = _resolve_column(params.get("column"), list(df.columns)) or params.get("column")
        n = params.get("n", 10) or 10
        ascending = params.get("ascending", False)

        if column not in df.columns:
            return {"status": "error", "message": f"Column '{column}' not found in dataset"}

        n = min(int(n), len(df))

        if ascending:
            result_df = df.nsmallest(n, column)
            direction = "bottom"
        else:
            result_df = df.nlargest(n, column)
            direction = "top"

        records = safe_json_serialize(result_df.to_dict("records"))
        result = {
            "status": "success",
            "data": records,
            "n": n,
            "column": column,
            "direction": direction,
        }
        # Add aggregate stats for LLM narration (prevents hallucination)
        if len(result_df) > 0:
            result["data_summary"] = _summarize_dataframe(result_df)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Skill 5: cross_tab
# ---------------------------------------------------------------------------

def cross_tab(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Cross-tabulate two categorical columns to show their relationship.

    Use this for questions about the relationship between two categorical variables.
    Examples: "Churn by contract type and internet service"
    """
    try:
        params = _extract_params_via_llm(
            question, llm_provider, "cross_tab",
            {"row": "str (row variable)", "col": "str (column variable)", "values": "str or null (values to aggregate)", "aggfunc": "str or null (aggregation function)"},
            list(df.columns),
        )

        if params and not params.get("row"):
            params = None

        if not params:
            # Deterministic fallback: find two columns mentioned in the question
            cols = list(df.columns)
            col_map = {c.lower(): c for c in cols}
            q_lower = question.lower()

            # Pattern: "X by Y", "X vs Y", "X and Y"
            mentioned = []
            for col_lower, col_orig in col_map.items():
                alt = col_lower.replace("_", " ")
                if col_lower in q_lower or alt in q_lower:
                    mentioned.append(col_orig)
            # Also match partial names (e.g. "gender" -> "Sex", "class" -> "Pclass")
            if len(mentioned) < 2:
                for word in re.findall(r'\b\w+\b', q_lower):
                    if len(word) < 3:
                        continue
                    for col_lower, col_orig in col_map.items():
                        if col_orig in mentioned:
                            continue
                        if word in col_lower or col_lower in word:
                            mentioned.append(col_orig)
                            break

            # Pick two categorical/low-cardinality columns from mentions
            if len(mentioned) >= 2:
                params = {"row": mentioned[0], "col": mentioned[1]}
            elif len(mentioned) == 1:
                # Pair with first other categorical column
                cat_cols = [c for c in cols if df[c].nunique() <= 20 and c != mentioned[0]]
                if cat_cols:
                    params = {"row": mentioned[0], "col": cat_cols[0]}

        if not params:
            return {"status": "error", "message": "Could not extract cross_tab parameters"}

        cols = list(df.columns)
        row = _resolve_column(params.get("row"), cols) or params.get("row")
        col = _resolve_column(params.get("col"), cols) or params.get("col")
        values_col = _resolve_column(params.get("values"), cols) if params.get("values") else None
        aggfunc = params.get("aggfunc")

        if row not in df.columns:
            return {"status": "error", "message": f"Column '{row}' not found in dataset"}
        if col not in df.columns:
            return {"status": "error", "message": f"Column '{col}' not found in dataset"}

        if values_col and aggfunc:
            if values_col not in df.columns:
                return {"status": "error", "message": f"Column '{values_col}' not found in dataset"}
            ct = pd.crosstab(df[row], df[col], values=df[values_col], aggfunc=aggfunc)
        else:
            ct = pd.crosstab(df[row], df[col])

        records = safe_json_serialize(ct.reset_index().to_dict("records"))
        return {
            "status": "success",
            "data": records,
            "row_column": row,
            "col_column": col,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Smart query sandbox helpers
# ---------------------------------------------------------------------------

def _validate_code(code: str) -> tuple:
    """Validate LLM-generated code for safety.

    Returns (is_safe: bool, reason: str).
    """
    lines = code.strip().split("\n")
    if len(lines) > _SMART_QUERY_MAX_LINES:
        return False, f"Code exceeds {_SMART_QUERY_MAX_LINES} line limit"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Forbidden: import statement"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_NAMES:
                return False, f"Forbidden function: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_ATTR_CALLS:
                return False, f"Forbidden method: {node.func.attr}"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"Forbidden name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"Forbidden dunder attribute: {node.attr}"
    return True, "OK"


def _extract_column_refs(code: str) -> list[str]:
    """Walk AST to extract df['col'] and df["col"] string references."""
    refs = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return refs
    for node in ast.walk(tree):
        # df['col'] or df["col"]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "df"
        ):
            # Simple string index: df['col']
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                refs.append(node.slice.value)
    return refs


def _fix_column_refs(code: str, df_columns: list) -> str:
    """Auto-correct column name typos in LLM-generated code.

    Finds df['col'] references, checks against actual columns, and replaces
    close-enough misspellings (difflib cutoff=0.6).
    """
    refs = _extract_column_refs(code)
    col_set = set(df_columns)
    col_lower_map = {c.lower(): c for c in df_columns}

    for ref in refs:
        if ref in col_set:
            continue  # exact match, no fix needed
        # Case-insensitive match
        if ref.lower() in col_lower_map:
            code = code.replace(f"'{ref}'", f"'{col_lower_map[ref.lower()]}'")
            code = code.replace(f'"{ref}"', f'"{col_lower_map[ref.lower()]}"')
            continue
        # Fuzzy match
        matches = difflib.get_close_matches(ref, df_columns, n=1, cutoff=0.6)
        if matches:
            code = code.replace(f"'{ref}'", f"'{matches[0]}'")
            code = code.replace(f'"{ref}"', f'"{matches[0]}"')
            logger.info(f"smart_query auto-corrected column: '{ref}' -> '{matches[0]}'")
    return code


def _is_id_column(series: pd.Series, col_name: str) -> bool:
    """Detect ID-like columns that aren't useful for narration."""
    name_lower = col_name.lower()
    if any(tag in name_lower for tag in ("id", "_id", "index", "row_num")):
        if pd.api.types.is_numeric_dtype(series) or series.nunique() > len(series) * 0.9:
            return True
    # High-cardinality text (e.g. names, tickets) — unique ratio > 80%
    if not pd.api.types.is_numeric_dtype(series) and len(series) > 0:
        if series.nunique() > len(series) * 0.8:
            return True
    return False


def _summarize_dataframe(df_output: pd.DataFrame, max_columns: int = 8) -> dict:
    """Compute aggregate stats from a DataFrame for LLM narration.

    Returns a compact summary dict so the LLM can narrate accurate statistics
    instead of hallucinating from truncated raw rows. Skips ID-like and
    high-cardinality text columns to keep the narrative focused.

    For small result sets (≤ 20 rows, typical of groupby/pivot/crosstab), includes
    the actual row data so the LLM can cite exact values instead of meaningless
    aggregate-of-aggregates.
    """
    # Normalise: bring index labels into columns so groupby/crosstab indexes
    # (e.g. Pclass, Survived) become visible data for narration.
    display_df = df_output
    try:
        if df_output.index.name or (
            hasattr(df_output.index, "names") and any(df_output.index.names)
        ):
            display_df = df_output.reset_index()
        # Flatten MultiIndex columns (crosstab produces tuples like (1, 'female'))
        if isinstance(display_df.columns, pd.MultiIndex):
            display_df.columns = [
                "_".join(str(c) for c in col).strip("_")
                for col in display_df.columns
            ]
    except Exception:
        display_df = df_output

    summary: dict = {
        "total_rows": len(display_df),
        "total_columns": len(display_df.columns),
        "columns": {},
    }

    # For small result sets (groupby, pivot, crosstab), include actual rows
    # so the LLM sees exact values instead of meaningless mean-of-means.
    if len(display_df) <= 20:
        try:
            rows_data = display_df.head(20).to_dict("records")
            summary["rows"] = safe_json_serialize(rows_data)
        except Exception:
            pass

    included = 0
    skipped: list = []
    for col in display_df.columns:
        if included >= max_columns:
            break
        series = display_df[col]
        if _is_id_column(series, str(col)):
            skipped.append(col)
            continue
        included += 1
        col_info: dict = {"dtype": str(series.dtype)}

        try:
            if pd.api.types.is_bool_dtype(series):
                vc = series.value_counts()
                col_info["type"] = "boolean"
                col_info["true_count"] = int(vc.get(True, 0))
                col_info["false_count"] = int(vc.get(False, 0))

            elif pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                col_info["type"] = "numeric"
                col_info["count"] = int(len(clean))
                col_info["mean"] = round(float(clean.mean()), 4) if len(clean) else None
                col_info["std"] = round(float(clean.std()), 4) if len(clean) else None
                col_info["min"] = float(clean.min()) if len(clean) else None
                col_info["max"] = float(clean.max()) if len(clean) else None
                col_info["median"] = float(clean.median()) if len(clean) else None
                if series.nunique() <= 20:
                    vc = series.value_counts().head(15)
                    unique_vals = set(clean.unique())
                    # Label binary 0/1 columns — only when the column name
                    # suggests a boolean flag (not arbitrary numeric codes).
                    _BINARY_HINTS = (
                        "surviv", "churn", "flag", "is_", "has_", "was_",
                        "active", "default", "cancel", "return", "success",
                        "fail", "pass", "approved", "reject", "attrition",
                        "target", "label", "outcome", "status", "result",
                    )
                    col_lower = str(col).lower()
                    looks_boolean = any(h in col_lower for h in _BINARY_HINTS)
                    if unique_vals <= {0, 1, 0.0, 1.0} and series.nunique() == 2 and looks_boolean:
                        col_info["value_labels"] = {"0": "No", "1": "Yes"}
                        col_info["value_counts"] = {
                            f"{int(k)} ({('No' if int(k) == 0 else 'Yes')})": int(v)
                            for k, v in vc.items()
                        }
                    elif unique_vals <= {0, 1, 0.0, 1.0} and series.nunique() == 2:
                        # Binary 0/1 but column name doesn't suggest boolean —
                        # show raw counts without Yes/No labels
                        col_info["value_counts"] = {str(int(k)): int(v) for k, v in vc.items()}
                        col_info["note"] = "Binary column (0/1) — check domain context for meaning"
                    elif unique_vals <= {0, 1, 0.0, 1.0} and series.nunique() == 1:
                        # Single-value binary column — likely used as a filter
                        only_val = int(clean.iloc[0])
                        if looks_boolean:
                            col_info["value_labels"] = {"0": "No", "1": "Yes"}
                            col_info["note"] = (
                                f"All {len(clean)} values are {only_val} "
                                f"({'Yes' if only_val else 'No'}) — "
                                f"this column was likely used as a filter"
                            )
                        else:
                            col_info["note"] = (
                                f"All {len(clean)} values are {only_val} — "
                                f"this column was likely used as a filter"
                            )
                    else:
                        col_info["value_counts"] = {str(k): int(v) for k, v in vc.items()}

            elif pd.api.types.is_datetime64_any_dtype(series):
                clean = series.dropna()
                col_info["type"] = "datetime"
                col_info["min"] = str(clean.min()) if len(clean) else None
                col_info["max"] = str(clean.max()) if len(clean) else None

            else:
                col_info["type"] = "categorical"
                col_info["n_unique"] = int(series.nunique())
                total = int(series.count())
                vc = series.value_counts().head(15)
                col_info["value_counts"] = {
                    str(k): {"count": int(v), "pct": round(v / total * 100, 1) if total else 0}
                    for k, v in vc.items()
                }
        except Exception:
            col_info["type"] = "unknown"

        summary["columns"][str(col)] = col_info

    if skipped:
        summary["_skipped_columns"] = skipped
    remaining = len(display_df.columns) - included - len(skipped)
    if remaining > 0:
        summary["_columns_truncated"] = f"Showing {included} of {len(display_df.columns)} columns"

    return safe_json_serialize(summary)


def _kill_thread(thread: threading.Thread) -> bool:
    """Raise SystemExit in the target thread to stop it (CPython only).

    Returns True if the exception was successfully set.
    Works for Python-level loops like `while True: pass` because the
    async exception is checked between bytecode instructions.
    """
    if not thread.is_alive():
        return True
    tid = thread.ident
    if tid is None:
        return False
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(SystemExit)
    )
    return res == 1


def _execute_safe(code: str, df: pd.DataFrame) -> dict:
    """Execute validated code in a sandboxed namespace.

    Returns result dict with status, data, and code.
    Uses daemon threads with ctypes kill to stop runaway code on timeout.
    """
    if not _is_smart_query_available():
        return {"status": "error", "message": "smart_query temporarily unavailable: too many recent timeouts (retries in 60s)"}

    safe_df = df.copy()
    # Single namespace dict so pd/np/df are visible inside lambdas and
    # comprehensions (nested scopes only see globals, not exec locals).
    namespace = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": _SafePandas(),
        "np": _SafeNumpy(),
        "df": safe_df,
    }
    result_container = {"status": "error", "message": "Unknown error"}

    def _run():
        try:
            exec(code, namespace)
            output = namespace.get("result", None)
            if output is None:
                result_container.update({"status": "error", "message": "No 'result' variable found"})
                return
            if isinstance(output, pd.DataFrame):
                result_container.update({
                    "status": "success",
                    "data": safe_json_serialize(output.head(_SMART_QUERY_MAX_ROWS).to_dict("records")),
                    "total_rows": len(output),
                })
                result_container["_raw_output"] = output
            elif isinstance(output, pd.Series):
                result_container.update({
                    "status": "success",
                    "data": safe_json_serialize(output.head(_SMART_QUERY_MAX_ROWS).to_dict()),
                    "total_rows": len(output),
                })
                result_container["_raw_output"] = output
            elif isinstance(output, (int, float, str, bool, np.integer, np.floating)):
                result_container.update({"status": "success", "data": safe_json_serialize(output)})
            else:
                result_container.update({"status": "error", "message": f"Unexpected output type: {type(output).__name__}"})
        except SystemExit:
            result_container.update({"status": "error", "message": f"Code execution timeout after {_SMART_QUERY_TIMEOUT}s"})
        except Exception as e:
            result_container.update({"status": "error", "message": str(e)[:200]})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=_SMART_QUERY_TIMEOUT)
    if thread.is_alive():
        _kill_thread(thread)
        thread.join(timeout=1)  # give it a moment to die
        _record_timeout()
        return {"status": "error", "message": f"Code execution timeout after {_SMART_QUERY_TIMEOUT}s"}
    return result_container


# ---------------------------------------------------------------------------
# Skill 6: smart_query
# ---------------------------------------------------------------------------

def smart_query(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """LLM-generated pandas query for questions no other skill can answer.

    Use this only when no other analysis skill matches the question.
    Generates safe pandas code, validates it, and executes in a sandbox.
    """
    if llm_provider is None:
        return {"status": "error", "message": "smart_query requires an LLM provider"}

    try:
        # Build prompt with column info and sample rows
        cols = list(df.columns)[:50]
        col_info = ", ".join(f"{c} ({df[c].dtype})" for c in cols)
        if len(df.columns) > 50:
            col_info += f"... and {len(df.columns) - 50} more columns"
        sample = df.head(3).to_string(index=False)
        prompt = (
            f"Dataset columns: {col_info}\n"
            f"Sample rows:\n{sample}\n\n"
            f"User question: {question}\n\n"
            "Write pandas code to answer this question. The DataFrame is available as 'df'.\n"
            "You can use 'pd' (pandas) and 'np' (numpy).\n"
            "Store the final answer in a variable called 'result'.\n"
            "Keep it under 10 lines. Do NOT import anything.\n"
            "Respond ONLY with the Python code, no explanation."
        )

        response = llm_provider.generate_plan(prompt)
        if response is None:
            return {"status": "error", "message": "LLM failed to generate code"}

        code = response.strip()
        # Strip markdown fences
        if code.startswith("```"):
            code = code.split("\n", 1)[1] if "\n" in code else code[3:]
            code = code.rsplit("```", 1)[0].strip()

        # Validate
        is_safe, reason = _validate_code(code)
        if not is_safe:
            return {"status": "error", "message": reason, "generated_code": code}

        # Auto-correct column name typos before execution
        code = _fix_column_refs(code, list(df.columns))

        # Execute
        result = _execute_safe(code, df)
        result["generated_code"] = code
        result["question_asked"] = question

        # Compute aggregate stats for narration (prevents LLM hallucination)
        raw_output = result.pop("_raw_output", None)
        if result.get("status") == "success" and raw_output is not None:
            if isinstance(raw_output, pd.DataFrame) and len(raw_output) > 0:
                result["data_summary"] = _summarize_dataframe(raw_output)
            elif isinstance(raw_output, pd.Series) and len(raw_output) > 0:
                result["data_summary"] = _summarize_dataframe(raw_output.to_frame())

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}

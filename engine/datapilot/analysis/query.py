"""
Query skills — interactive data querying via natural language.

5 fixed skills for common patterns + 1 smart_query LLM fallback.
"""

import ast
import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional

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
_SMART_QUERY_MAX_LEAKED_THREADS = 3

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


# Counter for leaked threads from timeouts
_leaked_thread_count = 0


# ---------------------------------------------------------------------------
# LLM parameter extraction helper
# ---------------------------------------------------------------------------

def _extract_params_via_llm(
    question: str,
    llm_provider,
    skill_name: str,
    param_schema: dict,
    df_columns: list,
) -> Optional[dict]:
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
            {"filter_expression": "pandas query string", "columns": "list of column names or null"},
            list(df.columns),
        )

        filter_expression = None
        columns = None

        if params:
            filter_expression = params.get("filter_expression")
            columns = params.get("columns")
        else:
            # Heuristic fallback: look for "where COL = VAL" patterns
            match = re.search(
                r"where\s+(\w+)\s*=\s*(\w+)",
                question, re.IGNORECASE,
            )
            if match:
                col, val = match.group(1), match.group(2)
                if col in df.columns:
                    filter_expression = f"{col} == '{val}'"

        filtered_df = df

        if filter_expression:
            # Validate: reject function calls in filter expressions
            try:
                filter_tree = ast.parse(filter_expression, mode="eval")
                for node in ast.walk(filter_tree):
                    if isinstance(node, ast.Call):
                        return {"status": "error", "message": "Function calls are not allowed in filter expressions"}
            except SyntaxError:
                return {"status": "error", "message": f"Invalid filter expression syntax"}

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
        return {
            "status": "success",
            "data": records,
            "total_rows": len(filtered_df),
            "query_description": filter_expression or "all rows",
        }
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
                values = match.group(2)
                index = match.group(3)
                if values in df.columns and index in df.columns:
                    params = {"values": values, "index": index, "aggfunc": aggfunc}

        if not params:
            return {"status": "error", "message": "Could not extract pivot parameters"}

        values = params.get("values")
        index = params.get("index")
        aggfunc = params.get("aggfunc", "mean")

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

        if not params:
            return {"status": "error", "message": "Could not extract column for value_counts"}

        column = params.get("column")
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

        if not params:
            return {"status": "error", "message": "Could not extract top_n parameters"}

        column = params.get("column")
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
        return {
            "status": "success",
            "data": records,
            "n": n,
            "column": column,
            "direction": direction,
        }
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

        if not params:
            return {"status": "error", "message": "Could not extract cross_tab parameters"}

        row = params.get("row")
        col = params.get("col")
        values_col = params.get("values")
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


def _execute_safe(code: str, df: pd.DataFrame) -> dict:
    """Execute validated code in a sandboxed namespace.

    Returns result dict with status, data, and code.
    Uses daemon threads with a leak counter to prevent runaway threads.
    """
    global _leaked_thread_count

    if _leaked_thread_count >= _SMART_QUERY_MAX_LEAKED_THREADS:
        return {"status": "error", "message": "smart_query unavailable: too many timed-out executions"}

    safe_df = df.copy()
    namespace = {"pd": _SafePandas(), "np": _SafeNumpy(), "df": safe_df}
    result_container = {"status": "error", "message": "Unknown error"}

    def _run():
        try:
            exec(code, {"__builtins__": {}}, namespace)
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
            elif isinstance(output, pd.Series):
                result_container.update({
                    "status": "success",
                    "data": safe_json_serialize(output.head(_SMART_QUERY_MAX_ROWS).to_dict()),
                    "total_rows": len(output),
                })
            elif isinstance(output, (int, float, str, bool, np.integer, np.floating)):
                result_container.update({"status": "success", "data": safe_json_serialize(output)})
            else:
                result_container.update({"status": "error", "message": f"Unexpected output type: {type(output).__name__}"})
        except Exception as e:
            result_container.update({"status": "error", "message": str(e)[:200]})

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=_SMART_QUERY_TIMEOUT)
    if thread.is_alive():
        _leaked_thread_count += 1
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

        # Execute
        result = _execute_safe(code, df)
        result["generated_code"] = code
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}

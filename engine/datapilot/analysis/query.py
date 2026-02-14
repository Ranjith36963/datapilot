"""
Query skills — interactive data querying via natural language.

5 fixed skills for common patterns + 1 smart_query LLM fallback.
"""

import ast
import json
import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.serializer import safe_json_serialize

logger = logging.getLogger("datapilot.analysis.query")


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
    raise NotImplementedError


def query_data(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Filter and select rows from the dataset based on natural language conditions.

    Use this when the user asks to filter, show specific rows, or count rows matching criteria.
    Examples: "Show me rows where churn = Yes", "How many customers have >3 service calls"
    """
    raise NotImplementedError


def pivot_table(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Aggregate data by groups using pivot tables.

    Use this for questions about averages, sums, or counts grouped by categories.
    Examples: "Average monthly charges by contract type", "Total revenue by state"
    """
    raise NotImplementedError


def value_counts(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Compute frequency distribution of a categorical column.

    Use this when the user asks about counts per category or frequency distributions.
    Examples: "How many customers per state?", "Distribution of contract types"
    """
    raise NotImplementedError


def top_n(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Rank and return the top or bottom N records by a column.

    Use this for ranking questions like top/bottom/highest/lowest.
    Examples: "Top 10 customers by monthly charges", "Bottom 5 states by churn rate"
    """
    raise NotImplementedError


def cross_tab(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """Cross-tabulate two categorical columns to show their relationship.

    Use this for questions about the relationship between two categorical variables.
    Examples: "Churn by contract type and internet service"
    """
    raise NotImplementedError


def smart_query(df: pd.DataFrame, question: str, llm_provider=None) -> dict:
    """LLM-generated pandas query for questions no other skill can answer.

    Use this only when no other analysis skill matches the question.
    Generates safe pandas code, validates it, and executes in a sandbox.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Smart query sandbox helpers
# ---------------------------------------------------------------------------

_SMART_QUERY_TIMEOUT = 5
_SMART_QUERY_MAX_LINES = 10

_FORBIDDEN_NAMES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "open", "eval", "exec", "compile", "__import__",
    "globals", "locals", "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit",
})


def _validate_code(code: str) -> tuple:
    """Validate LLM-generated code for safety.

    Returns (is_safe: bool, reason: str).
    """
    raise NotImplementedError


def _execute_safe(code: str, df: pd.DataFrame) -> dict:
    """Execute validated code in a sandboxed namespace.

    Returns result dict with status, data, and code.
    """
    raise NotImplementedError

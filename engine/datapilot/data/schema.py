"""
Schema inference — detect column types with semantic understanding.

Goes beyond pandas dtypes to identify: numeric_continuous, numeric_discrete,
categorical_nominal, categorical_ordinal, datetime, text, boolean, id,
email, phone, url.
"""

import re
from typing import Any

import pandas as pd

from ..utils.helpers import load_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result

logger = setup_logging("datapilot.schema")

# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
_PHONE_RE = re.compile(r"^[\+]?[\d\s\-().]{7,20}$")
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)

_ORDINAL_KEYWORDS = {
    "low", "medium", "high",
    "small", "large",
    "poor", "fair", "good", "excellent",
    "bad", "average",
    "none", "mild", "moderate", "severe",
}


def _match_pattern(values: pd.Series, regex: re.Pattern, min_match: float = 0.8) -> bool:
    """Return True if >= min_match fraction of non-null string values match regex."""
    strs = values.dropna().astype(str)
    if len(strs) == 0:
        return False
    matches = strs.apply(lambda v: bool(regex.match(v))).sum()
    return matches / len(strs) >= min_match


def _is_ordinal(values: pd.Series) -> bool:
    """Heuristic: column is ordinal if its lowered unique values overlap with known ordinal keywords."""
    uniq = {str(v).lower().strip() for v in values.dropna().unique()}
    return len(uniq & _ORDINAL_KEYWORDS) >= 2


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _infer_semantic_type(series: pd.Series, col_name: str) -> tuple:
    """
    Return (semantic_type, confidence, patterns_found, suggested_transformations).
    """
    clean = series.dropna()
    n_total = len(clean)
    n_unique = clean.nunique()
    patterns: list[str] = []

    if n_total == 0:
        return "unknown", 0.0, [], ["Drop column — entirely null"]

    # Boolean
    low = {str(v).lower() for v in clean.unique()}
    if low <= {"yes", "no", "true", "false", "1", "0", "1.0", "0.0"} and n_unique <= 2:
        return "boolean", 0.99, ["boolean_values"], ["Map to 0/1"]

    # ID-like
    if n_unique / n_total > 0.95 and n_unique > 20:
        return "id", 0.85, ["all_unique"], ["Exclude from modelling"]

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", 0.99, ["datetime_dtype"], []

    # Try parse datetime from object
    if series.dtype == object:
        sample = clean.head(30)
        try:
            pd.to_datetime(sample)
            patterns.append("parseable_datetime")
            return "datetime", 0.80, patterns, ["Parse with pd.to_datetime"]
        except Exception:
            pass

    # String-pattern types
    if series.dtype == object:
        if _match_pattern(clean, _EMAIL_RE):
            return "email", 0.95, ["email_format"], []
        if _match_pattern(clean, _PHONE_RE):
            return "phone", 0.85, ["phone_format"], []
        if _match_pattern(clean, _URL_RE):
            return "url", 0.90, ["url_format"], []

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series) and n_unique <= 20:
            return "numeric_discrete", 0.90, [], []
        if pd.api.types.is_float_dtype(series) or n_unique > 20:
            return "numeric_continuous", 0.90, [], []
        return "numeric_continuous", 0.80, [], []

    # Categorical
    if n_total > 0 and n_unique / n_total < 0.05:
        if _is_ordinal(clean):
            return "categorical_ordinal", 0.80, ["ordinal_keywords"], ["Ordinal-encode"]
        return "categorical_nominal", 0.85, [], ["One-hot encode"]

    if n_unique <= 50:
        if _is_ordinal(clean):
            return "categorical_ordinal", 0.75, ["ordinal_keywords"], ["Ordinal-encode"]
        return "categorical_nominal", 0.75, [], ["One-hot encode"]

    # Fallback: free text
    return "text", 0.60, [], ["Tokenize / embed for NLP"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_schema(file_path: str) -> dict:
    """
    Smart schema detection — goes beyond pandas dtypes.

    Returns per-column: name, pandas_dtype, semantic_type, confidence,
    nullable, patterns_found, suggested_transformations.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Inferring schema for {file_path}: {df.shape}")

        columns: list[dict[str, Any]] = []
        for col in df.columns:
            sem_type, conf, patterns, transforms = _infer_semantic_type(df[col], col)
            columns.append({
                "name": col,
                "pandas_dtype": str(df[col].dtype),
                "semantic_type": sem_type,
                "confidence": round(conf, 2),
                "nullable": bool(df[col].isna().any()),
                "patterns_found": patterns,
                "suggested_transformations": transforms,
            })

        return safe_json_serialize({"status": "success", "columns": columns})

    except Exception as e:
        return {"status": "error", "message": str(e)}


def detect_datetime_columns(df: pd.DataFrame) -> list:
    """
    Find columns that could be parsed as datetime.

    Accepts a DataFrame (not file_path) for composability.
    """
    candidates: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().head(30)
            try:
                pd.to_datetime(sample)
                candidates.append(col)
            except Exception:
                pass
    return candidates


def detect_categorical_threshold(df: pd.DataFrame, column: str, threshold: int = 50) -> bool:
    """Return True if *column* should be treated as categorical (unique count <= threshold)."""
    if column not in df.columns:
        return False
    return df[column].nunique() <= threshold


def infer_and_upload(file_path: str, output_name: str = "schema.json") -> dict:
    """Convenience function: infer_schema + upload."""
    result = infer_schema(file_path)
    upload_result(result, output_name)
    return result

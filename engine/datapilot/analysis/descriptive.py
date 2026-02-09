"""
Descriptive statistics — basic statistics for any column.

Provides numeric summaries (mean, median, skew, kurtosis, normality),
categorical summaries (mode, entropy, value counts), and group comparisons.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..utils.helpers import load_data, setup_logging, get_numeric_columns, get_categorical_columns
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.descriptive")


# ---------------------------------------------------------------------------
# Per-column helpers
# ---------------------------------------------------------------------------

def describe_numeric(df: pd.DataFrame, column: str) -> dict:
    """Detailed stats for one numeric column."""
    clean = df[column].dropna()
    if clean.dtype == bool:
        clean = clean.astype(int)
    if len(clean) == 0:
        return {"column": column, "count": 0}

    # Normality test (sample if large)
    sample = clean.sample(min(len(clean), 5000), random_state=42) if len(clean) > 5000 else clean
    try:
        _, norm_p = sp_stats.shapiro(sample) if len(sample) <= 5000 else (None, None)
    except Exception:
        norm_p = None

    return {
        "column": column,
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "q25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "q75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
        "skewness": float(clean.skew()),
        "kurtosis": float(clean.kurtosis()),
        "is_normal": bool(norm_p > 0.05) if norm_p is not None else None,
        "normality_pvalue": float(norm_p) if norm_p is not None else None,
    }


def describe_categorical(df: pd.DataFrame, column: str) -> dict:
    """Detailed stats for one categorical column."""
    clean = df[column].dropna()
    n = len(clean)
    if n == 0:
        return {"column": column, "count": 0}

    vc = clean.value_counts()
    mode_val = vc.index[0] if len(vc) else None
    mode_freq = int(vc.iloc[0]) if len(vc) else 0

    # Shannon entropy
    probs = vc.values / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    top_values = [
        {"value": str(v), "count": int(c), "pct": round(c / n * 100, 2)}
        for v, c in vc.head(20).items()
    ]

    return {
        "column": column,
        "count": n,
        "unique": int(clean.nunique()),
        "mode": str(mode_val) if mode_val is not None else None,
        "mode_freq": mode_freq,
        "mode_pct": round(mode_freq / n * 100, 2) if n else 0.0,
        "entropy": round(entropy, 4),
        "value_counts": top_values,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def describe_data(file_path: str, columns: Optional[List[str]] = None) -> dict:
    """
    Comprehensive descriptive statistics for all (or selected) columns.

    Returns numeric_summary and categorical_summary.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Describing {file_path}: {df.shape}")

        target_cols = columns or df.columns.tolist()

        numeric_summary = []
        categorical_summary = []

        for col in target_cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_summary.append(describe_numeric(df, col))
            else:
                categorical_summary.append(describe_categorical(df, col))

        result = {
            "status": "success",
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def compare_groups(file_path: str, group_column: str, value_column: str) -> dict:
    """
    Compare statistics of *value_column* across groups defined by *group_column*.

    Adapted from src/analysis/analyzer.py _compare_churned_retained.
    """
    try:
        df = load_data(file_path)
        if group_column not in df.columns or value_column not in df.columns:
            return {"status": "error", "message": f"Column not found: {group_column} or {value_column}"}

        groups = df.groupby(group_column)[value_column]
        comparison: List[Dict[str, Any]] = []

        overall_mean = float(df[value_column].mean()) if pd.api.types.is_numeric_dtype(df[value_column]) else None

        for name, grp in groups:
            clean = grp.dropna()
            entry: Dict[str, Any] = {"group": str(name), "count": int(len(clean))}
            if pd.api.types.is_numeric_dtype(clean):
                entry.update({
                    "mean": float(clean.mean()),
                    "median": float(clean.median()),
                    "std": float(clean.std()),
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                })
                if overall_mean and overall_mean != 0:
                    entry["diff_from_overall_pct"] = round(
                        (clean.mean() - overall_mean) / abs(overall_mean) * 100, 2
                    )
            else:
                entry["mode"] = str(clean.mode().iloc[0]) if len(clean.mode()) else None

            comparison.append(entry)

        comparison.sort(key=lambda x: x.get("mean", 0), reverse=True)

        result = {
            "status": "success",
            "group_column": group_column,
            "value_column": value_column,
            "n_groups": len(comparison),
            "overall_mean": overall_mean,
            "groups": comparison,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def describe_and_upload(file_path: str, output_name: str = "descriptive_stats.json", **kwargs) -> dict:
    """Convenience function: describe_data + upload."""
    result = describe_data(file_path, **kwargs)
    upload_result(result, output_name)
    return result

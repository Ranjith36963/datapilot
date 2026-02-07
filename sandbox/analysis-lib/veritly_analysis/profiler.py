"""
Dataset profiler — the first skill GPT-4 calls on ANY new dataset.

Provides a complete overview: shape, types, nulls, distributions,
correlations, quality score, warnings, and recommendations.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from veritly_analysis.utils import (
    load_data,
    safe_json_serialize,
    setup_logging,
    get_numeric_columns,
    get_categorical_columns,
    upload_result,
)


logger = setup_logging("profiler")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _semantic_type(series: pd.Series, col_name: str) -> str:
    """Infer a human-friendly semantic type for a column."""
    unique = series.dropna().unique()
    n_unique = len(unique)
    n_total = len(series.dropna())

    # Boolean-like
    low = {str(v).lower() for v in unique}
    if low <= {"yes", "no", "true", "false", "1", "0", "1.0", "0.0"} and 1 <= n_unique <= 2:
        return "boolean"

    # ID-like: every value unique and either int or string
    if n_total > 0 and n_unique / n_total > 0.95 and n_unique > 20:
        return "id"

    # Datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 2:
            return "boolean"
        if n_unique <= 20 or (n_total > 0 and n_unique / n_total < 0.01):
            return "categorical"
        return "numeric"

    # Object/string
    if n_total > 0 and n_unique / n_total < 0.05:
        return "categorical"

    # Try datetime parsing (sample)
    sample = series.dropna().head(20)
    try:
        pd.to_datetime(sample, infer_datetime_format=True)
        return "datetime"
    except Exception:
        pass

    if series.dtype == object and n_unique > 50:
        return "text"

    return "categorical"


def _column_profile(series: pd.Series, col_name: str) -> Dict[str, Any]:
    """Build a full profile dict for a single column."""
    n = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = round(missing_count / n * 100, 2) if n > 0 else 0.0
    unique_count = int(series.nunique())
    unique_pct = round(unique_count / n * 100, 2) if n > 0 else 0.0
    sem_type = _semantic_type(series, col_name)

    profile: Dict[str, Any] = {
        "name": col_name,
        "dtype": str(series.dtype),
        "semantic_type": sem_type,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "unique_count": unique_count,
        "unique_pct": unique_pct,
        "sample_values": series.dropna().head(5).tolist(),
    }

    clean = series.dropna()

    if sem_type == "numeric" and pd.api.types.is_numeric_dtype(series):
        profile.update({
            "mean": float(clean.mean()) if len(clean) else None,
            "std": float(clean.std()) if len(clean) else None,
            "min": float(clean.min()) if len(clean) else None,
            "max": float(clean.max()) if len(clean) else None,
            "median": float(clean.median()) if len(clean) else None,
            "q25": float(clean.quantile(0.25)) if len(clean) else None,
            "q75": float(clean.quantile(0.75)) if len(clean) else None,
            "skewness": float(clean.skew()) if len(clean) > 2 else None,
            "kurtosis": float(clean.kurtosis()) if len(clean) > 3 else None,
        })

    if sem_type in ("categorical", "boolean"):
        vc = clean.value_counts().head(10)
        profile["top_values"] = [
            {"value": str(v), "count": int(c), "pct": round(c / n * 100, 2)}
            for v, c in vc.items()
        ]
        if len(vc):
            profile["mode"] = str(vc.index[0])

    return profile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_data(file_path: str) -> dict:
    """
    Complete dataset profile.  GPT-4 calls this FIRST on any data.

    Returns a dict with keys: status, overview, columns, quality_score,
    warnings, recommendations, correlations.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Profiling {file_path}: {df.shape[0]} rows x {df.shape[1]} cols")

        # --- overview ---
        dup_rows = int(df.duplicated().sum())
        overview = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "duplicate_rows": dup_rows,
            "duplicate_pct": round(dup_rows / len(df) * 100, 2) if len(df) else 0.0,
        }

        # --- per-column profiles ---
        columns = [_column_profile(df[col], col) for col in df.columns]

        # --- quality score ---
        total_cells = len(df) * len(df.columns) if len(df.columns) else 1
        null_cells = int(df.isna().sum().sum())
        completeness = (1 - null_cells / total_cells) * 100 if total_cells else 100
        dup_penalty = min(overview["duplicate_pct"], 20)
        quality_score = round(max(completeness - dup_penalty, 0), 1)

        # --- warnings ---
        warnings: List[Dict[str, str]] = []
        recommendations: List[str] = []

        for cp in columns:
            name = cp["name"]
            if cp["missing_pct"] > 50:
                warnings.append({"type": "high_missing", "column": name,
                                 "detail": f"{cp['missing_pct']}% missing"})
                recommendations.append(f"Consider removing column '{name}' ({cp['missing_pct']}% missing)")
            if cp["semantic_type"] == "id":
                warnings.append({"type": "potential_id", "column": name,
                                 "detail": "Appears to be a unique identifier"})
            if cp["unique_count"] == 1:
                warnings.append({"type": "constant_column", "column": name,
                                 "detail": "Only one unique value"})
                recommendations.append(f"Remove constant column '{name}'")
            if cp["semantic_type"] == "categorical" and cp["unique_count"] > 100:
                warnings.append({"type": "high_cardinality", "column": name,
                                 "detail": f"{cp['unique_count']} unique values"})

        # --- correlations ---
        num_cols = get_numeric_columns(df)
        high_corrs: List[Dict[str, Any]] = []
        if len(num_cols) >= 2:
            try:
                corr = df[num_cols].corr()
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        r = corr.iloc[i, j]
                        if abs(r) > 0.7:
                            high_corrs.append({
                                "col1": num_cols[i],
                                "col2": num_cols[j],
                                "correlation": round(float(r), 4),
                            })
                high_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            except Exception:
                pass

        result = {
            "status": "success",
            "overview": overview,
            "columns": columns,
            "quality_score": quality_score,
            "warnings": warnings,
            "recommendations": recommendations,
            "correlations": {"high_correlations": high_corrs},
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def detect_target_candidates(file_path: str) -> dict:
    """
    Suggest which columns could be prediction targets.

    Returns binary_targets, multiclass_targets, and numeric_targets.
    """
    try:
        df = load_data(file_path)

        binary: List[Dict[str, Any]] = []
        multiclass: List[Dict[str, Any]] = []
        numeric: List[Dict[str, Any]] = []

        for col in df.columns:
            clean = df[col].dropna()
            n_unique = clean.nunique()

            if n_unique == 2:
                vals = clean.unique().tolist()
                balance = round(float(clean.value_counts(normalize=True).min()), 4)
                binary.append({"column": col, "values": vals, "balance": balance})

            elif 3 <= n_unique <= 20 and not pd.api.types.is_float_dtype(clean):
                multiclass.append({"column": col, "n_classes": n_unique})

            elif pd.api.types.is_numeric_dtype(clean) and n_unique > 20:
                try:
                    _, p = sp_stats.shapiro(clean.sample(min(len(clean), 5000), random_state=42))
                    dist = "normal" if p > 0.05 else "non-normal"
                except Exception:
                    dist = "unknown"
                numeric.append({"column": col, "distribution": dist})

        result = {
            "status": "success",
            "binary_targets": binary,
            "multiclass_targets": multiclass,
            "numeric_targets": numeric,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def profile_and_upload(file_path: str, output_name: str = "profile.json") -> dict:
    """Convenience function: profile_data + upload."""
    result = profile_data(file_path)
    upload_result(result, output_name)
    return result

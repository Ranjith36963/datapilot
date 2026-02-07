"""
Outlier detection — find anomalies in data.

Methods: Isolation Forest, Local Outlier Factor, Z-score, IQR, DBSCAN.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from veritly_analysis.utils import (
    load_data,
    save_data,
    safe_json_serialize,
    setup_logging,
    get_numeric_columns,
    upload_result,
)


logger = setup_logging("outlier_detection")


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _detect_iqr(df: pd.DataFrame, columns: List[str], threshold: float = 1.5):
    """IQR-based outlier detection. Returns boolean mask and scores."""
    outlier_mask = pd.Series(False, index=df.index)
    scores = pd.Series(0.0, index=df.index)
    col_details = []

    for col in columns:
        clean = df[col].dropna()
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_mask |= mask.fillna(False)

        n_out = int(mask.sum())
        if n_out > 0:
            col_details.append({
                "column": col,
                "outlier_count": n_out,
                "outlier_values": df.loc[mask, col].head(10).tolist(),
                "threshold_low": float(lower),
                "threshold_high": float(upper),
            })
        # Score = max IQR deviation across columns
        deviation = ((df[col] - clean.median()) / iqr).abs().fillna(0)
        scores = scores.combine(deviation, max)

    return outlier_mask, scores, col_details


def _detect_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0):
    """Z-score outlier detection."""
    outlier_mask = pd.Series(False, index=df.index)
    scores = pd.Series(0.0, index=df.index)
    col_details = []

    for col in columns:
        clean = df[col].dropna()
        mean, std = clean.mean(), clean.std()
        if std == 0:
            continue
        z = ((df[col] - mean) / std).abs().fillna(0)
        mask = z > threshold
        outlier_mask |= mask

        n_out = int(mask.sum())
        if n_out > 0:
            col_details.append({
                "column": col,
                "outlier_count": n_out,
                "outlier_values": df.loc[mask, col].head(10).tolist(),
                "threshold_low": float(mean - threshold * std),
                "threshold_high": float(mean + threshold * std),
            })
        scores = scores.combine(z, max)

    return outlier_mask, scores, col_details


def _detect_isolation_forest(df: pd.DataFrame, columns: List[str], contamination: float = 0.05):
    """Isolation Forest outlier detection."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = df[columns].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X_scaled)
    raw_scores = iso.decision_function(X_scaled)

    outlier_mask = pd.Series(preds == -1, index=df.index)
    scores = pd.Series(-raw_scores, index=df.index)  # higher = more anomalous
    return outlier_mask, scores, []


def _detect_lof(df: pd.DataFrame, columns: List[str], contamination: float = 0.05):
    """Local Outlier Factor detection."""
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    X = df[columns].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lof = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
    preds = lof.fit_predict(X_scaled)
    raw_scores = lof.negative_outlier_factor_

    outlier_mask = pd.Series(preds == -1, index=df.index)
    scores = pd.Series(-raw_scores, index=df.index)
    return outlier_mask, scores, []


def _detect_dbscan(df: pd.DataFrame, columns: List[str], eps: float = 0.5):
    """DBSCAN-based outlier detection (noise points = outliers)."""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    X = df[columns].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_scaled)

    outlier_mask = pd.Series(labels == -1, index=df.index)
    scores = pd.Series((labels == -1).astype(float), index=df.index)
    return outlier_mask, scores, []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_outliers(
    file_path: str,
    method: str = "isolation_forest",
    columns: Optional[List[str]] = None,
    contamination: float = 0.05,
) -> dict:
    """
    Detect outliers/anomalies in data.

    Methods: isolation_forest, lof, zscore, iqr, dbscan.
    """
    try:
        df = load_data(file_path)
        cols = columns or get_numeric_columns(df)
        if not cols:
            return {"status": "error", "message": "No numeric columns found"}

        logger.info(f"Detecting outliers via {method} on {len(cols)} columns")

        if method == "iqr":
            mask, scores, col_details = _detect_iqr(df, cols)
        elif method == "zscore":
            mask, scores, col_details = _detect_zscore(df, cols)
        elif method == "lof":
            mask, scores, col_details = _detect_lof(df, cols, contamination)
        elif method == "dbscan":
            mask, scores, col_details = _detect_dbscan(df, cols)
        else:
            mask, scores, col_details = _detect_isolation_forest(df, cols, contamination)

        n_outliers = int(mask.sum())
        recommendations = []
        if n_outliers > len(df) * 0.10:
            recommendations.append("High outlier rate — consider adjusting contamination or reviewing data quality")
        if n_outliers == 0:
            recommendations.append("No outliers detected — data appears clean for the chosen method")

        result = {
            "status": "success",
            "method": method,
            "total_rows": len(df),
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers / len(df) * 100, 2) if len(df) else 0.0,
            "outlier_indices": df.index[mask].tolist()[:500],
            "outlier_scores": scores.tolist(),
            "column_analysis": col_details,
            "recommendations": recommendations,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def flag_anomalies(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "isolation_forest",
) -> pd.DataFrame:
    """Return df with 'is_outlier' and 'outlier_score' columns added."""
    cols = columns or get_numeric_columns(df)
    if method == "iqr":
        mask, scores, _ = _detect_iqr(df, cols)
    elif method == "zscore":
        mask, scores, _ = _detect_zscore(df, cols)
    elif method == "lof":
        mask, scores, _ = _detect_lof(df, cols)
    else:
        mask, scores, _ = _detect_isolation_forest(df, cols)

    out = df.copy()
    out["is_outlier"] = mask
    out["outlier_score"] = scores
    return out


def get_outlier_summary(file_path: str) -> dict:
    """Quick summary of outliers across all numeric columns using IQR."""
    try:
        df = load_data(file_path)
        cols = get_numeric_columns(df)
        _, _, col_details = _detect_iqr(df, cols)
        return safe_json_serialize({
            "status": "success",
            "method": "iqr",
            "columns_with_outliers": len(col_details),
            "details": col_details,
        })
    except Exception as e:
        return {"status": "error", "message": str(e)}


def detect_and_upload(file_path: str, output_name: str = "outliers.json", **kwargs) -> dict:
    """Convenience function: detect_outliers + upload."""
    result = detect_outliers(file_path, **kwargs)
    upload_result(result, output_name)
    return result

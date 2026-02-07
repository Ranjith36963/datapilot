"""
Correlation analysis — find relationships between variables.

Supports Pearson, Spearman, Kendall, auto-selection, partial correlation,
and multicollinearity warnings.
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
    upload_result,
)


logger = setup_logging("correlation")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Full correlation matrix for numeric columns."""
    num_cols = get_numeric_columns(df)
    return df[num_cols].corr(method=method)


def find_top_correlations(df: pd.DataFrame, target: str, n: int = 10) -> list:
    """Top N features correlated with target."""
    num_cols = get_numeric_columns(df)
    if target not in num_cols:
        return []
    corrs = []
    for col in num_cols:
        if col == target:
            continue
        clean = df[[col, target]].dropna()
        if len(clean) < 5:
            continue
        r, p = sp_stats.pearsonr(clean[col], clean[target])
        corrs.append({"column": col, "correlation": round(float(r), 4), "pvalue": round(float(p), 6)})
    corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return corrs[:n]


def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    controlling: List[str],
) -> dict:
    """Correlation between x and y, controlling for other variables."""
    try:
        import pingouin as pg
        cols = [x, y] + controlling
        clean = df[cols].dropna()
        result = pg.partial_corr(data=clean, x=x, y=y, covar=controlling)
        return {
            "x": x,
            "y": y,
            "controlling": controlling,
            "partial_r": round(float(result["r"].values[0]), 4),
            "pvalue": round(float(result["p-val"].values[0]), 6),
            "n": int(result["n"].values[0]),
        }
    except ImportError:
        # Manual partial correlation via linear regression residuals
        from numpy.linalg import lstsq
        cols = [x, y] + controlling
        clean = df[cols].dropna()
        if len(clean) < len(controlling) + 3:
            return {"x": x, "y": y, "partial_r": None, "pvalue": None,
                    "message": "Not enough data for partial correlation"}
        C = clean[controlling].values
        C_ext = np.column_stack([C, np.ones(len(C))])
        # Residualise x
        bx, _, _, _ = lstsq(C_ext, clean[x].values, rcond=None)
        res_x = clean[x].values - C_ext @ bx
        # Residualise y
        by, _, _, _ = lstsq(C_ext, clean[y].values, rcond=None)
        res_y = clean[y].values - C_ext @ by
        r, p = sp_stats.pearsonr(res_x, res_y)
        return {
            "x": x, "y": y, "controlling": controlling,
            "partial_r": round(float(r), 4),
            "pvalue": round(float(p), 6),
            "n": len(clean),
        }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def analyze_correlations(
    file_path: str,
    target: Optional[str] = None,
    method: str = "auto",
) -> dict:
    """
    Correlation analysis.

    Methods: pearson, spearman, kendall, auto (pearson for normal, spearman otherwise).
    """
    try:
        df = load_data(file_path)
        num_cols = get_numeric_columns(df)
        if len(num_cols) < 2:
            return {"status": "error", "message": "Need at least 2 numeric columns"}

        logger.info(f"Analyzing correlations for {file_path}: {len(num_cols)} numeric cols")

        # Auto-select method
        chosen = method
        if method == "auto":
            # Check normality on a sample column
            sample_col = num_cols[0]
            sample = df[sample_col].dropna().head(5000)
            try:
                _, p = sp_stats.shapiro(sample)
                chosen = "pearson" if p > 0.05 else "spearman"
            except Exception:
                chosen = "spearman"

        corr = df[num_cols].corr(method=chosen)

        # Correlation matrix as nested dict
        corr_dict = {c: {c2: round(float(corr.loc[c, c2]), 4) for c2 in num_cols} for c in num_cols}

        # Top pairwise correlations
        top_pairs: List[Dict[str, Any]] = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = float(corr.iloc[i, j])
                clean = df[[num_cols[i], num_cols[j]]].dropna()
                if len(clean) < 5:
                    continue
                if chosen == "pearson":
                    _, p = sp_stats.pearsonr(clean.iloc[:, 0], clean.iloc[:, 1])
                elif chosen == "kendall":
                    _, p = sp_stats.kendalltau(clean.iloc[:, 0], clean.iloc[:, 1])
                else:
                    _, p = sp_stats.spearmanr(clean.iloc[:, 0], clean.iloc[:, 1])
                top_pairs.append({
                    "col1": num_cols[i],
                    "col2": num_cols[j],
                    "correlation": round(r, 4),
                    "pvalue": round(float(p), 6),
                })
        top_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Target correlations
        target_corrs = []
        if target and target in num_cols:
            for pair in top_pairs:
                if pair["col1"] == target or pair["col2"] == target:
                    other = pair["col2"] if pair["col1"] == target else pair["col1"]
                    target_corrs.append({
                        "column": other,
                        "correlation": pair["correlation"],
                        "pvalue": pair["pvalue"],
                        "significant": pair["pvalue"] < 0.05,
                    })
            target_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Multicollinearity warnings (|r| > 0.8, excluding target)
        warnings = [
            {"col1": p["col1"], "col2": p["col2"], "correlation": p["correlation"]}
            for p in top_pairs
            if abs(p["correlation"]) > 0.8 and p["col1"] != target and p["col2"] != target
        ]

        result = {
            "status": "success",
            "method": chosen,
            "correlation_matrix": corr_dict,
            "top_correlations": top_pairs[:20],
            "target_correlations": target_corrs if target else [],
            "multicollinearity_warning": warnings,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def correlate_and_upload(file_path: str, output_name: str = "correlations.json", **kwargs) -> dict:
    """Convenience function: analyze_correlations + upload."""
    result = analyze_correlations(file_path, **kwargs)
    upload_result(result, output_name)
    return result

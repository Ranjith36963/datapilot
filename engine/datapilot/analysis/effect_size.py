"""
Effect size — measure practical significance (how big is the difference?).

Provides Cohen's d, Hedges' g, odds ratio, relative risk, Cramer's V,
eta-squared, r-squared with interpretation and confidence intervals.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..utils.helpers import load_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.effect_size")


# ---------------------------------------------------------------------------
# Interpreters
# ---------------------------------------------------------------------------

_D_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}
_V_THRESHOLDS = {"small": 0.1, "medium": 0.3, "large": 0.5}
_ETA_THRESHOLDS = {"small": 0.01, "medium": 0.06, "large": 0.14}


def _interpret(value: float, thresholds: dict) -> str:
    v = abs(value)
    if v < thresholds["small"]:
        return "negligible"
    if v < thresholds["medium"]:
        return "small"
    if v < thresholds["large"]:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Individual effect sizes
# ---------------------------------------------------------------------------

def cohens_d(group1, group2) -> dict:
    """
    Cohen's d standardised mean difference.

    Accepts lists, arrays, or Series.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return {"status": "error", "message": "Each group needs \u22652 observations"}

    pooled = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return {"effect_size": 0.0, "interpretation": "negligible"}

    d = float((g1.mean() - g2.mean()) / pooled)

    # CI via non-central t approximation
    se = np.sqrt(n1 + n2) / np.sqrt(n1 * n2) * np.sqrt(1 + d ** 2 / (2 * (n1 + n2)))
    ci = (round(d - 1.96 * se, 4), round(d + 1.96 * se, 4))

    return {
        "effect_type": "cohens_d",
        "effect_size": round(d, 4),
        "confidence_interval": list(ci),
        "interpretation": _interpret(d, _D_THRESHOLDS),
        "interpretation_guide": _D_THRESHOLDS,
    }


def hedges_g(group1, group2) -> dict:
    """Hedges' g -- bias-corrected Cohen's d for small samples."""
    res = cohens_d(group1, group2)
    if "status" in res and res["status"] == "error":
        return res

    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n = len(g1[~np.isnan(g1)]) + len(g2[~np.isnan(g2)])
    # Correction factor
    j = 1 - 3 / (4 * (n - 2) - 1) if n > 3 else 1.0
    g = res["effect_size"] * j
    ci = [round(v * j, 4) for v in res["confidence_interval"]]

    return {
        "effect_type": "hedges_g",
        "effect_size": round(g, 4),
        "confidence_interval": ci,
        "interpretation": _interpret(g, _D_THRESHOLDS),
        "interpretation_guide": _D_THRESHOLDS,
    }


def odds_ratio(df: pd.DataFrame, exposure_col: str, outcome_col: str) -> dict:
    """Odds ratio for binary exposure and outcome."""
    ct = pd.crosstab(df[exposure_col], df[outcome_col])
    if ct.shape != (2, 2):
        return {"status": "error", "message": "Need exactly 2x2 table for odds ratio"}

    a, b = ct.iloc[1, 1], ct.iloc[1, 0]
    c, d_ = ct.iloc[0, 1], ct.iloc[0, 0]

    if b * c == 0:
        return {"status": "error", "message": "Cannot compute OR: zero cell in table"}

    oratio = (a * d_) / (b * c)
    se_ln = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d_) if min(a, b, c, d_) > 0 else 0
    ci_lo = np.exp(np.log(oratio) - 1.96 * se_ln) if se_ln > 0 else 0
    ci_hi = np.exp(np.log(oratio) + 1.96 * se_ln) if se_ln > 0 else 0

    return {
        "effect_type": "odds_ratio",
        "effect_size": round(float(oratio), 4),
        "confidence_interval": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "interpretation": (
            "large" if oratio > 3 or oratio < 1 / 3
            else "medium" if oratio > 1.5 or oratio < 1 / 1.5
            else "small"
        ),
        "interpretation_guide": {"small": "OR near 1", "medium": "OR 1.5-3", "large": "OR > 3"},
    }


def relative_risk(df: pd.DataFrame, exposure_col: str, outcome_col: str) -> dict:
    """Relative risk for binary exposure and outcome."""
    ct = pd.crosstab(df[exposure_col], df[outcome_col])
    if ct.shape != (2, 2):
        return {"status": "error", "message": "Need exactly 2x2 table"}

    a, b = ct.iloc[1, 1], ct.iloc[1, 0]
    c, d_ = ct.iloc[0, 1], ct.iloc[0, 0]

    risk_exposed = a / (a + b) if (a + b) > 0 else 0
    risk_unexposed = c / (c + d_) if (c + d_) > 0 else 0

    if risk_unexposed == 0:
        return {"status": "error", "message": "Cannot compute RR: zero risk in unexposed group"}

    rr = risk_exposed / risk_unexposed

    return {
        "effect_type": "relative_risk",
        "effect_size": round(float(rr), 4),
        "risk_exposed": round(float(risk_exposed), 4),
        "risk_unexposed": round(float(risk_unexposed), 4),
        "interpretation": "large" if rr > 2 else ("medium" if rr > 1.5 else "small"),
    }


def cramers_v(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Cramer's V for categorical association."""
    ct = pd.crosstab(df[col1], df[col2])
    stat, pval, dof, _ = sp_stats.chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(stat / (n * k))) if n * k > 0 else 0.0

    return {
        "effect_type": "cramers_v",
        "effect_size": round(v, 4),
        "confidence_interval": None,
        "interpretation": _interpret(v, _V_THRESHOLDS),
        "interpretation_guide": _V_THRESHOLDS,
        "chi2_pvalue": round(float(pval), 6),
    }


def eta_squared(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """Eta-squared for ANOVA-style group comparison."""
    groups = [grp[value_col].dropna().values for _, grp in df.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 1]
    if len(groups) < 2:
        return {"status": "error", "message": "Need at least 2 groups"}

    grand = np.concatenate(groups)
    grand_mean = grand.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((grand - grand_mean) ** 2)
    eta = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return {
        "effect_type": "eta_squared",
        "effect_size": round(eta, 4),
        "confidence_interval": None,
        "interpretation": _interpret(eta, _ETA_THRESHOLDS),
        "interpretation_guide": _ETA_THRESHOLDS,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def calculate_effect_size(file_path: str, effect_type: str, **kwargs) -> dict:
    """
    Calculate effect sizes.

    effect_type: cohens_d, hedges_g, odds_ratio, relative_risk,
                 cramers_v, eta_squared, r_squared.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Calculating {effect_type} for {file_path}")

        if effect_type == "cohens_d":
            g1 = df.loc[df[kwargs["group_col"]] == kwargs.get("group1", df[kwargs["group_col"]].unique()[0]),
                        kwargs["value_col"]].values
            g2 = df.loc[df[kwargs["group_col"]] == kwargs.get("group2", df[kwargs["group_col"]].unique()[1]),
                        kwargs["value_col"]].values
            res = cohens_d(g1, g2)
        elif effect_type == "hedges_g":
            g1 = df.loc[df[kwargs["group_col"]] == kwargs.get("group1", df[kwargs["group_col"]].unique()[0]),
                        kwargs["value_col"]].values
            g2 = df.loc[df[kwargs["group_col"]] == kwargs.get("group2", df[kwargs["group_col"]].unique()[1]),
                        kwargs["value_col"]].values
            res = hedges_g(g1, g2)
        elif effect_type == "odds_ratio":
            res = odds_ratio(df, kwargs["exposure_col"], kwargs["outcome_col"])
        elif effect_type == "relative_risk":
            res = relative_risk(df, kwargs["exposure_col"], kwargs["outcome_col"])
        elif effect_type == "cramers_v":
            res = cramers_v(df, kwargs["col1"], kwargs["col2"])
        elif effect_type == "eta_squared":
            res = eta_squared(df, kwargs["group_col"], kwargs["value_col"])
        elif effect_type == "r_squared":
            from scipy.stats import pearsonr
            pair = df[[kwargs["col1"], kwargs["col2"]]].dropna()
            r, p = pearsonr(pair[kwargs["col1"]], pair[kwargs["col2"]])
            r2 = r ** 2
            res = {
                "effect_type": "r_squared",
                "effect_size": round(float(r2), 4),
                "confidence_interval": None,
                "interpretation": "small" if r2 < 0.09 else ("medium" if r2 < 0.25 else "large"),
                "interpretation_guide": {"small": 0.01, "medium": 0.09, "large": 0.25},
            }
        else:
            return {"status": "error", "message": f"Unknown effect_type: {effect_type}"}

        res["status"] = res.get("status", "success")
        return safe_json_serialize(res)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def effect_and_upload(file_path: str, output_name: str = "effect_size.json", **kwargs) -> dict:
    """Convenience function: calculate_effect_size + upload."""
    result = calculate_effect_size(file_path, **kwargs)
    upload_result(result, output_name)
    return result

"""
Hypothesis testing — test statistical significance.

Tests: t-test, paired t-test, ANOVA, chi-square, Mann-Whitney, Kruskal,
normality, Levene.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..utils.helpers import load_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.hypothesis")


# ---------------------------------------------------------------------------
# Effect-size helpers (light versions for interpretation)
# ---------------------------------------------------------------------------

def _cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


def _interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def t_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group1: Optional[str] = None,
    group2: Optional[str] = None,
) -> dict:
    """Independent samples t-test."""
    groups = df[group_col].dropna().unique()
    if group1 is None or group2 is None:
        if len(groups) != 2:
            return {"status": "error", "message": f"Expected 2 groups, found {len(groups)}. Specify group1/group2."}
        group1, group2 = groups[0], groups[1]

    g1 = df.loc[df[group_col] == group1, value_col].dropna().values
    g2 = df.loc[df[group_col] == group2, value_col].dropna().values
    if len(g1) < 2 or len(g2) < 2:
        return {"status": "error", "message": "Each group needs at least 2 observations"}

    stat, pval = sp_stats.ttest_ind(g1, g2, equal_var=False)  # Welch's
    d = _cohens_d(g1, g2)

    return {
        "test": "independent_t_test",
        "statistic": float(stat),
        "pvalue": float(pval),
        "significant": pval < 0.05,
        "effect_size": round(abs(d), 4),
        "effect_interpretation": _interpret_d(d),
        "conclusion": (
            f"Significant difference between {group1} and {group2} (p={pval:.4f}, d={abs(d):.2f})"
            if pval < 0.05
            else f"No significant difference between {group1} and {group2} (p={pval:.4f})"
        ),
        "details": {
            "group1": str(group1), "group1_mean": float(g1.mean()), "group1_n": len(g1),
            "group2": str(group2), "group2_mean": float(g2.mean()), "group2_n": len(g2),
        },
    }


def paired_t_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Paired samples t-test."""
    clean = df[[col1, col2]].dropna()
    if len(clean) < 2:
        return {"status": "error", "message": "Need at least 2 paired observations"}

    stat, pval = sp_stats.ttest_rel(clean[col1], clean[col2])
    diff = clean[col1] - clean[col2]
    d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0

    return {
        "test": "paired_t_test",
        "statistic": float(stat),
        "pvalue": float(pval),
        "significant": pval < 0.05,
        "effect_size": round(abs(d), 4),
        "effect_interpretation": _interpret_d(d),
        "conclusion": (
            f"Significant difference between {col1} and {col2} (p={pval:.4f})"
            if pval < 0.05
            else f"No significant difference between {col1} and {col2} (p={pval:.4f})"
        ),
        "details": {"mean_diff": float(diff.mean()), "std_diff": float(diff.std()), "n": len(clean)},
    }


def anova(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """One-way ANOVA with post-hoc info."""
    groups = [grp[value_col].dropna().values for _, grp in df.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"status": "error", "message": "Need at least 2 groups with 2+ observations each"}

    stat, pval = sp_stats.f_oneway(*groups)

    # Eta-squared
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return {
        "test": "one_way_anova",
        "statistic": float(stat),
        "pvalue": float(pval),
        "significant": pval < 0.05,
        "effect_size": round(eta_sq, 4),
        "effect_interpretation": "small" if eta_sq < 0.06 else ("medium" if eta_sq < 0.14 else "large"),
        "conclusion": (
            f"Significant difference across groups (p={pval:.4f}, \u03b7\u00b2={eta_sq:.3f})"
            if pval < 0.05
            else f"No significant difference across groups (p={pval:.4f})"
        ),
        "details": {"n_groups": len(groups), "group_sizes": [len(g) for g in groups]},
    }


def chi_square(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Chi-square test of independence."""
    ct = pd.crosstab(df[col1], df[col2])
    stat, pval, dof, expected = sp_stats.chi2_contingency(ct)

    # Cramer's V
    n = ct.sum().sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(stat / (n * k))) if n * k > 0 else 0.0

    return {
        "test": "chi_square",
        "statistic": float(stat),
        "pvalue": float(pval),
        "significant": pval < 0.05,
        "effect_size": round(v, 4),
        "effect_interpretation": "small" if v < 0.1 else ("medium" if v < 0.3 else "large"),
        "conclusion": (
            f"Significant association between {col1} and {col2} (p={pval:.4f}, V={v:.3f})"
            if pval < 0.05
            else f"No significant association between {col1} and {col2} (p={pval:.4f})"
        ),
        "details": {"dof": int(dof), "contingency_table_shape": list(ct.shape)},
    }


def mann_whitney(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """Non-parametric alternative to t-test."""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return {"status": "error", "message": f"Expected 2 groups, found {len(groups)}"}

    g1 = df.loc[df[group_col] == groups[0], value_col].dropna().values
    g2 = df.loc[df[group_col] == groups[1], value_col].dropna().values
    if len(g1) < 1 or len(g2) < 1:
        return {"status": "error", "message": "Each group needs at least 1 observation"}

    stat, pval = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    # Rank-biserial r
    n1, n2 = len(g1), len(g2)
    r = 1 - (2 * stat) / (n1 * n2)

    return {
        "test": "mann_whitney",
        "statistic": float(stat),
        "pvalue": float(pval),
        "significant": pval < 0.05,
        "effect_size": round(abs(float(r)), 4),
        "effect_interpretation": "small" if abs(r) < 0.1 else ("medium" if abs(r) < 0.3 else "large"),
        "conclusion": (
            f"Significant difference (p={pval:.4f}, r={abs(r):.3f})"
            if pval < 0.05
            else f"No significant difference (p={pval:.4f})"
        ),
        "details": {
            "group1": str(groups[0]), "group1_median": float(np.median(g1)),
            "group2": str(groups[1]), "group2_median": float(np.median(g2)),
        },
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_hypothesis_test(file_path: str, test_type: str, **kwargs) -> dict:
    """
    Run a statistical hypothesis test.

    test_type: t_test, t_test_paired, anova, chi_square, mann_whitney,
               normality, levene, kruskal.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Running {test_type} on {file_path}")

        if test_type == "t_test":
            res = t_test(df, kwargs["group_col"], kwargs["value_col"],
                         kwargs.get("group1"), kwargs.get("group2"))
        elif test_type == "t_test_paired":
            res = paired_t_test(df, kwargs["col1"], kwargs["col2"])
        elif test_type == "anova":
            res = anova(df, kwargs["group_col"], kwargs["value_col"])
        elif test_type == "chi_square":
            res = chi_square(df, kwargs["col1"], kwargs["col2"])
        elif test_type == "mann_whitney":
            res = mann_whitney(df, kwargs["group_col"], kwargs["value_col"])
        elif test_type == "normality":
            col = kwargs["column"]
            sample = df[col].dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=42)
            stat, pval = sp_stats.shapiro(sample)
            res = {
                "test": "shapiro_wilk",
                "statistic": float(stat),
                "pvalue": float(pval),
                "significant": pval < 0.05,
                "effect_size": None,
                "effect_interpretation": None,
                "conclusion": (
                    f"Data is NOT normally distributed (p={pval:.4f})"
                    if pval < 0.05
                    else f"Data appears normally distributed (p={pval:.4f})"
                ),
                "details": {"n": len(sample)},
            }
        elif test_type == "levene":
            groups = [grp[kwargs["value_col"]].dropna().values
                      for _, grp in df.groupby(kwargs["group_col"])]
            stat, pval = sp_stats.levene(*groups)
            res = {
                "test": "levene",
                "statistic": float(stat),
                "pvalue": float(pval),
                "significant": pval < 0.05,
                "effect_size": None,
                "effect_interpretation": None,
                "conclusion": (
                    f"Variances are NOT equal (p={pval:.4f})"
                    if pval < 0.05
                    else f"Variances are approximately equal (p={pval:.4f})"
                ),
                "details": {"n_groups": len(groups)},
            }
        elif test_type == "kruskal":
            groups = [grp[kwargs["value_col"]].dropna().values
                      for _, grp in df.groupby(kwargs["group_col"])]
            groups = [g for g in groups if len(g) >= 1]
            stat, pval = sp_stats.kruskal(*groups)
            res = {
                "test": "kruskal_wallis",
                "statistic": float(stat),
                "pvalue": float(pval),
                "significant": pval < 0.05,
                "effect_size": None,
                "effect_interpretation": None,
                "conclusion": (
                    f"Significant difference across groups (p={pval:.4f})"
                    if pval < 0.05
                    else f"No significant difference across groups (p={pval:.4f})"
                ),
                "details": {"n_groups": len(groups)},
            }
        else:
            return {"status": "error", "message": f"Unknown test_type: {test_type}"}

        res["status"] = res.get("status", "success")
        return safe_json_serialize(res)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def test_and_upload(file_path: str, output_name: str = "hypothesis_test.json", **kwargs) -> dict:
    """Convenience function: run_hypothesis_test + upload."""
    result = run_hypothesis_test(file_path, **kwargs)
    upload_result(result, output_name)
    return result

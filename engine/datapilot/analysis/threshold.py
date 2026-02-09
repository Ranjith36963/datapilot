"""
Threshold finder — find tipping points where target rate changes significantly.

Adapted from src/analysis/threshold_finder.py ThresholdFinder class.
Methods: decision_tree, brute_force, optbinning, change_point.
Includes bootstrap confidence intervals.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..utils.helpers import load_data, setup_logging, get_numeric_columns
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.threshold")


# ---------------------------------------------------------------------------
# Core methods
# ---------------------------------------------------------------------------

def _brute_force_threshold(
    df: pd.DataFrame,
    feature: str,
    target: str,
    min_samples: int = 50,
    min_impact: float = 0.05,
) -> Optional[Dict]:
    """
    Brute-force search over unique values for the best split.

    Adapted from src/analysis/threshold_finder.py ThresholdFinder.find_optimal_threshold.
    """
    unique_values = sorted(df[feature].dropna().unique())
    if len(unique_values) < 2:
        return None

    best = None
    best_impact = 0

    for value in unique_values:
        below = df[df[feature] < value]
        above = df[df[feature] >= value]

        if len(below) < min_samples or len(above) < min_samples:
            continue

        below_rate = float(below[target].mean())
        above_rate = float(above[target].mean())
        impact = above_rate - below_rate

        if impact > best_impact and impact >= min_impact:
            best_impact = impact
            multiplier = above_rate / below_rate if below_rate > 0.001 else float("inf")
            best = {
                "feature": feature,
                "threshold": float(value),
                "direction": "above",
                "rate_below": round(below_rate, 4),
                "rate_above": round(above_rate, 4),
                "lift": round(min(float(multiplier), 100), 2),
                "samples_below": len(below),
                "samples_above": len(above),
            }

    return best


def _decision_tree_threshold(df: pd.DataFrame, feature: str, target: str) -> Optional[Dict]:
    """Use a depth-1 decision tree to find the single best split."""
    from sklearn.tree import DecisionTreeClassifier

    clean = df[[feature, target]].dropna()
    if len(clean) < 10:
        return None

    X = clean[[feature]].values
    y = clean[target].values

    tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    tree.fit(X, y)

    if tree.tree_.feature[0] < 0:
        return None  # no split found

    threshold = float(tree.tree_.threshold[0])
    below = clean[clean[feature] <= threshold]
    above = clean[clean[feature] > threshold]

    if len(below) == 0 or len(above) == 0:
        return None

    below_rate = float(below[target].mean())
    above_rate = float(above[target].mean())
    lift = above_rate / below_rate if below_rate > 0.001 else float("inf")

    return {
        "feature": feature,
        "threshold": round(threshold, 4),
        "direction": "above",
        "rate_below": round(below_rate, 4),
        "rate_above": round(above_rate, 4),
        "lift": round(min(float(lift), 100), 2),
        "samples_below": len(below),
        "samples_above": len(above),
    }


def _optbinning_threshold(df: pd.DataFrame, feature: str, target: str) -> Optional[Dict]:
    """Use optbinning for optimal binning threshold."""
    try:
        from optbinning import OptimalBinning

        clean = df[[feature, target]].dropna()
        x = clean[feature].values
        y = clean[target].values.astype(int)

        ob = OptimalBinning(name=feature, dtype="numerical", solver="cp")
        ob.fit(x, y)

        splits = ob.splits
        if len(splits) == 0:
            return None

        # Use the split with highest rate difference
        best = None
        best_lift = 0
        for sp in splits:
            below = clean[clean[feature] <= sp]
            above = clean[clean[feature] > sp]
            if len(below) < 10 or len(above) < 10:
                continue
            br = float(below[target].mean())
            ar = float(above[target].mean())
            lift = ar / br if br > 0.001 else 0
            if lift > best_lift:
                best_lift = lift
                best = {
                    "feature": feature,
                    "threshold": round(float(sp), 4),
                    "direction": "above",
                    "rate_below": round(br, 4),
                    "rate_above": round(ar, 4),
                    "lift": round(min(float(lift), 100), 2),
                    "samples_below": len(below),
                    "samples_above": len(above),
                }
        return best

    except ImportError:
        return None


def _change_point_threshold(df: pd.DataFrame, feature: str, target: str) -> Optional[Dict]:
    """Statistical change-point detection for threshold."""
    try:
        import ruptures as rpt

        clean = df[[feature, target]].dropna().sort_values(feature)
        signal = clean[target].values.astype(float)

        algo = rpt.Pelt(model="rbf").fit(signal)
        bkps = algo.predict(pen=10)

        if len(bkps) <= 1:
            return None

        # Best change point
        best = None
        best_diff = 0
        for bp in bkps[:-1]:
            before = signal[:bp]
            after = signal[bp:]
            if len(before) < 10 or len(after) < 10:
                continue
            diff = abs(after.mean() - before.mean())
            if diff > best_diff:
                best_diff = diff
                threshold = float(clean[feature].iloc[bp])
                br = float(before.mean())
                ar = float(after.mean())
                lift = ar / br if br > 0.001 else 0
                best = {
                    "feature": feature,
                    "threshold": round(threshold, 4),
                    "direction": "above",
                    "rate_below": round(br, 4),
                    "rate_above": round(ar, 4),
                    "lift": round(min(float(lift), 100), 2),
                    "samples_below": len(before),
                    "samples_above": len(after),
                }
        return best

    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_optimal_split(
    df: pd.DataFrame,
    feature: str,
    target: str,
    method: str = "decision_tree",
) -> Optional[Dict]:
    """Find single best threshold for one feature."""
    if method == "decision_tree":
        return _decision_tree_threshold(df, feature, target)
    elif method == "brute_force":
        return _brute_force_threshold(df, feature, target)
    elif method == "optbinning":
        result = _optbinning_threshold(df, feature, target)
        return result if result else _decision_tree_threshold(df, feature, target)
    elif method == "change_point":
        result = _change_point_threshold(df, feature, target)
        return result if result else _decision_tree_threshold(df, feature, target)
    else:
        return _decision_tree_threshold(df, feature, target)


def find_thresholds(
    file_path: str,
    target: str,
    features: Optional[List[str]] = None,
    method: str = "decision_tree",
) -> dict:
    """
    Find optimal thresholds where target rate changes significantly.

    Adapted from src/analysis/threshold_finder.py find_all_tipping_points.
    """
    try:
        df = load_data(file_path)
        if target not in df.columns:
            return {"status": "error", "message": f"Target '{target}' not found"}

        cols = features or get_numeric_columns(df)
        cols = [c for c in cols if c != target and c in df.columns]

        logger.info(f"Finding thresholds for {len(cols)} features via {method}")

        thresholds: List[Dict[str, Any]] = []
        for col in cols:
            result = find_optimal_split(df, col, target, method)
            if result:
                # Add p-value via chi-square test
                below = df[df[col] < result["threshold"]]
                above = df[df[col] >= result["threshold"]]
                try:
                    ct = pd.crosstab(df[col] >= result["threshold"], df[target])
                    _, pval, _, _ = sp_stats.chi2_contingency(ct)
                    result["pvalue"] = round(float(pval), 6)
                except Exception:
                    result["pvalue"] = None

                # Human-readable insight
                feature_name = col.replace("_", " ").title()
                result["insight"] = (
                    f"Customers with {feature_name} >= {result['threshold']} "
                    f"have {result['rate_above']*100:.1f}% rate vs "
                    f"{result['rate_below']*100:.1f}% below "
                    f"({result['lift']:.1f}x higher)"
                )
                thresholds.append(result)

        thresholds.sort(key=lambda x: x.get("lift", 0), reverse=True)

        best = thresholds[0] if thresholds else None

        return safe_json_serialize({
            "status": "success",
            "target": target,
            "method": method,
            "thresholds": thresholds,
            "best_threshold": best,
        })

    except Exception as e:
        return {"status": "error", "message": str(e)}


def threshold_confidence_interval(
    df: pd.DataFrame,
    feature: str,
    target: str,
    threshold: float,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Bootstrap confidence interval for a given threshold's lift.
    """
    try:
        lifts = []
        n = len(df)
        for _ in range(n_bootstrap):
            sample = df.sample(n, replace=True)
            below = sample[sample[feature] < threshold]
            above = sample[sample[feature] >= threshold]
            if len(below) < 5 or len(above) < 5:
                continue
            br = below[target].mean()
            ar = above[target].mean()
            lift = ar / br if br > 0.001 else 0
            lifts.append(float(min(lift, 100)))

        if not lifts:
            return {"status": "error", "message": "Not enough valid bootstrap samples"}

        ci_lower = round(float(np.percentile(lifts, 2.5)), 4)
        ci_upper = round(float(np.percentile(lifts, 97.5)), 4)
        mean_lift = round(float(np.mean(lifts)), 4)

        return safe_json_serialize({
            "status": "success",
            "feature": feature,
            "threshold": threshold,
            "mean_lift": mean_lift,
            "confidence_interval": [ci_lower, ci_upper],
            "n_bootstrap": n_bootstrap,
        })

    except Exception as e:
        return {"status": "error", "message": str(e)}


def thresholds_and_upload(file_path: str, output_name: str = "thresholds.json", **kwargs) -> dict:
    """Convenience function: find_thresholds + upload."""
    result = find_thresholds(file_path, **kwargs)
    upload_result(result, output_name)
    return result

"""
Feature selection — find the most important features for prediction.

Methods: tree-based importance, RFE, mutual information, chi-squared,
f_classif, lasso, SHAP.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from veritly_analysis.utils import (
    load_data,
    safe_json_serialize,
    setup_logging,
    get_numeric_columns,
    upload_result,
)


logger = setup_logging("feature_selection")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_Xy(df: pd.DataFrame, target: str, features: Optional[List[str]] = None):
    """Prepare X, y: encode categoricals, fill nulls."""
    cols = features if features else [c for c in df.columns if c != target]
    cols = [c for c in cols if c in df.columns and c != target]
    X = df[cols].copy()
    y = df[target].copy()

    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("__MISSING__").astype(str))
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)

    return X, y


# ---------------------------------------------------------------------------
# Individual methods
# ---------------------------------------------------------------------------

def rfe_selection(df: pd.DataFrame, target: str, n_features: int = 10) -> list:
    """Recursive Feature Elimination using Random Forest."""
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier

    X, y = _prepare_Xy(df, target)
    n_features = min(n_features, X.shape[1])

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe = RFE(model, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)

    ranking = []
    for i, col in enumerate(X.columns):
        ranking.append({
            "feature": col,
            "rank": int(rfe.ranking_[i]),
            "selected": bool(rfe.support_[i]),
        })
    ranking.sort(key=lambda x: x["rank"])
    return ranking


def shap_importance(df: pd.DataFrame, target: str) -> list:
    """SHAP-based feature importance."""
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier

        X, y = _prepare_Xy(df, target)
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        sample = X.sample(min(len(X), 500), random_state=42)
        shap_values = explainer.shap_values(sample)

        # For binary: shap_values is list of 2 arrays; use class 1
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[1]).mean(axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)

        ranking = [{"feature": col, "mean_shap": round(float(v), 6)}
                    for col, v in zip(X.columns, vals)]
        ranking.sort(key=lambda x: x["mean_shap"], reverse=True)
        return ranking

    except ImportError:
        # Fallback to tree importance
        from sklearn.ensemble import RandomForestClassifier
        X, y = _prepare_Xy(df, target)
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        ranking = [{"feature": col, "mean_shap": round(float(v), 6)}
                    for col, v in zip(X.columns, model.feature_importances_)]
        ranking.sort(key=lambda x: x["mean_shap"], reverse=True)
        return ranking


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def select_features(
    file_path: str,
    target: str,
    method: str = "auto",
    n_features: Optional[int] = None,
) -> dict:
    """
    Feature selection.

    Methods: auto, rfe, mutual_info, chi2, f_classif, tree_importance, lasso, shap.
    """
    try:
        df = load_data(file_path)
        if target not in df.columns:
            return {"status": "error", "message": f"Target '{target}' not found"}

        df_clean = df.dropna(subset=[target])
        X, y = _prepare_Xy(df_clean, target)
        n_feat = n_features or max(5, X.shape[1] // 2)
        n_feat = min(n_feat, X.shape[1])

        logger.info(f"Selecting features for {target} via {method}: {X.shape[1]} candidates")

        if method in ("auto", "tree_importance"):
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            scores = model.feature_importances_

        elif method == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif
            scores = mutual_info_classif(X, y, random_state=42)

        elif method == "chi2":
            from sklearn.feature_selection import chi2
            X_pos = X - X.min()  # chi2 needs non-negative
            scores, _ = chi2(X_pos, y)

        elif method == "f_classif":
            from sklearn.feature_selection import f_classif
            scores, _ = f_classif(X, y)

        elif method == "lasso":
            from sklearn.linear_model import LogisticRegression
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = LogisticRegression(penalty="l1", solver="saga", max_iter=1000, random_state=42, C=0.1)
            model.fit(Xs, y)
            scores = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)

        elif method == "rfe":
            ranking = rfe_selection(df_clean, target, n_feat)
            selected = [r["feature"] for r in ranking if r["selected"]]
            eliminated = [r["feature"] for r in ranking if not r["selected"]]
            return safe_json_serialize({
                "status": "success",
                "method": "rfe",
                "target": target,
                "original_features": X.shape[1],
                "selected_features": len(selected),
                "feature_ranking": ranking,
                "recommended_features": selected,
                "eliminated_features": eliminated,
                "selection_rationale": f"RFE selected top {len(selected)} features",
            })

        elif method == "shap":
            ranking = shap_importance(df_clean, target)
            selected = [r["feature"] for r in ranking[:n_feat]]
            eliminated = [r["feature"] for r in ranking[n_feat:]]
            for i, r in enumerate(ranking):
                r["rank"] = i + 1
                r["score"] = r.pop("mean_shap")
                r["selected"] = r["feature"] in selected
            return safe_json_serialize({
                "status": "success",
                "method": "shap",
                "target": target,
                "original_features": X.shape[1],
                "selected_features": len(selected),
                "feature_ranking": ranking,
                "recommended_features": selected,
                "eliminated_features": eliminated,
                "selection_rationale": f"SHAP selected top {len(selected)} features",
            })
        else:
            return {"status": "error", "message": f"Unknown method: {method}"}

        # Build ranking from scores
        ranking = []
        for i, col in enumerate(X.columns):
            ranking.append({"feature": col, "rank": 0, "score": round(float(scores[i]), 6), "selected": False})
        ranking.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(ranking):
            r["rank"] = i + 1
            r["selected"] = i < n_feat

        selected = [r["feature"] for r in ranking if r["selected"]]
        eliminated = [r["feature"] for r in ranking if not r["selected"]]

        result = {
            "status": "success",
            "method": method if method != "auto" else "tree_importance",
            "target": target,
            "original_features": X.shape[1],
            "selected_features": len(selected),
            "feature_ranking": ranking,
            "recommended_features": selected,
            "eliminated_features": eliminated,
            "selection_rationale": f"Selected top {len(selected)} features by {method} score",
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def select_and_upload(file_path: str, output_name: str = "feature_selection.json", **kwargs) -> dict:
    """Convenience function: select_features + upload."""
    result = select_features(file_path, **kwargs)
    upload_result(result, output_name)
    return result

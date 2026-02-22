"""
Explainability — explain WHY the model made a prediction.

Uses SHAP for global and local explanations. Falls back to tree-based
feature importance if SHAP is unavailable.
"""


import joblib
import numpy as np
import pandas as pd

from ..utils.helpers import load_data, setup_logging
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result

logger = setup_logging("datapilot.explain")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    """Load model artifact saved by classifier/regressor."""
    data = joblib.load(model_path)
    return data["model"], data.get("scaler"), data["features"], data.get("target_le"), data.get("label_encoders", {})


def _prepare_X(df: pd.DataFrame, features: list[str], scaler, label_encoders=None):
    """Prepare feature matrix matching training pipeline."""
    from sklearn.preprocessing import LabelEncoder
    if label_encoders is None:
        label_encoders = {}
    X = df[features].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].fillna("__MISSING__").astype(str)
        if col in label_encoders:
            le = label_encoders[col]
            known_classes = set(le.classes_)
            X[col] = X[col].map(lambda x, kc=known_classes: x if x in kc else "__UNKNOWN__")
            if "__UNKNOWN__" not in known_classes:
                le.classes_ = np.append(le.classes_, "__UNKNOWN__")
            X[col] = le.transform(X[col])
        else:
            X[col] = LabelEncoder().fit_transform(X[col])
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())
    if scaler:
        X = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)
    return X


def _get_shap_explainer(shap, model, background_data):
    """Pick the right SHAP explainer for the model type."""
    tree_types = {
        "RandomForestClassifier", "RandomForestRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "CatBoostClassifier", "CatBoostRegressor",
    }
    linear_types = {
        "LogisticRegression", "LinearRegression",
        "Ridge", "Lasso", "ElasticNet",
    }
    model_name = type(model).__name__
    if model_name in tree_types:
        return shap.TreeExplainer(model)
    elif model_name in linear_types:
        return shap.LinearExplainer(model, background_data)
    else:
        # KernelExplainer fallback (SVM, NaiveBayes, etc.)
        bg = shap.sample(background_data, min(50, len(background_data)))
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        return shap.KernelExplainer(predict_fn, bg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_model(model_path: str, file_path: str) -> dict:
    """
    Global model explanation using SHAP.

    Returns feature importance, direction of impact, and interaction effects.
    """
    try:
        model, scaler, features, target_le, label_encoders = _load_model(model_path)
        df = load_data(file_path)
        X = _prepare_X(df, features, scaler, label_encoders)
        sample = X.sample(min(len(X), 500), random_state=42)

        logger.info(f"Explaining model globally: {len(sample)} samples, {len(features)} features")

        try:
            import shap
            explainer = _get_shap_explainer(shap, model, sample)
            shap_values = explainer.shap_values(sample)

            if isinstance(shap_values, list):
                vals = shap_values[1]  # class 1 for binary
            else:
                vals = shap_values

            mean_abs = np.abs(vals).mean(axis=0)
            mean_signed = vals.mean(axis=0)

            feat_imp = []
            for i, f in enumerate(features):
                feat_imp.append({
                    "feature": f,
                    "mean_shap": round(float(mean_abs[i]), 6),
                    "direction": "increases" if mean_signed[i] > 0 else "decreases",
                })
            feat_imp.sort(key=lambda x: x["mean_shap"], reverse=True)

            # Interaction effects (top pairs by correlation of SHAP values)
            interactions = []
            if vals.shape[1] >= 2:
                shap_df = pd.DataFrame(vals, columns=features)
                corr = shap_df.corr().abs()
                for i in range(len(features)):
                    for j in range(i + 1, len(features)):
                        r = corr.iloc[i, j]
                        if r > 0.3:
                            interactions.append({
                                "feature1": features[i],
                                "feature2": features[j],
                                "interaction_strength": round(float(r), 4),
                            })
                interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)

            result = {
                "status": "success",
                "explanation_type": "global",
                "feature_importance": feat_imp,
                "interaction_effects": interactions[:10],
            }
            return safe_json_serialize(result)

        except ImportError:
            # Fallback: tree importance
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                return {"status": "error", "message": "Model has no feature_importances_ or coef_"}

            feat_imp = [{"feature": f, "mean_shap": round(float(imp[i]), 6), "direction": "unknown"}
                        for i, f in enumerate(features)]
            feat_imp.sort(key=lambda x: x["mean_shap"], reverse=True)

            return safe_json_serialize({
                "status": "success",
                "explanation_type": "global",
                "note": "SHAP not available, using model feature_importances_",
                "feature_importance": feat_imp,
                "interaction_effects": [],
            })

    except Exception as e:
        return {"status": "error", "message": str(e)}


def explain_prediction(model_path: str, file_path: str, row_index: int = 0) -> dict:
    """
    Explain a single prediction using SHAP local explanation.

    Returns per-feature contributions, top positive/negative factors.
    """
    try:
        model, scaler, features, target_le, label_encoders = _load_model(model_path)
        df = load_data(file_path)
        X = _prepare_X(df, features, scaler, label_encoders)

        if row_index >= len(X):
            return {"status": "error", "message": f"row_index {row_index} out of range (max {len(X) - 1})"}

        row = X.iloc[[row_index]]
        pred = model.predict(row)[0]
        prob = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None

        if target_le:
            pred_label = target_le.inverse_transform([pred])[0]
        else:
            pred_label = pred

        logger.info(f"Explaining prediction for row {row_index}: predicted={pred_label}")

        try:
            import shap
            explainer = _get_shap_explainer(shap, model, X)
            shap_values = explainer.shap_values(row)

            if isinstance(shap_values, list):
                vals = shap_values[1][0]  # class 1
                base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                vals = shap_values[0]
                base = explainer.expected_value

            contributions = []
            for i, f in enumerate(features):
                contributions.append({
                    "feature": f,
                    "value": safe_json_serialize(row.iloc[0, i]),
                    "contribution": round(float(vals[i]), 6),
                    "direction": "positive" if vals[i] > 0 else "negative",
                })

            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            top_pos = [c for c in contributions if c["direction"] == "positive"][:5]
            top_neg = [c for c in contributions if c["direction"] == "negative"][:5]

            result = {
                "status": "success",
                "explanation_type": "local",
                "row_index": row_index,
                "prediction": safe_json_serialize(pred_label),
                "probability": round(float(prob.max()), 4) if prob is not None else None,
                "base_value": round(float(base), 6),
                "contributions": contributions,
                "top_positive_factors": top_pos,
                "top_negative_factors": top_neg,
            }
            return safe_json_serialize(result)

        except ImportError:
            return safe_json_serialize({
                "status": "success",
                "explanation_type": "local",
                "row_index": row_index,
                "prediction": safe_json_serialize(pred_label),
                "probability": round(float(prob.max()), 4) if prob is not None else None,
                "note": "SHAP not available — install shap for per-feature contributions",
                "contributions": [],
            })

    except Exception as e:
        return {"status": "error", "message": str(e)}


def explain_and_upload(model_path: str, file_path: str, output_name: str = "explanation.json", **kwargs) -> dict:
    """Convenience function: explain_model + upload."""
    result = explain_model(model_path, file_path, **kwargs)
    upload_result(result, output_name)
    return result

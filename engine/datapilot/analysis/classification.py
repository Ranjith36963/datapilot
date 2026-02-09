"""
Classifier — predict categorical outcomes (churn, fraud, buy, risk level).

Supports: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost,
Decision Tree, Naive Bayes, SVM. Auto mode tries multiple and picks best.
Optuna for hyperparameter tuning.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
)

from ..utils.helpers import load_data, setup_logging, get_numeric_columns, get_categorical_columns
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.classification")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _get_model(algorithm: str, random_state: int = 42):
    """Return an unfitted model instance."""
    if algorithm == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif algorithm == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=random_state)
    elif algorithm == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    elif algorithm == "svm":
        from sklearn.svm import SVC
        return SVC(probability=True, random_state=random_state)
    elif algorithm == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric="logloss", random_state=random_state, n_jobs=-1, verbosity=0,
        )
    elif algorithm == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1)
    elif algorithm == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=100, random_state=random_state, verbose=0)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _prepare_data(df: pd.DataFrame, target: str, features: Optional[List[str]] = None):
    """Prepare X, y — impute, encode categoricals, scale numerics."""
    if features:
        cols = [c for c in features if c in df.columns and c != target]
    else:
        cols = [c for c in df.columns if c != target]

    X = df[cols].copy()
    y = df[target].copy()

    # Drop columns with all nulls
    X = X.dropna(axis=1, how="all")

    # Encode object columns
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = X[col].fillna("__MISSING__")
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Fill remaining numeric nulls with median
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # Encode target if non-numeric
    target_le = None
    if not pd.api.types.is_numeric_dtype(y):
        target_le = LabelEncoder()
        y = pd.Series(target_le.fit_transform(y.astype(str)), index=y.index)

    return X, y, X.columns.tolist(), target_le, label_encoders


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify(
    file_path: str,
    target: str,
    features: Optional[List[str]] = None,
    algorithm: str = "auto",
    tune: bool = False,
    cv_folds: int = 5,
    model_params: Optional[Dict] = None,
) -> dict:
    """
    Universal classifier for ANY binary/multiclass target.

    algorithm: auto, logistic, random_forest, xgboost, lightgbm, catboost,
               decision_tree, naive_bayes, svm.
    """
    try:
        df = load_data(file_path)
        if target not in df.columns:
            return {"status": "error", "message": f"Target '{target}' not found"}

        df_clean = df.dropna(subset=[target])
        X, y, used_features, target_le, label_encoders = _prepare_data(df_clean, target, features)
        n_classes = y.nunique()
        class_labels = sorted(y.unique().tolist())

        logger.info(f"Classifying {target}: {len(X)} rows, {len(used_features)} features, {n_classes} classes")

        if algorithm == "auto":
            return auto_classify(file_path, target, features)

        if tune:
            return tune_classifier(file_path, target, algorithm)

        t0 = time.time()
        model = _get_model(algorithm)
        if model_params:
            model.set_params(**model_params)

        # Scale for algorithms that benefit
        scaler = None
        if algorithm in ("logistic", "svm", "naive_bayes"):
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        else:
            X_scaled = X

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Probabilities
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # Metrics
        avg = "binary" if n_classes == 2 else "weighted"
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        }
        if y_prob is not None:
            try:
                if n_classes == 2:
                    metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                metrics["auc_roc"] = None
            try:
                metrics["log_loss"] = float(log_loss(y_test, y_prob))
            except Exception:
                metrics["log_loss"] = None

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

        # Confusion matrix & classification report
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Feature importance
        feat_imp = []
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for i, f in enumerate(used_features):
                feat_imp.append({"feature": f, "importance": round(float(imp[i]), 6)})
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            for i, f in enumerate(used_features):
                feat_imp.append({"feature": f, "importance": round(float(imp[i]), 6)})
        feat_imp.sort(key=lambda x: x["importance"], reverse=True)

        # Save model
        from pathlib import Path
        model_dir = Path(file_path).parent
        model_path = str(model_dir / f"model_{algorithm}.pkl")
        joblib.dump({"model": model, "scaler": scaler, "features": used_features,
                      "target_le": target_le, "algorithm": algorithm,
                      "label_encoders": label_encoders}, model_path)

        elapsed = time.time() - t0

        result = {
            "status": "success",
            "algorithm": algorithm,
            "target": target,
            "features_used": used_features,
            "n_classes": n_classes,
            "class_labels": [int(c) if isinstance(c, (np.integer,)) else c for c in class_labels],
            "metrics": metrics,
            "cross_validation": {
                "cv_folds": cv_folds,
                "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
                "cv_accuracy_std": round(float(cv_scores.std()), 4),
                "cv_scores": cv_scores.tolist(),
            },
            "confusion_matrix": cm,
            "classification_report": cr,
            "feature_importance": feat_imp[:20],
            "predictions": {
                "sample": y_pred[:10].tolist(),
                "probabilities_sample": y_prob[:10].tolist() if y_prob is not None else [],
            },
            "model_path": model_path,
            "hyperparameters": model.get_params(),
            "training_time_seconds": round(elapsed, 2),
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def auto_classify(file_path: str, target: str, features: Optional[List[str]] = None) -> dict:
    """Try multiple algorithms, return comparison and best."""
    try:
        df = load_data(file_path)
        df_clean = df.dropna(subset=[target])
        X, y, used_features, target_le, label_encoders = _prepare_data(df_clean, target, features)

        candidates = ["logistic", "random_forest", "decision_tree"]
        # Try boosting libs if available
        for lib_name, algo in [("xgboost", "xgboost"), ("lightgbm", "lightgbm")]:
            try:
                __import__(lib_name)
                candidates.append(algo)
            except ImportError:
                pass

        results = []
        best_score = -1
        best_algo = None

        for algo in candidates:
            try:
                model = _get_model(algo)
                scaler = None
                if algo in ("logistic", "svm"):
                    scaler = StandardScaler()
                    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                else:
                    Xs = X
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, Xs, y, cv=cv, scoring="accuracy")
                mean_acc = float(scores.mean())
                results.append({
                    "algorithm": algo,
                    "cv_accuracy_mean": round(mean_acc, 4),
                    "cv_accuracy_std": round(float(scores.std()), 4),
                })
                if mean_acc > best_score:
                    best_score = mean_acc
                    best_algo = algo
            except Exception as exc:
                results.append({"algorithm": algo, "error": str(exc)})

        # Train best model fully
        if best_algo:
            best_result = classify(file_path, target, features, algorithm=best_algo)
            best_result["auto_comparison"] = results
            return best_result

        return {"status": "error", "message": "All algorithms failed", "details": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def tune_classifier(
    file_path: str,
    target: str,
    algorithm: str = "random_forest",
    n_trials: int = 50,
) -> dict:
    """Hyperparameter tuning with Optuna."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        df = load_data(file_path)
        df_clean = df.dropna(subset=[target])
        X, y, used_features, target_le, label_encoders = _prepare_data(df_clean, target)

        def objective(trial):
            if algorithm == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif algorithm == "xgboost":
                from xgboost import XGBClassifier
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                }
                model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss",
                                      random_state=42, n_jobs=-1, verbosity=0)
            elif algorithm == "lightgbm":
                from lightgbm import LGBMClassifier
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                }
                model = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
            else:
                from sklearn.ensemble import RandomForestClassifier
                params = {"n_estimators": trial.suggest_int("n_estimators", 50, 300)}
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Train with best params
        best_params = study.best_params
        logger.info(f"Best params: {best_params}")

        # Return full classify result with best model using tuned params
        result = classify(file_path, target, algorithm=algorithm, model_params=best_params)
        result["tuning"] = {
            "n_trials": n_trials,
            "best_params": best_params,
            "best_cv_accuracy": round(study.best_value, 4),
        }
        return result

    except ImportError:
        return {"status": "error", "message": "Optuna not installed. Run: pip install optuna"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def predict(model_path: str, file_path: str) -> dict:
    """Load saved model and predict on new data."""
    try:
        data = joblib.load(model_path)
        model = data["model"]
        scaler = data.get("scaler")
        features = data["features"]
        target_le = data.get("target_le")

        df = load_data(file_path)
        X = df[features].copy()
        label_encoders = data.get("label_encoders", {})

        # Handle missing / encoding same as training
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
            X = pd.DataFrame(scaler.transform(X), columns=features)

        preds = model.predict(X)
        probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        if target_le:
            preds = target_le.inverse_transform(preds)

        result = {
            "status": "success",
            "predictions": preds.tolist(),
            "probabilities": probs.tolist() if probs is not None else [],
            "n_samples": len(X),
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def classify_and_upload(file_path: str, output_name: str = "classification.json", **kwargs) -> dict:
    """Convenience function: classify + upload."""
    result = classify(file_path, **kwargs)
    upload_result(result, output_name)
    return result

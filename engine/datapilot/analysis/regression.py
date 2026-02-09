"""
Regressor — predict numeric values (sales, price, revenue).

Supports: Linear, Ridge, Lasso, Elastic Net, Random Forest, XGBoost,
LightGBM, SVR. Auto mode tries multiple.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, explained_variance_score,
)
from scipy import stats as sp_stats

from ..utils.helpers import load_data, setup_logging, get_numeric_columns
from ..utils.serializer import safe_json_serialize
from ..utils.uploader import upload_result


logger = setup_logging("datapilot.regression")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _get_model(algorithm: str, random_state: int = 42):
    if algorithm == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif algorithm == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(random_state=random_state)
    elif algorithm == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(random_state=random_state, max_iter=5000)
    elif algorithm == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(random_state=random_state, max_iter=5000)
    elif algorithm == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif algorithm == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbosity=0)
    elif algorithm == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1)
    elif algorithm == "svr":
        from sklearn.svm import SVR
        return SVR()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _prepare_data(df: pd.DataFrame, target: str, features: Optional[List[str]] = None):
    cols = features if features else [c for c in df.columns if c != target]
    cols = [c for c in cols if c in df.columns and c != target]
    X = df[cols].copy()
    y = df[target].copy().astype(float)

    X = X.dropna(axis=1, how="all")
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].fillna("__MISSING__").astype(str))
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X, y, X.columns.tolist()


# ---------------------------------------------------------------------------
# Core regression
# ---------------------------------------------------------------------------

def predict_numeric(
    file_path: str,
    target: str,
    features: Optional[List[str]] = None,
    algorithm: str = "auto",
    tune: bool = False,
    model_params: Optional[Dict] = None,
) -> dict:
    """
    Universal regressor for ANY numeric target.

    algorithm: auto, linear, ridge, lasso, elastic_net, random_forest,
               xgboost, lightgbm, svr.
    """
    try:
        df = load_data(file_path)
        if target not in df.columns:
            return {"status": "error", "message": f"Target '{target}' not found"}

        df_clean = df.dropna(subset=[target])
        X, y, used_features = _prepare_data(df_clean, target, features)

        logger.info(f"Regressing {target}: {len(X)} rows, {len(used_features)} features")

        if algorithm == "auto":
            return auto_regress(file_path, target, features)

        if tune:
            return tune_regressor(file_path, target, algorithm)

        t0 = time.time()
        model = _get_model(algorithm)
        if model_params:
            model.set_params(**model_params)

        scaler = None
        if algorithm in ("linear", "ridge", "lasso", "elastic_net", "svr"):
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        else:
            X_scaled = X

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        ev = float(explained_variance_score(y_test, y_pred))

        # MAPE (avoid div by zero)
        nonzero = y_test[y_test != 0]
        if len(nonzero) > 0:
            mape = float(np.mean(np.abs((nonzero - y_pred[y_test != 0]) / nonzero)) * 100)
        else:
            mape = None

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")

        # Coefficients (linear models)
        coefficients = []
        if hasattr(model, "coef_"):
            coef = model.coef_ if model.coef_.ndim == 1 else model.coef_[0]
            for i, f in enumerate(used_features):
                coefficients.append({
                    "feature": f,
                    "coefficient": round(float(coef[i]), 6),
                })

        # Feature importance
        feat_imp = []
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            for i, f in enumerate(used_features):
                feat_imp.append({"feature": f, "importance": round(float(imp[i]), 6)})
        elif coefficients:
            for c in coefficients:
                feat_imp.append({"feature": c["feature"], "importance": round(abs(c["coefficient"]), 6)})
        feat_imp.sort(key=lambda x: x["importance"], reverse=True)

        # Residual analysis
        residuals = y_test.values - y_pred
        try:
            _, norm_p = sp_stats.shapiro(residuals[:5000]) if len(residuals) <= 5000 else (None, None)
        except Exception:
            norm_p = None

        # Save model
        from pathlib import Path
        model_path = str(Path(file_path).parent / f"regressor_{algorithm}.pkl")
        joblib.dump({"model": model, "scaler": scaler, "features": used_features, "algorithm": algorithm}, model_path)

        elapsed = time.time() - t0

        result = {
            "status": "success",
            "algorithm": algorithm,
            "target": target,
            "features_used": used_features,
            "metrics": {
                "r2": round(r2, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "mape": round(mape, 2) if mape is not None else None,
                "explained_variance": round(ev, 4),
            },
            "cross_validation": {
                "cv_r2_mean": round(float(cv_scores.mean()), 4),
                "cv_r2_std": round(float(cv_scores.std()), 4),
                "cv_scores": cv_scores.tolist(),
            },
            "coefficients": coefficients,
            "feature_importance": feat_imp[:20],
            "residual_analysis": {
                "mean_residual": round(float(residuals.mean()), 4),
                "std_residual": round(float(residuals.std()), 4),
                "normality_test_pvalue": round(float(norm_p), 6) if norm_p is not None else None,
            },
            "predictions_sample": y_pred[:10].tolist(),
            "model_path": model_path,
            "training_time_seconds": round(elapsed, 2),
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def auto_regress(file_path: str, target: str, features: Optional[List[str]] = None) -> dict:
    """Try all algorithms, return comparison."""
    try:
        df = load_data(file_path)
        df_clean = df.dropna(subset=[target])
        X, y, used_features = _prepare_data(df_clean, target, features)

        candidates = ["linear", "ridge", "random_forest"]
        for lib, algo in [("xgboost", "xgboost"), ("lightgbm", "lightgbm")]:
            try:
                __import__(lib)
                candidates.append(algo)
            except ImportError:
                pass

        results = []
        best_r2 = -np.inf
        best_algo = None

        for algo in candidates:
            try:
                model = _get_model(algo)
                scaler = None
                if algo in ("linear", "ridge", "lasso", "svr"):
                    scaler = StandardScaler()
                    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
                else:
                    Xs = X
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, Xs, y, cv=cv, scoring="r2")
                mean_r2 = float(scores.mean())
                results.append({
                    "algorithm": algo,
                    "cv_r2_mean": round(mean_r2, 4),
                    "cv_r2_std": round(float(scores.std()), 4),
                })
                if mean_r2 > best_r2:
                    best_r2 = mean_r2
                    best_algo = algo
            except Exception as exc:
                results.append({"algorithm": algo, "error": str(exc)})

        if best_algo:
            best_result = predict_numeric(file_path, target, features, algorithm=best_algo)
            best_result["auto_comparison"] = results
            return best_result

        return {"status": "error", "message": "All algorithms failed", "details": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def tune_regressor(
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
        X, y, _ = _prepare_data(df_clean, target)

        def objective(trial):
            if algorithm == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif algorithm == "xgboost":
                from xgboost import XGBRegressor
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                }
                model = XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=trial.suggest_int("n_estimators", 50, 300),
                                              random_state=42, n_jobs=-1)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            return cross_val_score(model, X, y, cv=cv, scoring="r2").mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        result = predict_numeric(file_path, target, algorithm=algorithm, model_params=study.best_params)
        result["tuning"] = {
            "n_trials": n_trials,
            "best_params": study.best_params,
            "best_cv_r2": round(study.best_value, 4),
        }
        return result

    except ImportError:
        return {"status": "error", "message": "Optuna not installed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def regress_and_upload(file_path: str, output_name: str = "regression.json", **kwargs) -> dict:
    """Convenience function: predict_numeric + upload."""
    result = predict_numeric(file_path, **kwargs)
    upload_result(result, output_name)
    return result

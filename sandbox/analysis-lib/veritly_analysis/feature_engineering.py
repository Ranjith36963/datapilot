"""
Feature engineering — create new features to improve ML performance.

Includes: date feature extraction, interaction terms, binning, encoding.
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
    get_categorical_columns,
    upload_result,
)


logger = setup_logging("feature_engineering")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Extract year, month, day, weekday, quarter, is_weekend, etc."""
    df = df.copy()
    dt = pd.to_datetime(df[date_column], errors="coerce")
    prefix = date_column
    df[f"{prefix}_year"] = dt.dt.year
    df[f"{prefix}_month"] = dt.dt.month
    df[f"{prefix}_day"] = dt.dt.day
    df[f"{prefix}_weekday"] = dt.dt.weekday
    df[f"{prefix}_quarter"] = dt.dt.quarter
    df[f"{prefix}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    df[f"{prefix}_day_of_year"] = dt.dt.dayofyear
    return df


def create_interaction_features(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """Create col1*col2, col1/col2, col1-col2 if both numeric."""
    df = df.copy()
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        denom = df[col2].replace(0, np.nan)
        df[f"{col1}_div_{col2}"] = df[col1] / denom
        df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
    return df


def bin_numeric(
    df: pd.DataFrame,
    column: str,
    bins: int = 5,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Bin numeric column into categories (quantile or uniform)."""
    df = df.copy()
    if strategy == "quantile":
        df[f"{column}_binned"], _ = pd.qcut(
            df[column], q=bins, duplicates="drop", retbins=True, labels=False
        )
    else:
        df[f"{column}_binned"], _ = pd.cut(
            df[column], bins=bins, retbins=True, labels=False
        )
    return df


def encode_categorical(
    df: pd.DataFrame,
    column: str,
    method: str = "onehot",
) -> pd.DataFrame:
    """
    Encode categorical column.

    Methods: onehot, label, ordinal.
    (target encoding requires a target column — use engineer_features for that.)
    """
    df = df.copy()

    if method == "label":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        mask = df[column].notna()
        df.loc[mask, column] = le.fit_transform(df.loc[mask, column].astype(str))

    elif method == "ordinal":
        from sklearn.preprocessing import OrdinalEncoder
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        vals = df[[column]].fillna("__MISSING__")
        df[column] = oe.fit_transform(vals).astype(int)

    else:  # onehot
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)

    return df


# ---------------------------------------------------------------------------
# Auto feature engineering
# ---------------------------------------------------------------------------

def engineer_features(file_path: str, operations: Optional[List[str]] = None) -> dict:
    """
    Auto-generate useful features.

    Default operations (if none specified):
    - Date features from datetime columns
    - Interaction terms for top correlated numeric pairs
    - Binning for skewed numeric columns

    Returns: status, original/new column counts, features_created list, output_path.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Engineering features for {file_path}: {df.shape}")
        original_cols = set(df.columns)
        features_created: List[Dict[str, Any]] = []

        ops = operations or ["date", "interaction", "binning"]

        # --- Date features ---
        if "date" in ops:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df = create_date_features(df, col)
                    new = set(df.columns) - original_cols
                    for nc in new:
                        features_created.append({
                            "name": nc,
                            "type": "date_part",
                            "source_columns": [col],
                            "description": f"Date component from {col}",
                        })
                    original_cols = set(df.columns)
                elif df[col].dtype == object:
                    sample = df[col].dropna().head(20)
                    try:
                        pd.to_datetime(sample, infer_datetime_format=True)
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        df = create_date_features(df, col)
                        new = set(df.columns) - original_cols
                        for nc in new:
                            features_created.append({
                                "name": nc,
                                "type": "date_part",
                                "source_columns": [col],
                                "description": f"Date component from {col}",
                            })
                        original_cols = set(df.columns)
                    except Exception:
                        pass

        # --- Interaction features (top correlated numeric pairs) ---
        if "interaction" in ops:
            num_cols = get_numeric_columns(df)
            if len(num_cols) >= 2:
                try:
                    corr = df[num_cols].corr().abs()
                    pairs = []
                    for i in range(len(num_cols)):
                        for j in range(i + 1, len(num_cols)):
                            r = corr.iloc[i, j]
                            if 0.3 < r < 0.95:
                                pairs.append((num_cols[i], num_cols[j], r))
                    pairs.sort(key=lambda x: x[2], reverse=True)
                    for c1, c2, _ in pairs[:3]:
                        df = create_interaction_features(df, c1, c2)
                        new = set(df.columns) - original_cols
                        for nc in new:
                            features_created.append({
                                "name": nc,
                                "type": "interaction",
                                "source_columns": [c1, c2],
                                "description": f"Interaction of {c1} and {c2}",
                            })
                        original_cols = set(df.columns)
                except Exception:
                    pass

        # --- Binning for skewed columns ---
        if "binning" in ops:
            num_cols = get_numeric_columns(df)
            for col in num_cols:
                try:
                    skew = df[col].skew()
                    if abs(skew) > 1.5 and df[col].nunique() > 10:
                        df = bin_numeric(df, col, bins=5, strategy="quantile")
                        new_col = f"{col}_binned"
                        if new_col in df.columns:
                            features_created.append({
                                "name": new_col,
                                "type": "binned",
                                "source_columns": [col],
                                "description": f"Quantile binning of {col} (skew={skew:.2f})",
                            })
                            original_cols = set(df.columns)
                except Exception:
                    pass

        # Save
        from pathlib import Path
        out_path = str(Path(file_path).parent / "engineered_features.csv")
        save_data(df, out_path)

        result = {
            "status": "success",
            "original_columns": len(load_data(file_path).columns),
            "new_columns": len(df.columns),
            "features_created": features_created,
            "output_path": out_path,
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def engineer_and_upload(file_path: str, output_name: str = "features.json", **kwargs) -> dict:
    """Convenience function: engineer_features + upload."""
    result = engineer_features(file_path, **kwargs)
    upload_result(result, output_name)
    return result

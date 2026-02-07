"""
Data curation — clean and prepare data for analysis.

Adapted from src/processing/curation.py: column standardization, missing-value
imputation, duplicate removal. Extended with KNN imputation and configurable
strategies.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from veritly_analysis.utils import (
    load_data,
    save_data,
    safe_json_serialize,
    setup_logging,
    upload_result,
)


logger = setup_logging("curation")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Lowercase, replace spaces/hyphens with underscores, strip special chars."""
    original = df.columns.tolist()
    new_cols = []
    for col in original:
        c = col.strip().lower()
        c = re.sub(r"[\s\-]+", "_", c)
        c = re.sub(r"[^a-z0-9_]", "", c)
        c = re.sub(r"_+", "_", c).strip("_")
        new_cols.append(c if c else f"col_{len(new_cols)}")
    df = df.copy()
    df.columns = new_cols
    renamed = {o: n for o, n in zip(original, new_cols) if o != n}
    return df, {"columns_renamed": renamed, "count": len(renamed)}


def impute_missing(df: pd.DataFrame, strategy: str = "auto") -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values.

    Strategies:
        auto   — KNN for numeric (if sklearn available), mode for categorical
        knn    — KNN imputation for all numeric columns
        median — median for numeric, mode for categorical
        mean   — mean for numeric, mode for categorical
        drop   — drop rows with any missing
    """
    df = df.copy()
    changes: Dict[str, Dict] = {}

    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        return df, {"rows_dropped": before - len(df)}

    use_knn = strategy in ("auto", "knn")
    knn_imputer = None
    if use_knn:
        try:
            from sklearn.impute import KNNImputer
            knn_imputer = KNNImputer(n_neighbors=5)
        except ImportError:
            use_knn = False

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        null_counts_before = df[num_cols].isna().sum()
        cols_with_nulls = null_counts_before[null_counts_before > 0].index.tolist()

        if cols_with_nulls:
            if use_knn and knn_imputer is not None:
                df[num_cols] = pd.DataFrame(
                    knn_imputer.fit_transform(df[num_cols]),
                    columns=num_cols,
                    index=df.index,
                )
                for col in cols_with_nulls:
                    changes[col] = {"count": int(null_counts_before[col]), "method": "knn"}
            else:
                fill = "median" if strategy in ("auto", "median") else "mean"
                for col in cols_with_nulls:
                    fill_val = df[col].median() if fill == "median" else df[col].mean()
                    df[col] = df[col].fillna(fill_val)
                    changes[col] = {"count": int(null_counts_before[col]), "method": fill}

    # Categorical / object columns — always use mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            mode_vals = df[col].mode()
            fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"
            df[col] = df[col].fillna(fill_val)
            changes[col] = {"count": n_null, "method": "mode"}

    return df, changes


# ---------------------------------------------------------------------------
# Main curation function
# ---------------------------------------------------------------------------

def curate_dataframe(
    file_path: str,
    impute_strategy: str = "auto",
    remove_duplicates: bool = True,
    standardize_columns: bool = True,
) -> dict:
    """
    Comprehensive data cleaning.

    Returns: status, input/output shape, changes log, output_path, sample rows.
    """
    try:
        df = load_data(file_path)
        logger.info(f"Curating {file_path}: {df.shape}")
        input_shape = list(df.shape)
        all_changes: Dict[str, Any] = {
            "rows_removed": 0,
            "duplicates_removed": 0,
            "nulls_imputed": {},
            "columns_standardized": [],
        }

        # 1. Standardize column names
        if standardize_columns:
            df, rename_info = standardize_column_names(df)
            all_changes["columns_standardized"] = list(rename_info.get("columns_renamed", {}).values())

        # 2. Convert yes/no to 1/0
        for col in df.select_dtypes(include=["object"]).columns:
            uniq = {str(v).lower() for v in df[col].dropna().unique()}
            if uniq <= {"yes", "no"} and len(uniq) > 0:
                df[col] = df[col].apply(
                    lambda x: 1 if str(x).lower() == "yes" else (0 if str(x).lower() == "no" else x)
                )

        # 3. Impute missing
        df, impute_changes = impute_missing(df, strategy=impute_strategy)
        all_changes["nulls_imputed"] = impute_changes

        # 4. Remove duplicates
        if remove_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            all_changes["duplicates_removed"] = before - len(df)
            all_changes["rows_removed"] += before - len(df)

        # 5. Save curated data
        from pathlib import Path
        out_dir = Path(file_path).parent
        out_path = str(out_dir / "curated_data.csv")
        save_data(df, out_path)

        result = {
            "status": "success",
            "input_shape": input_shape,
            "output_shape": list(df.shape),
            "changes": all_changes,
            "output_path": out_path,
            "curated_data_sample": df.head(5).to_dict(orient="records"),
        }
        return safe_json_serialize(result)

    except Exception as e:
        return {"status": "error", "message": str(e)}


def curate_and_upload(file_path: str, output_name: str = "curation_report.json", **kwargs) -> dict:
    """Convenience function: curate_dataframe + upload."""
    result = curate_dataframe(file_path, **kwargs)
    upload_result(result, output_name)
    return result

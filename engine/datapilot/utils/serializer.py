"""
JSON serialization utilities for DataPilot.

Converts numpy/pandas types to JSON-serializable Python types.
"""

from typing import Any

import numpy as np
import pandas as pd


def safe_json_serialize(obj: Any) -> Any:
    """
    Recursively convert numpy / pandas types to JSON-serializable Python types.

    Handles: dict, list, tuple, np.integer, np.floating, np.bool_,
    np.ndarray, pd.Series, pd.DataFrame, pd.Timestamp, np.datetime64, NaN/None.
    """
    if obj is None:
        return None

    # Check pandas NA / numpy nan first (before dict/list since pd.NA is truthy-weird)
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return None
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        pass  # pd.isna can raise on some types

    if isinstance(obj, dict):
        return {safe_json_serialize(k): safe_json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json_serialize(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, pd.Categorical):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.generic):
        return obj.item()

    return obj

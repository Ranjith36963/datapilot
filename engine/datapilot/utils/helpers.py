"""
Shared utility functions for DataPilot.

Provides file loading/saving, logging, and common DataFrame helpers.
All other modules depend on this.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str = "datapilot") -> logging.Logger:
    """Set up logging with console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_execution(func_name: str, params: Optional[Dict] = None, result: Optional[Dict] = None) -> Dict:
    """
    Create an execution log entry.

    Can be used as a simple logger factory when called with just func_name,
    or to produce a structured log dict when params/result are provided.
    """
    entry: Dict[str, Any] = {
        "function": func_name,
        "timestamp": datetime.now().isoformat(),
    }
    if params is not None:
        entry["params"] = {k: str(v)[:200] for k, v in params.items()}
    if result is not None:
        entry["status"] = result.get("status", "unknown")
    return entry


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV, Excel, JSON, or Parquet files into a DataFrame.

    Args:
        file_path: Path to the data file.

    Returns:
        pandas DataFrame.

    Raises:
        ValueError: If the file format is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p)
    elif ext == ".xlsx":
        return pd.read_excel(p, engine="openpyxl")
    elif ext == ".xls":
        return pd.read_excel(p, engine="xlrd")
    elif ext == ".json":
        return pd.read_json(p)
    elif ext == ".parquet":
        return pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def save_data(df: pd.DataFrame, file_path: str) -> str:
    """
    Save a DataFrame to CSV, Excel, JSON, or Parquet.

    Args:
        df: DataFrame to save.
        file_path: Destination path (extension determines format).

    Returns:
        The resolved file path as a string.
    """
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ext = p.suffix.lower()
    if ext == ".csv":
        df.to_csv(p, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(p, index=False, engine="openpyxl")
    elif ext == ".json":
        df.to_json(p, orient="records", indent=2)
    elif ext == ".parquet":
        df.to_parquet(p, index=False)
    else:
        # Default to CSV
        p = p.with_suffix(".csv")
        df.to_csv(p, index=False)

    return str(p)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return list of numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame, threshold: int = 50) -> List[str]:
    """
    Return list of categorical column names.

    Includes object/category dtype columns, plus numeric columns with
    fewer than *threshold* unique values.
    """
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= threshold and col not in cats:
            cats.append(col)
    return cats


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return list of datetime column names."""
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

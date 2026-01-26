"""
Schema inference for automatically detecting column types and generating schemas.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from src.config import SCHEMAS_DIR
from src.utils import setup_logging, save_json, get_timestamp


class SchemaInference:
    """Infers schema from DataFrames and generates JSON schemas."""

    def __init__(self):
        self.logger = setup_logging("schema_inference")

    def infer_and_save(self, df: pd.DataFrame, dataset_name: str) -> Path:
        """Infer schema from DataFrame and save to JSON."""
        schema = self.infer_schema(df)
        schema["dataset_name"] = dataset_name
        schema["inferred_at"] = get_timestamp()

        filename = f"{dataset_name.lower().replace(' ', '_')}_schema.json"
        schema_path = SCHEMAS_DIR / filename

        save_json(schema, schema_path)
        self.logger.info(f"Saved schema to {schema_path}")

        return schema_path

    def infer_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Infer complete schema from DataFrame."""
        columns = []
        for col in df.columns:
            col_info = self._analyze_column(df[col], col)
            columns.append(col_info)

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": columns,
            "quality_summary": self._compute_quality_summary(df, columns)
        }

    def _analyze_column(self, series: pd.Series, name: str) -> Dict[str, Any]:
        """Analyze a single column and return its schema."""
        dtype = str(series.dtype)
        inferred_type = self._infer_semantic_type(series)

        col_info = {
            "name": name,
            "pandas_dtype": dtype,
            "inferred_type": inferred_type,
            "nullable": str(series.isna().any()),
            "null_count": int(series.isna().sum()),
            "null_percentage": round(series.isna().sum() / len(series) * 100, 2),
            "unique_count": int(series.nunique()),
            "sample_values": self._get_sample_values(series)
        }

        if inferred_type == "numeric":
            col_info["statistics"] = self._compute_numeric_stats(series)
        elif inferred_type in ["categorical", "boolean"]:
            col_info["value_distribution"] = self._compute_value_distribution(series)

        return col_info

    def _infer_semantic_type(self, series: pd.Series) -> str:
        """Infer the semantic type of a column."""
        unique_vals = set(series.dropna().unique())

        # Check for boolean-like columns
        bool_patterns = [
            frozenset({True, False}),
            frozenset({"yes", "no"}),
            frozenset({"Yes", "No"}),
            frozenset({1, 0}),
        ]

        frozen_unique = frozenset(str(v).lower() for v in unique_vals)
        if frozen_unique in [frozenset({"yes", "no"}), frozenset({"true", "false"})]:
            return "boolean"
        if frozenset(unique_vals) in bool_patterns:
            return "boolean"

        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 10 and series.nunique() / len(series) < 0.01:
                return "categorical"
            return "numeric"

        if series.nunique() / len(series) < 0.05:
            return "categorical"

        return "text"

    def _compute_numeric_stats(self, series: pd.Series) -> Dict[str, float]:
        """Compute statistics for numeric columns."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}
        return {
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std()),
        }

    def _compute_value_distribution(self, series: pd.Series) -> Dict[str, int]:
        """Compute value counts for categorical columns."""
        counts = series.value_counts()
        return {str(k): int(v) for k, v in counts.items()}

    def _get_sample_values(self, series: pd.Series, n: int = 5) -> List[Any]:
        """Get sample values from a column."""
        unique = series.dropna().unique()
        samples = unique[:n].tolist()
        return [self._convert_to_native(v) for v in samples]

    def _convert_to_native(self, value: Any) -> Any:
        """Convert numpy types to native Python types."""
        if isinstance(value, (np.integer,)):
            return int(value)
        elif isinstance(value, (np.floating,)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        return value

    def _compute_quality_summary(self, df: pd.DataFrame, columns: List[Dict]) -> Dict:
        """Compute overall data quality summary."""
        total_cells = len(df) * len(df.columns)
        null_cells = df.isna().sum().sum()

        return {
            "completeness_percentage": round((1 - null_cells / total_cells) * 100, 2),
            "columns_with_nulls": sum(1 for c in columns if c["null_count"] > 0),
            "total_null_values": int(null_cells),
            "numeric_columns": sum(1 for c in columns if c["inferred_type"] == "numeric"),
            "categorical_columns": sum(1 for c in columns if c["inferred_type"] == "categorical"),
            "boolean_columns": sum(1 for c in columns if c["inferred_type"] == "boolean"),
        }

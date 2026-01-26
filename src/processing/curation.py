"""
Data curation module for cleaning and standardizing raw data.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config import CURATED_ZONE
from src.utils import setup_logging, MetadataManager


class DataCurator:
    """Cleans and standardizes raw data, saving to curated zone."""

    def __init__(self):
        self.logger = setup_logging("curator")
        self.metadata = MetadataManager()
        self.curation_log: List[Dict] = []

    def curate(
        self,
        df: pd.DataFrame,
        source_dataset_id: str,
        output_name: str = "telecom_customers",
        version: str = "v1"
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Curate a DataFrame and save to curated zone.

        Returns:
            Tuple of (curated DataFrame, output path)
        """
        self.curation_log = []
        self.logger.info(f"Starting curation of {len(df)} rows")

        curated = df.copy()

        # Step 1: Standardize column names
        curated = self._standardize_column_names(curated)

        # Step 2: Handle missing values
        curated = self._handle_missing_values(curated)

        # Step 3: Convert yes/no to binary
        curated = self._standardize_categorical_binary(curated)

        # Step 4: Standardize boolean target
        curated = self._standardize_boolean_target(curated)

        # Step 5: Validate data types
        curated = self._validate_data_types(curated)

        # Step 6: Remove duplicates
        curated = self._remove_duplicates(curated)

        # Save to curated zone
        filename = f"{output_name}_{version}.xlsx"
        output_path = CURATED_ZONE / filename
        curated.to_excel(output_path, index=False, engine="openpyxl")
        self.logger.info(f"Saved curated data to {output_path}")

        # Register in metadata
        dataset_id = f"curated_{output_name}_{version}"
        self.metadata.register_dataset(
            dataset_id=dataset_id,
            name=f"{output_name} (Curated)",
            zone="curated",
            path=str(output_path),
            row_count=len(curated),
            column_count=len(curated.columns),
            description=f"Curated telecom customer data, version {version}"
        )

        # Track lineage
        self.metadata.add_lineage(
            source_id=source_dataset_id,
            target_id=dataset_id,
            transformation="curation",
            details={"steps": self.curation_log, "input_rows": len(df), "output_rows": len(curated)}
        )

        return curated, output_path

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores."""
        original_cols = df.columns.tolist()
        df.columns = [col.lower().strip().replace(" ", "_").replace("-", "_") for col in df.columns]
        renamed = [(o, n) for o, n in zip(original_cols, df.columns) if o != n]
        if renamed:
            self._log_step("standardize_column_names", f"Renamed {len(renamed)} columns")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately by column type."""
        null_before = df.isna().sum().sum()

        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                    df[col] = df[col].fillna(fill_val)

        null_after = df.isna().sum().sum()
        if null_before > null_after:
            self._log_step("handle_missing_values", f"Filled {null_before - null_after} null values")
        return df

    def _standardize_categorical_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert yes/no categorical columns to 1/0."""
        converted = []
        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            if 'object' in dtype_str or 'str' in dtype_str:
                try:
                    unique_vals = set(str(v).lower() for v in df[col].dropna().unique())
                    if unique_vals.issubset({"yes", "no"}) and len(unique_vals) > 0:
                        df[col] = df[col].apply(
                            lambda x: 1 if str(x).lower() == "yes" else (0 if str(x).lower() == "no" else x)
                        )
                        converted.append(col)
                except Exception:
                    pass

        if converted:
            self._log_step("standardize_categorical_binary", f"Converted {len(converted)} yes/no columns to binary: {converted}")
        return df

    def _standardize_boolean_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize boolean target column."""
        if "churn" in df.columns:
            if df["churn"].dtype == bool:
                df["churn"] = df["churn"].astype(int)
                self._log_step("standardize_boolean_target", "Converted bool to int (0/1)")
            elif df["churn"].dtype == object:
                bool_map = {"True": 1, "False": 0, "true": 1, "false": 0, True: 1, False: 0}
                df["churn"] = df["churn"].map(bool_map).fillna(df["churn"])
                self._log_step("standardize_boolean_target", "Mapped string booleans to int")
        return df

    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for known columns."""
        int_columns = ["account_length", "area_code", "number_vmail_messages",
                       "total_day_calls", "total_eve_calls", "total_night_calls",
                       "total_intl_calls", "customer_service_calls", "churn"]

        for col in int_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(int)
                except (ValueError, TypeError):
                    pass

        self._log_step("validate_data_types", "Validated and corrected data types")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before > after:
            self._log_step("remove_duplicates", f"Removed {before - after} duplicate rows")
        return df

    def _log_step(self, step_name: str, description: str) -> None:
        """Log a curation step."""
        self.curation_log.append({"step": step_name, "description": description})
        self.logger.info(f"[{step_name}] {description}")

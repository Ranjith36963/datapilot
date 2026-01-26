"""
Data derivation module for creating analysis outputs and derived datasets.
Uses Pele's logistic regression model for churn prediction (86.38% accuracy).
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DERIVED_ZONE,
    LOGISTIC_COEFFICIENTS,
    STANDARDISATION_PARAMS,
    MODEL_FEATURES,
    RISK_TIERS
)
from src.utils import setup_logging, MetadataManager


class DataDeriver:
    """Creates derived datasets from curated data using Pele's logistic regression model."""

    def __init__(self):
        self.logger = setup_logging("deriver")
        self.metadata = MetadataManager()

    def _standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize features using z-score: z = (value - mean) / std."""
        standardized = pd.DataFrame(index=df.index)

        for feature in MODEL_FEATURES:
            if feature in df.columns:
                params = STANDARDISATION_PARAMS[feature]
                standardized[f"{feature}_z"] = (
                    (df[feature] - params['mean']) / params['std']
                )
            else:
                self.logger.warning(f"Feature {feature} not found in dataframe")
                standardized[f"{feature}_z"] = 0

        return standardized

    def _calculate_churn_probability(self, standardized_df: pd.DataFrame) -> np.ndarray:
        """Calculate churn probability using logistic regression.

        Formula:
        1. Linear combination: L = intercept + sum(coefficient * z)
        2. Probability: P = 1 / (1 + exp(-L))
        """
        linear_combination = np.full(len(standardized_df), LOGISTIC_COEFFICIENTS['intercept'])

        for feature in MODEL_FEATURES:
            z_col = f"{feature}_z"
            if z_col in standardized_df.columns:
                linear_combination += (
                    LOGISTIC_COEFFICIENTS[feature] * standardized_df[z_col].values
                )

        probability = 1 / (1 + np.exp(-linear_combination))
        return probability

    def _assign_risk_tier(self, probability: float) -> str:
        """Assign risk tier based on churn probability."""
        for tier, (low, high) in RISK_TIERS.items():
            if low <= probability < high:
                return tier
        return "critical"

    def create_risk_scores(
        self,
        df: pd.DataFrame,
        source_dataset_id: str,
        version: str = "v1"
    ) -> Tuple[pd.DataFrame, Path]:
        """Create customer risk scores using Pele's logistic regression model."""
        self.logger.info("Calculating churn probabilities using Pele's logistic regression model")

        risk_df = df.copy()

        # Step 1: Standardize features
        standardized = self._standardize_features(risk_df)

        # Step 2: Calculate churn probability (0 to 1)
        risk_df["churn_probability"] = self._calculate_churn_probability(standardized)

        # Step 3: Calculate risk score as percentage (0 to 100)
        risk_df["risk_score"] = (risk_df["churn_probability"] * 100).round(2)

        # Step 4: Add predicted churn (1 if probability > 0.5 else 0)
        risk_df["predicted_churn"] = (risk_df["churn_probability"] > 0.5).astype(int)

        # Step 5: Assign risk tier
        risk_df["risk_tier"] = risk_df["churn_probability"].apply(self._assign_risk_tier)

        # Calculate model accuracy for logging
        if "churn" in risk_df.columns:
            correct_predictions = (risk_df["predicted_churn"] == risk_df["churn"]).sum()
            accuracy = (correct_predictions / len(risk_df)) * 100
            self.logger.info(f"Model accuracy: {accuracy:.2f}%")

            high_risk_count = (risk_df["predicted_churn"] == 1).sum()
            self.logger.info(f"High-risk customers (predicted churn): {high_risk_count}")

        # Select output columns
        output_cols = [
            "phone_number", "state", "account_length", "customer_service_calls",
            "international_plan", "voice_mail_plan", "total_day_minutes", "total_day_charge",
            "total_eve_minutes", "total_night_minutes", "total_intl_minutes", "total_intl_calls",
            "number_vmail_messages", "churn_probability", "risk_score", "predicted_churn",
            "risk_tier", "churn"
        ]
        output_cols = [c for c in output_cols if c in risk_df.columns]
        risk_output = risk_df[output_cols]

        # Save to derived zone
        filename = f"customer_risk_scores_{version}.xlsx"
        output_path = DERIVED_ZONE / filename
        risk_output.to_excel(output_path, index=False, engine="openpyxl")
        self.logger.info(f"Saved risk scores to {output_path}")

        # Register dataset
        dataset_id = f"derived_risk_scores_{version}"
        self.metadata.register_dataset(
            dataset_id=dataset_id,
            name="Customer Risk Scores",
            zone="derived",
            path=str(output_path),
            row_count=len(risk_output),
            column_count=len(risk_output.columns),
            description="Customer churn risk scores using Pele's logistic regression model (86.38% accuracy)"
        )

        # Track lineage
        self.metadata.add_lineage(
            source_id=source_dataset_id,
            target_id=dataset_id,
            transformation="logistic_regression",
            details={
                "model": "Pele's Logistic Regression",
                "accuracy": "86.38%",
                "features": MODEL_FEATURES,
                "risk_tiers": RISK_TIERS
            }
        )

        return risk_output, output_path

    def create_churn_analysis(
        self,
        df: pd.DataFrame,
        analysis_results: Dict,
        source_dataset_id: str,
        version: str = "v1"
    ) -> Tuple[pd.DataFrame, Path]:
        """Create churn analysis summary dataset."""
        self.logger.info("Creating churn analysis derived dataset")

        summaries = []

        # Overall churn summary
        overall = pd.DataFrame([{
            "metric": "Overall Churn Rate",
            "segment": "All Customers",
            "value": analysis_results["overall_churn_rate"],
            "churned_count": analysis_results["churned_count"],
            "total_count": analysis_results["total_count"]
        }])
        summaries.append(overall)

        # Churn by segment
        for segment_name, segment_data in analysis_results.get("segment_analysis", {}).items():
            for segment_value, stats in segment_data.items():
                row = {
                    "metric": f"Churn Rate by {segment_name}",
                    "segment": str(segment_value),
                    "value": stats["churn_rate"],
                    "churned_count": stats["churned"],
                    "total_count": stats["total"]
                }
                summaries.append(pd.DataFrame([row]))

        # Tipping points
        for tp in analysis_results.get("tipping_points", []):
            row = {
                "metric": f"Tipping Point: {tp['factor']}",
                "segment": f"Threshold: {tp['threshold']}",
                "value": tp["churn_above"],
                "churned_count": None,
                "total_count": None
            }
            summaries.append(pd.DataFrame([row]))

        analysis_df = pd.concat(summaries, ignore_index=True)

        # Save to derived zone
        filename = f"churn_analysis_{version}.xlsx"
        output_path = DERIVED_ZONE / filename
        analysis_df.to_excel(output_path, index=False, engine="openpyxl")
        self.logger.info(f"Saved churn analysis to {output_path}")

        # Register dataset
        dataset_id = f"derived_churn_analysis_{version}"
        self.metadata.register_dataset(
            dataset_id=dataset_id,
            name="Churn Analysis Summary",
            zone="derived",
            path=str(output_path),
            row_count=len(analysis_df),
            column_count=len(analysis_df.columns),
            description="Aggregated churn analysis results and insights"
        )

        self.metadata.add_lineage(
            source_id=source_dataset_id,
            target_id=dataset_id,
            transformation="churn_analysis",
            details={"analysis_types": list(analysis_results.keys())}
        )

        return analysis_df, output_path

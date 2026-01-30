"""
Data Derivation - Creates derived datasets using TRAINED sklearn model.

IMPORTANT: All predictions are made using sklearn-trained model.
NO hardcoded coefficients. Everything is LEARNED from data.

The model:
- CALCULATES standardisation parameters via StandardScaler.fit()
- LEARNS coefficients via LogisticRegression.fit()
- CALCULATES predictions via model.predict_proba()
- CALCULATES metrics via sklearn.metrics
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import DERIVED_ZONE, MODELS_DIR, DEFAULT_MODEL_PATH
from src.utils import setup_logging, MetadataManager
from src.ml.trainer import ChurnModelTrainer


class DataDeriver:
    """
    Creates derived datasets using sklearn-trained model.

    CRITICAL: NO hardcoded coefficients. All values are LEARNED from data.
    """

    def __init__(self):
        self.logger = setup_logging("deriver")
        self.metadata = MetadataManager()
        self.trainer = ChurnModelTrainer()
        self.training_results: Optional[Dict] = None

    def create_risk_scores(
        self,
        df: pd.DataFrame,
        source_dataset_id: str,
        version: str = "v1",
        force_retrain: bool = True,
        model_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Create customer risk scores using sklearn-trained model.

        This method either:
        1. TRAINS a new model from scratch (force_retrain=True)
        2. LOADS an existing trained model (force_retrain=False)

        All coefficients are LEARNED from data, not hardcoded.

        Args:
            df: Curated customer DataFrame with features and 'churn' column
            source_dataset_id: ID of source dataset for lineage tracking
            version: Version string for output files
            force_retrain: If True, always train new model. If False, try to load existing.
            model_path: Path to model file. Defaults to models/churn_model.pkl

        Returns:
            Tuple of (result_DataFrame, output_path)
        """
        model_path = model_path or DEFAULT_MODEL_PATH

        # Decide whether to train or load
        if not force_retrain and model_path.exists():
            self.logger.info("=" * 60)
            self.logger.info("LOADING EXISTING TRAINED MODEL")
            self.logger.info("=" * 60)
            self.trainer.load(model_path)
            self.training_results = self.trainer.training_results
        else:
            self.logger.info("=" * 60)
            self.logger.info("TRAINING NEW MODEL FROM SCRATCH")
            self.logger.info("All coefficients will be LEARNED from data")
            self.logger.info("=" * 60)

            # TRAIN model - this CALCULATES everything from data
            self.training_results = self.trainer.train(df)

            # Save trained model for future use
            self.trainer.save(model_path)

        # Generate predictions using TRAINED model
        self.logger.info("")
        self.logger.info("Generating predictions using TRAINED model...")
        result_df = self.trainer.predict(df)

        # Select and order output columns
        output_cols = [
            "phone_number", "state", "account_length", "customer_service_calls",
            "international_plan", "voice_mail_plan", "total_day_minutes", "total_day_charge",
            "total_eve_minutes", "total_night_minutes", "total_intl_minutes", "total_intl_calls",
            "number_vmail_messages", "churn_probability", "risk_score", "predicted_churn",
            "risk_tier", "churn"
        ]
        output_cols = [c for c in output_cols if c in result_df.columns]
        result_df = result_df[output_cols]

        # Save to Excel
        filename = f"customer_risk_scores_{version}.xlsx"
        output_path = DERIVED_ZONE / filename
        result_df.to_excel(output_path, index=False, engine="openpyxl")
        self.logger.info(f"Saved risk scores to: {output_path}")

        # Register in metadata
        dataset_id = f"derived_risk_scores_{version}"
        self.metadata.register_dataset(
            dataset_id=dataset_id,
            name="Customer Risk Scores",
            zone="derived",
            path=str(output_path),
            row_count=len(result_df),
            column_count=len(result_df.columns),
            description="Customer churn risk scores using sklearn-trained logistic regression"
        )

        # Track lineage
        self.metadata.add_lineage(
            source_id=source_dataset_id,
            target_id=dataset_id,
            transformation="sklearn_logistic_regression",
            details={
                "method": "sklearn.linear_model.LogisticRegression",
                "accuracy": self.training_results.get('accuracy', 0),
                "training_type": "REAL ML - coefficients LEARNED from data"
            }
        )

        return result_df, output_path

    def create_churn_analysis(
        self,
        df: pd.DataFrame,
        analysis_results: Dict,
        source_dataset_id: str,
        version: str = "v1"
    ) -> Tuple[pd.DataFrame, Path]:
        """
        Create churn analysis summary dataset.

        Uses CALCULATED metrics from training, not hardcoded values.
        """
        self.logger.info("Creating churn analysis derived dataset")

        summaries = []

        # Overall churn summary
        overall = pd.DataFrame([{
            "metric": "Overall Churn Rate",
            "segment": "All Customers",
            "value": analysis_results.get("overall_churn_rate", 0),
            "churned_count": analysis_results.get("churned_count", 0),
            "total_count": analysis_results.get("total_count", 0)
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

        # Add model metrics if available
        if self.training_results:
            model_metrics = [
                {"metric": "Model Accuracy", "segment": "sklearn.LogisticRegression",
                 "value": round(self.training_results.get('accuracy', 0) * 100, 2),
                 "churned_count": None, "total_count": None},
                {"metric": "Model Precision", "segment": "sklearn.metrics",
                 "value": round(self.training_results.get('precision', 0) * 100, 2),
                 "churned_count": None, "total_count": None},
                {"metric": "Model Recall", "segment": "sklearn.metrics",
                 "value": round(self.training_results.get('recall', 0) * 100, 2),
                 "churned_count": None, "total_count": None},
                {"metric": "High Risk Count", "segment": "prob > 0.5",
                 "value": self.training_results.get('high_risk_count', 0),
                 "churned_count": None, "total_count": None},
            ]
            summaries.append(pd.DataFrame(model_metrics))

        analysis_df = pd.concat(summaries, ignore_index=True)

        # Save to derived zone
        filename = f"churn_analysis_{version}.xlsx"
        output_path = DERIVED_ZONE / filename
        analysis_df.to_excel(output_path, index=False, engine="openpyxl")
        self.logger.info(f"Saved churn analysis to: {output_path}")

        # Register dataset
        dataset_id = f"derived_churn_analysis_{version}"
        self.metadata.register_dataset(
            dataset_id=dataset_id,
            name="Churn Analysis Summary",
            zone="derived",
            path=str(output_path),
            row_count=len(analysis_df),
            column_count=len(analysis_df.columns),
            description="Aggregated churn analysis with sklearn model metrics"
        )

        self.metadata.add_lineage(
            source_id=source_dataset_id,
            target_id=dataset_id,
            transformation="churn_analysis",
            details={"analysis_types": list(analysis_results.keys())}
        )

        return analysis_df, output_path

    def get_training_results(self) -> Dict:
        """Return training results with all CALCULATED values."""
        if self.training_results is None:
            raise ValueError("No training results. Call create_risk_scores() first.")
        return self.training_results

    def get_model_info(self) -> Dict:
        """Return information about the trained model."""
        if not self.trainer.is_trained:
            raise ValueError("Model not trained yet.")

        return {
            'learned_coefficients': self.trainer.get_learned_coefficients(),
            'calculated_standardisation': self.trainer.get_calculated_standardisation(),
            'feature_importance': self.trainer.get_feature_importance(),
            'metrics': self.training_results
        }

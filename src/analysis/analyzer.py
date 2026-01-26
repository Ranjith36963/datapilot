"""
Churn analysis module for detecting patterns and insights in customer data.
Includes validation of Pele's logistic regression model.
"""

from typing import Any, Dict, List

import pandas as pd
import numpy as np

from src.config import (
    CUSTOMER_SERVICE_CALL_THRESHOLD,
    LOGISTIC_COEFFICIENTS,
    STANDARDISATION_PARAMS,
    MODEL_FEATURES,
    RISK_TIERS
)
from src.utils import setup_logging


class ChurnAnalyzer:
    """Analyzes customer churn patterns and identifies key insights."""

    def __init__(self):
        self.logger = setup_logging("analyzer")
        self.results: Dict[str, Any] = {}
        self.model_validation: Dict[str, Any] = {}

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run complete churn analysis on customer data."""
        self.logger.info(f"Starting churn analysis on {len(df)} customers")

        # Overall metrics
        self.results["total_count"] = len(df)
        self.results["churned_count"] = int(df["churn"].sum())
        self.results["retained_count"] = int(len(df) - df["churn"].sum())
        self.results["overall_churn_rate"] = round(df["churn"].mean() * 100, 2)

        self.logger.info(f"Overall churn rate: {self.results['overall_churn_rate']}%")

        # Segment analysis
        self.results["segment_analysis"] = self._analyze_segments(df)

        # Tipping points
        self.results["tipping_points"] = self._find_tipping_points(df)

        # Churned vs retained comparison
        self.results["comparison"] = self._compare_churned_retained(df)

        # Key insights
        self.results["insights"] = self._generate_insights(df)

        # Validate logistic regression model
        self.results["model_validation"] = self.validate_model(df)

        return self.results

    def _analyze_segments(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze churn by customer segments."""
        segments = {}

        # International plan
        if "international_plan" in df.columns:
            intl_stats = {}
            for val in df["international_plan"].unique():
                subset = df[df["international_plan"] == val]
                label = "With International Plan" if val == 1 else "Without International Plan"
                intl_stats[label] = {
                    "total": int(len(subset)),
                    "churned": int(subset["churn"].sum()),
                    "churn_rate": round(subset["churn"].mean() * 100, 2)
                }
            segments["international_plan"] = intl_stats

            with_intl = intl_stats.get("With International Plan", {})
            without_intl = intl_stats.get("Without International Plan", {})
            self.logger.info(
                f"International Plan churn: With={with_intl.get('churn_rate', 0)}%, "
                f"Without={without_intl.get('churn_rate', 0)}%"
            )

        # Voicemail plan
        if "voice_mail_plan" in df.columns:
            vm_stats = {}
            for val in df["voice_mail_plan"].unique():
                subset = df[df["voice_mail_plan"] == val]
                label = "With Voicemail Plan" if val == 1 else "Without Voicemail Plan"
                vm_stats[label] = {
                    "total": int(len(subset)),
                    "churned": int(subset["churn"].sum()),
                    "churn_rate": round(subset["churn"].mean() * 100, 2)
                }
            segments["voice_mail_plan"] = vm_stats

        # Top states by churn
        if "state" in df.columns:
            state_churn = df.groupby("state").agg({"churn": ["sum", "count", "mean"]}).reset_index()
            state_churn.columns = ["state", "churned", "total", "churn_rate"]
            state_churn["churn_rate"] = (state_churn["churn_rate"] * 100).round(2)
            state_churn = state_churn.sort_values("churn_rate", ascending=False)

            state_stats = {}
            for _, row in state_churn.head(10).iterrows():
                state_stats[row["state"]] = {
                    "total": int(row["total"]),
                    "churned": int(row["churned"]),
                    "churn_rate": float(row["churn_rate"])
                }
            segments["top_states_by_churn"] = state_stats

        return segments

    def _find_tipping_points(self, df: pd.DataFrame) -> List[Dict]:
        """Find tipping points where churn rate spikes."""
        tipping_points = []

        # Customer service calls
        if "customer_service_calls" in df.columns:
            threshold = CUSTOMER_SERVICE_CALL_THRESHOLD
            below = df[df["customer_service_calls"] < threshold]
            above = df[df["customer_service_calls"] >= threshold]

            below_churn = round(below["churn"].mean() * 100, 2)
            above_churn = round(above["churn"].mean() * 100, 2)

            self.logger.info(f"Service calls tipping point: <{threshold}={below_churn}% churn, >={threshold}={above_churn}% churn")

            tipping_points.append({
                "factor": "Customer Service Calls",
                "threshold": threshold,
                "churn_below": below_churn,
                "churn_above": above_churn,
                "impact_multiplier": round(above_churn / max(below_churn, 0.01), 2),
                "insight": f"Customers with {threshold}+ service calls have {above_churn}% churn vs {below_churn}% for others"
            })

        # High day usage
        if "total_day_minutes" in df.columns:
            q75 = df["total_day_minutes"].quantile(0.75)
            below = df[df["total_day_minutes"] <= q75]
            above = df[df["total_day_minutes"] > q75]

            below_churn = round(below["churn"].mean() * 100, 2)
            above_churn = round(above["churn"].mean() * 100, 2)

            if above_churn > below_churn:
                tipping_points.append({
                    "factor": "High Day Usage",
                    "threshold": round(q75, 1),
                    "churn_below": below_churn,
                    "churn_above": above_churn,
                    "impact_multiplier": round(above_churn / max(below_churn, 0.01), 2),
                    "insight": f"Customers with >={round(q75, 0)} day minutes have higher churn"
                })

        return tipping_points

    def _compare_churned_retained(self, df: pd.DataFrame) -> Dict:
        """Compare metrics between churned and retained customers."""
        churned = df[df["churn"] == 1]
        retained = df[df["churn"] == 0]

        comparison = {}
        numeric_cols = ["account_length", "total_day_minutes", "total_day_calls",
                        "total_eve_minutes", "total_night_minutes", "total_intl_minutes",
                        "customer_service_calls", "number_vmail_messages"]

        for col in numeric_cols:
            if col in df.columns:
                churned_mean = float(churned[col].mean())
                retained_mean = float(retained[col].mean())
                diff_pct = round((churned_mean - retained_mean) / max(retained_mean, 0.01) * 100, 2)
                comparison[col] = {
                    "churned_avg": round(churned_mean, 2),
                    "retained_avg": round(retained_mean, 2),
                    "difference_percent": diff_pct
                }

        comparison["significant_differences"] = [
            k for k, v in comparison.items() if isinstance(v, dict) and abs(v.get("difference_percent", 0)) > 10
        ]

        return comparison

    def _generate_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate actionable insights from the analysis."""
        insights = []

        # International plan risk
        if "international_plan" in df.columns:
            intl_churn = df[df["international_plan"] == 1]["churn"].mean() * 100
            non_intl_churn = df[df["international_plan"] == 0]["churn"].mean() * 100

            if intl_churn > non_intl_churn * 1.5:
                insights.append({
                    "category": "High Risk Segment",
                    "title": "International Plan Customers at Risk",
                    "finding": f"International plan customers have {round(intl_churn, 1)}% churn vs {round(non_intl_churn, 1)}% for others",
                    "recommendation": "Review international plan pricing. Consider targeted retention offers.",
                    "priority": "high"
                })

        # Service calls warning
        if "customer_service_calls" in df.columns:
            high_call_churn = df[df["customer_service_calls"] >= CUSTOMER_SERVICE_CALL_THRESHOLD]["churn"].mean() * 100
            insights.append({
                "category": "Early Warning Indicator",
                "title": "Customer Service Calls Predict Churn",
                "finding": f"Customers with {CUSTOMER_SERVICE_CALL_THRESHOLD}+ service calls have {round(high_call_churn, 1)}% churn",
                "recommendation": "Implement proactive outreach when customers reach 3 service calls.",
                "priority": "critical"
            })

        # Voicemail protection
        if "voice_mail_plan" in df.columns:
            vm_churn = df[df["voice_mail_plan"] == 1]["churn"].mean() * 100
            no_vm_churn = df[df["voice_mail_plan"] == 0]["churn"].mean() * 100

            if no_vm_churn > vm_churn:
                insights.append({
                    "category": "Retention Opportunity",
                    "title": "Voicemail Plan Reduces Churn",
                    "finding": f"Customers without voicemail have {round(no_vm_churn, 1)}% churn vs {round(vm_churn, 1)}% with",
                    "recommendation": "Promote voicemail plan adoption among customers without it.",
                    "priority": "medium"
                })

        return insights

    def validate_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate Pele's logistic regression model against actual churn data.

        Returns accuracy metrics comparing model predictions to actual outcomes.
        Expected accuracy: 86.38%
        """
        self.logger.info("Validating Pele's logistic regression model")

        # Standardize features
        standardized = pd.DataFrame(index=df.index)
        for feature in MODEL_FEATURES:
            if feature in df.columns:
                params = STANDARDISATION_PARAMS[feature]
                standardized[f"{feature}_z"] = (
                    (df[feature] - params['mean']) / params['std']
                )
            else:
                standardized[f"{feature}_z"] = 0

        # Calculate linear combination
        linear_combination = np.full(len(df), LOGISTIC_COEFFICIENTS['intercept'])
        for feature in MODEL_FEATURES:
            z_col = f"{feature}_z"
            if z_col in standardized.columns:
                linear_combination += LOGISTIC_COEFFICIENTS[feature] * standardized[z_col].values

        # Calculate probability using sigmoid
        churn_probability = 1 / (1 + np.exp(-linear_combination))

        # Make predictions (1 if probability > 0.5)
        predicted_churn = (churn_probability > 0.5).astype(int)
        actual_churn = df["churn"].values

        # Calculate metrics
        correct_predictions = (predicted_churn == actual_churn).sum()
        accuracy = (correct_predictions / len(df)) * 100

        true_positives = ((predicted_churn == 1) & (actual_churn == 1)).sum()
        true_negatives = ((predicted_churn == 0) & (actual_churn == 0)).sum()
        false_positives = ((predicted_churn == 1) & (actual_churn == 0)).sum()
        false_negatives = ((predicted_churn == 0) & (actual_churn == 1)).sum()

        precision = true_positives / max(true_positives + false_positives, 1) * 100
        recall = true_positives / max(true_positives + false_negatives, 1) * 100
        f1_score = 2 * (precision * recall) / max(precision + recall, 1)

        # Risk tier distribution
        tier_distribution = {}
        for tier, (low, high) in RISK_TIERS.items():
            count = ((churn_probability >= low) & (churn_probability < high)).sum()
            tier_distribution[tier] = int(count)

        self.model_validation = {
            "accuracy": round(accuracy, 2),
            "expected_accuracy": 86.38,
            "accuracy_match": abs(accuracy - 86.38) < 0.5,
            "total_predictions": int(len(df)),
            "correct_predictions": int(correct_predictions),
            "true_positives": int(true_positives),
            "true_negatives": int(true_negatives),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1_score, 2),
            "high_risk_count": int((predicted_churn == 1).sum()),
            "risk_tier_distribution": tier_distribution
        }

        self.logger.info(f"Model accuracy: {accuracy:.2f}% (expected: 86.38%)")
        self.logger.info(f"High-risk customers (predicted churn): {(predicted_churn == 1).sum()}")

        return self.model_validation

    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of analysis results."""
        if not self.results:
            return {"error": "No analysis results. Run analyze() first."}

        summary = {
            "total_customers": self.results["total_count"],
            "churn_rate": self.results["overall_churn_rate"],
            "churned_count": self.results["churned_count"],
            "key_tipping_points": [
                f"{tp['factor']}: {tp['churn_above']}% churn above threshold"
                for tp in self.results.get("tipping_points", [])
            ],
            "high_priority_insights": [
                i["title"] for i in self.results.get("insights", [])
                if i.get("priority") in ["high", "critical"]
            ]
        }

        if self.model_validation:
            summary["model_accuracy"] = self.model_validation.get("accuracy")
            summary["high_risk_count"] = self.model_validation.get("high_risk_count")

        return summary

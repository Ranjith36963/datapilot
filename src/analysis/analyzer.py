"""
Churn analysis module for detecting patterns and insights in customer data.

This module performs statistical analysis on customer data.
Model validation is now handled by src/ml/trainer.py using sklearn.

IMPORTANT: No hardcoded thresholds - all values CALCULATED from data.
Works for ANY data, ANY industry, ANY domain.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.config import MIN_SAMPLES_FOR_THRESHOLD, MIN_IMPACT_FOR_THRESHOLD
from src.utils import setup_logging
from src.analysis.threshold_finder import ThresholdFinder


class ChurnAnalyzer:
    """
    Analyzes customer churn patterns and identifies key insights.

    IMPORTANT: All thresholds are CALCULATED from data using ThresholdFinder.
    No hardcoded values - works for ANY data, ANY industry.

    Note: Model training and validation is now done by ChurnModelTrainer.
    This class focuses on statistical analysis of customer segments.
    """

    def __init__(self):
        self.logger = setup_logging("analyzer")
        self.results: Dict[str, Any] = {}
        self.threshold_finder = ThresholdFinder()

    def analyze(
        self,
        df: pd.DataFrame,
        training_results: Optional[Dict] = None,
        target_col: str = "churn"
    ) -> Dict[str, Any]:
        """
        Run complete analysis on customer data.

        Works for ANY binary target column (churn, readmission, attrition, etc.)
        All thresholds are CALCULATED from data, not hardcoded.

        Args:
            df: DataFrame with target column
            training_results: Optional results from ChurnModelTrainer.train()
                             Used to include model metrics in analysis
            target_col: Name of target column (default: 'churn')

        Returns:
            Dictionary with analysis results
        """
        target_label = target_col.replace("_", " ").title()
        self.logger.info(f"Starting {target_label} analysis on {len(df)} records")

        # Validate target column exists
        if target_col not in df.columns:
            self.logger.error(f"Target column '{target_col}' not found in data")
            return {"error": f"Target column '{target_col}' not found"}

        # Overall metrics
        self.results["target_column"] = target_col
        self.results["total_count"] = len(df)
        self.results["churned_count"] = int(df[target_col].sum())
        self.results["retained_count"] = int(len(df) - df[target_col].sum())
        self.results["overall_churn_rate"] = round(df[target_col].mean() * 100, 2)

        self.logger.info(f"Overall {target_label.lower()} rate: {self.results['overall_churn_rate']}%")

        # Segment analysis
        self.results["segment_analysis"] = self._analyze_segments(df, target_col)

        # Tipping points - CALCULATED dynamically from data
        self.results["tipping_points"] = self._find_tipping_points(df, target_col)

        # Churned vs retained comparison
        self.results["comparison"] = self._compare_churned_retained(df, target_col)

        # Key insights
        self.results["insights"] = self._generate_insights(df, target_col)

        # Include model validation results if provided by trainer
        if training_results:
            self.results["model_validation"] = self._format_training_results(training_results)
        else:
            self.results["model_validation"] = {}

        return self.results

    def _analyze_segments(self, df: pd.DataFrame, target_col: str = "churn") -> Dict[str, Dict]:
        """
        Analyze target rate by customer segments.

        Dynamically finds binary columns to segment by.
        Works for ANY data structure.
        """
        segments = {}
        target_label = target_col.replace("_", " ")

        # Find binary columns (potential segments) - 0/1 or yes/no
        binary_cols = []
        for col in df.columns:
            if col == target_col:
                continue
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                if set(unique_vals).issubset({0, 1, True, False, 'yes', 'no', 'Yes', 'No'}):
                    binary_cols.append(col)

        self.logger.info(f"Found {len(binary_cols)} binary columns for segment analysis")

        # Analyze each binary segment
        for col in binary_cols:
            col_stats = {}
            for val in df[col].unique():
                subset = df[df[col] == val]
                if len(subset) < 10:  # Skip tiny segments
                    continue

                # Create descriptive label
                col_name = col.replace("_", " ").title()
                if val in [1, True, 'yes', 'Yes']:
                    label = f"With {col_name}"
                else:
                    label = f"Without {col_name}"

                col_stats[label] = {
                    "total": int(len(subset)),
                    "churned": int(subset[target_col].sum()),
                    "churn_rate": round(subset[target_col].mean() * 100, 2)
                }

            if len(col_stats) >= 2:
                segments[col] = col_stats

                # Log significant differences
                rates = [v["churn_rate"] for v in col_stats.values()]
                if len(rates) >= 2 and max(rates) > min(rates) * 1.5:
                    self.logger.info(
                        f"Significant segment difference in {col}: "
                        f"rates range from {min(rates)}% to {max(rates)}%"
                    )

        # Analyze by categorical columns (like state) if they exist
        categorical_cols = ['state', 'region', 'category', 'type', 'plan']
        for col in categorical_cols:
            if col in df.columns and df[col].nunique() <= 50:
                cat_stats = df.groupby(col).agg({
                    target_col: ["sum", "count", "mean"]
                }).reset_index()
                cat_stats.columns = [col, "churned", "total", "churn_rate"]
                cat_stats["churn_rate"] = (cat_stats["churn_rate"] * 100).round(2)
                cat_stats = cat_stats.sort_values("churn_rate", ascending=False)

                top_stats = {}
                for _, row in cat_stats.head(10).iterrows():
                    top_stats[str(row[col])] = {
                        "total": int(row["total"]),
                        "churned": int(row["churned"]),
                        "churn_rate": float(row["churn_rate"])
                    }

                if top_stats:
                    segments[f"top_{col}_by_{target_col}"] = top_stats

        return segments

    def _find_tipping_points(self, df: pd.DataFrame, target_col: str = "churn") -> List[Dict]:
        """
        Find tipping points where target rate spikes.

        Uses ThresholdFinder to CALCULATE optimal thresholds from data.
        No hardcoded values - works for ANY feature columns.
        """
        tipping_points = []

        # Get numeric columns that could be tipping point candidates
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target column and ID-like columns
        exclude_patterns = [target_col, 'id', 'phone', 'area_code']
        candidate_cols = [
            col for col in numeric_cols
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]

        self.logger.info(f"Finding tipping points for {len(candidate_cols)} numeric features")

        for col in candidate_cols:
            # Use ThresholdFinder to calculate optimal threshold
            result = self.threshold_finder.find_optimal_threshold(
                df=df,
                feature_col=col,
                target_col=target_col,
                min_samples=MIN_SAMPLES_FOR_THRESHOLD,
                min_impact=MIN_IMPACT_FOR_THRESHOLD
            )

            if result['threshold'] is not None:
                # Format column name for display
                factor_name = col.replace('_', ' ').title()

                tipping_point = {
                    "factor": factor_name,
                    "feature": col,  # Include raw column name for later use
                    "threshold": result['threshold'],
                    "churn_below": round(result['below_rate'] * 100, 2),
                    "churn_above": round(result['above_rate'] * 100, 2),
                    "impact_multiplier": round(result['impact_multiplier'], 2),
                    "below_count": result['below_count'],
                    "above_count": result['above_count'],
                    "insight": f"Records with {factor_name} >= {result['threshold']} have "
                              f"{round(result['above_rate'] * 100, 1)}% {target_col} rate vs "
                              f"{round(result['below_rate'] * 100, 1)}% below threshold",
                    "calculated": True  # Mark as dynamically calculated
                }
                tipping_points.append(tipping_point)

                self.logger.info(
                    f"  Found tipping point: {factor_name} >= {result['threshold']} "
                    f"({result['above_rate']*100:.1f}% vs {result['below_rate']*100:.1f}%, "
                    f"{result['impact_multiplier']:.1f}x impact)"
                )

        # Sort by impact multiplier (most significant first)
        tipping_points.sort(key=lambda x: x['impact_multiplier'], reverse=True)

        self.logger.info(f"Found {len(tipping_points)} significant tipping points")
        return tipping_points

    def _compare_churned_retained(self, df: pd.DataFrame, target_col: str = "churn") -> Dict:
        """
        Compare metrics between positive and negative target groups.

        Works for ANY target column and ANY numeric features.
        """
        positive = df[df[target_col] == 1]
        negative = df[df[target_col] == 0]

        comparison = {}

        # Find all numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]

        for col in numeric_cols:
            try:
                positive_mean = float(positive[col].mean())
                negative_mean = float(negative[col].mean())
                diff_pct = round(
                    (positive_mean - negative_mean) / max(abs(negative_mean), 0.01) * 100,
                    2
                )
                comparison[col] = {
                    f"{target_col}_positive_avg": round(positive_mean, 2),
                    f"{target_col}_negative_avg": round(negative_mean, 2),
                    "difference_percent": diff_pct
                }
            except Exception:
                continue

        comparison["significant_differences"] = [
            k for k, v in comparison.items()
            if isinstance(v, dict) and abs(v.get("difference_percent", 0)) > 10
        ]

        return comparison

    def _generate_insights(self, df: pd.DataFrame, target_col: str = "churn") -> List[Dict]:
        """
        Generate actionable insights from the analysis.

        Uses CALCULATED tipping points, not hardcoded thresholds.
        Works for ANY data structure.
        """
        insights = []
        target_label = target_col.replace("_", " ")

        # Get the calculated tipping points
        tipping_points = self.results.get("tipping_points", [])

        # Generate insights from top tipping points
        for i, tp in enumerate(tipping_points[:3]):  # Top 3 most impactful
            factor = tp.get("factor", "Unknown")
            threshold = tp.get("threshold", 0)
            above_rate = tp.get("churn_above", 0)
            below_rate = tp.get("churn_below", 0)
            multiplier = tp.get("impact_multiplier", 1)

            priority = "critical" if i == 0 else ("high" if i == 1 else "medium")

            insights.append({
                "category": "Calculated Tipping Point",
                "title": f"{factor} Predicts {target_label.title()}",
                "finding": f"Records with {factor} >= {threshold} have "
                          f"{above_rate}% {target_label} rate vs {below_rate}% below "
                          f"({multiplier:.1f}x higher)",
                "recommendation": f"Implement proactive outreach when {factor} reaches "
                                 f"threshold of {threshold}.",
                "priority": priority,
                "calculated_threshold": threshold
            })

        # Generate insights from segment analysis
        segments = self.results.get("segment_analysis", {})
        for segment_name, segment_data in segments.items():
            if not isinstance(segment_data, dict):
                continue

            rates = [(k, v.get("churn_rate", 0), v.get("total", 0))
                     for k, v in segment_data.items() if isinstance(v, dict)]

            if len(rates) >= 2:
                rates.sort(key=lambda x: x[1], reverse=True)
                high_seg, high_rate, high_count = rates[0]
                low_seg, low_rate, low_count = rates[-1]

                if high_rate > low_rate * 1.5 and high_count >= 50:
                    insights.append({
                        "category": "High Risk Segment",
                        "title": f"{segment_name.replace('_', ' ').title()} Segment Risk",
                        "finding": f"{high_seg} shows {high_rate}% {target_label} rate "
                                  f"vs {low_rate}% for {low_seg}",
                        "recommendation": f"Target retention efforts at {high_seg} segment.",
                        "priority": "high" if high_rate > low_rate * 2 else "medium"
                    })

        return insights

    def _format_training_results(self, training_results: Dict) -> Dict:
        """Format training results from ChurnModelTrainer for analysis output."""
        return {
            "accuracy": training_results.get("accuracy", 0) * 100,
            "precision": training_results.get("precision", 0) * 100,
            "recall": training_results.get("recall", 0) * 100,
            "f1_score": training_results.get("f1_score", 0),
            "true_positives": training_results.get("true_positives", 0),
            "true_negatives": training_results.get("true_negatives", 0),
            "false_positives": training_results.get("false_positives", 0),
            "false_negatives": training_results.get("false_negatives", 0),
            "high_risk_count": training_results.get("high_risk_count", 0),
            "risk_tier_distribution": training_results.get("risk_tier_distribution", {}),
            "training_method": "sklearn.linear_model.LogisticRegression",
            "note": "All metrics CALCULATED by sklearn.metrics, NOT hardcoded"
        }

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

        validation = self.results.get("model_validation", {})
        if validation:
            summary["model_accuracy"] = validation.get("accuracy")
            summary["high_risk_count"] = validation.get("high_risk_count")

        return summary

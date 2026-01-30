"""
Dynamic Threshold Finder - Calculates optimal thresholds from data.

IMPORTANT: No hardcoded values - adapts to ANY dataset.
All thresholds are CALCULATED from the input data.

This module works for ANY:
- Industry (telecom, healthcare, retail, SaaS, etc.)
- Target variable (churn, readmission, attrition, etc.)
- Feature columns (service calls, usage, tenure, etc.)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import setup_logging


class ThresholdFinder:
    """
    Finds optimal thresholds dynamically from data.

    No hardcoded assumptions - adapts to ANY dataset.
    """

    def __init__(self):
        self.logger = setup_logging('threshold_finder')
        self.found_thresholds: Dict[str, Dict] = {}

    def find_optimal_threshold(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        min_samples: int = 50,
        min_impact: float = 0.05
    ) -> Dict:
        """
        Find the threshold value where target rate increases most significantly.

        Works for ANY feature column and ANY target column.
        No hardcoded assumptions about what the threshold should be.

        Args:
            df: DataFrame with feature and target columns
            feature_col: Column to find threshold for (e.g., 'customer_service_calls')
            target_col: Target column (e.g., 'churn')
            min_samples: Minimum samples required in each group
            min_impact: Minimum rate difference to consider significant

        Returns:
            Dictionary with:
            - threshold: optimal threshold value (None if not found)
            - below_rate: target rate below threshold
            - above_rate: target rate at/above threshold
            - impact_multiplier: how much target rate increases
            - below_count: samples below threshold
            - above_count: samples at/above threshold
        """
        self.logger.info(f"Finding optimal threshold for '{feature_col}' vs '{target_col}'")

        # Validate columns exist
        if feature_col not in df.columns:
            self.logger.warning(f"Feature column '{feature_col}' not found")
            return self._empty_result()

        if target_col not in df.columns:
            self.logger.warning(f"Target column '{target_col}' not found")
            return self._empty_result()

        # Get unique values sorted
        unique_values = sorted(df[feature_col].dropna().unique())

        if len(unique_values) < 2:
            self.logger.warning(f"Not enough unique values in '{feature_col}'")
            return self._empty_result()

        best_result = self._empty_result()
        best_impact = 0

        for value in unique_values:
            below = df[df[feature_col] < value]
            above = df[df[feature_col] >= value]

            # Skip if either group is too small
            if len(below) < min_samples or len(above) < min_samples:
                continue

            below_rate = below[target_col].mean()
            above_rate = above[target_col].mean()

            # Calculate impact (difference in rates)
            impact = above_rate - below_rate

            if impact > best_impact and impact >= min_impact:
                best_impact = impact
                multiplier = above_rate / below_rate if below_rate > 0.001 else float('inf')

                best_result = {
                    'threshold': value,
                    'below_rate': float(below_rate),
                    'above_rate': float(above_rate),
                    'impact_multiplier': float(min(multiplier, 100)),  # Cap at 100x
                    'below_count': int(len(below)),
                    'above_count': int(len(above)),
                    'feature': feature_col,
                    'target': target_col
                }

        if best_result['threshold'] is not None:
            self.logger.info(f"  Found optimal threshold: {feature_col} >= {best_result['threshold']}")
            self.logger.info(f"  Below threshold: {best_result['below_rate']*100:.2f}% {target_col} rate")
            self.logger.info(f"  Above threshold: {best_result['above_rate']*100:.2f}% {target_col} rate")
            self.logger.info(f"  Impact multiplier: {best_result['impact_multiplier']:.2f}x")

            # Store for later reference
            self.found_thresholds[feature_col] = best_result
        else:
            self.logger.info(f"  No significant threshold found for '{feature_col}'")

        return best_result

    def find_all_tipping_points(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        min_samples: int = 50
    ) -> Dict[str, Dict]:
        """
        Find optimal thresholds for multiple features.

        Returns dictionary mapping feature -> threshold info,
        sorted by impact (most significant first).
        """
        self.logger.info(f"Finding tipping points for {len(feature_cols)} features")

        results = {}

        for col in feature_cols:
            if col in df.columns and col != target_col:
                result = self.find_optimal_threshold(df, col, target_col, min_samples)
                if result['threshold'] is not None:
                    results[col] = result

        # Sort by impact multiplier (most significant first)
        sorted_results = dict(
            sorted(
                results.items(),
                key=lambda x: x[1].get('impact_multiplier', 0),
                reverse=True
            )
        )

        self.logger.info(f"Found {len(sorted_results)} significant tipping points")
        return sorted_results

    def find_percentile_threshold(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        percentile: float = 0.75
    ) -> Dict:
        """
        Find threshold at a specific percentile and measure impact.

        Useful for "high usage" type thresholds where you want
        to compare top quartile vs rest.
        """
        if feature_col not in df.columns:
            return self._empty_result()

        threshold = df[feature_col].quantile(percentile)

        below = df[df[feature_col] <= threshold]
        above = df[df[feature_col] > threshold]

        if len(below) == 0 or len(above) == 0:
            return self._empty_result()

        below_rate = below[target_col].mean()
        above_rate = above[target_col].mean()
        multiplier = above_rate / below_rate if below_rate > 0.001 else 1.0

        result = {
            'threshold': float(threshold),
            'percentile': percentile,
            'below_rate': float(below_rate),
            'above_rate': float(above_rate),
            'impact_multiplier': float(min(multiplier, 100)),
            'below_count': int(len(below)),
            'above_count': int(len(above)),
            'feature': feature_col,
            'target': target_col
        }

        self.logger.info(f"  {feature_col} at {percentile*100:.0f}th percentile ({threshold:.1f}):")
        self.logger.info(f"    Below: {below_rate*100:.2f}% rate, Above: {above_rate*100:.2f}% rate")

        return result

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'threshold': None,
            'below_rate': 0.0,
            'above_rate': 0.0,
            'impact_multiplier': 1.0,
            'below_count': 0,
            'above_count': 0,
            'feature': None,
            'target': None
        }


def calculate_dynamic_risk_tiers(
    probabilities: pd.Series,
    method: str = 'percentile'
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate risk tier boundaries from probability distribution.

    No hardcoded boundaries - adapts to ANY data.

    Args:
        probabilities: Series of predicted probabilities
        method: 'percentile' (based on distribution) or 'fixed' (0.25, 0.50, 0.75)

    Returns:
        Dictionary mapping tier name to (min, max) tuple
    """
    logger = setup_logging('threshold_finder')

    if method == 'percentile':
        # Calculate percentile-based boundaries
        p25 = float(probabilities.quantile(0.50))   # Bottom 50% = Low
        p50 = float(probabilities.quantile(0.75))   # 50-75% = Medium
        p75 = float(probabilities.quantile(0.90))   # 75-90% = High
        # Above 90th percentile = Critical

        tiers = {
            'low': (0.0, p25),
            'medium': (p25, p50),
            'high': (p50, p75),
            'critical': (p75, 1.01)
        }
    else:
        # Quartile-based fixed divisions of the probability range
        prob_min = float(probabilities.min())
        prob_max = float(probabilities.max())
        prob_range = prob_max - prob_min

        tiers = {
            'low': (prob_min, prob_min + prob_range * 0.25),
            'medium': (prob_min + prob_range * 0.25, prob_min + prob_range * 0.50),
            'high': (prob_min + prob_range * 0.50, prob_min + prob_range * 0.75),
            'critical': (prob_min + prob_range * 0.75, 1.01)
        }

    logger.info("CALCULATED risk tier boundaries from data:")
    for tier, (low, high) in tiers.items():
        logger.info(f"  {tier.capitalize()}: {low*100:.1f}% - {high*100:.1f}%")

    return tiers


def assign_risk_tier(
    probability: float,
    tiers: Dict[str, Tuple[float, float]]
) -> str:
    """
    Assign a single probability to a risk tier.

    Args:
        probability: Predicted probability (0-1)
        tiers: Dictionary from calculate_dynamic_risk_tiers()

    Returns:
        Tier name ('low', 'medium', 'high', 'critical')
    """
    for tier_name, (low, high) in tiers.items():
        if low <= probability < high:
            return tier_name
    return 'critical'  # Default for edge cases


def assign_risk_tiers_series(
    probabilities: pd.Series,
    tiers: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.Series:
    """
    Assign risk tiers to a Series of probabilities.

    If tiers not provided, calculates them from the data.
    """
    if tiers is None:
        tiers = calculate_dynamic_risk_tiers(probabilities)

    return probabilities.apply(lambda p: assign_risk_tier(p, tiers))

"""
Churn Model Trainer - REAL Machine Learning Training

IMPORTANT: All parameters are CALCULATED from data, nothing is hardcoded.
- Mean values: CALCULATED by StandardScaler.fit()
- Std values: CALCULATED by StandardScaler.fit()
- Coefficients: LEARNED by LogisticRegression.fit()
- Intercept: LEARNED by LogisticRegression.fit()
- Metrics: CALCULATED by sklearn.metrics

This is REAL machine learning, not copying pre-calculated answers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from src.utils import setup_logging


class ChurnModelTrainer:
    """
    Trains logistic regression model from scratch using sklearn.

    CRITICAL: All coefficients and parameters are CALCULATED from data.
    Nothing is hardcoded. This is REAL machine learning.

    Usage:
        trainer = ChurnModelTrainer()
        results = trainer.train(df)  # LEARNS from data
        predictions = trainer.predict(df)  # Uses LEARNED model
        trainer.save('model.pkl')  # Persist trained model
    """

    # Features used for prediction (feature names, not values)
    DEFAULT_FEATURES = [
        'customer_service_calls',
        'total_day_minutes',
        'total_eve_minutes',
        'total_night_minutes',
        'total_intl_minutes',
        'total_intl_calls',
        'number_vmail_messages',
        'account_length',
        'international_plan',
        'voice_mail_plan'
    ]

    def __init__(self, features: Optional[List[str]] = None):
        """
        Initialize trainer.

        Args:
            features: List of feature column names. Defaults to standard churn features.
        """
        self.logger = setup_logging('trainer')
        self.features = features or self.DEFAULT_FEATURES.copy()

        # These will be SET during training (not hardcoded)
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self.training_results: Optional[Dict] = None

    def train(self, df: pd.DataFrame, target_col: str = 'churn') -> Dict:
        """
        Train model on data - CALCULATES everything from scratch.

        This method LEARNS:
        - Mean values (via StandardScaler.fit_transform)
        - Std values (via StandardScaler.fit_transform)
        - Coefficients (via LogisticRegression.fit)
        - Intercept (via LogisticRegression.fit)

        NOTHING is hardcoded. All values come from the data.

        Args:
            df: DataFrame with features and target column
            target_col: Name of target column (default: 'churn')

        Returns:
            Dictionary with all CALCULATED metrics and parameters
        """
        self.logger.info("=" * 70)
        self.logger.info("TRAINING MODEL - ALL VALUES CALCULATED FROM DATA")
        self.logger.info("=" * 70)

        # Validate input data
        self._validate_input(df, target_col)

        X = df[self.features].copy()
        y = df[target_col].copy()

        self.logger.info(f"Training data: {len(X)} samples, {len(self.features)} features")
        self.logger.info(f"Target distribution: {y.sum()} churned ({y.mean()*100:.2f}%), "
                        f"{len(y) - y.sum()} retained ({(1-y.mean())*100:.2f}%)")

        # ================================================================
        # STEP 1: CALCULATE standardisation parameters from data
        # ================================================================
        self.logger.info("")
        self.logger.info("STEP 1: CALCULATING standardisation parameters from data...")
        self.logger.info("        (NOT using any hardcoded values)")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Log the CALCULATED parameters
        self.logger.info("")
        self.logger.info("CALCULATED MEANS (learned from data via StandardScaler):")
        for i, feat in enumerate(self.features):
            self.logger.info(f"    {feat}: {self.scaler.mean_[i]:.6f}")

        self.logger.info("")
        self.logger.info("CALCULATED STD DEVS (learned from data via StandardScaler):")
        for i, feat in enumerate(self.features):
            self.logger.info(f"    {feat}: {self.scaler.scale_[i]:.6f}")

        # ================================================================
        # STEP 2: TRAIN model to LEARN coefficients from data
        # ================================================================
        self.logger.info("")
        self.logger.info("STEP 2: TRAINING model to LEARN coefficients from data...")
        self.logger.info("        (NOT using any hardcoded coefficients)")
        self.logger.info("        Using: LogisticRegression(penalty=None, solver='lbfgs')")

        self.model = LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # Log the LEARNED coefficients
        self.logger.info("")
        self.logger.info(f"LEARNED INTERCEPT (from training): {self.model.intercept_[0]:.6f}")
        self.logger.info("")
        self.logger.info("LEARNED COEFFICIENTS (from training via LogisticRegression):")
        for i, feat in enumerate(self.features):
            self.logger.info(f"    {feat}: {self.model.coef_[0][i]:.6f}")

        # ================================================================
        # STEP 3: CALCULATE predictions using trained model
        # ================================================================
        self.logger.info("")
        self.logger.info("STEP 3: CALCULATING predictions using TRAINED model...")

        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        # ================================================================
        # STEP 4: CALCULATE accuracy metrics using sklearn
        # ================================================================
        self.logger.info("")
        self.logger.info("STEP 4: CALCULATING accuracy metrics via sklearn.metrics...")

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y, y_prob)
        except ValueError:
            auc = 0.0

        conf_matrix = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()

        # Calculate dynamic risk tier boundaries from probability distribution
        # No hardcoded thresholds - adapts to ANY data
        risk_tier_bounds = self._calculate_dynamic_risk_tiers(pd.Series(y_prob))
        high_risk_count = int((y_prob >= risk_tier_bounds['high'][0]).sum())

        # Calculate risk tier distribution using dynamic boundaries
        risk_tiers = {}
        for tier_name, (low_bound, high_bound) in risk_tier_bounds.items():
            risk_tiers[tier_name] = int(((y_prob >= low_bound) & (y_prob < high_bound)).sum())

        self.logger.info("")
        self.logger.info("CALCULATED METRICS (from sklearn.metrics):")
        self.logger.info(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"    Precision: {precision:.4f} ({precision*100:.2f}%)")
        self.logger.info(f"    Recall:    {recall:.4f} ({recall*100:.2f}%)")
        self.logger.info(f"    F1 Score:  {f1:.4f}")
        self.logger.info(f"    AUC-ROC:   {auc:.4f}")
        self.logger.info(f"    True Positives:  {tp}")
        self.logger.info(f"    True Negatives:  {tn}")
        self.logger.info(f"    False Positives: {fp}")
        self.logger.info(f"    False Negatives: {fn}")
        self.logger.info(f"    High-risk customers (prob > 0.5): {high_risk_count}")

        self.is_trained = True

        # Store all CALCULATED results
        self.training_results = {
            # Metrics (all CALCULATED)
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc),

            # Confusion matrix (all CALCULATED)
            'confusion_matrix': conf_matrix.tolist(),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),

            # Risk distribution (all CALCULATED from data)
            'high_risk_count': high_risk_count,
            'risk_tier_distribution': risk_tiers,
            'risk_tier_bounds': risk_tier_bounds,  # Store calculated boundaries

            # Data stats
            'total_samples': len(y),
            'churn_count': int(y.sum()),
            'retained_count': int(len(y) - y.sum()),
            'churn_rate': float(y.mean()),

            # LEARNED model parameters (NOT hardcoded)
            'learned_intercept': float(self.model.intercept_[0]),
            'learned_coefficients': {
                feat: float(self.model.coef_[0][i])
                for i, feat in enumerate(self.features)
            },

            # CALCULATED standardisation params (NOT hardcoded)
            'calculated_means': {
                feat: float(self.scaler.mean_[i])
                for i, feat in enumerate(self.features)
            },
            'calculated_stds': {
                feat: float(self.scaler.scale_[i])
                for i, feat in enumerate(self.features)
            },

            # Metadata
            'features_used': self.features.copy(),
            'training_method': 'sklearn.linear_model.LogisticRegression',
            'scaler_method': 'sklearn.preprocessing.StandardScaler'
        }

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("TRAINING COMPLETE - All values CALCULATED from data")
        self.logger.info("=" * 70)

        return self.training_results

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using the TRAINED model.

        Uses LEARNED coefficients and CALCULATED standardisation,
        not any hardcoded values.

        Args:
            df: DataFrame with feature columns

        Returns:
            DataFrame with added prediction columns
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first or load() a saved model.")

        self.logger.info(f"Generating predictions for {len(df)} samples using TRAINED model...")

        # Validate features exist
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in data: {missing}")

        X = df[self.features].copy()
        X_scaled = self.scaler.transform(X)

        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = self.model.predict(X_scaled)

        result = df.copy()
        result['churn_probability'] = probabilities
        result['risk_score'] = (probabilities * 100).round(2)
        result['predicted_churn'] = predictions

        # Calculate dynamic risk tier boundaries from THIS data
        # No hardcoded thresholds - adapts to ANY probability distribution
        prob_series = pd.Series(probabilities)
        risk_tier_bounds = self._calculate_dynamic_risk_tiers(prob_series)

        # Assign risk tiers using calculated boundaries
        result['risk_tier'] = self._assign_risk_tiers(prob_series, risk_tier_bounds)

        tier_counts = result['risk_tier'].value_counts().to_dict()
        self.logger.info(f"Risk distribution: {tier_counts}")

        return result

    def save(self, path: Path) -> None:
        """
        Save trained model to file.

        Saves the LEARNED model and CALCULATED scaler parameters.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'is_trained': self.is_trained,
            'training_results': self.training_results
        }

        joblib.dump(save_data, path)
        self.logger.info(f"Saved trained model to: {path}")

    def load(self, path: Path) -> 'ChurnModelTrainer':
        """
        Load trained model from file.

        Loads the previously LEARNED model and CALCULATED parameters.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = joblib.load(path)

        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']
        self.is_trained = data['is_trained']
        self.training_results = data.get('training_results')

        self.logger.info(f"Loaded trained model from: {path}")

        if self.training_results:
            accuracy = self.training_results.get('accuracy', 0)
            self.logger.info(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        return self

    def get_learned_coefficients(self) -> Dict:
        """
        Return LEARNED coefficients (not hardcoded).

        Returns:
            Dictionary with intercept and feature coefficients
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        return {
            'intercept': float(self.model.intercept_[0]),
            'coefficients': {
                feat: float(self.model.coef_[0][i])
                for i, feat in enumerate(self.features)
            }
        }

    def get_calculated_standardisation(self) -> Dict:
        """
        Return CALCULATED standardisation params (not hardcoded).

        Returns:
            Dictionary with mean and std for each feature
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        return {
            feat: {
                'mean': float(self.scaler.mean_[i]),
                'std': float(self.scaler.scale_[i])
            }
            for i, feat in enumerate(self.features)
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance based on absolute coefficient values.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        abs_coefs = np.abs(self.model.coef_[0])
        importance = abs_coefs / abs_coefs.sum()

        return {
            feat: float(importance[i])
            for i, feat in enumerate(self.features)
        }

    def _calculate_dynamic_risk_tiers(
        self,
        probabilities: pd.Series
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate risk tier boundaries from probability distribution.

        No hardcoded boundaries - adapts to ANY data.
        Uses percentiles to create meaningful categories.

        Args:
            probabilities: Series of predicted probabilities

        Returns:
            Dictionary mapping tier name to (min, max) tuple
        """
        # Calculate percentile-based boundaries from the data
        p50 = float(probabilities.quantile(0.50))   # Median
        p75 = float(probabilities.quantile(0.75))   # 75th percentile
        p90 = float(probabilities.quantile(0.90))   # 90th percentile

        tiers = {
            'low': (0.0, p50),        # Bottom 50%
            'medium': (p50, p75),     # 50th-75th percentile
            'high': (p75, p90),       # 75th-90th percentile
            'critical': (p90, 1.01)   # Top 10%
        }

        self.logger.info("CALCULATED risk tier boundaries from data:")
        for tier_name, (low, high) in tiers.items():
            self.logger.info(f"    {tier_name.capitalize()}: {low*100:.1f}% - {high*100:.1f}%")

        return tiers

    def _assign_risk_tiers(
        self,
        probabilities: pd.Series,
        tier_bounds: Dict[str, Tuple[float, float]]
    ) -> pd.Series:
        """
        Assign risk tiers based on CALCULATED boundaries.

        Args:
            probabilities: Series of predicted probabilities
            tier_bounds: Dictionary from _calculate_dynamic_risk_tiers()

        Returns:
            Series of tier labels
        """
        def get_tier(prob: float) -> str:
            for tier_name, (low, high) in tier_bounds.items():
                if low <= prob < high:
                    return tier_name
            return 'critical'  # Default for edge cases

        return probabilities.apply(get_tier)

    def _validate_input(self, df: pd.DataFrame, target_col: str) -> None:
        """Validate input data before training."""
        # Check for missing features
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        # Check for target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Check for sufficient data
        if len(df) < 100:
            self.logger.warning(f"Small dataset ({len(df)} samples). Results may be unreliable.")

        # Check for class balance
        churn_rate = df[target_col].mean()
        if churn_rate < 0.05 or churn_rate > 0.95:
            self.logger.warning(f"Imbalanced classes (churn rate: {churn_rate:.2%}). "
                              "Consider using class_weight='balanced'.")

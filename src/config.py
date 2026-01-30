"""
Configuration settings for the Veritly data platform.

IMPORTANT: This file contains ONLY path configurations and analysis thresholds.
NO hardcoded ML coefficients or model parameters.
All ML values are CALCULATED from data by src/ml/trainer.py
"""

from pathlib import Path

# =============================================================================
# PATH CONFIGURATIONS
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
METADATA_DIR = BASE_DIR / "metadata"
PRODUCTS_DIR = BASE_DIR / "data_products"

# Data zones
RAW_ZONE = DATA_DIR / "raw"
CURATED_ZONE = DATA_DIR / "curated"
DERIVED_ZONE = DATA_DIR / "derived"

# Metadata paths
SCHEMAS_DIR = METADATA_DIR / "schemas"
REGISTRY_PATH = METADATA_DIR / "registry.json"
LINEAGE_PATH = METADATA_DIR / "lineage.json"

# Reports
REPORTS_DIR = PRODUCTS_DIR / "reports"

# Visualisations
VISUALISATIONS_DIR = PRODUCTS_DIR / "visualisations"

# =============================================================================
# MODEL STORAGE (for trained sklearn models)
# =============================================================================

MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "churn_model.pkl"

# =============================================================================
# ENSURE DIRECTORIES EXIST
# =============================================================================

for dir_path in [RAW_ZONE, CURATED_ZONE, DERIVED_ZONE, SCHEMAS_DIR,
                 REPORTS_DIR, VISUALISATIONS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================
#
# NOTE: All thresholds are now CALCULATED dynamically from input data.
# See src/analysis/threshold_finder.py for dynamic threshold calculation.
#
# REMOVED (now calculated):
# - CUSTOMER_SERVICE_CALL_THRESHOLD: Now found by ThresholdFinder
# - HIGH_RISK_SCORE_THRESHOLD: Now calculated from probability distribution
# - MEDIUM_RISK_SCORE_THRESHOLD: Now calculated from probability distribution
# - RISK_TIERS: Now calculated by calculate_dynamic_risk_tiers()
#
# This makes the pipeline work for ANY data, ANY industry, ANY domain.
# =============================================================================

# Minimum samples required for threshold calculation
MIN_SAMPLES_FOR_THRESHOLD = 50

# Minimum impact (rate difference) to consider a threshold significant
MIN_IMPACT_FOR_THRESHOLD = 0.05

# =============================================================================
# REMOVED: HARDCODED ML PARAMETERS
# =============================================================================
#
# The following were REMOVED because they were hardcoded from Pele's Excel.
# All ML parameters are now CALCULATED from data by src/ml/trainer.py
#
# REMOVED: LOGISTIC_COEFFICIENTS - Now LEARNED by sklearn.LogisticRegression.fit()
# REMOVED: STANDARDISATION_PARAMS - Now CALCULATED by sklearn.StandardScaler.fit()
# REMOVED: MODEL_FEATURES - Now defined in src/ml/trainer.py
#
# This is REAL machine learning - all values come from training on data.
# =============================================================================

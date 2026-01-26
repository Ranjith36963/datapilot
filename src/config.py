"""
Configuration settings for the Veritly data platform.
"""

from pathlib import Path

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

# Ensure all directories exist
for dir_path in [RAW_ZONE, CURATED_ZONE, DERIVED_ZONE, SCHEMAS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Analysis thresholds
CUSTOMER_SERVICE_CALL_THRESHOLD = 4
HIGH_RISK_SCORE_THRESHOLD = 0.7
MEDIUM_RISK_SCORE_THRESHOLD = 0.4

# =============================================================================
# PELE'S LOGISTIC REGRESSION MODEL PARAMETERS
# Model achieves 86.38% accuracy on telecom churn prediction
# =============================================================================

LOGISTIC_COEFFICIENTS = {
    'intercept': -2.311676,
    'customer_service_calls': 0.670108,
    'total_day_minutes': 0.705463,
    'total_eve_minutes': 0.362806,
    'total_night_minutes': 0.187327,
    'total_intl_minutes': 0.244144,
    'total_intl_calls': -0.225838,
    'number_vmail_messages': 0.481997,
    'account_length': 0.034674,
    'international_plan': 0.599908,
    'voice_mail_plan': -0.890693
}

STANDARDISATION_PARAMS = {
    'customer_service_calls': {'mean': 1.562856, 'std': 1.315491},
    'total_day_minutes': {'mean': 179.775098, 'std': 54.467389},
    'total_eve_minutes': {'mean': 200.980348, 'std': 50.713844},
    'total_night_minutes': {'mean': 200.872037, 'std': 50.573847},
    'total_intl_minutes': {'mean': 10.237294, 'std': 2.791840},
    'total_intl_calls': {'mean': 4.479448, 'std': 2.461214},
    'number_vmail_messages': {'mean': 8.099010, 'std': 13.688365},
    'account_length': {'mean': 101.064806, 'std': 39.822106},
    'international_plan': {'mean': 0.096910, 'std': 0.295879},
    'voice_mail_plan': {'mean': 0.276628, 'std': 0.447398}
}

# Model features (order matters for consistency)
MODEL_FEATURES = [
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

# Risk tier thresholds (based on churn probability)
RISK_TIERS = {
    'low': (0.0, 0.15),
    'medium': (0.15, 0.30),
    'high': (0.30, 0.50),
    'critical': (0.50, 1.01)
}

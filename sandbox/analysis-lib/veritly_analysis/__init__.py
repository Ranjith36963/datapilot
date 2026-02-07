"""
Veritly Analysis Library — Sandbox skill modules for data analytics.

Runs in offline Daytona sandbox. Pure computation, no external API calls.
All functions return JSON-serializable dicts with {"status": "success|error", ...}.
"""

__version__ = "1.0.0"

# Data Understanding
from veritly_analysis.profiler import profile_data, detect_target_candidates, profile_and_upload
from veritly_analysis.schema_inference import infer_schema, detect_datetime_columns, detect_categorical_threshold, infer_and_upload
from veritly_analysis.data_validator import validate_data, check_nulls, check_duplicates, check_outliers, validate_and_upload

# Data Preparation
from veritly_analysis.curation import curate_dataframe, impute_missing, standardize_column_names, curate_and_upload
from veritly_analysis.outlier_detection import detect_outliers, flag_anomalies, get_outlier_summary, detect_and_upload
from veritly_analysis.feature_engineering import engineer_features, create_date_features, create_interaction_features, bin_numeric, encode_categorical, engineer_and_upload

# Statistics
from veritly_analysis.descriptive_stats import describe_data, describe_numeric, describe_categorical, compare_groups, describe_and_upload
from veritly_analysis.correlation import analyze_correlations, correlation_matrix, find_top_correlations, partial_correlation, correlate_and_upload
from veritly_analysis.hypothesis_testing import run_hypothesis_test, t_test, paired_t_test, anova, chi_square, mann_whitney, test_and_upload
from veritly_analysis.effect_size import calculate_effect_size, cohens_d, odds_ratio, cramers_v, effect_and_upload

# ML — Classification
from veritly_analysis.classifier import classify, auto_classify, tune_classifier, predict, classify_and_upload
from veritly_analysis.feature_selection import select_features, rfe_selection, shap_importance, select_and_upload
from veritly_analysis.explainability import explain_model, explain_prediction, explain_and_upload

# ML — Regression
from veritly_analysis.regressor import predict_numeric, auto_regress, tune_regressor, regress_and_upload

# ML — Clustering
from veritly_analysis.clustering import find_clusters, optimal_clusters, describe_clusters, cluster_and_upload
from veritly_analysis.dimensionality import reduce_dimensions, pca_analysis, visualize_2d, reduce_and_upload

# Time Series
from veritly_analysis.time_series import analyze_time_series, forecast, detect_change_points, timeseries_and_upload

# Survival Analysis
from veritly_analysis.survival_analysis import survival_analysis, cox_regression, survival_and_upload

# Threshold Finding
from veritly_analysis.threshold_finder import find_thresholds, find_optimal_split, threshold_confidence_interval, thresholds_and_upload

# NLP
from veritly_analysis.sentiment import analyze_sentiment, batch_sentiment, sentiment_and_upload
from veritly_analysis.text_stats import analyze_text, batch_text_stats, extract_keywords, text_stats_and_upload
from veritly_analysis.topic_model import extract_topics, assign_topics, topics_and_upload
from veritly_analysis.entity_extractor import extract_entities, batch_extract_entities, entities_and_upload
from veritly_analysis.intent_detector import detect_intent, train_intent_classifier, batch_classify_intent, intent_and_upload

# Image
from veritly_analysis.ocr import extract_text_from_image, batch_ocr, ocr_and_upload

# Visualization
from veritly_analysis.charts import create_chart, auto_chart, create_dashboard, chart_and_upload

# Output Formatting
from veritly_analysis.report_data import format_for_narrative, create_executive_summary_data, create_detailed_findings_data

# Utilities
from veritly_analysis.utils import load_data, save_data, upload_result, safe_json_serialize

__all__ = [
    # Data Understanding
    "profile_data", "detect_target_candidates", "profile_and_upload",
    "infer_schema", "detect_datetime_columns", "detect_categorical_threshold", "infer_and_upload",
    "validate_data", "check_nulls", "check_duplicates", "check_outliers", "validate_and_upload",
    # Data Preparation
    "curate_dataframe", "impute_missing", "standardize_column_names", "curate_and_upload",
    "detect_outliers", "flag_anomalies", "get_outlier_summary", "detect_and_upload",
    "engineer_features", "create_date_features", "create_interaction_features", "bin_numeric", "encode_categorical", "engineer_and_upload",
    # Statistics
    "describe_data", "describe_numeric", "describe_categorical", "compare_groups", "describe_and_upload",
    "analyze_correlations", "correlation_matrix", "find_top_correlations", "partial_correlation", "correlate_and_upload",
    "run_hypothesis_test", "t_test", "paired_t_test", "anova", "chi_square", "mann_whitney", "test_and_upload",
    "calculate_effect_size", "cohens_d", "odds_ratio", "cramers_v", "effect_and_upload",
    # ML — Classification
    "classify", "auto_classify", "tune_classifier", "predict", "classify_and_upload",
    "select_features", "rfe_selection", "shap_importance", "select_and_upload",
    "explain_model", "explain_prediction", "explain_and_upload",
    # ML — Regression
    "predict_numeric", "auto_regress", "tune_regressor", "regress_and_upload",
    # ML — Clustering
    "find_clusters", "optimal_clusters", "describe_clusters", "cluster_and_upload",
    "reduce_dimensions", "pca_analysis", "visualize_2d", "reduce_and_upload",
    # Time Series
    "analyze_time_series", "forecast", "detect_change_points", "timeseries_and_upload",
    # Survival
    "survival_analysis", "cox_regression", "survival_and_upload",
    # Threshold
    "find_thresholds", "find_optimal_split", "threshold_confidence_interval", "thresholds_and_upload",
    # NLP
    "analyze_sentiment", "batch_sentiment", "sentiment_and_upload",
    "analyze_text", "batch_text_stats", "extract_keywords", "text_stats_and_upload",
    "extract_topics", "assign_topics", "topics_and_upload",
    "extract_entities", "batch_extract_entities", "entities_and_upload",
    "detect_intent", "train_intent_classifier", "batch_classify_intent", "intent_and_upload",
    # Image
    "extract_text_from_image", "batch_ocr", "ocr_and_upload",
    # Visualization
    "create_chart", "auto_chart", "create_dashboard", "chart_and_upload",
    # Output
    "format_for_narrative", "create_executive_summary_data", "create_detailed_findings_data",
    # Utilities
    "load_data", "save_data", "upload_result", "safe_json_serialize",
]

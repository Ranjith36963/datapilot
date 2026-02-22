"""
DataPilot — AI-powered data analysis engine.

Provides 34 analytical skills across data profiling, statistics,
machine learning, NLP, visualization, and more.

All functions return JSON-serializable dicts with {"status": "success|error", ...}.
"""

__version__ = "0.1.0"

# Data Understanding
from .analysis.anomaly import detect_and_upload, detect_outliers, flag_anomalies, get_outlier_summary

# ML — Classification
from .analysis.classification import auto_classify, classify, classify_and_upload, predict, tune_classifier

# ML — Clustering
from .analysis.clustering import cluster_and_upload, describe_clusters, find_clusters, optimal_clusters
from .analysis.correlation import (
    analyze_correlations,
    correlate_and_upload,
    correlation_matrix,
    find_top_correlations,
    partial_correlation,
)

# Statistics
from .analysis.descriptive import (
    compare_groups,
    describe_and_upload,
    describe_categorical,
    describe_data,
    describe_numeric,
)
from .analysis.dimensionality import pca_analysis, reduce_and_upload, reduce_dimensions, visualize_2d
from .analysis.effect_size import calculate_effect_size, cohens_d, cramers_v, effect_and_upload, odds_ratio
from .analysis.engineering import (
    bin_numeric,
    create_date_features,
    create_interaction_features,
    encode_categorical,
    engineer_and_upload,
    engineer_features,
)
from .analysis.explain import explain_and_upload, explain_model, explain_prediction
from .analysis.hypothesis import (
    anova,
    chi_square,
    mann_whitney,
    paired_t_test,
    run_hypothesis_test,
    t_test,
    test_and_upload,
)

# Data Querying
from .analysis.query import cross_tab, pivot_table, query_data, smart_query, top_n, value_counts

# ML — Regression
from .analysis.regression import auto_regress, predict_numeric, regress_and_upload, tune_regressor
from .analysis.selection import rfe_selection, select_and_upload, select_features, shap_importance

# Survival Analysis
from .analysis.survival import cox_regression, survival_analysis, survival_and_upload

# Threshold Finding
from .analysis.threshold import (
    find_optimal_split,
    find_thresholds,
    threshold_confidence_interval,
    thresholds_and_upload,
)

# Time Series
from .analysis.timeseries import analyze_time_series, detect_change_points, forecast, timeseries_and_upload

# Core — Analyst class
from .core.analyst import Analyst, AnalystResult
from .core.executor import Executor
from .core.router import Router

# Data Preparation
from .data.cleaner import curate_and_upload, curate_dataframe, impute_missing, standardize_column_names
from .data.fingerprint import fingerprint_dataset

# Image
from .data.ocr import batch_ocr, extract_text_from_image, ocr_and_upload
from .data.profiler import detect_target_candidates, profile_and_upload, profile_data
from .data.schema import detect_categorical_threshold, detect_datetime_columns, infer_and_upload, infer_schema
from .data.validator import check_duplicates, check_nulls, check_outliers, validate_and_upload, validate_data
from .llm.claude import ClaudeProvider
from .llm.groq import GroqProvider
from .llm.ollama import OllamaProvider
from .llm.openai import OpenAIProvider

# LLM Providers
from .llm.provider import LLMProvider
from .nlp.entities import batch_extract_entities, entities_and_upload, extract_entities
from .nlp.intent import batch_classify_intent, detect_intent, intent_and_upload, train_intent_classifier

# NLP
from .nlp.sentiment import analyze_sentiment, batch_sentiment, sentiment_and_upload
from .nlp.text_stats import analyze_text, batch_text_stats, extract_keywords, text_stats_and_upload
from .nlp.topics import assign_topics, extract_topics, topics_and_upload

# Utilities
from .utils.helpers import load_data, save_data

# Output Formatting
from .utils.report_data import create_detailed_findings_data, create_executive_summary_data, format_for_narrative
from .utils.serializer import safe_json_serialize
from .utils.uploader import upload_result

# Visualization
from .viz.charts import auto_chart, chart_and_upload, create_chart, create_dashboard

__all__ = [
    # Data Understanding
    "profile_data", "detect_target_candidates", "profile_and_upload",
    "infer_schema", "detect_datetime_columns", "detect_categorical_threshold", "infer_and_upload",
    "validate_data", "check_nulls", "check_duplicates", "check_outliers", "validate_and_upload",
    "fingerprint_dataset",
    # Data Querying
    "query_data", "pivot_table", "value_counts", "top_n", "cross_tab", "smart_query",
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
    # Core
    "Analyst", "AnalystResult", "Router", "Executor",
    # LLM
    "LLMProvider", "GroqProvider", "OllamaProvider", "ClaudeProvider", "OpenAIProvider",
]

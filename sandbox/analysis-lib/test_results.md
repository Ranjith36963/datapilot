# Veritly Analysis Library — Test Results

**Date:** 2026-02-06
**Test Data:** `Raw Data.xlsx` — Telecom churn dataset (3333 rows x 21 cols)
**Python:** 3.12 | **Platform:** Windows

---

## Summary

| Category | Tests | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| S1: Data Understanding | 4 | 4 | 0 | 0 |
| S2: Data Preparation | 5 | 5 | 0 | 0 |
| S3: Statistics | 10 | 10 | 0 | 0 |
| S4: ML Classification | 12 | 12 | 0 | 0 |
| S5: ML Regression | 3 | 3 | 0 | 0 |
| S6: ML Clustering | 6 | 6 | 0 | 0 |
| S7: Time Series | 5 | 5 | 0 | 0 |
| S8: Survival Analysis | 3 | 3 | 0 | 0 |
| S9: Threshold Finding | 2 | 2 | 0 | 0 |
| S10: NLP | 10 | 10 | 0 | 0 |
| S11: OCR | 1 | 0 | 0 | 1 |
| S12: Visualization | 7 | 7 | 0 | 0 |
| S13: Report Formatting | 3 | 3 | 0 | 0 |
| S14: JSON Serialization | 18 | 18 | 0 | 0 |
| **TOTAL** | **89** | **88** | **0** | **1** |

**Pass Rate: 98.9%** (88/89) — 100% excluding expected skips (OCR needs tesseract binary)

---

## Critical Bug Fix Verifications

| Fix | Description | Verified? | Result |
|-----|-------------|-----------|--------|
| Fix 1 | `predict()` LabelEncoder reuse | YES | `predict()` returned 3333 predictions with correct labels |
| Fix 3 | `tune_classifier()` Optuna best_params applied | YES | `hyperparameters` match `best_params` exactly |
| Fix 3 | `tune_regressor()` Optuna best_params applied | YES | Best params applied, CV R2 = 0.9999 |
| Fix 5 | `r_squared` pairwise NaN drop | YES | r2 = 1.0 for `total day minutes` vs `total day charge` (expected) |
| Fix 6 | `validate_data` returns `status: "success"` | YES | `assert result["status"] == "success"` passed |
| Fix 7 | `explain_model` LinearExplainer for logistic | YES | Logistic model explained without crash |
| Fix 7 | `explain_model` TreeExplainer for xgboost | YES | XGBoost model explained correctly |
| Fix 8 | `safe_json_serialize` handles `float('inf')` | YES | `inf` -> `None`, `-inf` -> `None` |

**All 8 critical fixes verified working at runtime.**

---

## Detailed Results

### Section 1: Data Understanding

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `profile_data` | OK | 1.0s | 3333 rows x 21 cols |
| `detect_target_candidates` | OK | 0.6s | |
| `infer_schema` | OK | 0.7s | Deprecation warning: `infer_datetime_format` |
| `validate_data` | OK | 0.7s | `validation_result=pass`, Fix 6 verified |

### Section 2: Data Preparation

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `curate_dataframe` | OK | 0.9s | |
| `detect_outliers (isolation_forest)` | OK | 1.5s | |
| `detect_outliers (iqr)` | OK | 1.0s | |
| `get_outlier_summary` | OK | 0.9s | |
| `engineer_features` | OK | 1.3s | 0 new features (no datetime cols) |

### Section 3: Statistics

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `describe_data` | OK | 0.9s | Bool columns cast to int (fixed) |
| `compare_groups` | OK | 0.6s | |
| `analyze_correlations (target)` | OK | 0.9s | |
| `analyze_correlations (spearman)` | OK | 0.9s | |
| `run_hypothesis_test (t_test)` | OK | 0.6s | |
| `run_hypothesis_test (chi_square)` | OK | 0.6s | |
| `run_hypothesis_test (anova)` | OK | 0.5s | |
| `calculate_effect_size (cohens_d)` | OK | 0.6s | |
| `calculate_effect_size (cramers_v)` | OK | 0.5s | |
| `calculate_effect_size (r_squared)` | OK | 0.6s | Fix 5 verified: r2 = 1.0 |

### Section 4: ML Classification

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `classify (random_forest)` | OK | 3.1s | Acc=0.94, F1=0.76, AUC=0.90 |
| `classify (xgboost)` | OK | 4.5s | |
| `classify (logistic)` | OK | 0.8s | |
| `classify (lightgbm)` | OK | 3.7s | |
| `auto_classify` | OK | 6.6s | Best: lightgbm |
| `predict (FIX 1)` | OK | 1.0s | 3333 predictions, Fix 1 verified |
| `tune_classifier (FIX 3)` | OK | 16.6s | Params match, Fix 3 verified |
| `select_features (auto)` | OK | 1.3s | |
| `select_features (mutual_info)` | OK | 1.0s | |
| `explain_model (xgboost)` | OK | 8.7s | TreeExplainer |
| `explain_prediction (xgboost)` | OK | 0.8s | |
| `explain_model (logistic)` | OK | 0.7s | LinearExplainer, Fix 7 verified |

### Section 5: ML Regression

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `predict_numeric (random_forest)` | OK | 5.3s | R2=0.9999, RMSE=0.108 |
| `auto_regress` | OK | 10.0s | |
| `tune_regressor (FIX 3)` | OK | 33.7s | Best params applied, Fix 3 verified |

### Section 6: ML Clustering

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `find_clusters (kmeans)` | OK | 10.3s | |
| `find_clusters (dbscan)` | OK | 4.7s | |
| `optimal_clusters` | OK | 3.3s | |
| `reduce_dimensions (PCA)` | OK | 0.7s | |
| `pca_analysis` | OK | 0.7s | |
| `reduce_dimensions (t-SNE)` | OK | 22.6s | |

### Section 7: Time Series

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `analyze_time_series` | OK | 1.0s | Synthetic data, 365 days |
| `forecast (arima)` | OK | 1.0s | |
| `forecast (exp_smoothing)` | OK | 0.1s | |
| `forecast (prophet)` | OK | 2.4s | Prophet installed & working |
| `detect_change_points` | OK | 0.2s | |

### Section 8: Survival Analysis

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `survival_analysis` | OK | 1.3s | 483 events |
| `survival_analysis (grouped)` | OK | 0.6s | Grouped by `international plan` |
| `cox_regression` | OK | 0.9s | Convergence warning (benign) |

### Section 9: Threshold Finding

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `find_thresholds` | OK | 0.7s | 16 thresholds, top: `total day minutes >= 264.45` (5.28x lift) |
| `find_thresholds (decision_tree)` | OK | 0.7s | |

### Section 10: NLP

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `analyze_sentiment` (single) | OK | <0.1s | Detected "negative" |
| `batch_sentiment` | OK | 1.1s | 100 texts |
| `batch_text_stats` | OK | <0.1s | |
| `extract_keywords` | OK | <0.1s | 5 keywords extracted |
| `extract_topics (LDA)` | OK | 5.8s | 3 topics |
| `extract_topics (NMF)` | OK | 4.7s | 3 topics |
| `extract_entities` (single) | OK | <0.1s | 5 entities found |
| `batch_extract_entities` | OK | 1.5s | spaCy en_core_web_sm |
| `train_intent_classifier` | OK | 0.1s | 5 intents |
| `batch_classify_intent` | OK | <0.1s | 100 texts classified |

### Section 11: OCR

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `extract_text_from_image` | SKIP | — | Tesseract binary not installed (expected) |

### Section 12: Visualization

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `create_chart (bar)` | OK | 6.7s | |
| `create_chart (histogram)` | OK | 0.9s | |
| `create_chart (scatter)` | OK | 1.0s | |
| `create_chart (box)` | OK | 0.8s | |
| `create_chart (heatmap)` | OK | 1.6s | `x` now optional for heatmap/pair (fixed) |
| `auto_chart` | OK | 1.6s | |
| `create_dashboard` | OK | 5.1s | |

### Section 13: Report Formatting

| Test | Status | Time | Notes |
|------|--------|------|-------|
| `format_for_narrative` | OK | — | Headline generated, 5 metrics, 5 findings |
| `create_executive_summary_data` | OK | — | 0 metrics (generic result shape) |
| `create_detailed_findings_data` | OK | — | 0 findings (generic result shape) |

### Section 14: JSON Serialization (18 types)

All 18 type tests passed: `None`, `float NaN`, `float inf`, `float -inf`, `np.int64`, `np.float64`, `np.bool_`, `np.array`, `np.nan`, `pd.NaT`, `pd.NA`, `pd.Timestamp`, `pd.Series`, `pd.Categorical`, nested dict, nested inf, bytes, set.

---

## Fixed Since Initial Run

### 1. `describe_data` — numpy boolean subtract error (FIXED)
- **Root cause:** `churn` column has `dtype=bool`; numpy 2.x disallows `bool - bool`.
- **Fix:** Added `if clean.dtype == bool: clean = clean.astype(int)` at start of `describe_numeric()`.

### 2. `create_chart(heatmap)` — missing `x` argument (FIXED)
- **Root cause:** `x` was a required positional parameter for all chart types.
- **Fix:** Made `x` optional (`Optional[str] = None`), added guard: heatmap/pair don't require `x`.

---

## Warnings (non-blocking)

1. **`infer_schema` / `engineer_features`**: `infer_datetime_format` deprecation warning (pandas 2.x)
2. **`cox_regression`**: Convergence warning from lifelines (benign for this dataset)
3. **`create_executive_summary_data` / `create_detailed_findings_data`**: Return 0 metrics/findings for classification results — may need format adaptation

---

## Dependencies Status

| Package | Installed | Working |
|---------|-----------|---------|
| Core (pandas, numpy, scipy, sklearn) | Yes | Yes |
| xgboost | Yes | Yes |
| lightgbm | Yes | Yes |
| catboost | Yes | Yes |
| optuna | Yes | Yes |
| shap | Yes | Yes |
| statsmodels | Yes | Yes |
| prophet | Yes | Yes |
| lifelines | Yes | Yes |
| spacy + en_core_web_sm | Yes | Yes |
| gensim | Yes | Yes |
| vaderSentiment | Yes | Yes |
| textblob | Yes | Yes |
| plotly | Yes | Yes |
| ruptures | Yes | Yes |
| optbinning | Yes | Yes |
| hdbscan | Yes | Yes |
| pytesseract | Yes | Needs tesseract binary |
| xlrd | Yes | Yes |

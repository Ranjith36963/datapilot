# Veritly Analysis Library — Comprehensive V2 Audit

**Date:** 2026-02-06
**Auditor:** Claude Code (Opus 4.6)
**Codebase:** `sandbox/analysis-lib/veritly_analysis/` — 29 files, ~6,600 lines, 114 exported functions

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Foundation Layer (utils.py, \_\_init\_\_.py)](#2-foundation-layer)
3. [Module-by-Module Deep Dive](#3-module-by-module-deep-dive)
4. [Requirements.txt Cross-Reference](#4-requirementstxt-cross-reference)
5. [Sandbox Compatibility Matrix](#5-sandbox-compatibility-matrix)
6. [JSON Serialization Stress Test](#6-json-serialization-stress-test)
7. [GPT-4 Interface Quality Assessment](#7-gpt-4-interface-quality-assessment)
8. [Cross-Cutting Concerns](#8-cross-cutting-concerns)
9. [Consolidated Bug Report](#9-consolidated-bug-report)
10. [Recommendations & Fix Priority](#10-recommendations--fix-priority)

---

## 1. Architecture Overview

### System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Dennis's Layer (OUTSIDE SANDBOX)                                │
│  ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────────┐  │
│  │ React UI │──│ Convex DB │──│ GPT-4   │──│ Code Generator   │  │
│  │ Frontend │  │ Backend   │  │ Planner │  │ writes Python    │  │
│  └──────────┘  └──────────┘  └─────────┘  └───────┬──────────┘  │
│                                                     │            │
│                              ┌───────────────────────┘           │
│                              ▼                                   │
│              ┌──────────────────────────────┐                    │
│              │   Daytona Sandbox Container  │                    │
│              │   Python 3.12  |  NO INTERNET│                    │
│              │                              │                    │
│              │   from veritly_analysis      │                    │
│              │     import profile_data      │                    │
│              │   result = profile_data(fp)  │                    │
│              │   # → {"status":"success"..} │                    │
│              │                              │                    │
│              │   upload_result(result, name) │                   │
│              │      │                       │                    │
│              │      ▼                       │                    │
│              │   /home/daytona/workspace/   │                    │
│              │   data-upload-lib/uploader.py │                   │
│              └──────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

### Skill Contract

Every public function MUST return:
```python
{"status": "success", ...}  # on success
{"status": "error", "message": "..."} # on failure
```

Every module has an `_and_upload()` variant that calls the main function + `upload_result()`.

### Dependency Graph (no circular imports)

```
utils.py (foundation — all modules import from here)
  ├── profiler.py
  ├── schema_inference.py
  ├── data_validator.py
  ├── curation.py
  ├── outlier_detection.py
  ├── feature_engineering.py
  ├── descriptive_stats.py
  ├── correlation.py
  ├── hypothesis_testing.py
  ├── effect_size.py
  ├── classifier.py
  ├── regressor.py
  ├── feature_selection.py
  ├── explainability.py (also imports from classifier via joblib artifacts)
  ├── clustering.py
  ├── dimensionality.py
  ├── time_series.py
  ├── survival_analysis.py
  ├── threshold_finder.py
  ├── sentiment.py
  ├── text_stats.py
  ├── topic_model.py
  ├── entity_extractor.py
  ├── intent_detector.py
  ├── ocr.py
  ├── charts.py
  └── report_data.py
```

No circular dependencies. Clean leaf-module structure.

---

## 2. Foundation Layer

### 2.1 `__init__.py` (109 lines)

**Purpose:** Package entry point. Imports and re-exports 114 functions from 28 modules.

**Analysis:**
- Clean flat import structure — every function imported explicitly
- `__all__` matches imports exactly (verified)
- **Gap:** 3 functions from `effect_size.py` are NOT exported: `hedges_g`, `relative_risk`, `eta_squared`. They exist in the module but are only reachable via the `calculate_effect_size()` dispatcher. This is intentional (dispatcher pattern) but worth documenting.
- **Gap:** `report_data.py` functions (`format_for_narrative`, `create_executive_summary_data`, `create_detailed_findings_data`) don't follow the `_and_upload()` pattern — intentional since they format results, not produce them.

### 2.2 `utils.py` (220 lines)

**Purpose:** Shared file I/O, JSON serialization, DataFrame helpers, logging, upload mechanism.

#### Functions:

**`setup_logging(name)` — line 23**
- Creates one StreamHandler per logger name; idempotent via `if not logger.handlers` check.
- Correct.

**`log_execution(func_name, params, result)` — line 37**
- **DEAD CODE.** Never called by any module. No module imports `log_execution`.
- Truncates param values to 200 chars — good practice if it were used.

**`load_data(file_path)` — line 59**
- **BUG [CRITICAL]: Line 81** — `.xls` files use `engine="openpyxl"` but openpyxl cannot read legacy `.xls` format. Needs `engine="xlrd"` for `.xls`, `engine="openpyxl"` for `.xlsx`.
  ```python
  # Current (broken for .xls):
  elif ext in (".xlsx", ".xls"):
      return pd.read_excel(p, engine="openpyxl")

  # Fix:
  elif ext == ".xlsx":
      return pd.read_excel(p, engine="openpyxl")
  elif ext == ".xls":
      return pd.read_excel(p, engine="xlrd")
  ```
- Does NOT catch encoding issues for CSV (no `encoding` parameter, no fallback).
- Raises `FileNotFoundError` and `ValueError` instead of returning `{"status": "error"}`. This is fine since every caller wraps in `try/except`.

**`save_data(df, file_path)` — line 90**
- Creates parent dirs automatically — good.
- Falls back to CSV for unknown extensions — good.
- `.xls` save uses openpyxl — this actually works for writing (openpyxl writes xlsx format).

**`upload_result(result, filename)` — line 121**
- **WARNING: Line 130** — `sys.path.append(...)` called on EVERY invocation. This appends a duplicate path entry each time. Should use `if path not in sys.path:` guard.
- Gracefully falls back to a warning log if not in sandbox — correct behavior.
- No return value / no error status returned — the caller doesn't know if upload failed.

**`get_numeric_columns(df)` — line 145**
- Simple `select_dtypes(include=[np.number])`. Correct.

**`get_categorical_columns(df, threshold=50)` — line 150**
- Includes object/category columns PLUS numeric columns with ≤50 unique values.
- **NOTE:** threshold=50 is aggressive — columns like `age`, `quantity` might be misclassified as categorical. This is a design choice, not a bug.

**`get_datetime_columns(df)` — line 164**
- Only returns columns already typed as datetime; doesn't detect string dates. That's fine — schema_inference handles detection.

**`safe_json_serialize(obj)` — line 173**
- Handles: dict, list, tuple, np.integer, np.floating, np.bool_, np.ndarray, pd.Series, pd.DataFrame, pd.Timestamp, np.datetime64, pd.Categorical, bytes, set, np.generic, NaN, None.
- **BUG [CRITICAL]:** Does NOT handle `float('inf')` or `float('-inf')`. When `json.dumps()` encounters infinity, it raises `ValueError` (with default settings). Many statistical calculations can produce infinity (e.g., division by zero producing `inf` lift values in threshold_finder.py:63).
- **Missing types:** `pd.Timedelta`, `pd.Interval`, `datetime.date`, `datetime.time`, `Decimal`.
- **Unused imports at module level:** `json`, `os` — imported but never used.

---

## 3. Module-by-Module Deep Dive

### 3.1 `profiler.py` (263 lines) — Data Understanding

**Role:** THE entry point for any dataset. GPT-4 calls `profile_data()` FIRST.

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_semantic_type(series, col_name)` | 31-73 | N/A (helper) |
| `_column_profile(series, col_name)` | 76-120 | N/A (helper) |
| `profile_data(file_path)` | 127-210 | Yes |
| `detect_target_candidates(file_path)` | 213-255 | Yes |
| `profile_and_upload(file_path, output_name)` | 258-262 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 65** — `pd.to_datetime(sample, infer_datetime_format=True)` — `infer_datetime_format` is deprecated since pandas 2.0 and removed in 2.2. Should use `format="mixed"` or `format="ISO8601"`.
2. **Quality Score formula** (line 156): `completeness - min(dup_penalty, 20)`. The cap at 20 means a dataset that's 100% duplicates gets a score of 80. Consider capping at a higher penalty.
3. `_semantic_type()` uses simple heuristics — may misclassify postal codes (numeric but categorical) or currency strings. Acceptable for a first pass.

**GPT-4 Interface Quality:** EXCELLENT. Returns `overview`, `columns`, `quality_score`, `warnings`, `recommendations`, `correlations` — everything GPT-4 needs to plan next steps.

---

### 3.2 `schema_inference.py` (197 lines) — Data Understanding

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_match_pattern(values, regex, min_match)` | 42-48 | N/A (helper) |
| `_is_ordinal(values)` | 51-54 | N/A (helper) |
| `_infer_semantic_type(series, col_name)` | 61-127 | N/A (returns tuple) |
| `infer_schema(file_path)` | 134-161 | Yes |
| `detect_datetime_columns(df)` | 164-182 | N/A (returns list) |
| `detect_categorical_threshold(df, column, threshold)` | 185-189 | N/A (returns bool) |
| `infer_and_upload(file_path, output_name)` | 192-196 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 92** — Same `infer_datetime_format=True` deprecation as profiler.
2. **WARNING: Line 178** — Same deprecation in `detect_datetime_columns()`.
3. Semantic types detected: `unknown, boolean, id, datetime, email, phone, url, numeric_discrete, numeric_continuous, categorical_ordinal, categorical_nominal, text`. Good coverage.
4. `_is_ordinal()` checks against a fixed keyword set (low/medium/high/poor/fair/good/excellent, etc.) — reasonable heuristic.

**GPT-4 Interface Quality:** GOOD. Returns per-column confidence scores and suggested_transformations.

---

### 3.3 `data_validator.py` (270 lines) — Data Understanding

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `check_nulls(df)` | 29-47 | No (helper, returns dict without status) |
| `check_duplicates(df, subset)` | 50-60 | No (helper) |
| `check_outliers(df, method, threshold)` | 63-105 | No (helper) |
| `validate_data(file_path, rules)` | 112-262 | **BROKEN** |
| `validate_and_upload(file_path, output_name, rules)` | 265-269 | Inherits broken |

**Bugs/Warnings:**
1. **BUG [CRITICAL]: Line 245** — Returns `"status": "pass"` or `"status": "fail"` instead of `"status": "success"`. This VIOLATES the skill contract. Every consumer (GPT-4, Dennis's layer, report_data.py) checks `status == "success"` to determine if the call worked.
   ```python
   # Current (broken):
   status = "pass" if failed == 0 else "fail"

   # Fix — use "success" always, convey pass/fail in a separate field:
   status = "success"
   result["validation_passed"] = failed == 0
   ```
2. Quality score (line 244): `round(passed / total * 100, 1)` — this is a check-pass percentage, NOT the same quality_score as profiler.py. Naming collision could confuse GPT-4.

**GPT-4 Interface Quality:** POOR due to contract violation. GPT-4 will think the call failed.

---

### 3.4 `curation.py` (185 lines) — Data Preparation

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `standardize_column_names(df)` | 31-44 | N/A (returns tuple) |
| `impute_missing(df, strategy)` | 47-107 | N/A (returns tuple) |
| `curate_dataframe(file_path, ...)` | 114-177 | Yes |
| `curate_and_upload(file_path, output_name, ...)` | 180-184 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 83-84** — KNN imputation (`KNNImputer`) without prior scaling. KNNImputer uses Euclidean distance, so columns with large ranges will dominate. Should scale before imputing, then inverse-transform.
2. **WARNING: Line 163** — Always saves to `curated_data.csv`. If called multiple times with different strategies, it silently overwrites the previous output.
3. Line 144-147: Yes/no → 1/0 conversion — handles case insensitivity correctly via `.lower()`.
4. `standardize_column_names()` strips special chars and normalizes — good.

**GPT-4 Interface Quality:** GOOD. Returns `input_shape`, `output_shape`, `changes` log, `output_path`, `curated_data_sample`.

---

### 3.5 `outlier_detection.py` (248 lines) — Data Preparation

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_detect_iqr(df, columns, threshold)` | 29-59 | N/A (helper) |
| `_detect_zscore(df, columns, threshold)` | 62-88 | N/A (helper) |
| `_detect_isolation_forest(df, columns, contamination)` | 91-107 | N/A (helper) |
| `_detect_lof(df, columns, contamination)` | 110-126 | N/A (helper) |
| `_detect_dbscan(df, columns, eps)` | 129-144 | N/A (helper) |
| `detect_outliers(file_path, method, columns, contamination)` | 151-202 | Yes |
| `flag_anomalies(df, columns, method)` | 205-224 | N/A (returns DataFrame) |
| `get_outlier_summary(file_path)` | 227-240 | Yes |
| `detect_and_upload(file_path, output_name, ...)` | 243-247 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 195** — `outlier_scores: scores.tolist()` returns ALL scores for ALL rows. For a 1M-row dataset, this is a 1M-element JSON array. Should truncate or summarize.
2. **WARNING: Line 194** — `outlier_indices` capped at 500 (good), but `outlier_scores` is not capped.
3. Five methods available: IQR, z-score, Isolation Forest, LOF, DBSCAN — good coverage.

**GPT-4 Interface Quality:** GOOD, except the unbounded `outlier_scores` array could cause JSON size issues.

---

### 3.6 `feature_engineering.py` (232 lines) — Data Preparation

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `create_date_features(df, date_column)` | 30-42 | N/A (returns DataFrame) |
| `create_interaction_features(df, col1, col2)` | 45-53 | N/A (returns DataFrame) |
| `bin_numeric(df, column, bins, strategy)` | 56-72 | N/A (returns DataFrame) |
| `encode_categorical(df, column, method)` | 75-104 | N/A (returns DataFrame) |
| `engineer_features(file_path, operations)` | 111-224 | Yes |
| `engineer_and_upload(file_path, output_name, ...)` | 227-231 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 216** — `load_data(file_path)` called TWICE. Once at line 123 for processing, and again at line 216 just to count original columns. Should save original column count at the start.
   ```python
   # Line 216 (wasteful):
   "original_columns": len(load_data(file_path).columns),
   ```
2. **WARNING: Line 147** — `infer_datetime_format=True` deprecation (same as profiler/schema).
3. Date features extracted: year, month, day, weekday, quarter, is_weekend, day_of_year — comprehensive.
4. Interaction features limited to top 3 correlated pairs (0.3 < r < 0.95) — sensible guard.

**GPT-4 Interface Quality:** GOOD. Returns `features_created` list with name, type, source_columns, description.

---

### 3.7 `descriptive_stats.py` (189 lines) — Statistics

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `describe_numeric(df, column)` | 32-59 | No (returns dict without status) |
| `describe_categorical(df, column)` | 62-91 | No (returns dict without status) |
| `describe_data(file_path, columns)` | 98-129 | Yes |
| `compare_groups(file_path, group_column, value_column)` | 132-181 | Yes |
| `describe_and_upload(file_path, output_name, ...)` | 184-188 | Yes |

**Bugs/Warnings:**
1. **BUG [MEDIUM]: Line 159** — `if overall_mean and overall_mean != 0:` — This is a falsy check. If `overall_mean == 0.0`, this evaluates to `False`, and `diff_from_overall_pct` is silently skipped. Should be `if overall_mean is not None and overall_mean != 0:`.
2. **WARNING: Line 41** — Shapiro test: `if len(sample) <= 5000` condition. `sample` is already capped at 5000 by line 39, so this condition is always True. The else branch is dead code. Harmless but confusing.
3. Shannon entropy calculation (line 75): Uses `+ 1e-12` to avoid log(0) — correct.

**GPT-4 Interface Quality:** GOOD. Numeric summary includes normality test result.

---

### 3.8 `correlation.py` (200 lines) — Statistics

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `correlation_matrix(df, method)` | 30-33 | N/A (returns DataFrame) |
| `find_top_correlations(df, target, n)` | 36-51 | N/A (returns list) |
| `partial_correlation(df, x, y, controlling)` | 54-96 | No (returns dict without status) |
| `analyze_correlations(file_path, target, method)` | 103-192 | Yes |
| `correlate_and_upload(file_path, output_name, ...)` | 195-199 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 125-129** — Auto method selection checks normality of ONLY the first numeric column. If the first column is normal but others aren't, Pearson may be inappropriate. Should check multiple columns or default to Spearman.
2. Multicollinearity warnings for |r| > 0.8 (line 175-178) — good feature for GPT-4 to use.
3. Manual partial correlation fallback using OLS residualization (lines 76-96) — mathematically correct.

**GPT-4 Interface Quality:** EXCELLENT. Returns full correlation matrix, top pairs with p-values, target correlations, and multicollinearity warnings.

---

### 3.9 `hypothesis_testing.py` (309 lines) — Statistics

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_cohens_d(g1, g2)` | 29-34 | N/A (helper) |
| `_interpret_d(d)` | 37-45 | N/A (helper) |
| `t_test(df, group_col, value_col, group1, group2)` | 52-90 | No (returns dict without status) |
| `paired_t_test(df, col1, col2)` | 93-116 | No (returns dict without status) |
| `anova(df, group_col, value_col)` | 119-147 | No (returns dict without status) |
| `chi_square(df, col1, col2)` | 150-173 | No (returns dict without status) |
| `mann_whitney(df, group_col, value_col)` | 176-208 | No (returns dict without status) |
| `run_hypothesis_test(file_path, test_type, **kwargs)` | 215-301 | Yes (via dispatcher) |
| `test_and_upload(file_path, output_name, ...)` | 304-308 | Yes |

**Bugs/Warnings:**
1. **WARNING:** No multiple testing correction (Bonferroni, FDR). If GPT-4 runs many tests, false positives inflate.
2. **WARNING:** ANOVA has no post-hoc test (Tukey HSD). A significant ANOVA only says "some groups differ" but not which ones.
3. **WARNING:** Chi-square (line 150-173) doesn't check if expected counts are ≥5 (Cochran's rule). With sparse data, chi-square can be unreliable.
4. **WARNING:** `run_hypothesis_test()` raises `KeyError` if required kwargs are missing (e.g., calling `t_test` without `group_col`). Should validate kwargs before dispatching.
5. Individual test functions (`t_test`, `anova`, etc.) don't set `"status"` — the dispatcher adds it at line 297. Inconsistent if called directly.
6. Uses Welch's t-test by default (`equal_var=False`) — correct choice.
7. Every test returns `conclusion` as a human-readable string — excellent for GPT-4 narrative.

**GPT-4 Interface Quality:** GOOD. Clear `conclusion` strings. Missing post-hoc details for ANOVA.

---

### 3.10 `effect_size.py` (262 lines) — Statistics

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `cohens_d(group1, group2)` | 49-79 | Partial (no status on success) |
| `hedges_g(group1, group2)` | 82-102 | Partial |
| `odds_ratio(df, exposure_col, outcome_col)` | 105-132 | Partial |
| `relative_risk(df, exposure_col, outcome_col)` | 135-158 | Partial |
| `cramers_v(df, col1, col2)` | 161-176 | Partial |
| `eta_squared(df, group_col, value_col)` | 179-198 | Partial |
| `calculate_effect_size(file_path, effect_type, **kwargs)` | 205-254 | Yes (dispatcher adds status) |
| `effect_and_upload(file_path, output_name, ...)` | 257-261 | Yes |

**Bugs/Warnings:**
1. **BUG [CRITICAL]: Line 238** — `r_squared` drops NaN independently per column:
   ```python
   r, p = pearsonr(df[kwargs["col1"]].dropna(), df[kwargs["col2"]].dropna())
   ```
   If col1 has NaN at row 5 and col2 has NaN at row 10, the resulting arrays are different lengths → `pearsonr` crashes with `ValueError`. Fix: `df[[col1, col2]].dropna()`.
2. **WARNING:** `hedges_g()` (line 92) correction factor formula: `1 - 3 / (4 * (n - 2) - 1)`. Standard Hedges' correction is `1 - 3 / (4 * df - 1)` where df = n1 + n2 - 2. The code uses `n = n1 + n2`, so `4 * (n - 2) - 1 = 4*(n1+n2-2)-1` — this is correct.
3. `cohens_d()` confidence interval uses non-central t approximation — correct method.
4. Individual functions don't include `"status"` on success — the dispatcher adds it at line 250.

**GPT-4 Interface Quality:** GOOD. Every effect size includes interpretation thresholds and human-readable interpretation.

---

### 3.11 `classifier.py` (420 lines) — ML Classification

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_get_model(algorithm, random_state)` | 39-69 | N/A (helper) |
| `_prepare_data(df, target, features)` | 72-104 | N/A (helper) |
| `classify(file_path, target, features, algorithm, tune, cv_folds)` | 111-243 | Yes |
| `auto_classify(file_path, target, features)` | 246-298 | Yes |
| `tune_classifier(file_path, target, algorithm, n_trials)` | 301-372 | Yes |
| `predict(model_path, file_path)` | 375-412 | Yes |
| `classify_and_upload(file_path, output_name, ...)` | 415-419 | Yes |

**Bugs/Warnings:**
1. **BUG [CRITICAL]: Line 390** — `predict()` creates NEW LabelEncoders instead of reusing the ones from training:
   ```python
   X[col] = LabelEncoder().fit_transform(X[col].astype(str))  # NEW encoder!
   ```
   The training LabelEncoders are created in `_prepare_data()` (line 88) and stored in `label_encoders` dict, but they are NOT saved in the model artifact. The joblib dump (line 210) saves `model, scaler, features, target_le, algorithm` — NO `label_encoders`. This means predictions on new data will use different integer mappings than training → **wrong predictions**.

   **Fix:** Save `label_encoders` in the joblib artifact and reuse them in `predict()`:
   ```python
   # In classify(), add to joblib.dump:
   joblib.dump({"model": model, "scaler": scaler, "features": used_features,
                 "target_le": target_le, "label_encoders": label_encoders, ...}, model_path)

   # In predict(), reuse:
   label_encoders = data.get("label_encoders", {})
   for col in X.select_dtypes(include=["object", "category"]).columns:
       if col in label_encoders:
           X[col] = label_encoders[col].transform(X[col].fillna("__MISSING__").astype(str))
   ```

2. **BUG [CRITICAL]: Line 361** — `tune_classifier()` finds best_params via Optuna but DISCARDS them:
   ```python
   study.optimize(objective, n_trials=n_trials)
   best_params = study.best_params  # e.g., {"n_estimators": 250, "max_depth": 12}

   # BUT: calls classify() with DEFAULT params:
   result = classify(file_path, target, algorithm=algorithm)  # ← uses default n_estimators=100!
   ```
   The `best_params` are returned in the response dict but never applied. The model saved to disk was trained with defaults, not the tuned hyperparameters.

3. **WARNING: Line 59** — `use_label_encoder=False` is deprecated in XGBoost ≥1.6 and removed in ≥2.0. Since requirements.txt specifies `xgboost>=2.0.3`, this parameter will cause a warning or error.

4. **WARNING: Line 155** — `stratify=y` in `train_test_split`. If any class has fewer than 2 samples, stratification fails. No guard for this.

5. `auto_classify()` tries: logistic, random_forest, decision_tree, xgboost, lightgbm. Does NOT try catboost, svm, naive_bayes. Acceptable for speed.

6. Model artifact saved via joblib includes: model, scaler, features, target_le, algorithm. Missing: label_encoders (bug #1).

**GPT-4 Interface Quality:** EXCELLENT. Returns metrics, cross-validation, confusion matrix, classification report, feature importance, predictions sample, hyperparameters, training time.

---

### 3.12 `regressor.py` (331 lines) — ML Regression

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_get_model(algorithm, random_state)` | 37-63 | N/A (helper) |
| `_prepare_data(df, target, features)` | 66-79 | N/A (helper) |
| `predict_numeric(file_path, target, features, algorithm, tune)` | 86-214 | Yes |
| `auto_regress(file_path, target, features)` | 217-267 | Yes |
| `tune_regressor(file_path, target, algorithm, n_trials)` | 270-323 | Yes |
| `regress_and_upload(file_path, output_name, ...)` | 326-330 | Yes |

**Bugs/Warnings:**
1. **BUG [CRITICAL]: Line 312** — Same Optuna bug as classifier:
   ```python
   result = predict_numeric(file_path, target, algorithm=algorithm)  # DEFAULT params!
   ```
   `study.best_params` found but never applied to the final model.

2. **WARNING: Line 74** — Same LabelEncoder-not-saved issue as classifier. However, regressor doesn't have a separate `predict()` function — predictions are only done in `predict_numeric()` itself. Less severe since predictions happen in the same function call, but the saved model artifact can't be reused for prediction on new data via `explainability.py`.

3. **WARNING: Line 172** — Shapiro test on residuals: `sp_stats.shapiro(residuals[:5000]) if len(residuals) <= 5000`. The conditional cap `[:5000]` is applied even when `len(residuals) <= 5000` — no bug, just redundant.

4. Metrics: R², RMSE, MAE, MAPE, explained_variance, residual analysis with normality test — comprehensive.

**GPT-4 Interface Quality:** EXCELLENT. Includes residual analysis and normality test for model diagnostics.

---

### 3.13 `feature_selection.py` (238 lines) — ML Classification

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_prepare_Xy(df, target, features)` | 30-45 | N/A (helper) |
| `rfe_selection(df, target, n_features)` | 52-72 | N/A (returns list) |
| `shap_importance(df, target)` | 75-109 | N/A (returns list) |
| `select_features(file_path, target, method, n_features)` | 116-230 | Yes |
| `select_and_upload(file_path, output_name, ...)` | 233-237 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 38** — Same standalone LabelEncoder pattern (creates new encoders). Not a bug here since feature_selection trains its own model — but the encoding may differ from classifier.py's encoding.
2. **WARNING:** `rfe_selection()` always uses `RandomForestClassifier`. If the target is regression, this will fail. No check for task type.
3. 8 methods supported: auto, rfe, mutual_info, chi2, f_classif, tree_importance, lasso, shap — excellent coverage.

**GPT-4 Interface Quality:** EXCELLENT. Returns ranking, selected/eliminated features, and rationale.

---

### 3.14 `explainability.py` (220 lines) — ML Explainability

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_load_model(model_path)` | 29-32 | N/A (helper) |
| `_prepare_X(df, features, scaler)` | 35-45 | N/A (helper) |
| `explain_model(model_path, file_path)` | 52-134 | Yes |
| `explain_prediction(model_path, file_path, row_index)` | 137-212 | Yes |
| `explain_and_upload(model_path, file_path, output_name, ...)` | 215-219 | Yes |

**Bugs/Warnings:**
1. **BUG [CRITICAL]: Line 40** — Same LabelEncoder bug as `predict()`:
   ```python
   X[col] = LabelEncoder().fit_transform(X[col].fillna("__MISSING__").astype(str))
   ```
   Creates NEW encoders instead of reusing training ones. SHAP values will be computed on incorrectly encoded data → **wrong explanations**.

2. **BUG [CRITICAL]: Line 68** — Always uses `shap.TreeExplainer(model)`:
   ```python
   explainer = shap.TreeExplainer(model)
   ```
   This works for tree-based models (RandomForest, XGBoost, LightGBM, DecisionTree, CatBoost) but CRASHES for:
   - `LogisticRegression` → should use `shap.LinearExplainer`
   - `SVM` → should use `shap.KernelExplainer`
   - `NaiveBayes` → should use `shap.KernelExplainer`

   The ImportError fallback (line 112-131) handles SHAP not installed, but there's no fallback for TreeExplainer crashing on non-tree models.

3. Line 72: For binary classification, always uses `shap_values[1]` (class 1). Correct for standard binary SHAP.
4. Interaction detection via SHAP correlation (lines 89-102) — clever approach.

**GPT-4 Interface Quality:** GOOD when it works. The explainer crash for linear/SVM models is a significant issue.

---

### 3.15 `clustering.py` (251 lines) — ML Clustering

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_prepare_features(df, features)` | 33-41 | N/A (helper) |
| `_profile_clusters(df, labels, features)` | 44-92 | N/A (helper) |
| `find_clusters(file_path, n_clusters, algorithm, features)` | 99-186 | Yes |
| `optimal_clusters(file_path, max_k, features)` | 189-230 | Yes |
| `describe_clusters(df, cluster_column)` | 233-243 | Yes |
| `cluster_and_upload(file_path, output_name, ...)` | 246-250 | Yes |

**Bugs/Warnings:**
1. **BUG [MEDIUM]: Line 213** — Elbow detection finds the biggest absolute drop in inertia, NOT the elbow point:
   ```python
   diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
   elbow_k = diffs.index(max(diffs)) + 2
   ```
   The biggest drop is almost always between k=2 and k=3 (first iteration), so this always recommends k=2. The elbow method should find where the rate of decrease changes most (second derivative). However, `recommended_k` uses silhouette (line 218), which is more reliable, so the practical impact is limited.

2. **WARNING: Line 180** — `cluster_assignments: labels.tolist()` returns ALL cluster labels for every row. For large datasets, this bloats the JSON response.

3. 5 algorithms: kmeans, dbscan, hdbscan, hierarchical, gmm — good coverage.
4. `_profile_clusters()` auto-generates label suggestions (e.g., "High Revenue Group") — excellent for GPT-4 narrative.

**GPT-4 Interface Quality:** EXCELLENT. Cluster profiles with auto-labels, distinguishing features, and quality metrics.

---

### 3.16 `dimensionality.py` (177 lines) — ML Clustering

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `reduce_dimensions(file_path, method, n_components, features)` | 26-138 | Yes |
| `pca_analysis(file_path, n_components)` | 141-143 | Yes |
| `visualize_2d(file_path, method, color_by)` | 146-169 | Yes |
| `reduce_and_upload(file_path, output_name, ...)` | 172-176 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 99** — t-SNE perplexity: `perplexity=min(30, len(X_scaled) - 1)`. If `len(X_scaled)` is 1, perplexity = 0, which causes a ValueError. Should be `max(1, min(30, len(X_scaled) - 1))`. Realistically, nobody runs t-SNE on <5 samples, but defensive coding matters.
2. **WARNING: Line 155** — `visualize_2d()` reads back the CSV file it just wrote via `reduce_dimensions()`. This double I/O is wasteful. Could return the array directly.
3. PCA auto-mode retains 95% variance — good default.
4. 5 methods: PCA, Truncated SVD, Factor Analysis, t-SNE, UMAP — comprehensive.

**GPT-4 Interface Quality:** GOOD. PCA returns top feature loadings per component.

---

### 3.17 `time_series.py` (331 lines) — Time Series

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_detect_freq(dates)` | 29-41 | N/A (helper) |
| `analyze_time_series(file_path, date_column, value_column, freq)` | 48-141 | Yes |
| `forecast(file_path, date_column, value_column, periods, method)` | 144-277 | Yes |
| `detect_change_points(file_path, date_column, value_column)` | 280-323 | Yes |
| `timeseries_and_upload(file_path, output_name, ...)` | 326-330 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 215** — ARIMA hardcoded to order (1,1,1). No auto_arima, no model selection, no AIC comparison. This is a "good enough" default but may produce poor forecasts for many series.
2. **WARNING: Line 236** — `ExponentialSmoothing(values.values, trend="add", seasonal=None)` — always uses `seasonal=None` even when seasonality was detected in `analyze_time_series()`. The forecast function doesn't use the seasonality detection results.
3. **WARNING: Line 97-100** — Seasonality period map: `{"daily": 7, ...}`. For daily data, period=7 assumes weekly seasonality. Monthly data with period=12 is fine. Hourly data is not handled.
4. Forecast methods: prophet, arima, exponential_smoothing, naive — auto selects prophet first.
5. Change point detection uses ruptures PELT algorithm — good choice.

**GPT-4 Interface Quality:** GOOD. Trend direction, stationarity, seasonality strength, forecast with confidence intervals.

---

### 3.18 `survival_analysis.py` (208 lines) — Survival Analysis

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `survival_analysis(file_path, duration_column, event_column, group_column)` | 23-131 | Yes |
| `cox_regression(file_path, duration_column, event_column, features)` | 134-200 | Yes |
| `survival_and_upload(file_path, output_name, ...)` | 203-207 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 78** — Hardcoded survival-at time points: 30, 90, 180, 365. Labels say "30_days", "90_days" etc., but the duration column might be in months, weeks, or hours. No unit detection or parameter.
2. **WARNING: Line 155** — Cox regression excludes categorical covariates: `use_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]`. CoxPH can handle categoricals via dummy encoding. Important covariates could be lost.
3. **WARNING:** No proportional hazards assumption check (Schoenfeld residuals). A violated PH assumption invalidates the Cox model.
4. Log-rank test only compares first two groups (line 104-107) even if there are more.
5. KM curve data capped at 200 points (line 122) — good for JSON size.

**GPT-4 Interface Quality:** GOOD. Survival curve data, median survival, group comparison with log-rank.

---

### 3.19 `threshold_finder.py` (343 lines) — Threshold Finding

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_brute_force_threshold(df, feature, target, ...)` | 31-75 | N/A (helper) |
| `_decision_tree_threshold(df, feature, target)` | 78-115 | N/A (helper) |
| `_optbinning_threshold(df, feature, target)` | 118-161 | N/A (helper) |
| `_change_point_threshold(df, feature, target)` | 163-205 | N/A (helper) |
| `find_optimal_split(df, feature, target, method)` | 212-230 | N/A (helper) |
| `find_thresholds(file_path, target, features, method)` | 233-291 | Yes |
| `threshold_confidence_interval(df, feature, target, threshold, n_bootstrap)` | 294-335 | Yes |
| `thresholds_and_upload(file_path, output_name, ...)` | 338-342 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 59** — `_brute_force_threshold()` only checks `impact = above_rate - below_rate > 0`. This means it ONLY finds "above > below" thresholds. If the pattern is "below > above" (rate decreases above threshold), it's missed. Should check `abs(impact)`.
2. **WARNING: Line 63** — `float("inf")` can be returned as lift when `below_rate` is near zero. Combined with the `safe_json_serialize` infinity bug, this could crash JSON encoding. The `min(float(multiplier), 100)` cap at 100 helps but `float("inf") * anything` paths aren't all capped.
3. Bootstrap CI (lines 294-335) — well-implemented with 1000 samples, percentile method.
4. 4 methods: decision_tree, brute_force, optbinning, change_point — with fallback to decision_tree.
5. Automatic insight generation (line 269-274) — excellent for GPT-4 narrative.

**GPT-4 Interface Quality:** EXCELLENT. Human-readable insights like "Customers with Monthly Charges >= 70.5 have 45.2% rate vs 12.1% below (3.7x higher)".

---

### 3.20 `sentiment.py` (152 lines) — NLP

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `analyze_sentiment(text, method)` | 23-80 | Partial (no "status" on success) |
| `batch_sentiment(file_path, text_column, method)` | 83-144 | Yes |
| `sentiment_and_upload(file_path, output_name, ...)` | 147-151 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 28-29** — Creates a new `SentimentIntensityAnalyzer()` instance for EVERY text in `batch_sentiment()` (called via `analyze_sentiment()`). VADER's initializer loads a lexicon file each time. Should create ONE instance outside the loop.
2. **WARNING:** `analyze_sentiment()` doesn't return `"status": "success"` on success. Only returns `"status": "error"` on failure. Contract violation (minor since batch_sentiment adds status).
3. VADER threshold: compound >= 0.05 positive, <= -0.05 negative. Standard thresholds.
4. TextBlob threshold: polarity > 0.1 positive, < -0.1 negative. Standard.

**GPT-4 Interface Quality:** GOOD. Distribution, average compound, most positive/negative examples.

---

### 3.21 `text_stats.py` (177 lines) — NLP

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_tokenize(text)` | 41-43 | N/A (helper) |
| `_sentences(text)` | 46-48 | N/A (helper) |
| `analyze_text(text)` | 51-109 | No (missing "status") |
| `batch_text_stats(file_path, text_column)` | 112-147 | Yes |
| `extract_keywords(text, n)` | 150-169 | N/A (returns list) |
| `text_stats_and_upload(file_path, output_name, ...)` | 172-176 | Yes |

**Bugs/Warnings:**
1. **WARNING:** `analyze_text()` doesn't return `"status"` key on success. Contract violation.
2. Self-contained stopwords (77 words, lines 24-38) — no NLTK dependency needed. Smart design for no-internet sandbox.
3. Readability: Uses `textstat` library if available, falls back to manual Flesch approximation — good.
4. TF-IDF keyword extraction via sklearn — works offline.

**GPT-4 Interface Quality:** GOOD. Readability scores with interpretation.

---

### 3.22 `topic_model.py` (138 lines) — NLP

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `extract_topics(file_path, text_column, n_topics, method)` | 26-112 | Yes |
| `assign_topics(df, text_column, model_path)` | 115-131 | N/A (returns DataFrame) |
| `topics_and_upload(file_path, output_name, ...)` | 133-137 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 56** — LDA `max_iter=20` is very low. Standard is 100-500 iterations. Topics may not converge at 20 iterations, producing unstable results.
2. **WARNING: Line 120** — `assign_topics()` hardcodes `n_components=5` and ignores the `model_path` parameter entirely. It always trains a NEW LDA model with 5 topics. The `model_path` parameter is accepted but never used.
3. Coherence scoring via gensim (optional) — good but gensim is a heavy dependency.
4. Topic auto-labeling from top 3 words — good for GPT-4 narrative.

**GPT-4 Interface Quality:** GOOD. Topic words with weights, document distribution, optional coherence score.

---

### 3.23 `entity_extractor.py` (177 lines) — NLP

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_get_nlp()` | 32-44 | N/A (helper) |
| `_spacy_ner(text)` | 47-56 | N/A (helper) |
| `_regex_ner(text)` | 70-81 | N/A (helper) |
| `extract_entities(text)` | 88-104 | No (missing "status" on success) |
| `batch_extract_entities(file_path, text_column)` | 107-169 | Yes |
| `entities_and_upload(file_path, output_name, ...)` | 172-176 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 150** — `batch_extract_entities()` runs NER TWICE per row: once in the initial loop (lines 118-122) to collect all entities, and again in the save loop (line 150) to build the entities column. Should store results from the first pass.
2. **WARNING:** `extract_entities()` doesn't return `"status": "success"` on success. Contract violation.
3. spaCy model lazy-loading with global `_NLP` singleton — correct pattern for sandbox.
4. Falls back from `en_core_web_sm` → `en_core_web_md` → regex — good graceful degradation.
5. Regex fallback detects: DATE, MONEY, EMAIL, PERCENT — limited but useful.

**GPT-4 Interface Quality:** GOOD. Entity summary by type, most mentioned people/orgs/locations.

---

### 3.24 `intent_detector.py` (186 lines) — NLP

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_heuristic_intent(text)` | 47-64 | No (missing "status") |
| `detect_intent(text, intents)` | 71-76 | No (missing "status") |
| `train_intent_classifier(file_path, text_column, intent_column)` | 79-117 | Yes |
| `batch_classify_intent(file_path, text_column, model_path)` | 120-178 | Yes |
| `intent_and_upload(file_path, output_name, ...)` | 181-185 | Yes |

**Bugs/Warnings:**
1. **WARNING: Line 71** — `detect_intent()` accepts `intents` parameter but NEVER uses it. The parameter is silently ignored. Should either use it to filter _INTENT_KEYWORDS or remove it from the signature.
2. **WARNING: Line 98** — `cv=min(5, len(set(labels)))` — if there's only 1 unique intent class, cv=1 which causes a cross-validation error. Should require ≥2 classes.
3. 5 heuristic intent categories: complaint, inquiry, feedback, request, praise — reasonable defaults.
4. Pipeline: TF-IDF + LogisticRegression — lightweight and effective.

**GPT-4 Interface Quality:** GOOD. Intent distribution with confidence scores.

---

### 3.25 `ocr.py` (99 lines) — Image

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `extract_text_from_image(image_path, language)` | 21-55 | Yes |
| `batch_ocr(folder_path, language)` | 58-91 | Yes |
| `ocr_and_upload(image_path, output_name, ...)` | 94-98 | Yes |

**Bugs/Warnings:**
1. **WARNING:** No image preprocessing (deskew, binarization, denoising). Raw images passed directly to Tesseract. Quality will be poor for scanned documents.
2. **WARNING:** Tesseract binary must be installed at the OS level in the sandbox container. Not just a pip package — requires `apt-get install tesseract-ocr`.
3. Confidence via `image_to_data()` — good quality signal.
4. Batch OCR processes all images in a folder — simple and effective.

**GPT-4 Interface Quality:** GOOD. Extracted text, confidence score, word count.

---

### 3.26 `charts.py` (257 lines) — Visualization

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `_save_fig(fig, file_path, chart_name)` | 28-35 | N/A (helper) |
| `create_chart(file_path, chart_type, x, y, hue, title, ...)` | 38-186 | Yes |
| `auto_chart(file_path, x, y)` | 189-210 | Yes |
| `create_dashboard(file_path, charts)` | 213-249 | Yes |
| `chart_and_upload(file_path, output_name, ...)` | 252-256 | Yes |

**Bugs/Warnings:**
1. No significant bugs found.
2. 13 chart types: bar, barh, line, scatter, histogram, box, violin, heatmap, pie, area, density, count, pair.
3. Dual output: matplotlib PNG + optional Plotly HTML — excellent.
4. `auto_chart()` smartly selects chart type based on data types — good for GPT-4.
5. `create_dashboard()` auto-generates charts from data types — good default.

**GPT-4 Interface Quality:** EXCELLENT. Chart paths returned for embedding in reports.

---

### 3.27 `report_data.py` (245 lines) — Output Formatting

**Functions:**
| Function | Lines | Returns `status`? |
|---|---|---|
| `format_for_narrative(results, context)` | 19-56 | No (formats other results) |
| `create_executive_summary_data(results)` | 59-71 | Partial |
| `create_detailed_findings_data(results)` | 74-87 | Partial |
| `_generate_headline(results, context)` | 94-126 | N/A (helper) |
| `_extract_metrics(results)` | 129-159 | N/A (helper) |
| `_extract_findings(results)` | 162-198 | N/A (helper) |
| `_extract_recommendations(results)` | 201-221 | N/A (helper) |
| `_extract_viz_paths(results)` | 224-244 | N/A (helper) |

**Bugs/Warnings:**
1. **WARNING:** `_generate_headline()` only handles ~6 result patterns (algorithm, n_clusters, quality_score, thresholds, trend, distribution). Missing patterns for: survival analysis, hypothesis testing, correlation, feature selection, NLP results. Falls back to "Analysis complete" — generic but not helpful.
2. `_extract_viz_paths()` checks 12 known path keys — comprehensive.
3. `_extract_findings()` handles warnings, thresholds, feature importance, cluster profiles — good coverage.

**GPT-4 Interface Quality:** GOOD. Structured for narrative generation.

---

## 4. Requirements.txt Cross-Reference

### File: `sandbox/analysis-lib/requirements.txt` (59 lines)

| Package | Version | Used In | Status |
|---|---|---|---|
| pandas>=2.1.0 | Core | utils, all modules | **OK** |
| numpy>=1.26.0 | Core | utils, all modules | **OK** |
| openpyxl>=3.1.2 | Core | utils (Excel I/O) | **OK** (but .xls needs xlrd) |
| pyarrow>=14.0.1 | Core | utils (parquet) | **OK** |
| scikit-learn>=1.4.0 | ML | classifier, regressor, clustering, etc. | **OK** |
| xgboost>=2.0.3 | ML | classifier, regressor | **OK** (but use_label_encoder deprecated) |
| lightgbm>=4.3.0 | ML | classifier, regressor | **OK** |
| catboost>=1.2.2 | ML | classifier | **OK** |
| optuna>=3.5.0 | ML | classifier, regressor | **OK** |
| shap>=0.44.0 | ML | explainability, feature_selection | **OK** |
| hdbscan>=0.8.33 | Clustering | clustering | **OK** |
| scipy>=1.12.0 | Stats | hypothesis_testing, effect_size, etc. | **OK** |
| statsmodels>=0.14.1 | Stats/TS | time_series, correlation | **OK** |
| pingouin>=0.5.4 | Stats | correlation (partial_correlation) | **OK** |
| prophet>=1.1.5 | TS | time_series | **CONCERN** (see sandbox matrix) |
| **sktime>=0.26.0** | TS | **NOWHERE** | **DEAD DEPENDENCY** |
| lifelines>=0.27.8 | Survival | survival_analysis | **OK** |
| optbinning>=0.19.0 | Threshold | threshold_finder | **OK** |
| ruptures>=1.1.9 | TS/Threshold | time_series, threshold_finder | **OK** |
| textblob>=0.17.1 | NLP | sentiment | **OK** |
| vaderSentiment>=3.3.2 | NLP | sentiment | **OK** |
| **nltk>=3.8.1** | NLP | **NOWHERE** | **DEAD DEPENDENCY** |
| spacy>=3.7.2 | NLP | entity_extractor | **CONCERN** (needs model download) |
| gensim>=4.3.2 | NLP | topic_model (optional coherence) | **OK** |
| textstat>=0.7.3 | NLP | text_stats | **OK** |
| matplotlib>=3.8.2 | Viz | charts | **OK** |
| seaborn>=0.13.1 | Viz | charts | **OK** |
| plotly>=5.18.0 | Viz | charts (interactive) | **OK** |
| **pandera>=0.18.0** | DQ | **NOWHERE** | **DEAD DEPENDENCY** |
| pytesseract>=0.3.10 | Image | ocr | **OK** (needs tesseract binary) |
| pillow>=10.2.0 | Image | ocr | **OK** |
| joblib>=1.3.2 | Util | classifier, regressor, intent | **OK** |
| **tenacity>=8.2.3** | Util | **NOWHERE** | **DEAD DEPENDENCY** |
| **httpx>=0.26.0** | Util | **NOWHERE** | **DEAD + SECURITY CONCERN** |

### Missing from requirements.txt:
| Package | Used In | Impact |
|---|---|---|
| **xlrd** | utils.py (needed for .xls files) | **CRITICAL** — .xls files will fail |
| **umap-learn** | dimensionality.py | Low — handled by try/except ImportError |

### Dead Dependencies Summary:
- **sktime** — imported nowhere, never referenced. 33MB+ package with many transitive deps. Remove.
- **pandera** — data_validator.py uses custom validation, not pandera. 8MB+ package. Remove.
- **nltk** — text_stats.py uses custom stopwords, not NLTK. entity_extractor uses spacy. Remove.
- **tenacity** — retry library, never used. Remove.
- **httpx** — HTTP client library. **This is a security concern in a no-internet sandbox.** No module uses HTTP calls. Remove immediately.

---

## 5. Sandbox Compatibility Matrix

| Package | Pip Install OK? | Needs Internet at Runtime? | Needs OS-Level Install? | Pre-download Required? |
|---|---|---|---|---|
| pandas | Yes | No | No | No |
| numpy | Yes | No | No | No |
| scikit-learn | Yes | No | No | No |
| xgboost | Yes | No | No | No |
| lightgbm | Yes | No | No | No |
| catboost | Yes | No | No | No |
| optuna | Yes | No | No | No |
| shap | Yes | No | No | No |
| hdbscan | Yes | No | No | No |
| scipy | Yes | No | No | No |
| statsmodels | Yes | No | No | No |
| pingouin | Yes | No | No | No |
| **prophet** | **Problematic** | No | **Needs cmdstan/C++ compiler** | **cmdstanpy compilation** |
| lifelines | Yes | No | No | No |
| optbinning | Yes | No | No | No |
| ruptures | Yes | No | No | No |
| textblob | Yes | No | No | No |
| vaderSentiment | Yes | No | No | No |
| **spacy** | Yes | **Yes (model download)** | No | **Must pre-install `en_core_web_sm`** |
| gensim | Yes | No | No | No |
| textstat | Yes | No | No | No |
| matplotlib | Yes | No | No | No |
| seaborn | Yes | No | No | No |
| plotly | Yes | No | No | No |
| **pytesseract** | Yes | No | **Needs `tesseract-ocr` binary** | **`apt-get install tesseract-ocr`** |
| pillow | Yes | No | No | No |
| joblib | Yes | No | No | No |
| openpyxl | Yes | No | No | No |
| pyarrow | Yes | No | No | No |

### Sandbox Pre-Installation Checklist:
1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm` (needs internet DURING build, not runtime)
3. `apt-get install -y tesseract-ocr` (Dockerfile)
4. Prophet: `pip install prophet` may require `cmdstan` compilation. Consider using `prophet` with `backend='cmdstanpy'` and ensuring C++ compiler is available in the container build.
5. Remove dead dependencies to reduce image size.

---

## 6. JSON Serialization Stress Test

### `safe_json_serialize()` handles:

| Type | Handled? | Conversion |
|---|---|---|
| `None` | Yes | `None` |
| `float('nan')` | Yes | `None` |
| `np.nan` | Yes | `None` |
| `pd.NA` | Yes | `None` |
| `dict` | Yes | Recursive |
| `list` / `tuple` | Yes | Recursive |
| `np.integer` | Yes | `int()` |
| `np.floating` | Yes | `float()` |
| `np.bool_` | Yes | `bool()` |
| `np.ndarray` | Yes | `.tolist()` |
| `pd.Series` | Yes | `.tolist()` |
| `pd.DataFrame` | Yes | `.to_dict("records")` |
| `pd.Timestamp` | Yes | `str()` |
| `np.datetime64` | Yes | `str()` |
| `pd.Categorical` | Yes | `.tolist()` |
| `bytes` | Yes | `.decode("utf-8")` |
| `set` | Yes | `list()` |
| `np.generic` | Yes | `.item()` |
| **`float('inf')`** | **NO** | **CRASHES `json.dumps()`** |
| **`float('-inf')`** | **NO** | **CRASHES `json.dumps()`** |
| **`pd.Timedelta`** | **NO** | **Falls through as-is** |
| **`pd.Interval`** | **NO** | **Falls through as-is** |
| **`datetime.date`** | **NO** | **Falls through as-is** |
| **`datetime.time`** | **NO** | **Falls through as-is** |
| **`Decimal`** | **NO** | **Falls through as-is** |

### Where `float('inf')` can appear:
- `threshold_finder.py:63` — `float("inf")` lift when `below_rate` near 0
- `threshold_finder.py:104` — Same pattern in decision_tree_threshold
- `survival_analysis.py:58` — `kmf.median_survival_time_` can be `inf` when median is undefined (handled with `if not np.isinf()` check — good)
- `effect_size.py:64` — pooled std of 0 returns 0, but intermediate calculations could produce inf
- Any division by zero in statistics

### Recommended Fix:
```python
# Add to safe_json_serialize, after the NaN check:
if isinstance(obj, float) and (np.isinf(obj)):
    return None  # or "Infinity" as a string
```

---

## 7. GPT-4 Interface Quality Assessment

### Scoring (1-5):

| Module | Docstring Quality | Return Schema Clarity | Error Messages | Status Contract | Score |
|---|---|---|---|---|---|
| profiler | 5 | 5 | 5 | OK | **5/5** |
| schema_inference | 4 | 4 | 4 | OK | **4/5** |
| data_validator | 4 | 4 | 4 | **BROKEN** | **2/5** |
| curation | 4 | 4 | 4 | OK | **4/5** |
| outlier_detection | 4 | 3 (unbounded arrays) | 4 | OK | **3.5/5** |
| feature_engineering | 4 | 4 | 4 | OK | **4/5** |
| descriptive_stats | 4 | 4 | 4 | OK | **4/5** |
| correlation | 5 | 5 | 4 | OK | **5/5** |
| hypothesis_testing | 5 | 5 | 3 (KeyError) | OK | **4/5** |
| effect_size | 4 | 4 | 4 | OK | **4/5** |
| classifier | 5 | 5 | 4 | OK | **5/5** |
| regressor | 5 | 5 | 4 | OK | **5/5** |
| feature_selection | 5 | 5 | 4 | OK | **5/5** |
| explainability | 4 | 4 | 3 (crashes) | OK | **3/5** |
| clustering | 5 | 4 (big arrays) | 4 | OK | **4/5** |
| dimensionality | 4 | 4 | 4 | OK | **4/5** |
| time_series | 4 | 4 | 4 | OK | **4/5** |
| survival_analysis | 4 | 4 | 4 | OK | **4/5** |
| threshold_finder | 5 | 5 | 4 | OK | **5/5** |
| sentiment | 4 | 4 | 4 | Partial | **3.5/5** |
| text_stats | 4 | 4 | 4 | Missing | **3/5** |
| topic_model | 4 | 4 | 4 | OK | **4/5** |
| entity_extractor | 4 | 4 | 4 | Missing | **3/5** |
| intent_detector | 3 | 4 | 4 | Missing | **3/5** |
| ocr | 4 | 4 | 4 | OK | **4/5** |
| charts | 5 | 5 | 4 | OK | **5/5** |
| report_data | 4 | 4 | 4 | N/A | **4/5** |

**Average: 4.0/5** — Good overall. Key issues are the data_validator contract violation and several NLP modules missing `"status"` on success.

---

## 8. Cross-Cutting Concerns

### 8.1 Error Handling Pattern
Every module follows: `try: ... except Exception as e: return {"status": "error", "message": str(e)}`. Consistent and correct. Optional library imports use nested `try/except ImportError`.

### 8.2 Memory Concerns
- `outlier_detection.py` returns ALL outlier_scores (unbounded)
- `clustering.py` returns ALL cluster_assignments (unbounded)
- `topic_model.py` returns ALL document_topics (unbounded)
- For a 1M-row dataset, each of these produces a multi-MB JSON array

### 8.3 Security
- **httpx in requirements.txt** — HTTP client with no use in a no-internet sandbox. Remove.
- `upload_result()` uses `sys.path.append()` — acceptable in sandbox but not ideal.
- No user input sanitization needed (sandbox processes files, not web requests).
- File path handling uses `pathlib.Path` — safe against path injection in sandbox.

### 8.4 Concurrency
- No threading/multiprocessing used. All sequential.
- `n_jobs=-1` used in sklearn estimators — uses all CPU cores. May cause issues if sandbox has limited CPU. Consider capping at `n_jobs=4`.

### 8.5 Reproducibility
- `random_state=42` used consistently across all ML modules — good.
- `n_init=10` for KMeans — good default.

### 8.6 Unused Code
- `utils.py:log_execution()` — dead code, never called
- `utils.py` imports `json` and `os` — never used
- `intent_detector.py:detect_intent(intents=...)` — parameter accepted but ignored

---

## 9. Consolidated Bug Report

### CRITICAL (will produce wrong results or crash):

| # | Module | Line | Description | Impact |
|---|---|---|---|---|
| C1 | classifier.py | 390 | `predict()` creates NEW LabelEncoders instead of reusing training ones | Wrong predictions on new data |
| C2 | explainability.py | 40 | `_prepare_X()` same LabelEncoder bug | Wrong SHAP explanations |
| C3 | classifier.py | 361 | `tune_classifier()` best_params never applied to final model | Tuned hyperparameters discarded |
| C4 | regressor.py | 312 | `tune_regressor()` same Optuna bug | Tuned hyperparameters discarded |
| C5 | utils.py | 81 | `load_data()` uses openpyxl for .xls files (needs xlrd) | .xls files crash with "not xlsx" error |
| C6 | effect_size.py | 238 | `r_squared` drops NaN independently per column | Length mismatch crash |
| C7 | data_validator.py | 245 | Returns "pass"/"fail" instead of "success" | Contract violation — GPT-4 thinks call failed |
| C8 | explainability.py | 68 | Always uses TreeExplainer — crashes for linear/SVM models | Explainability broken for 3 of 8 algorithms |
| C9 | utils.py | 184 | `safe_json_serialize` doesn't handle `float('inf')` | JSON encoding crash on infinity values |

### WARNING (suboptimal but won't crash in normal use):

| # | Module | Line | Description |
|---|---|---|---|
| W1 | profiler.py | 65 | `infer_datetime_format=True` deprecated in pandas ≥2.0 |
| W2 | schema_inference.py | 92, 178 | Same deprecation (2 locations) |
| W3 | feature_engineering.py | 147 | Same deprecation |
| W4 | curation.py | 83-84 | KNN imputation without prior scaling |
| W5 | outlier_detection.py | 195 | Unbounded `outlier_scores` array |
| W6 | clustering.py | 180 | Unbounded `cluster_assignments` array |
| W7 | clustering.py | 213 | Elbow detection always picks k=2 (uses max drop, not elbow) |
| W8 | descriptive_stats.py | 159 | Falsy check `if overall_mean` fails when mean is 0.0 |
| W9 | correlation.py | 125-129 | Auto method checks only first column for normality |
| W10 | hypothesis_testing.py | 215+ | No multiple testing correction |
| W11 | hypothesis_testing.py | 119+ | ANOVA has no post-hoc test |
| W12 | hypothesis_testing.py | 150+ | Chi-square no expected count check |
| W13 | classifier.py | 59 | XGBoost `use_label_encoder=False` deprecated in ≥2.0 |
| W14 | time_series.py | 215 | ARIMA hardcoded order (1,1,1) |
| W15 | time_series.py | 236 | ExponentialSmoothing ignores detected seasonality |
| W16 | survival_analysis.py | 78 | Hardcoded time points assume days |
| W17 | survival_analysis.py | 155 | Cox regression excludes categorical covariates |
| W18 | threshold_finder.py | 59 | Brute force only finds "above > below" direction |
| W19 | sentiment.py | 28-29 | New VADER instance per text in batch |
| W20 | topic_model.py | 56 | LDA max_iter=20 too low |
| W21 | topic_model.py | 120 | `assign_topics()` ignores model_path parameter |
| W22 | entity_extractor.py | 150 | NER runs twice per row in batch |
| W23 | intent_detector.py | 71 | `intents` parameter accepted but ignored |
| W24 | dimensionality.py | 99 | t-SNE perplexity can be 0 for tiny datasets |
| W25 | dimensionality.py | 155 | `visualize_2d()` reads back CSV it just wrote |
| W26 | utils.py | 130 | `sys.path.append` on every upload call (path bloat) |
| W27 | feature_engineering.py | 216 | `load_data()` called twice (wasteful) |
| W28 | utils.py | 8-10 | Unused imports: `json`, `os` |
| W29 | utils.py | 37-52 | `log_execution()` is dead code |
| W30 | requirements.txt | 27 | sktime — dead dependency (never imported) |
| W31 | requirements.txt | 39 | nltk — dead dependency (never imported) |
| W32 | requirements.txt | 50 | pandera — dead dependency (never used) |
| W33 | requirements.txt | 58 | tenacity — dead dependency (never used) |
| W34 | requirements.txt | 59 | httpx — dead dependency + security concern |
| W35 | requirements.txt | - | xlrd MISSING (needed for .xls files) |

### STATUS CONTRACT VIOLATIONS:

| Module | Function | Issue |
|---|---|---|
| data_validator.py | `validate_data()` | Returns "pass"/"fail" instead of "success" |
| sentiment.py | `analyze_sentiment()` | No "status" on success |
| text_stats.py | `analyze_text()` | No "status" on success |
| entity_extractor.py | `extract_entities()` | No "status" on success |
| intent_detector.py | `detect_intent()` | No "status" on success |

---

## 10. Recommendations & Fix Priority

### Priority 1 — Fix Before Production (Critical Bugs):

1. **Save and reuse LabelEncoders** in classifier.py and use them in predict() and explainability.py
2. **Apply Optuna best_params** to final model in both tune_classifier() and tune_regressor()
3. **Fix .xls loading** — use xlrd engine; add xlrd to requirements.txt
4. **Fix r_squared NaN handling** — use `df[[col1, col2]].dropna()`
5. **Fix data_validator status** — return "success" with a `validation_passed` boolean
6. **Fix explainability TreeExplainer** — detect model type, use appropriate SHAP explainer
7. **Handle float('inf') in safe_json_serialize** — convert to None or string

### Priority 2 — Fix Before Beta (Warnings):

8. Replace `infer_datetime_format=True` with `format="mixed"` (4 locations)
9. Remove `use_label_encoder=False` from XGBoost constructor
10. Cap unbounded arrays (outlier_scores, cluster_assignments) at 10,000 entries
11. Fix `compare_groups()` falsy check for overall_mean
12. Add VADER singleton to batch_sentiment loop
13. Fix entity_extractor double NER run
14. Increase LDA max_iter to at least 100
15. Add missing "status" keys to NLP single-text functions

### Priority 3 — Clean Up (Before Launch):

16. Remove dead dependencies from requirements.txt (sktime, pandera, nltk, tenacity, httpx)
17. Add xlrd and umap-learn to requirements.txt
18. Remove dead code (log_execution, unused imports)
19. Fix feature_engineering.py double load_data call
20. Fix dimensionality.py double I/O in visualize_2d
21. Add proportional hazards check to Cox regression (or warning in output)
22. Add ANOVA post-hoc test (Tukey HSD)
23. Fix elbow detection algorithm (use second derivative)
24. Guard against small datasets in t-SNE perplexity

---

*End of audit. 9 critical bugs, 35 warnings, 5 contract violations identified across 29 files.*

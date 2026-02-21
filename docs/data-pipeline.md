# Data Pipeline

## Overview

DataPilot processes data through a multi-stage pipeline: upload, profiling, domain understanding, and analysis.

```
Upload → Profile → [Fingerprint/Understand] → Analyze → Narrate → Response
                                                  ↑
                                           User question or
                                           Auto-pilot step
```

## Stage 1: Upload

**Endpoint:** `POST /api/upload`
**Files:** `backend/app/services/data_service.py`, `backend/app/api/data.py`

1. Frontend sends file via multipart/form-data
2. `DataService.save_upload()` writes to `/tmp/datapilot/{session_id}/`
3. `SessionManager.create_session()` creates an `Analyst` instance
4. Analyst loads the file into a pandas DataFrame
5. Session persisted to SQLite + in-memory cache
6. Returns: session_id, shape, columns, preview

**Supported formats:** CSV, XLSX, XLS, JSON, Parquet (max 100 MB)

## Stage 2: Profiling

**File:** `engine/datapilot/data/profiler.py`

Runs automatically on upload. Characterizes the dataset:

- Column count and types (numeric, categorical, datetime, boolean, text)
- Summary statistics (mean, std, min, max, quartiles)
- Distribution analysis (skewness, kurtosis)
- Missing value percentages
- Unique value counts
- Duplicate detection

**Type detection order** (critical):
1. `is_bool_dtype` — FIRST (numpy treats bool as numeric)
2. `is_numeric_dtype`
3. `is_datetime64_any_dtype`
4. Categorical / text (by unique ratio)

## Stage 3: Domain Understanding

### Current Implementation (v2 — keyword-based)

**File:** `engine/datapilot/data/fingerprint.py` (543 lines)
**Endpoint:** `POST /api/fingerprint/{session_id}`

3-layer ensemble detection:

```
Layer 1: Column Keywords (0ms, zero cost)
    ↓ confidence < threshold
Layer 2: Value Profiling (~100ms)
    ↓ confidence < 0.6
Layer 3: LLM Confirmation (Gemini → Groq)
```

- **Layer 1:** Matches column names against keyword dictionaries for 6 domains
- **Layer 2:** Analyzes value distributions, binary targets, ID patterns
- **Layer 3:** LLM classifies from column names + sample rows (only for ambiguous cases)

Returns: `FingerprintResult` with domain, confidence, explanation, suggested_target

### Planned Implementation (v3 — LLM-first)

Per `phase2fixed.md`, this will be **rewritten** to an LLM-first approach:

```
Upload → Profile → Build Data Snapshot → LLM UNDERSTAND → LLM PLAN → EXECUTE → LLM NARRATE → LLM SUMMARIZE
```

Key changes:
- No hardcoded domains — LLM freely classifies any dataset
- No keyword dictionaries — LLM reads a data snapshot (<2000 tokens)
- Domain is free text ("telecom customer churn") not enum ("telecom_saas")
- No deterministic fallback — LLM or nothing

## Stage 4: Analysis

**Files:** `engine/datapilot/core/router.py`, `engine/datapilot/core/executor.py`

### Manual Analysis (User Questions)

```
User question
    ↓
Router (keyword priority → LLM routing → deterministic fallback)
    ↓
{skill: "analyze_correlations", params: {target: "churn"}, confidence: 0.92}
    ↓
Executor (param filtering, signature inspection)
    ↓
Skill function → result dict
```

**Router priority chain:**
1. Chart keywords → chart creation skill
2. Keyword table → direct skill match
3. Primary LLM (Groq) → skill routing with confidence
4. Fallback LLM (Gemini) → skill routing
5. Default → `profile_data`

### Auto-Pilot Analysis (D3 — planned)

```
DatasetUnderstanding (from Stage 3)
    ↓
LLM generates analysis plan (max 5 steps)
    ↓
Serial execution: analyst.ask(step.question) for each step
    ↓ 1-second delay between steps (rate limit protection)
LLM generates executive summary
```

Safety rails:
- MAX_STEPS = 5
- STEP_TIMEOUT = 30s per step
- DELAY = 1s between steps
- Total worst case: ~13 LLM calls, ~20-40 seconds

## Stage 5: Narration

**File:** `engine/datapilot/core/analyst.py`

Each analysis result gets an LLM-generated narrative:

```
Skill result dict
    ↓
_sanitize_for_narration()  ← strips base64, paths; injects column names
    ↓
FailoverProvider.generate_narrative()  ← Gemini primary, Groq fallback
    ↓
narrative text + key_points + suggestions
```

If LLM fails → template fallback (basic but functional).

## 81 Analytical Skills

Organized across 6 modules:

| Module | Skills | Examples |
|--------|--------|----------|
| `analysis/descriptive.py` | 8 | describe_data, profile_data, compare_groups |
| `analysis/correlation.py` | 3 | analyze_correlations, correlation_matrix (helper) |
| `analysis/classification.py` | 5 | classify (8 algorithms), predict_numeric |
| `analysis/anomaly.py` | 3 | detect_outliers (IQR, isolation forest, LOF) |
| `analysis/clustering.py` | 3 | find_clusters (k-means, DBSCAN, hierarchical) |
| `analysis/timeseries.py` | 4 | analyze_time_series, forecast |
| `analysis/hypothesis.py` | 3 | run_hypothesis_test, test_normality |
| `nlp/` | 5 | sentiment, entities, topics, intent, text_stats |
| `viz/charts.py` | 10+ | bar, line, scatter, histogram, box, heatmap, pie, violin, area |
| `export/` | 3 | PDF, DOCX, PPTX report generation |

Every skill returns: `{"status": "success|error", ...}` — never raw DataFrames.

## Data Flow Diagram

```
┌─────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│ Frontend │───▶│ FastAPI  │───▶│ Analyst      │───▶│ Skills   │
│ (React)  │◀───│ Backend  │◀───│ (Router +    │◀───│ (81+)    │
│          │    │          │    │  Executor)   │    │          │
└─────────┘    └──────────┘    └──────┬───────┘    └──────────┘
                    │                 │
                    │          ┌──────┴───────┐
                    │          │ FailoverProv │
                    │          │ Groq+Gemini  │
                    ▼          └──────────────┘
              ┌──────────┐
              │ SQLite   │
              │ sessions │
              └──────────┘
```

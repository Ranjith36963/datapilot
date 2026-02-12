# DATAPILOT_DIAGNOSTIC.md — Complete Codebase Diagnostic & Architecture Documentation

> **Generated**: 2026-02-12 | **Files read**: 65+ | **Scope**: Every source file in engine/, backend/, frontend/, tests/

---

## 1. Executive Summary

DataPilot is an AI-powered data analysis platform with **81 analytical skills** spanning statistics, ML, NLP, time-series, survival analysis, and more. The stack is Python 3.13 (FastAPI backend + custom engine) and Next.js 16 (React 19 frontend). LLM intelligence is provided by Groq (default, via openai SDK), with Ollama, Claude, and OpenAI as alternatives.

**Architecture**: 3-layer — Engine (skills + routing + LLM) → Backend (REST API + WebSocket) → Frontend (4-page SPA). The engine is self-contained and can operate without the backend.

**Current state**: V1.0 is functional. Upload → Explore → Visualize → Export flow works end-to-end. The 3-layer hallucination defense is in place. 28 tests pass. Key issues remain in template narratives (recently patched), session management (in-memory only), and lack of deterministic chart rules.

---

## 2. File-by-File Documentation

### 2.1 Engine Core (`engine/datapilot/core/`)

| File | Purpose | Key Functions/Classes | Returns |
|------|---------|----------------------|---------|
| `__init__.py` | Core module exports | Exports `Analyst`, `Router`, `Executor` | — |
| `analyst.py` (627 lines) | **Main public API** — orchestrates routing, execution, narrative generation | `Analyst` class: `ask()`, `_generate_narrative()`, `_template_narrative()`, `_sanitize_for_narration()`, `_build_report_data()`, `_fmt()` helper. Dataclasses: `AnalystResult`, `ExecutionResult` | `AnalystResult` with routing, execution, narrative |
| `router.py` (350+ lines) | **Question routing** — maps NL questions to skill names | `Router` class: `route()`, `_try_keyword_route()`, `_enrich_parameters()`, `build_data_context()`. Constants: `_CHART_PRIORITY_PATTERNS`, `_CHART_TYPE_MAP`, `_KEYWORD_ROUTES` (20+ patterns) | `RoutingResult(skill_name, parameters, confidence, reasoning)` |
| `executor.py` (200+ lines) | **Skill dispatch** — calls skill functions with filtered params | `Executor` class: `execute()`, `_filter_params()` (signature inspection), `_df_to_temp_path()`, `_inject_chart_base64()`, `_humanize_error()`, `_build_code_snippet()` | `ExecutionResult(status, skill_name, result, elapsed_seconds)` |

**Routing priority chain**: Chart keywords → Keyword table (20+ regex patterns) → Primary LLM → Groq fallback → `profile_data` default

**Narrative pipeline**: LLM narrative → "?" rejection check (rejects if contains `"? "` or <30 chars) → Template fallback via `_template_narrative()`

### 2.2 Engine Data (`engine/datapilot/data/`)

| File | Purpose | Key Functions | Returns |
|------|---------|--------------|---------|
| `profiler.py` (300+ lines) | Dataset profiling — per-column stats, quality score, warnings | `profile_data(file_path)` | `{status, overview: {rows, columns, memory_mb, duplicate_rows, duplicate_pct}, columns: [...], quality_score, warnings, recommendations, correlations}` |
| `schema.py` (194 lines) | Semantic type inference — goes beyond pandas dtypes | `infer_schema(file_path)`, `_infer_semantic_type()`, `detect_datetime_columns()` | `{status, columns: [{name, pandas_dtype, semantic_type, confidence, nullable, patterns_found, suggested_transformations}]}` |
| `validator.py` (267 lines) | Data quality validation — nulls, duplicates, outliers, type consistency | `validate_data(file_path, rules)`, `check_nulls()`, `check_duplicates()`, `check_outliers()` | `{status, validation_result, score (0-100), checks, summary, recommendations}` |
| `cleaner.py` (181 lines) | Data cleaning — column standardization, imputation, dedup | `curate_dataframe(file_path, ...)`, `standardize_column_names()`, `impute_missing()` | `{status, input_shape, output_shape, changes, output_path}` |
| `ocr.py` (97 lines) | OCR — extract text from images | `extract_text_from_image()`, `batch_ocr()` | `{status, extracted_text, confidence, word_count}` |

**Key issue**: `profiler.py` returns `overview.rows` and `overview.columns` — NOT `total_rows`/`total_columns`. This mismatch caused the narrative regression (now patched in analyst.py).

### 2.3 Engine Analysis (`engine/datapilot/analysis/`)

| File | Purpose | Key Functions | Returns |
|------|---------|--------------|---------|
| `descriptive.py` | Descriptive stats | `describe_data(file_path, columns)`, `compare_groups(file_path, ...)` | `{status, numeric_summary: [{column, count, mean, std, min, q25, median, q75, max, skewness, kurtosis, is_normal}], categorical_summary}` |
| `correlation.py` | Correlation analysis | `analyze_correlations(file_path, method, target)` | `{status, method, correlation_matrix, top_correlations: [{col1, col2, correlation, pvalue}], target_correlations, multicollinearity_warning}` |
| `anomaly.py` | Outlier detection (IQR, Z-score, Isolation Forest, LOF, DBSCAN) | `detect_outliers(file_path, method, threshold)` | `{status, method, total_rows, outlier_count, outlier_pct, outlier_indices, column_analysis}` |
| `classification.py` | ML classification (7 algorithms + auto + tuning) | `classify(file_path, target, algorithm)`, `auto_classify()`, `tune_classifier()` | `{status, algorithm, target, metrics: {accuracy, precision, recall, f1, auc_roc}, feature_importance, confusion_matrix, model_path}` |
| `regression.py` | ML regression (7 algorithms + auto + tuning) | `predict_numeric(file_path, target, algorithm)`, `auto_regress()`, `tune_regressor()` | `{status, algorithm, metrics: {r2, rmse, mae, mape}, coefficients, feature_importance, model_path}` |
| `clustering.py` (246 lines) | Clustering (K-Means, DBSCAN, HDBSCAN, Hierarchical, GMM) | `find_clusters(file_path, n_clusters, algorithm)`, `optimal_clusters()`, `describe_clusters()` | `{status, algorithm, n_clusters, cluster_sizes, metrics: {silhouette, calinski_harabasz, davies_bouldin}, cluster_profiles}` |
| `timeseries.py` (328 lines) | Time series (trend, seasonality, stationarity, forecasting) | `analyze_time_series()`, `forecast(method)`, `detect_change_points()` | `{status, frequency, trend, seasonality, stationarity, summary_stats}` for analysis; `{status, method, forecast, metrics}` for forecast |
| `hypothesis.py` (306 lines) | Hypothesis testing (t-test, ANOVA, chi-square, Mann-Whitney, Kruskal, normality, Levene) | `run_hypothesis_test(file_path, test_type, **kwargs)` dispatcher + individual test functions | `{status, test, statistic, pvalue, significant, effect_size, effect_interpretation, conclusion, details}` |
| `effect_size.py` (260 lines) | Effect size measures (Cohen's d, Hedges' g, odds ratio, relative risk, Cramer's V, eta-squared, r-squared) | `calculate_effect_size(file_path, effect_type, **kwargs)` dispatcher | `{status, effect_type, effect_size, confidence_interval, interpretation}` |
| `survival.py` (205 lines) | Survival analysis (Kaplan-Meier, Cox PH) | `survival_analysis()`, `cox_regression()` | `{status, n_subjects, n_events, median_survival, survival_curve, group_comparison}` |
| `dimensionality.py` (172 lines) | Dimensionality reduction (PCA, SVD, Factor Analysis, t-SNE, UMAP) | `reduce_dimensions()`, `pca_analysis()`, `visualize_2d()` | `{status, method, original_dimensions, reduced_dimensions, explained_variance_ratio, components}` |
| `selection.py` (234 lines) | Feature selection (tree importance, RFE, mutual info, chi2, LASSO, SHAP) | `select_features(file_path, target, method)` | `{status, method, target, feature_ranking, recommended_features, eliminated_features}` |
| `threshold.py` (339 lines) | Threshold/tipping point finder (decision tree, brute force, optbinning, change point) | `find_thresholds(file_path, target)`, `threshold_confidence_interval()` | `{status, target, thresholds: [{feature, threshold, rate_below, rate_above, lift, pvalue, insight}], best_threshold}` |
| `engineering.py` (226 lines) | Feature engineering (date extraction, interactions, binning, encoding) | `engineer_features(file_path, operations)` | `{status, original_columns, new_columns, features_created, output_path}` |
| `explain.py` (254 lines) | Model explainability (SHAP global + local) | `explain_model(model_path, file_path)`, `explain_prediction(model_path, file_path, row_index)` | `{status, explanation_type, feature_importance, interaction_effects}` or `{status, prediction, contributions, top_positive/negative_factors}` |

### 2.4 Engine NLP (`engine/datapilot/nlp/`)

| File | Purpose | Key Functions | Deps |
|------|---------|--------------|------|
| `sentiment.py` (148 lines) | Sentiment analysis (VADER, TextBlob) | `analyze_sentiment(text)`, `batch_sentiment(file_path, text_column)` | vaderSentiment, textblob |
| `text_stats.py` (174 lines) | Text statistics (word count, readability, keywords, bigrams) | `analyze_text(text)`, `batch_text_stats()`, `extract_keywords()` | textstat (optional), sklearn TfidfVectorizer |
| `topics.py` (135 lines) | Topic modeling (LDA, NMF) | `extract_topics(file_path, text_column, n_topics)` | sklearn, gensim (optional for coherence) |
| `entities.py` (173 lines) | Named entity recognition (spaCy + regex fallback) | `extract_entities(text)`, `batch_extract_entities()` | spacy (optional, regex fallback) |
| `intent.py` (182 lines) | Intent detection (keyword heuristic + trainable classifier) | `detect_intent(text)`, `train_intent_classifier()`, `batch_classify_intent()` | sklearn Pipeline (TF-IDF + LogReg) |

### 2.5 Engine Viz (`engine/datapilot/viz/`)

| File | Purpose | Key Functions |
|------|---------|--------------|
| `charts.py` | Chart creation (13 chart types) | `create_chart(file_path, chart_type, x, y, hue, title, interactive)` — supports bar, barh, line, scatter, histogram, box, violin, heatmap, pie, area, density, count, pair. `_compute_chart_summary()` (Layer 1 of hallucination defense) |

**Chart output**: Static PNG via matplotlib/seaborn (base64-encoded), Interactive HTML via plotly.

### 2.6 Engine Export (`engine/datapilot/export/`)

| File | Purpose | Key Functions | Deps |
|------|---------|--------------|------|
| `pdf.py` | PDF export via reportlab | `export_to_pdf(analysis_results, output_path, ...)` — title page, TOC, executive summary, key metrics table, detailed sections with inline charts, `_NumberedCanvas` for page numbers | reportlab |
| `docx.py` (201 lines) | Word export via python-docx | `export_to_docx(analysis_results, output_path, ...)` — title, executive summary, key metrics table, detailed analysis sections, inline charts, page number footer via XML field codes | python-docx |
| `pptx.py` (313 lines) | PowerPoint export via python-pptx | `export_to_pptx(analysis_results, output_path, ...)` — title slide, executive summary (bullet points), key metrics (colored cards, 2x4 grid), analysis slides (text + chart split), closing slide with recommendations | python-pptx |

**Report data contract**: All export functions consume `analysis_results` dict with `{summary, sections: [{heading, question, narrative, key_points, skill}], metrics: [{label, value}], key_points}`.

### 2.7 Engine LLM (`engine/datapilot/llm/`)

| File | Purpose | API | Default Model |
|------|---------|-----|---------------|
| `provider.py` | ABC for LLM providers | `route_question()`, `generate_narrative()`, `suggest_chart()`, `generate_chart_insight()`. Dataclasses: `RoutingResult`, `NarrativeResult` | — |
| `groq.py` (400+ lines) | **Default provider** — Groq via openai SDK | `base_url="https://api.groq.com/openai/v1"`. `_truncate_for_narration()` (50KB limit), `suggest_chart()` with 4-6 ranked suggestions, smart fallback on API failure. 6-rule system prompt. | `llama-3.3-70b-versatile` |
| `ollama.py` (250+ lines) | Local fallback — Ollama via urllib HTTP | Direct HTTP to `http://localhost:11434`. JSON mode for routing/suggestions. | `llama3.2` |
| `claude.py` (200+ lines) | Claude — Anthropic SDK | Standard narrative/routing prompts with anthropic client. | `claude-sonnet-4-5-20250929` |
| `openai.py` (200+ lines) | OpenAI — openai SDK | Standard narrative/routing prompts with openai client. | `gpt-4o-mini` |
| `prompts.py` (200+ lines) | Skill catalog + prompt templates | `build_skill_registry()` auto-generates from `__all__`, excludes `_and_upload` variants and `correlation_matrix`. `get_skill_function()` looks up by name. `build_skill_catalog()` creates human-readable catalog for LLM prompts. | — |

### 2.8 Engine Utils (`engine/datapilot/utils/`)

| File | Purpose | Key Functions |
|------|---------|--------------|
| `config.py` | Configuration with env var overrides | `Config` class: `LLM_PROVIDER` (default "groq"), `GROQ_MODEL` (default "llama-3.3-70b-versatile"), `GROQ_API_KEY`, `OLLAMA_URL`, etc. |
| `helpers.py` | File I/O + DataFrame helpers | `load_data(file_path)` (CSV/Excel/JSON/Parquet), `save_data()`, `get_numeric_columns()`, `get_categorical_columns()`, `get_datetime_columns()`, `setup_logging()` |
| `serializer.py` | JSON serialization for numpy/pandas types | `safe_json_serialize(obj)` — recursively converts numpy int/float/bool, pandas Timestamp, ndarray, etc. |
| `report_data.py` | Report data formatting | `format_for_narrative()`, `create_executive_summary_data()`, `create_detailed_findings_data()` |
| `uploader.py` | External upload utility | `upload_result(result, filename)` — uses `DATAPILOT_UPLOAD_PATH` env var, graceful fallback if not set |

### 2.9 Backend (`backend/app/`)

| File | Purpose | Endpoints |
|------|---------|-----------|
| `main.py` | FastAPI app entry, CORS, lifespan | Health check `/api/health`, session endpoints |
| `api/data.py` | File upload + preview | `POST /api/upload` (creates session + auto-profile), `GET /api/preview`, `GET /api/profile` |
| `api/analysis.py` | NL analysis pipeline | `POST /api/ask` (full pipeline: route → execute → narrate, with conversation context from last 3 Q&As), `GET /api/narrative` (poll), `POST /api/analyze` (direct skill bypass) |
| `api/charts.py` | Chart creation + AI suggestions | `POST /api/chart/create` (with AI insight), `GET /api/chart/suggest` (LLM-powered, 4-6 ranked suggestions) |
| `api/export.py` | Report generation | `POST /api/export/{fmt}` (pdf/docx/pptx), `GET /api/export/download/{filename}` |
| `api/ws.py` | WebSocket streaming | `WS /api/ws/chat` — streams progress: routing → execution → result → narrative |
| `services/analyst.py` | Session management | `SessionManager` singleton — in-memory `{session_id: Analyst}` dict |
| `services/data_service.py` | File handling | Upload to `/tmp/datapilot/uploads/{session_id}/`, 100MB max |
| `models/requests.py` | Pydantic v2 request schemas | `AskRequest`, `ChartRequest`, `ExportRequest` |
| `models/responses.py` | Pydantic v2 response schemas | `AskResponse`, `SuggestChartResponse`, `ExportResponse` |

**Session flow**: Session ID passed via `x-session-id` header. Created at upload, all subsequent requests reference same session.

### 2.10 Frontend (`frontend/src/`)

| File | Purpose | Key Behavior |
|------|---------|-------------|
| `app/layout.tsx` | Root layout | Inter font, `<Providers>` wrapper (Theme + Session), `<Navbar />` + `<main>` |
| `app/providers.tsx` | Provider composition | `ThemeProvider` (next-themes) + `SessionProvider` (Zustand) |
| `app/page.tsx` | **Upload page** | react-dropzone, data preview table, column badges with semantic type colors (numeric=blue, categorical=purple, datetime=green, text=amber, boolean=pink, id=slate), "Start Exploring" CTA |
| `app/explore/page.tsx` | **Explore chat** | Sidebar (columns list, suggested questions), TrustHeader (confidence badge + code snippet + columns used), ResultCard (structured data display), key_points (clickable to copy), follow-up suggestions (clickable buttons). Builds conversation context from last 3 Q&A pairs. |
| `app/visualize/page.tsx` | **Chart builder** | 3-column layout: Chart Builder (left, type selector + column dropdowns), Chart Feed (center, Plotly interactive charts with download PNG + AI insight), AI Suggestions (right, auto-loads on mount, clickable cards). Chart history stored in Zustand. |
| `app/export/page.tsx` | **Export** | Format selector (PDF/Word/PowerPoint), title/subtitle inputs, generate + download |
| `components/navbar.tsx` | Navigation bar | 4 nav items (Upload/Explore/Visualize/Export), disabled when no file uploaded, theme toggle, filename + shape display |
| `components/result-card.tsx` (959 lines) | **Structured result display** | Handles: overview grid, quality score bar, warnings, recommendations, correlations, classification metrics/confusion matrix/feature importance/CV/algorithm comparison, clustering profiles, hypothesis test results, outlier detection, generic scalar stats, array tables, expandable raw data, chart images |
| `components/data-table.tsx` | Simple data table | Column headers + rows, max 50 rows, null styling, truncation |
| `components/theme-provider.tsx` | Theme wrapper | next-themes with system default, class attribute |
| `lib/api.ts` | API client | `apiFetch<T>()` helper with session header injection, methods: `uploadFile`, `askQuestion`, `createChart`, `suggestChart`, `exportReport`, `connectChat` (WebSocket) |
| `lib/store.tsx` | Zustand state | `useSession()`: sessionId, filename, shape, columns, preview, exploreMessages, chartHistory. Persisted to sessionStorage. `SessionProvider` wraps children. |

### 2.11 Tests (`tests/`)

| File | Tests | Coverage |
|------|-------|----------|
| `unit/test_router_correlation.py` (124 lines) | 7 tests: "relate" variants route to analyze_correlations, existing correlation keywords work, correlation_matrix excluded from skill registry | Router keyword matching, skill registry |
| `unit/test_export_content.py` (265 lines) | 8 tests: `_build_report_data` produces summary/sections/metrics/key_points, PDF/DOCX/PPTX export creates non-empty files with real content | Report data builder, all 3 export formats |
| `unit/test_chart_suggest.py` (224 lines) | 13 tests: Groq suggest_chart parsing, null-string cleanup, invalid chart type fallback, code block wrapper, multiple suggestions, reason field, API failure fallback, empty columns | Chart suggestion flow, Groq response parsing |

**Total: 28 tests**, all passing.

### 2.12 Configuration

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python packaging — setuptools, deps (pandas, numpy, sklearn, scipy, statsmodels, matplotlib, seaborn, plotly, reportlab, python-docx, python-pptx, textblob), optional-deps (ml, nlp, api, llm), ruff config, pytest config (`testpaths=["tests"]`, `pythonpath=["engine"]`) |
| `frontend/package.json` | Node deps — next 16.1.6, react 19.2.3, zustand 5.0.11, lucide-react, next-themes, react-dropzone. Dev: tailwindcss 4, typescript 5 |
| `frontend/next.config.ts` | Next.js config — standalone output, `ignoreBuildErrors: true` (Turbopack false positive) |
| `frontend/tsconfig.json` | TypeScript — ES2017 target, strict, bundler module resolution, `@/*` path alias |

---

## 3. Architecture Flowchart

```mermaid
graph TD
    subgraph Frontend ["Frontend (Next.js 16 + React 19)"]
        UP[Upload Page<br/>react-dropzone]
        EX[Explore Page<br/>Chat Interface]
        VZ[Visualize Page<br/>Chart Builder]
        XP[Export Page<br/>Report Generator]
        ST[Zustand Store<br/>sessionStorage]
        API_CLIENT[API Client<br/>apiFetch + WebSocket]
    end

    subgraph Backend ["Backend (FastAPI)"]
        CORS[CORS Middleware]
        DATA_API["/api/upload<br/>/api/preview<br/>/api/profile"]
        ASK_API["/api/ask<br/>/api/analyze"]
        CHART_API["/api/chart/create<br/>/api/chart/suggest"]
        EXPORT_API["/api/export/{fmt}<br/>/api/export/download"]
        WS_API["WS /api/ws/chat"]
        SM[SessionManager<br/>in-memory dict]
    end

    subgraph Engine ["Engine (datapilot)"]
        ANALYST[Analyst<br/>Public API]
        ROUTER[Router<br/>Keywords → LLM]
        EXECUTOR[Executor<br/>Param Filtering]

        subgraph Skills ["81 Skills"]
            DATA_SKILLS[Data: profiler, schema<br/>validator, cleaner, ocr]
            ANALYSIS_SKILLS[Analysis: descriptive<br/>correlation, anomaly<br/>classification, regression<br/>clustering, timeseries<br/>hypothesis, effect_size<br/>survival, dimensionality<br/>selection, threshold<br/>engineering, explain]
            NLP_SKILLS[NLP: sentiment, entities<br/>topics, intent, text_stats]
            VIZ_SKILLS[Viz: charts<br/>13 chart types]
            EXPORT_SKILLS[Export: pdf, docx, pptx]
        end

        subgraph LLM ["LLM Providers"]
            GROQ[Groq<br/>llama-3.3-70b]
            OLLAMA[Ollama<br/>llama3.2]
            CLAUDE_LLM[Claude<br/>sonnet-4-5]
            OPENAI_LLM[OpenAI<br/>gpt-4o-mini]
        end

        PROMPTS[Skill Catalog<br/>Auto-generated]
    end

    UP -->|POST /api/upload| DATA_API
    EX -->|POST /api/ask + WS| ASK_API
    EX -->|WS streaming| WS_API
    VZ -->|POST /api/chart/create| CHART_API
    VZ -->|GET /api/chart/suggest| CHART_API
    XP -->|POST /api/export/{fmt}| EXPORT_API

    API_CLIENT --> CORS
    CORS --> DATA_API
    CORS --> ASK_API
    CORS --> CHART_API
    CORS --> EXPORT_API

    DATA_API --> SM
    ASK_API --> SM
    CHART_API --> SM
    EXPORT_API --> SM
    WS_API --> SM

    SM -->|get_session| ANALYST
    ANALYST -->|route| ROUTER
    ROUTER -->|keyword match| EXECUTOR
    ROUTER -->|LLM fallback| GROQ
    ROUTER --> PROMPTS
    ANALYST -->|execute| EXECUTOR
    EXECUTOR --> Skills
    ANALYST -->|narrate| GROQ
    ANALYST -->|suggest_chart| GROQ

    style Frontend fill:#e3f2fd
    style Backend fill:#fff3e0
    style Engine fill:#e8f5e9
    style Skills fill:#f3e5f5
    style LLM fill:#fce4ec
```

---

## 4. User Journey Maps

### Journey 1: Upload

```
User drags CSV → react-dropzone onDrop
  → POST /api/upload (multipart, file + x-session-id header)
    → DataService.handle_upload() — saves to /tmp/datapilot/uploads/{session_id}/
    → SessionManager.create_session() — new Analyst(file_path)
    → Analyst.__init__() → load_data() → auto profile_data()
  ← {session_id, filename, shape, columns, preview}
→ Zustand store: set sessionId, filename, shape, columns, preview
→ UI: Shows data preview table + column badges + "Start Exploring" CTA
```

**AI involvement**: NONE. Upload is fully deterministic.

### Journey 2: Explore (Ask a Question)

```
User types "Are there any outliers?" → Submit
  → POST /api/ask {question, context: last 3 Q&As}
    → session_manager.get_session(session_id) → Analyst instance
    → Analyst.ask(question, context)
      → Router.route(question, data_context)
        → _try_keyword_route("outlier" matches) → RoutingResult(skill="detect_outliers")
        → _enrich_parameters(result, data_context) → adds numeric columns
      → Executor.execute("detect_outliers", params)
        → _filter_params() — inspects detect_outliers() signature
        → detect_outliers(file_path, method="isolation_forest")
        ← {status: "success", total_rows: 3333, outlier_count: 167, outlier_pct: 5.01, ...}
      → _generate_narrative(skill_name, result)
        → _sanitize_for_narration(result) — strips base64/paths, injects column context
        → _try_llm_narrative(skill, sanitized_result, context)
          → Groq.generate_narrative(prompt) → LLM response
          → Validate: no "?" placeholders, length > 30 chars
          ← NarrativeResult(text="...", key_points=[...], suggestions=[...])
        → If LLM fails: _template_narrative(skill, result) — template fallback
      ← AnalystResult(routing, execution, narrative)
  ← {narrative, key_points, suggestions, result, routing: {skill, confidence}, code_snippet}
→ UI: Renders narrative text, key_points as clickable bullets, ResultCard with structured data,
      follow-up suggestions as clickable buttons, TrustHeader with confidence badge
```

**AI involvement**: Routing (keyword match is deterministic, LLM fallback is non-deterministic), Narrative generation (LLM with template fallback).

### Journey 3: Visualize — Manual Chart

```
User selects: Chart type=scatter, X=age, Y=income, Hue=category
  → POST /api/chart/create {chart_type, x, y, hue, title, interactive: true}
    → Analyst.create_chart(chart_type="scatter", x="age", y="income", ...)
      → Executor.execute("create_chart", params)
        → create_chart(file_path, chart_type="scatter", x="age", y="income", hue="category", interactive=true)
          → _compute_chart_summary(df, chart_type, x, y, hue) — Layer 1 hallucination defense
          → Plotly scatter plot → HTML file saved
          → matplotlib scatter → PNG → base64
        ← {status, chart_type, chart_base64, chart_html_path, chart_summary, key_points}
    → Groq.generate_chart_insight(chart_summary) — AI insight for the chart
  ← {chart_base64, chart_html, insight, key_points}
→ UI: Plotly interactive chart in center feed, "Download PNG" button, AI insight text,
      chart added to chartHistory in Zustand
```

**AI involvement**: Chart creation is DETERMINISTIC. AI insight is non-deterministic (LLM).

### Journey 4: Visualize — AI Suggest

```
Page mounts → GET /api/chart/suggest
  → Analyst.suggest_chart()
    → Router.build_data_context(df) → column metadata
    → Groq.suggest_chart(data_context)
      → LLM prompt: "Suggest 4-6 ranked chart visualizations for this dataset..."
      → Parse JSON array, validate chart types, clean "null" strings
      → On API failure: deterministic fallback (histogram of first numeric + scatter of first two numerics)
    ← {suggestions: [{chart_type, x, y, hue, title, reason}, ...]}
  ← suggestions list
→ UI: Right sidebar shows suggestion cards with title + reason
→ User clicks suggestion → auto-fills Chart Builder → triggers manual chart flow
```

**AI involvement**: Suggestion generation is LLM-powered with deterministic fallback.

### Journey 5: Export

```
User selects PDF format, enters title "Q1 Report"
  → POST /api/export/pdf {title, subtitle, selected_analyses: null (all)}
    → Analyst._build_report_data()
      → Iterates history[], builds {summary, sections, metrics, key_points}
      → summary: concatenates first narrative sentences
      → sections: per-analysis {heading, question, narrative, key_points, skill}
      → metrics: extracts key numbers (strongest correlation, outlier count, accuracy, etc.)
    → export_to_pdf(report_data, output_path, title, ...)
      → reportlab: title page → TOC → executive summary → key metrics table →
        detailed analysis sections with inline chart images → additional visualizations
      → _NumberedCanvas for page numbers
    ← {filename, download_url}
  → GET /api/export/download/{filename}
  ← Binary file download
→ UI: Download triggers, file saves locally
```

**AI involvement**: NONE at export time. All content was generated during Explore phase and stored in history.

---

## 5. Intelligence Audit

### 5.1 Where AI (LLM) Is Used

| Location | Purpose | Provider | Deterministic Fallback? |
|----------|---------|----------|------------------------|
| `Router.route()` | Question → skill name mapping | Groq (primary), any provider (fallback) | YES — keyword table covers 20+ patterns, final fallback is `profile_data` |
| `Analyst._generate_narrative()` | Skill result → human-readable text | Groq (default) | YES — `_template_narrative()` with per-skill templates |
| `GroqProvider.suggest_chart()` | Dataset → 4-6 chart suggestions | Groq | YES — histogram + scatter fallback using actual column names |
| `GroqProvider.generate_chart_insight()` | Chart summary → insight text | Groq | NO — returns empty string on failure |
| `Router._enrich_parameters()` (partial) | Extract column names from question | Regex only | YES — pure regex, no LLM |

### 5.2 Where Pure Computation Is Used

| Location | Purpose |
|----------|---------|
| All 81 skills | Statistical analysis, ML training, NLP processing — pandas, sklearn, scipy, statsmodels |
| `create_chart()` | Chart rendering — matplotlib, seaborn, plotly |
| `_compute_chart_summary()` | Structured summary of chart data — pure pandas |
| All export functions | PDF/DOCX/PPTX generation — reportlab, python-docx, python-pptx |
| `profile_data()` | Dataset profiling — pure pandas + numpy |
| `_sanitize_for_narration()` | Strips base64, paths, limits size — pure string manipulation |
| `_filter_params()` | Inspects function signatures — pure `inspect` module |

### 5.3 Failure Points

| Component | Failure Mode | Impact | Mitigation |
|-----------|-------------|--------|------------|
| Groq API | Rate limit / timeout / 500 | Routing falls back to keyword → profile_data; narrative falls back to template | Keyword table + template fallback |
| LLM narrative | Contains "?" or too short (<30 chars) | Rejected, template fallback used | `_generate_narrative()` validation |
| LLM narrative | Hallucinates numbers not in result | User sees fabricated statistics | 3-layer defense: chart_summary → sanitizer → prompt constraint |
| Skill execution | Exception in skill function | Returns `{status: "error", message: ...}` | `_humanize_error()` in Executor |
| File upload | >100MB file | Rejected at backend | `data_service.py` size check |
| Session management | Server restart | All sessions lost (in-memory dict) | **NO MITIGATION** — critical gap |
| WebSocket | Connection drop during streaming | Partial response shown | Frontend should handle reconnection (not implemented) |

### 5.4 Deterministic vs Non-Deterministic

| Operation | Type | Notes |
|-----------|------|-------|
| Keyword routing | **Deterministic** | Regex pattern matching, priority order |
| LLM routing | **Non-deterministic** | Same question may route differently |
| Skill execution | **Deterministic** | Same data + params = same result (except random_state in ML) |
| Template narrative | **Deterministic** | Same result dict = same text |
| LLM narrative | **Non-deterministic** | Same result may produce different text |
| Chart creation | **Deterministic** | Same params = same chart |
| Chart suggestion | **Non-deterministic** | LLM-generated, fallback is deterministic |
| Export generation | **Deterministic** | Same history = same report |

---

## 6. Current State Assessment

### 6.1 Strengths

1. **Comprehensive skill catalog**: 81 skills covering descriptive stats, ML, NLP, time-series, survival, explainability — well beyond typical data analysis tools
2. **Robust routing**: Keyword table handles 20+ common patterns deterministically; LLM is fallback, not primary
3. **3-layer hallucination defense**: chart_summary → sanitizer → prompt constraints — principled approach to LLM grounding
4. **Graceful degradation**: Every LLM call has a deterministic fallback. System works (degraded) without any LLM
5. **Consistent skill contract**: Every skill returns `{status: "success|error", ...}` — uniform error handling
6. **Rich frontend**: ResultCard handles 10+ skill-specific result formats (correlations, classification, clustering, hypothesis, outliers, etc.)
7. **Export quality**: PDF/DOCX/PPTX all include structured sections, inline charts, key metrics, page numbers
8. **Auto-profiling on upload**: Immediate data quality score + warnings + recommendations
9. **Conversation context**: Last 3 Q&A pairs passed to routing and narrative — enables follow-up questions

### 6.2 Weaknesses

1. **In-memory session management**: Server restart loses ALL sessions. No persistence layer.
2. **No authentication/authorization**: Anyone can access any session if they know the ID.
3. **No concurrent request handling**: Single Analyst per session, no locking — concurrent requests to same session could corrupt state.
4. **Template narratives are shallow**: Even after fixes, templates produce 1-2 sentences with basic stats — far less insightful than LLM narratives.
5. **No retry logic for LLM calls**: Single attempt, then fallback. No exponential backoff or circuit breaker.
6. **Large result payloads**: Correlation matrices, cluster assignments, outlier indices all sent to frontend — no pagination or lazy loading.
7. **No rate limiting**: Backend has no request rate limiting — vulnerable to abuse.
8. **WebSocket reconnection**: No auto-reconnect logic in frontend if connection drops.
9. **next.config.ts `ignoreBuildErrors: true`**: Masks TypeScript errors in production builds.

### 6.3 Gaps

1. **No dataset classification/fingerprinting**: System doesn't auto-detect dataset type (financial, medical, marketing, etc.) to tailor suggestions.
2. **No deterministic chart rules**: Chart suggestions are entirely LLM-dependent — no rule engine like "if 2 numeric cols exist, always suggest scatter."
3. **No auto-pilot analysis mode**: User must ask each question manually — no "run a full analysis automatically."
4. **No caching**: Same question on same data re-runs entire pipeline (routing → execution → narrative).
5. **No data versioning**: If user uploads a new file, old session's history references stale data.
6. **No collaboration**: Single-user sessions only.
7. **No streaming for non-WS endpoints**: REST endpoints block until complete — no progress for long-running skills.
8. **No integration tests for full pipeline**: Tests only cover unit-level (router, export, chart suggest) — no upload→ask→visualize→export E2E.
9. **No monitoring/observability**: No metrics, no tracing, no health dashboard beyond basic `/api/health`.

### 6.4 Hardcoded Values

| Location | Value | Should Be |
|----------|-------|-----------|
| `anomaly.py` | `contamination=0.05` for Isolation Forest | Configurable or auto-detected |
| `classification.py` | `test_size=0.2`, `random_state=42` | Configurable |
| `timeseries.py` | Prophet `perplexity=min(30, n-1)` | Auto-tuned |
| `clustering.py` | `n_init=10` for KMeans | Configurable |
| `data_service.py` | `100 * 1024 * 1024` (100MB) max upload | Env var |
| `groq.py` | 50KB truncation limit for narration | Configurable |
| `charts.py` | `figsize=(10, 6)` for all charts | Responsive or configurable |
| `threshold.py` | `min_samples=50`, `min_impact=0.05` | Configurable |
| `entities.py` | `text[:100000]` spaCy input cap | Configurable |

---

## 7. Phase 2 Readiness Assessment

### 7.1 Auto-Pilot Analysis

**Readiness: 60%**. The infrastructure exists — `Analyst.ask()` can be called programmatically, `_build_report_data()` aggregates history. What's missing:
- **Analysis planner**: Given a dataset profile, generate an ordered list of questions (e.g., "if >5 numeric cols, run correlations; if has target col, run classification")
- **Completion criteria**: When to stop analyzing (diminishing returns, all key patterns covered)
- **Progress reporting**: Streaming updates as auto-pilot runs each step

### 7.2 Dataset Classification/Fingerprinting

**Readiness: 40%**. `infer_schema()` provides semantic types and `profile_data()` provides column distributions, but:
- No dataset-level classification (financial, medical, marketing, etc.)
- No automatic target column detection
- No pattern library for common dataset archetypes

### 7.3 Deterministic Chart Rules

**Readiness: 30%**. `suggest_chart()` has a deterministic fallback (histogram + scatter), but:
- No rule engine mapping column types → chart types
- No consideration of data distribution (skewed → log scale, many categories → horizontal bar)
- Frontend `_CHART_TYPE_MAP` in router.py maps keywords to chart types but not data patterns

### 7.4 Multi-Dataset Support

**Readiness: 20%**. Current architecture is strictly single-dataset per session:
- `Analyst.__init__()` takes one `file_path`
- No join/merge/compare capabilities
- No cross-dataset correlation

### 7.5 Persistent Sessions

**Readiness: 10%**. SessionManager is a plain Python dict:
- Need: Database-backed sessions (Redis for fast access, PostgreSQL for durability)
- Need: Session serialization (Analyst state, history, data reference)
- Need: Session expiry and cleanup

---

## 8. Recommended Fix Priority

### P0 — Critical (Do Now)

1. **Session persistence** — Add Redis or SQLite-backed session storage. Server restart currently loses ALL user work.
2. **Integration tests** — Add E2E test: upload CSV → ask 3 questions → create chart → export PDF. Verify each step.
3. **Remove `ignoreBuildErrors: true`** — Fix the underlying TypeScript issue instead of masking it.

### P1 — High (Next Sprint)

4. **Deterministic chart rules** — Add rule engine: `{2 numeric → scatter, 1 categorical + 1 numeric → bar, datetime + numeric → line, ...}`. Use as primary, LLM as enhancement.
5. **Auto-pilot analysis** — Build analysis planner that runs a standard sequence based on dataset profile: overview → describe → correlations → outliers → (if target) classify/predict.
6. **WebSocket reconnection** — Add auto-reconnect with exponential backoff in frontend.
7. **Rate limiting** — Add per-session and per-IP rate limits via FastAPI middleware.

### P2 — Medium (Backlog)

8. **Dataset fingerprinting** — Classify datasets by domain using column name patterns and value distributions. Use to customize suggested questions and default analyses.
9. **Result caching** — Cache skill results by (file_hash, skill_name, params_hash). Avoid re-running identical analyses.
10. **Streaming REST endpoints** — Use FastAPI StreamingResponse for long-running skills instead of blocking.
11. **Template narrative enrichment** — Improve templates to produce 3-5 sentence narratives with comparisons and context, reducing dependence on LLM.

### P3 — Low (Future)

12. **Multi-dataset support** — Allow joining/comparing datasets within a session.
13. **Authentication** — Add user accounts, API keys, session ownership.
14. **Monitoring** — Add Prometheus metrics, structured logging, error tracking (Sentry).
15. **Plugin architecture** — Allow custom skills to be registered without modifying engine code.

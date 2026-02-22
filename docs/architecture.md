# Architecture

## System Overview

```
User
 │
 ▼
┌──────────────────────────────┐
│  Frontend (Next.js :3000)    │
│  - Upload page               │
│  - Explore chat              │
│  - Chart builder             │
│  - Report export             │
└──────────┬───────────────────┘
           │ REST / WebSocket
           ▼
┌──────────────────────────────┐
│  Backend (FastAPI :8000)     │
│  - Session manager           │
│  - 17 API endpoints          │
│  - WebSocket streaming       │
│  - SQLite persistence (WAL)  │
└──────────┬───────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌──────────────────┐
│  Engine │  │  LLM Layer       │
│  34     │  │  FailoverProvider│
│  skills │  │  ├─ Groq (fast)  │
└─────────┘  │  ├─ Gemini (safe)│
             │  ├─ Ollama       │
             │  ├─ Claude       │
             │  └─ OpenAI       │
             └──────────────────┘
```

## Engine Package Structure

```
engine/datapilot/
├── core/           Main public API
│   ├── analyst.py         Analyst class — ask(), profile(), classify()
│   ├── router.py          Question → skill routing (keywords + LLM)
│   ├── semantic_router.py Embedding-based skill matching (all-MiniLM-L6-v2)
│   ├── executor.py        Safe skill execution with param filtering
│   └── autopilot.py       Auto-pilot (LLM-driven analysis plans)
│
├── data/           Data understanding & preparation
│   ├── profiler.py    Dataset profiling (types, stats, distributions)
│   ├── schema.py      Schema inference (email, phone, ordinal detection)
│   ├── validator.py   Data quality checks (nulls, duplicates, score 0-100)
│   ├── cleaner.py     Data cleaning (imputation, standardization)
│   ├── ocr.py         Image text extraction
│   └── fingerprint.py LLM-first domain understanding (no hardcoded domains)
│
├── analysis/       Statistical & ML analysis
│   ├── descriptive.py    Descriptive statistics, compare_groups
│   ├── correlation.py    Correlation analysis
│   ├── hypothesis.py     t-test, chi-square, ANOVA, Mann-Whitney
│   ├── effect_size.py    Cohen's d, Cramér's V, odds ratio
│   ├── anomaly.py        IQR, Z-score, Isolation Forest, LOF
│   ├── engineering.py    Feature engineering (binning, interactions)
│   ├── selection.py      Feature selection (mutual info, RFE, SHAP)
│   ├── classification.py 8 algorithms (RF, XGBoost, LightGBM, etc.)
│   ├── regression.py     Linear, Ridge, Lasso, RF, XGBoost
│   ├── clustering.py     K-Means, DBSCAN, hierarchical
│   ├── dimensionality.py PCA, t-SNE, UMAP
│   ├── explain.py        SHAP values, feature importance
│   ├── timeseries.py     Decomposition, forecasting, change points
│   ├── survival.py       Kaplan-Meier, Cox PH
│   ├── threshold.py      ROC-based optimal cutoffs
│   └── query.py          6 query skills (filter, pivot, top_n, smart_query)
│
├── nlp/            Natural language processing
│   ├── sentiment.py  VADER + TextBlob sentiment
│   ├── text_stats.py Word count, readability, keywords
│   ├── topics.py     LDA/NMF topic modeling
│   ├── entities.py   spaCy NER (fallback: regex)
│   └── intent.py     TF-IDF + LogReg intent classification
│
├── viz/            Visualization
│   └── charts.py     13 chart types, auto_chart, dashboards
│
├── export/         Report generation
│   ├── pdf.py        ReportLab PDF reports
│   ├── docx.py       python-docx Word documents
│   ├── pptx.py       python-pptx PowerPoint presentations
│   └── templates/    Executive + detailed templates
│
├── llm/            LLM integration
│   ├── provider.py    LLMProvider ABC
│   ├── failover.py    Task-aware failover across providers
│   ├── groq.py        Groq (Llama 3.3 70B, OpenAI SDK)
│   ├── gemini.py      Gemini Flash 2.0 (google-genai SDK)
│   ├── ollama.py      Local Ollama inference
│   ├── claude.py      Anthropic Claude
│   ├── openai.py      OpenAI GPT models
│   └── prompts/       Prompt templates + skill catalog
│
└── utils/          Shared utilities
    ├── helpers.py     Data loading, column helpers, logging
    ├── serializer.py  JSON serialization (numpy/pandas safe)
    ├── uploader.py    Result upload utility
    ├── config.py      Centralized configuration
    └── report_data.py Report data formatting
```

## Backend Structure

```
backend/app/
├── main.py          FastAPI app, CORS, lifespan, router registration
├── api/
│   ├── data.py      /upload, /preview, /profile, /fingerprint, /autopilot
│   ├── analysis.py  /ask, /analyze, /history, /narrative
│   ├── charts.py    /chart/create, /chart/suggest
│   ├── export.py    /export/{fmt}, /export/download/{filename}
│   └── ws.py        WS /ws/chat (streaming)
├── services/
│   ├── analyst.py       SessionManager — two-tier cache (in-memory + SQLite)
│   ├── session_store.py SQLite persistence (WAL mode, 24h expiry)
│   └── data_service.py  File upload handling, temp file management
└── models/
    ├── requests.py      Pydantic v2 request schemas
    └── responses.py     Pydantic v2 response schemas
```

---

## Data Pipeline

```
Upload → Profile → Fingerprint (LLM) → Autopilot (background)
                                            ↓
User question → Semantic Router → Executor → Skill → LLM Narrative → Response
```

### Stage 1: Upload
`POST /api/upload` → `DataService.save_upload()` → `SessionManager.create_session()` → Analyst loads DataFrame → session persisted to SQLite + in-memory cache.

### Stage 2: Profiling
`profiler.py` runs automatically. Detects column types (bool BEFORE numeric — numpy treats bool as numeric), computes stats, distributions, missing values, duplicates.

### Stage 3: Domain Understanding (LLM-first)
`POST /api/fingerprint/{session_id}` → builds data snapshot (<2000 tokens) → LLM classifies domain freely (no hardcoded domains) → returns `DatasetUnderstanding` with domain, target column, observations, suggested questions. Cached in SQLite.

### Stage 4: Auto-Pilot (background)
Triggered after fingerprinting. LLM generates analysis plan (max 5 steps) → serial execution with 1s delay → LLM summarizes findings. Safety: `MAX_STEPS=5`, `STEP_TIMEOUT=30s`.

### Stage 5: Question Routing

```
User question
    ↓
1. Keyword overrides (chart + query regex)  ← instant
2. Semantic router (embeddings)             ← ~5ms, local
3. Smart query (LLM-generated pandas)       ← ~300ms
4. Profile fallback (profile_data)          ← last resort
```

**Semantic router** uses sentence-transformers (all-MiniLM-L6-v2, 90MB, local). Encodes 30 skill descriptions into embeddings, matches by cosine similarity with 0.35 threshold. Degrades gracefully if not installed.

**Parameter enrichment** after routing: extracts target column, chart type, date column, mentioned columns from the question.

### Stage 6: Execution
`Executor` resolves skill name → inspects function signature → filters parameters → calls skill → returns `ExecutionResult`.

### Stage 7: Narration
`_sanitize_for_narration()` strips base64/paths → `FailoverProvider.generate_narrative()` (Gemini primary, Groq fallback) → narrative + key_points + suggestions. Template fallback if LLM fails.

---

## LLM Architecture

### Providers

| Provider | Model | SDK | Strengths |
|----------|-------|-----|-----------|
| **Groq** | Llama 3.3 70B | `openai` (base_url override) | Speed (~300 tok/s), reasoning |
| **Gemini** | Flash 2.0 | `google-genai` | Reliable JSON, low hallucination |
| **Ollama** | Local models | REST API | No API key, privacy |
| **Claude** | Anthropic Claude | `anthropic` | Quality reasoning |
| **OpenAI** | GPT models | `openai` | Broad capability |

### Task-Aware Routing

```
FailoverProvider
├── routing      → Groq (fast) → Gemini
├── narrative    → Gemini (safe) → Groq
├── chart_suggest → Gemini (JSON) → Groq
├── chart_insight → Groq (fast) → Gemini
├── understand   → Gemini → Groq  (D3, no deterministic)
├── plan         → Gemini → Groq  (D3, no deterministic)
└── summary      → Gemini → Groq  (D3, no deterministic)
```

**D1 tasks** (routing, narratives, charts): 3 layers — Primary → Fallback → Deterministic fallback
**D3 tasks** (understanding, plan, summary): 2 layers — Primary → Fallback only (LLM or nothing)

Every response includes `provider_used` metadata for debugging.

### Hallucination Defense (3 layers)
1. `chart_summary()` — structured data summary with actual numbers
2. `_sanitize_for_narration()` — strips base64/paths, injects column names
3. Narration prompt — "Only cite numbers that appear verbatim in the analysis results"

### Key availability

| Keys Available | Behavior |
|---|---|
| Both Groq + Gemini | Full task-aware routing |
| Groq only | Groq handles everything |
| Gemini only | Gemini handles everything |
| Neither | Deterministic fallback (no LLM) |

---

## Session Persistence

### Two-Tier Cache

```
┌───────────────────────┐
│ Tier 1: In-Memory     │  ← Fast path (<1ms)
│ {session_id: Analyst}  │
└───────────┬───────────┘
            │ cache miss
            ▼
┌───────────────────────┐
│ Tier 2: SQLite (WAL)  │  ← Cold start (<2s)
│ sessions.db           │
│ - session metadata    │
│ - analysis_history    │
│ - domain/autopilot    │
│ - 24h auto-expiry     │
└───────────────────────┘
```

### SQLite Schema

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    columns_json TEXT,
    shape_json TEXT,
    analysis_history TEXT DEFAULT '[]',
    domain TEXT DEFAULT NULL,
    domain_confidence REAL DEFAULT NULL,
    domain_explanation TEXT DEFAULT NULL,
    autopilot_status TEXT DEFAULT NULL,
    autopilot_results TEXT DEFAULT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    last_accessed TEXT DEFAULT (datetime('now'))
);
```

### Lifecycle
- **Startup:** Init SQLite, enable WAL, cleanup sessions >24h
- **Runtime:** Write to both tiers on upload/analysis. Cold start reconstructs Analyst from `file_path`.
- **Shutdown:** Close SQLite cleanly. Files kept on disk for restoration.

Configuration: WAL journal mode, 5000ms busy timeout. Override path: `DATAPILOT_DB_PATH` env var.

---

## Key Design Decisions

- **Groq + Gemini dual LLM**: Task-aware routing exploits each provider's strengths
- **Semantic routing**: Local embeddings handle most questions without LLM API calls
- **Auto-generated skill catalog**: Function docstrings → LLM prompt at import time
- **Safe execution**: Executor inspects function signatures to filter parameters
- **Session-based**: Each upload creates an isolated Analyst instance
- **Contract**: Every skill returns `{"status": "success|error", ...}`
- **SQLite persistence**: WAL mode, two-tier cache, 24h expiry
- **Hallucination defense**: 3-layer chain (chart_summary → sanitizer → prompt)
- **LLM-first understanding**: No hardcoded domains — LLM freely classifies any dataset
- **Smart query sandbox**: AST validation, safe proxies, thread timeout, circuit breaker

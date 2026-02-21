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
│  - File upload handling      │
│  - 16 API endpoints          │
│  - WebSocket streaming       │
│  - SQLite persistence (WAL)  │
└──────────┬───────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌──────────────────┐
│  Engine │  │  LLM Layer       │
│  81+    │  │  FailoverProvider│
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
│   ├── analyst.py    Analyst class — ask(), profile(), classify(), etc.
│   ├── router.py     LLM-powered question → skill routing
│   ├── executor.py   Safe skill execution with param filtering
│   └── autopilot.py  Auto-pilot analysis (D3: LLM-driven plans)
│
├── data/           Data understanding & preparation
│   ├── profiler.py   Dataset profiling
│   ├── schema.py     Schema inference
│   ├── validator.py  Data validation
│   ├── cleaner.py    Data cleaning / curation
│   ├── ocr.py        Image text extraction
│   └── fingerprint.py Dataset understanding (D3: LLM snapshot + classification)
│
├── analysis/       Statistical & ML analysis
│   ├── descriptive.py    Descriptive statistics
│   ├── correlation.py    Correlation analysis
│   ├── hypothesis.py     Hypothesis testing
│   ├── effect_size.py    Effect size calculations
│   ├── anomaly.py        Outlier detection
│   ├── engineering.py    Feature engineering
│   ├── selection.py      Feature selection
│   ├── classification.py Classification (8 algorithms)
│   ├── regression.py     Regression
│   ├── clustering.py     Clustering
│   ├── dimensionality.py PCA / dimensionality reduction
│   ├── explain.py        SHAP explanations
│   ├── timeseries.py     Time series & forecasting
│   ├── survival.py       Survival analysis
│   └── threshold.py      Optimal threshold finding
│
├── nlp/            Natural language processing
│   ├── sentiment.py  Sentiment analysis
│   ├── text_stats.py Text statistics & keywords
│   ├── topics.py     Topic modeling
│   ├── entities.py   Named entity extraction
│   └── intent.py     Intent classification
│
├── viz/            Visualization
│   └── charts.py     Chart creation (10+ types), auto_chart, dashboards
│
├── export/         Report generation
│   ├── pdf.py        PDF reports
│   ├── docx.py       Word documents
│   ├── pptx.py       PowerPoint presentations
│   └── templates/    Report templates
│
├── llm/            LLM integration
│   ├── provider.py    Abstract base class (LLMProvider ABC)
│   ├── groq.py        Groq provider — Llama 3.3 70B (routing, chart insights)
│   ├── gemini.py      Gemini Flash 2.0 (narratives, suggestions, D3 understanding)
│   ├── failover.py    Task-aware failover across providers
│   ├── ollama.py      Local Ollama provider
│   ├── claude.py      Anthropic Claude provider
│   ├── openai.py      OpenAI provider
│   └── prompts/       Prompt templates (package)
│       ├── __init__.py     Re-exports all prompts
│       ├── base.py         Core prompts (routing, narrative, chart, skill catalog)
│       └── fingerprint_prompts.py  Domain detection prompts
│
└── utils/          Shared utilities
    ├── helpers.py    Data loading, column helpers, logging
    ├── serializer.py JSON serialization (numpy/pandas safe)
    ├── uploader.py   Result upload utility
    ├── config.py     Centralized configuration
    └── report_data.py Report data formatting
```

## Backend Structure

```
backend/app/
├── main.py          FastAPI app, CORS, lifespan, router registration
├── api/
│   ├── data.py      POST /api/upload, GET /api/preview, GET /api/profile,
│   │                POST /api/fingerprint/{session_id}
│   ├── analysis.py  POST /api/ask, POST /api/analyze,
│   │                GET /api/history, GET /api/narrative
│   ├── charts.py    POST /api/chart/create, GET /api/chart/suggest
│   ├── export.py    POST /api/export/{fmt}, GET /api/export/download/{filename}
│   └── ws.py        WS /api/ws/chat (streaming)
├── services/
│   ├── analyst.py       SessionManager — two-tier cache (in-memory + SQLite)
│   ├── session_store.py SQLite persistence (WAL mode, 24h expiry)
│   └── data_service.py  File upload handling, temp file management
├── models/
│   ├── requests.py      Pydantic v2 request schemas
│   └── responses.py     Pydantic v2 response schemas
└── data/
    └── sessions.db      SQLite database (WAL mode, auto-created)
```

## Request Flow: `analyst.ask("What predicts churn?")`

1. **Router** sends the question + dataset context + skill catalog to the LLM
2. **LLM** returns `{skill: "classify", params: {target: "churn"}, confidence: 0.92}`
3. **Executor** looks up `classify()`, filters params, calls `classify(df, target="churn")`
4. **Skill** runs classification (auto-selects best algorithm), returns result dict
5. **LLM** generates a narrative from the result: text + key_points + suggestions
6. **AnalystResult** packages everything and returns to the caller

## Backend API Flow

```
POST /api/upload  →  DataService.save_upload()  →  SessionManager.create_session()
                                                         │
                                              SQLite persistence (session_store)
                                                         │
POST /api/ask     →  Analyst.ask(question)  ─────────────┘
                     ├── Router.route()          (LLM call via FailoverProvider)
                     ├── Executor.execute()      (skill execution)
                     └── Provider.generate_narrative()  (LLM call)

WS /api/ws/chat   →  Same flow with progress updates streamed back
```

## Session Persistence (D2)

```
┌─────────────────────┐
│ Tier 1: In-Memory   │  ← Fast path (hot sessions)
│ dict[str, Analyst]  │
└────────┬────────────┘
         │ miss
         ▼
┌─────────────────────┐
│ Tier 2: SQLite      │  ← Cold start recovery
│ sessions.db (WAL)   │
│ - session metadata  │
│ - analysis_history  │
│ - domain/autopilot  │
│ - 24h auto-expiry   │
└─────────────────────┘
```

- **Hot path**: In-memory dict lookup (<1ms)
- **Cold start**: Reconstruct Analyst from stored file_path (<2s)
- **Persistence**: All analysis history, domain info, and autopilot results survive restarts

## LLM Architecture (D1)

```
FailoverProvider
├── Task: routing      → Groq (fast) → Gemini
├── Task: narrative    → Gemini (safe) → Groq
├── Task: chart_suggest → Gemini (JSON) → Groq
├── Task: chart_insight → Groq (fast) → Gemini
├── Task: fingerprint  → Gemini (JSON) → Groq
├── Task: understand   → Gemini → Groq (D3, no deterministic)
├── Task: plan         → Gemini → Groq (D3, no deterministic)
└── Task: summary      → Gemini → Groq (D3, no deterministic)
```

Each task routes to the provider best suited for it, with automatic failover. D1 tasks have 3 layers (primary → fallback → deterministic). D3 tasks have 2 layers (primary → fallback only).

## Key Design Decisions

- **Groq + Gemini dual LLM**: Task-aware routing exploits each provider's strengths. FailoverProvider manages failover automatically.
- **Auto-generated skill catalog**: Function docstrings → LLM prompt at import time
- **Safe execution**: Executor inspects function signatures to filter parameters
- **Session-based**: Each upload creates an isolated Analyst instance
- **Contract**: Every skill returns `{"status": "success|error", ...}`
- **SQLite session persistence**: WAL mode, two-tier cache (in-memory + DB), 24h expiry
- **Hallucination defense**: 3-layer chain (chart_summary → sanitizer → prompt constraint)
- **D3 LLM-first approach**: No hardcoded domains — LLM freely classifies any dataset. No deterministic fallback for understanding (LLM or nothing).

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
└──────────┬───────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌──────────────┐
│  Engine │  │  LLM Provider│
│  81+    │  │  - Ollama    │
│  skills │  │  - Claude    │
└─────────┘  │  - OpenAI   │
             └──────────────┘
```

## Engine Package Structure

```
engine/datapilot/
├── core/           Main public API
│   ├── analyst.py    Analyst class — ask(), profile(), classify(), etc.
│   ├── router.py     LLM-powered question → skill routing
│   └── executor.py   Safe skill execution with param filtering
│
├── data/           Data understanding & preparation
│   ├── profiler.py   Dataset profiling
│   ├── schema.py     Schema inference
│   ├── validator.py  Data validation
│   ├── cleaner.py    Data cleaning / curation
│   └── ocr.py        Image text extraction
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
│   ├── provider.py   Abstract base class
│   ├── ollama.py     Local Ollama provider
│   ├── claude.py     Anthropic Claude provider
│   ├── openai.py     OpenAI provider
│   └── prompts.py    Auto-generated skill catalog & prompt templates
│
└── utils/          Shared utilities
    ├── helpers.py    Data loading, column helpers, logging
    ├── serializer.py JSON serialization (numpy/pandas safe)
    ├── uploader.py   Result upload utility
    ├── config.py     Centralized configuration
    └── report_data.py Report data formatting
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
POST /api/ask     →  Analyst.ask(question)  ─────────────┘
                     ├── Router.route()          (LLM call)
                     ├── Executor.execute()      (skill execution)
                     └── Provider.generate_narrative()  (LLM call)

WS /api/ws/chat   →  Same flow with progress updates streamed back
```

## Key Design Decisions

- **Local-first LLM**: Ollama is the default — no API keys required
- **Auto-generated skill catalog**: Function docstrings → LLM prompt at import time
- **Safe execution**: Executor inspects function signatures to filter parameters
- **Session-based**: Each upload creates an isolated Analyst instance
- **Contract**: Every skill returns `{"status": "success|error", ...}`

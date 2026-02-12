# DataPilot Engine

Core Python analysis library — 81 analytical skills across 29 modules.

## Package Structure
```
datapilot/
├── core/        Analyst (public API), Router (keyword→LLM), Executor (param filtering + dispatch), AutoPilot (domain recipes)
├── data/        profiler, schema, validator, cleaner, ocr, fingerprint
├── analysis/    descriptive, correlation, anomaly, classification, regression,
│                clustering, timeseries, hypothesis, effect_size, survival,
│                dimensionality, selection, threshold, engineering, explain
├── nlp/         sentiment, entities, topics, intent, text_stats
├── viz/         charts (static matplotlib/seaborn, interactive plotly)
├── export/      pdf (reportlab), docx (python-docx), pptx (python-pptx), templates/
├── llm/         provider ABC, groq, gemini, failover, ollama, claude, openai, prompts
└── utils/       helpers, serializer, uploader, config, report_data
```

## Contract
Every skill function returns: `{"status": "success|error", ...}` dict — never raw DataFrames.
Upload variants (`_and_upload()`) call `upload_result()` after success.

## Import Pattern
All internal imports use relative paths:
```python
from ..utils.helpers import load_data, setup_logging, get_numeric_columns
from ..utils.serializer import safe_json_serialize
```

## Key Classes
- **Analyst** (`core/analyst.py`): Public API. Loads data, routes questions via Router, executes via Executor, generates narratives via LLM provider. Maintains `history: List[AnalystResult]`.
- **Router** (`core/router.py`): Priority chain — chart keywords → keyword table → primary LLM → Groq fallback → profile_data. Enriches params (target, date_col, columns).
- **Executor** (`core/executor.py`): Dispatches to skill functions. Filters parameters to match function signatures.
- **LLMProvider** (`llm/provider.py`): ABC with `route_question()`, `generate_narrative()`, `suggest_chart()`.
- **FailoverProvider** (`llm/failover.py`): Task-aware routing across multiple LLM providers. Each method (routing, narrative, suggest_chart, chart_insight) has its own primary/fallback order.
- **GeminiProvider** (`llm/gemini.py`): Google Gemini Flash 2.0 via google-genai SDK. Primary for narratives and chart suggestions.
- **fingerprint_dataset** (`data/fingerprint.py`): 3-layer domain detection (column keywords → value profiling → LLM confirmation). Returns FingerprintResult with domain, confidence, explanation, suggested_target.
- **AutoPilot** (`core/autopilot.py`): Runs domain-specific analysis recipes using existing skills. Confidence-aware: HIGH=full domain, MEDIUM=hybrid, LOW=general. Sequential execution with rate limit protection.

## Hallucination Defense (3 layers)
1. `chart_summary` in viz/charts.py — structured data summary attached to chart results
2. `_sanitize_for_narration()` in core/analyst.py — strips base64/paths, injects column names
3. Narration prompt — "Only cite numbers that appear verbatim in the analysis results"

## Dependencies
Core: pandas, numpy, scikit-learn, scipy, statsmodels
ML: xgboost, lightgbm, optuna, shap
NLP: textblob, vaderSentiment, spacy, gensim
Viz: matplotlib, seaborn, plotly
Export: reportlab, python-docx, python-pptx
Optional: prophet, lifelines, hdbscan, pytesseract
LLM: google-genai (Gemini), openai (Groq)

## Testing
```bash
pytest tests/unit/ -v          # unit tests
pytest tests/integration/ -v   # integration tests
```
Test data lives in `tests/test_data/`.

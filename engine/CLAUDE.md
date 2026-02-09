# DataPilot Engine

Core Python analysis library with 114+ functions across 29 modules.

## Package Structure

```
datapilot/
├── core/        Analyst class, router, executor
├── data/        profiler, schema, validator, cleaner, ocr
├── analysis/    stats, ML (classification, regression, clustering, timeseries, survival)
├── nlp/         sentiment, entities, topics, intent, text_stats
├── viz/         charts (static, interactive, dashboard)
├── export/      pdf, docx, pptx with templates
├── llm/         provider ABC, ollama, claude, openai
└── utils/       helpers, serializer, uploader, config, report_data
```

## Contract

Every skill function returns: `{"status": "success|error", ...}`

Upload variants (`_and_upload()`) call `upload_result()` after success.

## Import Pattern

All internal imports use relative paths:
```python
from ..utils.helpers import load_data, setup_logging, get_numeric_columns
from ..utils.serializer import safe_json_serialize
```

## Dependencies

Core: pandas, numpy, scikit-learn, scipy, statsmodels
ML: xgboost, lightgbm, catboost, optuna, shap
NLP: textblob, vaderSentiment, spacy, gensim
Viz: matplotlib, seaborn, plotly
Export: reportlab, python-docx, python-pptx
Optional: prophet, lifelines, hdbscan, pytesseract

## Testing

```bash
pytest tests/unit/ -v          # unit tests
pytest tests/integration/ -v   # integration tests
```

Test data lives in `tests/test_data/` and `examples/`.

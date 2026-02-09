# DataPilot

AI-powered data analysis engine with 114+ analytical functions.

## Architecture

```
engine/datapilot/     Python analysis engine (core library)
backend/app/          FastAPI REST API
frontend/             Next.js 14 web UI
tests/                pytest test suite
```

## Tech Stack

- **Engine**: Python 3.12, pandas, scikit-learn, XGBoost, LightGBM, SHAP, Prophet
- **Backend**: FastAPI, uvicorn, Pydantic v2
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, shadcn/ui, Plotly.js
- **LLM**: Ollama (default, local), Claude API, OpenAI API
- **Infra**: Docker Compose, GitHub Actions CI

## Conventions

- All engine functions return `{"status": "success|error", ...}` dicts
- Relative imports within `engine/datapilot/` (e.g., `from ..utils.helpers import load_data`)
- Backend uses Pydantic v2 models for request/response validation
- Frontend uses App Router, server components by default
- Tests use pytest with fixtures in `conftest.py`

## Key Commands

- `pytest tests/` — run all tests
- `cd backend && uvicorn app.main:app --reload` — start API server
- `cd frontend && npm run dev` — start frontend dev server
- `docker compose up` — start full stack

## Code Quality

- Python: ruff for linting/formatting, mypy for type checking
- TypeScript: eslint + prettier
- All functions must have docstrings
- Type hints required on all public functions

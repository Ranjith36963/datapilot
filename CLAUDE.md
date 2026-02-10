# DataPilot — AI Data Analysis Platform

## Stack
- Backend: Python 3.13, FastAPI 0.115+, Pydantic v2
- Frontend: Next.js 16, React 19, TypeScript, Tailwind CSS 4
- Engine: 81 analytical skills (pandas, scikit-learn, statsmodels, XGBoost, LightGBM)
- LLM: Groq (default via openai SDK), Ollama (local fallback), Claude, OpenAI
- Testing: pytest (backend), Next.js build (frontend)

## Commands
- Backend: `cd backend && python -m uvicorn app.main:app --reload --port 8000`
- Frontend: `cd frontend && npm run dev`
- Tests: `python -m pytest tests/ -v`
- Frontend build: `cd frontend && npm run build`
- Syntax check: `python -c "import py_compile; py_compile.compile('path/to/file.py', doraise=True)"`

## Architecture
- /engine/datapilot/ — 81 analytical skills (core/, analysis/, data/, nlp/, viz/, export/, llm/, utils/)
- /backend/app/ — FastAPI REST API (16 endpoints) + WebSocket
- /frontend/src/ — Next.js app (4 pages: Upload, Explore, Visualize, Export)
- /tests/ — pytest unit tests
- /docs/ — API reference, architecture, quick start

## Request Flow
User question → Router (keyword priority → LLM fallback) → Executor (param filtering) → Skill → LLM narrative → Response

## Critical Rules
- NEVER modify working Explore page code without explicit approval
- NEVER change the 3-layer hallucination defense (chart_summary → sanitizer → prompt)
- Always run `python -m pytest tests/ -v` after changes
- Always run frontend build after TSX changes
- Skills must return {"status": "success|error", ...} dict — never raw DataFrames
- correlation_matrix is a helper, NOT a routable skill
- Groq uses openai SDK with base_url="https://api.groq.com/openai/v1"
- Column names in suggestions must match actual dataset columns (spaces, not underscores)

## What Claude Gets Wrong (add to this list)
- Tries to use `rmdir /s /q` instead of `Remove-Item -Recurse -Force` on PowerShell
- Forgets that numpy treats bool as numeric — check is_bool_dtype BEFORE is_numeric_dtype
- Suggests correlation_matrix when it should suggest analyze_correlations
- Hardcodes domain-specific column names instead of reading from data_context

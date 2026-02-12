# DataPilot — AI Data Analysis Platform

## Stack
- Backend: Python 3.13, FastAPI 0.115+, Pydantic v2
- Frontend: Next.js 16, React 19, TypeScript, Tailwind CSS 4
- Engine: 81 analytical skills (pandas, scikit-learn, statsmodels, XGBoost, LightGBM)
- LLM: Groq (routing/insights) + Gemini (narratives/suggestions) via FailoverProvider, with Ollama/Claude/OpenAI as alternatives
- Testing: pytest (backend), Next.js build (frontend)

## LLM Strategy (Phase 2)
- Two LLM providers: Groq (Llama 3.3 70B) + Gemini (Flash 2.0)
- Task-aware routing via FailoverProvider:
  - Routing: Groq primary (fast) → Gemini fallback
  - Narratives: Gemini primary (lower hallucination) → Groq fallback
  - Chart suggestions: Gemini primary (reliable JSON) → Groq fallback
  - Chart insights: Groq primary (fast) → Gemini fallback
  - Fingerprinting: Gemini primary (accurate JSON) → Groq fallback
- Every task has 3 layers: Primary → Failover → Deterministic fallback
- Response metadata logs which provider handled each task

## Commands
- Backend: `cd backend && python -m uvicorn app.main:app --reload --port 8000`
- Frontend: `cd frontend && npm run dev`
- Tests: `python -m pytest tests/ -v`
- Frontend build: `cd frontend && npm run build`
- Syntax check: `python -c "import py_compile; py_compile.compile('path/to/file.py', doraise=True)"`

## Architecture
- /engine/datapilot/ — 81 analytical skills (core/, analysis/, data/, nlp/, viz/, export/, llm/, utils/)
  - llm/ now includes: provider ABC, groq, gemini, failover, ollama, claude, openai, prompts
  - data/ now includes: profiler, schema, validator, cleaner, ocr, fingerprint
  - core/ now includes: analyst, router, executor, autopilot
- /backend/app/ — FastAPI REST API (18 endpoints) + WebSocket
- /frontend/src/ — Next.js app (4 pages: Upload, Explore, Visualize, Export)
- /tests/ — pytest unit tests
- /docs/ — API reference, architecture, quick start

## Phase 2 Deliverables
- D1: Multi-LLM (Gemini + task-aware failover) ← CURRENT
- D2: SQLite session persistence (replace in-memory dict)
- D3: Dataset fingerprinting (3-layer domain detection + explainability)
- D4: Auto-pilot analysis (domain-aware recipes, confidence tiers, skill budget)
- D5: Domain-specific narrative templates (45 templates across 9 skills × 6 domains)

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
- FailoverProvider wraps all LLM calls — never call GroqProvider or GeminiProvider directly from analyst.py
- Gemini uses google-genai SDK (NOT google-generativeai — that's the old package)
- Sessions backed by SQLite — SessionManager has two-tier cache (in-memory + DB)
- Every LLM response must include provider_used in metadata for debugging

## What Claude Gets Wrong (add to this list)
- Tries to use `rmdir /s /q` instead of `Remove-Item -Recurse -Force` on PowerShell
- Forgets that numpy treats bool as numeric — check is_bool_dtype BEFORE is_numeric_dtype
- Suggests correlation_matrix when it should suggest analyze_correlations
- Hardcodes domain-specific column names instead of reading from data_context

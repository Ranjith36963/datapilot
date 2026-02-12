# DataPilot Backend

FastAPI REST API serving the DataPilot engine.

## Structure
```
app/
├── main.py          FastAPI app, CORS, lifespan, router registration
├── api/
│   ├── data.py      POST /api/upload, GET /api/preview, GET /api/profile
│   ├── analysis.py  POST /api/ask, POST /api/analyze
│   ├── charts.py    POST /api/chart/create, GET /api/chart/suggest
│   ├── export.py    POST /api/export/{fmt}, GET /api/export/download/{filename}
│   └── ws.py        WS /api/ws/chat (streaming)
├── services/
│   ├── analyst.py   SessionManager — wraps engine Analyst, two-tier cache (in-memory + SQLite)
│   ├── session_store.py  SQLite persistence (WAL mode), session CRUD, 24h expiry
│   └── data_service.py  File upload handling, temp file management
├── models/
│   ├── requests.py  Pydantic v2 request schemas (AskRequest, ChartRequest, ExportRequest)
│   └── responses.py Pydantic v2 response schemas (AskResponse, SuggestChartResponse, ExportResponse)
└── middleware/       CORS configuration
```

## Conventions
- All routes prefixed with `/api/`
- Pydantic v2 models for all request/response schemas
- Session persistence: SQLite (WAL mode) via SessionStore + in-memory cache via SessionManager
  - Hot path: in-memory dict (no DB hit)
  - Cold start: reconstruct Analyst from file_path stored in SQLite
  - SessionStore at `backend/app/services/session_store.py`
- Files stored in `/tmp/datapilot/` (uploads) and `/tmp/datapilot/exports/` (reports)
- Session ID passed via `x-session-id` header
- Streaming responses via WebSocket for chat

## Request Flow
1. Frontend sends request with `x-session-id` header
2. API handler calls `session_manager.get_session(session_id)` to get Analyst instance
3. Analyst method called (ask, chart, export, suggest_chart, etc.)
4. Result serialized via Pydantic response model
5. Charts: base64-encoded PNG or Plotly JSON returned inline

## Running
```bash
cd backend && python -m uvicorn app.main:app --reload --port 8000
```
API docs at `http://localhost:8000/docs`

## Key Dependencies
- fastapi, uvicorn, python-multipart (for file uploads)
- aiosqlite (async SQLite for session persistence)
- google-genai (Gemini LLM provider)
- Engine imported via sys.path manipulation in services/analyst.py

## Phase 2 Endpoints (planned)
- `GET /api/autopilot/{session_id}` — auto-pilot analysis status/results
- `PUT /api/session/domain` — user domain correction

# DataPilot Backend

FastAPI REST API serving the DataPilot engine.

## Structure

```
app/
├── main.py          FastAPI app, CORS, lifespan
├── api/
│   ├── data.py      POST /api/upload, GET /api/preview, GET /api/profile
│   ├── analysis.py  POST /api/ask, POST /api/analyze
│   ├── charts.py    POST /api/chart/create
│   ├── export.py    POST /api/export/{format}
│   └── ws.py        WS /api/ws/chat (streaming)
├── services/
│   ├── analyst.py   Wraps engine Analyst, manages sessions
│   └── data_service.py  File upload handling
├── models/
│   ├── requests.py  Pydantic request schemas
│   └── responses.py Pydantic response schemas
└── middleware/
    └── cors.py      CORS configuration
```

## Conventions

- All routes prefixed with `/api/`
- Pydantic v2 models for all request/response schemas
- Session-based: `{session_id: Analyst}` in-memory dict
- Files stored in `/tmp/datapilot/`
- Streaming responses via WebSocket for chat

## Running

```bash
uvicorn app.main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

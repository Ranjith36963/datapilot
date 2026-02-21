# API Reference

Base URL: `http://localhost:8000`

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Authentication

No authentication required. Sessions are identified by `x-session-id` header.

---

## Data Endpoints

### POST /api/upload

Upload a dataset file. Creates a new analysis session.

**Request:** `multipart/form-data` with `file` field

**Supported formats:** CSV, XLSX, XLS, JSON, Parquet (max 100 MB)

**Response:**
```json
{
  "status": "success",
  "session_id": "a1b2c3d4e5f6",
  "filename": "sales.csv",
  "shape": [1000, 15],
  "columns": [
    {"name": "revenue", "dtype": "float64", "semantic_type": "numeric", "n_unique": 987, "null_pct": 0.0}
  ],
  "preview": [{"revenue": 1234.56, "region": "West"}]
}
```

### GET /api/preview

Preview dataset rows.

**Headers:** `x-session-id: <session_id>`

**Query params:** `rows` (default: 20, max: 500)

**Response:**
```json
{
  "shape": [1000, 15],
  "columns": ["revenue", "region", "date"],
  "data": [{"revenue": 1234.56, "region": "West"}]
}
```

### GET /api/profile

Full dataset profile (types, stats, distributions).

**Headers:** `x-session-id: <session_id>`

### POST /api/fingerprint/{session_id}

Understand the dataset using LLM-driven domain classification. No hardcoded domains — the LLM freely classifies any dataset into a domain, identifies the target column, and generates observations and suggested questions. Returns cached result if available. Kicks off autopilot in the background.

**Response:**
```json
{
  "domain": "telecom customer churn",
  "domain_short": "Telecom",
  "confidence": 0.9,
  "target_column": "Churn",
  "target_type": "classification",
  "key_observations": [
    "The churn rate is approximately 14.5%",
    "High correlation between tenure and churn"
  ],
  "suggested_questions": [
    "What factors drive customer churn?",
    "How does tenure affect churn rate?",
    "Are there anomalies in monthly charges?"
  ],
  "data_quality_notes": [
    "No missing values detected",
    "High correlation between TotalCharges and tenure"
  ],
  "provider_used": "gemini"
}
```

### GET /api/autopilot/{session_id}

Auto-pilot analysis status and results. Returns the current state of the background autopilot run triggered by fingerprinting.

**Response (planning):**
```json
{"status": "planning"}
```

**Response (running):**
```json
{"status": "running"}
```

**Response (complete):**
```json
{
  "status": "complete",
  "completed_steps": 5,
  "total_steps": 5,
  "results": [
    {
      "step": "describe_data",
      "status": "complete",
      "question": "Describe the dataset",
      "narrative": "This dataset contains 7043 customer records..."
    }
  ],
  "summary": "The analysis reveals that contract type and tenure are the strongest predictors of churn..."
}
```

**Response (failed):**
```json
{"status": "failed", "error": "Both LLM providers unavailable"}
```

---

## Analysis Endpoints

### POST /api/ask

Ask a natural-language question. Routes to the best skill automatically via semantic router + FailoverProvider.

**Headers:** `x-session-id: <session_id>`

**Body:**
```json
{
  "question": "What are the strongest correlations?",
  "narrate": true,
  "conversation_context": [
    {"question": "Describe the data", "summary": "7043 rows, 21 columns..."}
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "question": "What are the strongest correlations?",
  "skill": "analyze_correlations",
  "confidence": 0.95,
  "reasoning": "User asked about correlations",
  "route_method": "semantic",
  "result": {},
  "narrative": "The strongest correlation is between...",
  "narrative_pending": false,
  "key_points": ["Revenue and price are highly correlated (r=0.89)"],
  "suggestions": ["Try running a regression analysis"],
  "code_snippet": "analyst.ask('analyze correlations')",
  "columns_used": ["revenue", "price"],
  "elapsed_seconds": 2.3,
  "routing_ms": 340.5,
  "execution_ms": 1200.0,
  "narration_ms": 800.0,
  "error": null
}
```

### POST /api/analyze

Run a specific skill directly (bypasses LLM routing).

**Headers:** `x-session-id: <session_id>`

**Body:**
```json
{
  "skill": "describe_data",
  "params": {"columns": ["revenue", "cost"]}
}
```

**Response:**
```json
{
  "status": "success",
  "skill": "describe_data",
  "result": {},
  "elapsed_seconds": 0.5,
  "error": null
}
```

### GET /api/history

Return analysis history for session restoration on frontend refresh.

**Headers:** `x-session-id: <session_id>`

**Response:**
```json
{
  "history": [
    {
      "question": "Describe the data",
      "skill": "describe_data",
      "narrative": "This dataset contains...",
      "key_points": ["7043 rows", "21 columns"],
      "confidence": 0.95,
      "reasoning": "User asked for data description"
    }
  ]
}
```

### GET /api/narrative

Poll for the narrative of a previous /ask result. Blocks up to `timeout` seconds for background narration to finish.

**Headers:** `x-session-id: <session_id>`

**Query params:** `index` (default: -1, latest), `timeout` (default: 30s, max: 60s)

**Response:**
```json
{
  "ready": true,
  "narrative": "The analysis reveals...",
  "key_points": ["Finding 1", "Finding 2"],
  "suggestions": ["Try exploring..."],
  "narration_ms": 800.0
}
```

---

## Chart Endpoints

### POST /api/chart/create

Create a chart from the dataset. Supports 13 chart types.

**Headers:** `x-session-id: <session_id>`

**Body:**
```json
{
  "chart_type": "scatter",
  "x": "price",
  "y": "revenue",
  "hue": "region",
  "title": "Price vs Revenue"
}
```

**Chart types:** `bar`, `barh`, `line`, `scatter`, `histogram`, `box`, `violin`, `heatmap`, `pie`, `area`, `density`, `count`, `pair`

**Response:** Includes `image_base64` (PNG) and/or `plotly_json` (interactive).

### GET /api/chart/suggest

Ask the LLM to suggest the best charts for the data. Uses Gemini (primary) with Groq fallback.

**Headers:** `x-session-id: <session_id>`

**Response:**
```json
{
  "suggestions": [
    {"chart_type": "scatter", "x": "price", "y": "revenue", "title": "Price vs Revenue", "reason": "Strong positive correlation"}
  ]
}
```

---

## Export Endpoints

### POST /api/export/{format}

Generate a report. Format: `pdf`, `docx`, or `pptx`.

**Headers:** `x-session-id: <session_id>`

**Body:**
```json
{
  "title": "Q4 Analysis Report",
  "subtitle": "Prepared by DataPilot",
  "include_history": true
}
```

**Response:**
```json
{
  "status": "success",
  "format": "pdf",
  "filename": "report_abc12345.pdf",
  "download_url": "/api/export/download/report_abc12345.pdf"
}
```

### GET /api/export/download/{filename}

Download a generated report file.

---

## WebSocket

### WS /api/ws/chat

Streaming chat interface with progress updates.

**Client sends:**
```json
{"type": "question", "content": "Describe the data", "session_id": "abc123"}
```

**Server sends (in sequence):**
```json
{"type": "status", "content": "Routing question to best skill..."}
{"type": "status", "content": "Using skill: describe_data (confidence: 95%)"}
{"type": "status", "content": "Running describe_data..."}
{"type": "status", "content": "Generating insights..."}
{"type": "result", "data": {"question": "...", "skill": "...", "narrative": "..."}}
```

---

## System Endpoints

### GET /health

Health check. Returns `{"status": "ok", "version": "0.1.0"}`.

### GET /api/sessions

List active sessions. Returns count and session IDs.

### DELETE /api/sessions/{session_id}

Delete a session and its uploaded data. Removes from both in-memory cache and SQLite.

---

## Endpoint Summary

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/upload | Upload dataset, create session |
| GET | /api/preview | Preview rows |
| GET | /api/profile | Full dataset profile |
| POST | /api/fingerprint/{session_id} | LLM-driven domain understanding |
| GET | /api/autopilot/{session_id} | Auto-pilot status and results |
| POST | /api/ask | Natural language question |
| POST | /api/analyze | Run specific skill |
| GET | /api/history | Session analysis history |
| GET | /api/narrative | Poll for narrative |
| POST | /api/chart/create | Create chart |
| GET | /api/chart/suggest | LLM chart suggestions |
| POST | /api/export/{format} | Generate report |
| GET | /api/export/download/{filename} | Download report |
| WS | /api/ws/chat | Streaming chat |
| GET | /health | Health check |
| GET | /api/sessions | List sessions |
| DELETE | /api/sessions/{session_id} | Delete session |

**Total: 17 endpoints** (13 REST + 1 WebSocket + 2 system + 1 delete)

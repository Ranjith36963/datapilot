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
  "preview": [{"revenue": 1234.56, "region": "West", ...}]
}
```

### GET /api/preview

Preview dataset rows.

**Headers:** `x-session-id: <session_id>`

**Query params:** `rows` (default: 20, max: 500)

### GET /api/profile

Full dataset profile.

**Headers:** `x-session-id: <session_id>`

---

## Analysis Endpoints

### POST /api/ask

Ask a natural-language question. Routes to the best skill automatically.

**Headers:** `x-session-id: <session_id>`

**Body:**
```json
{
  "question": "What are the strongest correlations?",
  "narrate": true
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
  "result": { ... },
  "narrative": "The strongest correlation is between...",
  "key_points": ["Revenue and price are highly correlated (r=0.89)"],
  "suggestions": ["Try running a regression analysis"],
  "elapsed_seconds": 2.3
}
```

### POST /api/analyze

Run a specific skill directly (bypasses LLM routing).

**Body:**
```json
{
  "skill": "describe_data",
  "params": {"columns": ["revenue", "cost"]}
}
```

---

## Chart Endpoints

### POST /api/chart/create

Create a chart from the dataset.

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

**Response:** Includes `image_base64` (PNG) or `plotly_json`.

### GET /api/chart/suggest

Ask the LLM to suggest the best chart for the data.

---

## Export Endpoints

### POST /api/export/{format}

Generate a report. Format: `pdf`, `docx`, or `pptx`.

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
{"type": "result", "data": { "question": "...", "skill": "...", "narrative": "...", ... }}
```

---

## System Endpoints

### GET /health

Health check. Returns `{"status": "ok", "version": "0.1.0"}`.

### GET /api/sessions

List active sessions.

### DELETE /api/sessions/{session_id}

Delete a session and its uploaded data.

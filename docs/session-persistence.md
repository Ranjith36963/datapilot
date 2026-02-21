# Session Persistence (D2)

## Overview

DataPilot uses a **two-tier caching architecture** to balance speed and durability:

- **Tier 1: In-memory cache** — `dict[str, Analyst]` for fast access (<1ms)
- **Tier 2: SQLite persistence** — WAL-mode database for durability across restarts

Sessions survive backend restarts. Analysis history, domain info, and autopilot results are preserved.

## Architecture

```
┌─────────────────────────────┐
│      SessionManager          │
│                              │
│  ┌───────────────────────┐  │
│  │ Tier 1: In-Memory     │  │  ← Hot path: dict lookup
│  │ {session_id: Analyst}  │  │     No DB hit on active sessions
│  └───────────┬───────────┘  │
│              │ cache miss    │
│              ▼               │
│  ┌───────────────────────┐  │
│  │ Tier 2: SQLite (WAL)  │  │  ← Cold path: reconstruct from file_path
│  │ sessions table        │  │     Reads file, rebuilds Analyst, caches it
│  │ 24h auto-expiry       │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

## SessionStore

**File:** `backend/app/services/session_store.py`

Async SQLite store using `aiosqlite` with WAL mode for safe concurrent reads under FastAPI.

### Database Location

Default: `backend/data/sessions.db`

Override via environment variable:
```bash
export DATAPILOT_DB_PATH=/custom/path/sessions.db
```

### Schema

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    columns_json TEXT,              -- JSON array of column metadata
    shape_json TEXT,                -- JSON: {"rows": N, "columns": M}
    analysis_history TEXT DEFAULT '[]',  -- JSON array of analysis summaries
    domain TEXT DEFAULT NULL,            -- D3: domain classification (free text)
    domain_confidence REAL DEFAULT NULL, -- D3: 0.0-1.0
    domain_explanation TEXT DEFAULT NULL, -- D3: JSON of explainability data
    autopilot_status TEXT DEFAULT NULL,   -- D3: "planning"|"running"|"complete"|"failed"
    autopilot_results TEXT DEFAULT NULL,  -- D3: JSON of AutopilotResult
    created_at TEXT DEFAULT (datetime('now')),
    last_accessed TEXT DEFAULT (datetime('now'))
);
```

### Key Methods

```python
class SessionStore:
    async def init_db()                    # Create table, enable WAL mode
    async def create_session(...)          # Insert new session
    async def get_session(session_id)      # Fetch by ID, update last_accessed
    async def update_history(...)          # Update analysis_history JSON
    async def update_domain(...)           # Store domain/confidence/explanation
    async def update_autopilot(...)        # Store autopilot status/results
    async def delete_session(session_id)   # Remove session
    async def cleanup_expired(max_age=24h) # Delete old sessions + files
    async def list_sessions()              # List all session IDs
    async def close()                      # Close connection
```

## SessionManager

**File:** `backend/app/services/analyst.py`

Wraps engine `Analyst` instances with the two-tier cache.

### Access Pattern

```python
async def get_or_restore_session(session_id):
    # 1. Check in-memory dict → fast path
    # 2. Check SQLite → reconstruct Analyst from file_path
    # 3. Neither → raise SessionNotFound
```

### Lazy Reconstruction

Analysts are not serialized — they contain DataFrames. Instead:
1. Store `file_path` in SQLite
2. On cold start, call `pandas.read_csv(file_path)` to reconstruct
3. Restore analysis history from SQLite
4. Cache in-memory for subsequent requests

Reconstruction takes <2s for typical CSV files.

## Lifecycle

### Startup (lifespan hook in `main.py`)

1. Initialize SQLite DB (`session_store.init_db()`)
2. Enable WAL mode + busy timeout
3. Cleanup sessions older than 24 hours (files deleted too)
4. Log recoverable session count

### Runtime

- New upload → write to both in-memory + SQLite
- Analysis complete → update both in-memory history + SQLite
- Frontend refresh → GET /api/history reads from in-memory or SQLite

### Shutdown

- Close SQLite connection cleanly
- Uploaded files kept on disk for future restoration

## Analysis History Format

History is stored as compact JSON summaries (not full result dicts):

```json
[
  {
    "question": "Are there outliers?",
    "skill": "detect_outliers",
    "narrative": "Outlier detection identified...",
    "key_points": ["167 outliers found", "5% of dataset"],
    "confidence": 0.95,
    "reasoning": "User asked about outliers"
  }
]
```

> **D2 Enhancement (planned):** Store the full `result` dict alongside compact history so `<ResultCard>` renders fully on restoration, instead of showing "Restored from previous session" placeholder.

## SQLite Configuration

- **Journal mode:** WAL (Write-Ahead Logging) — concurrent reads while writing
- **Busy timeout:** 5000ms — prevents "database is locked" errors
- **Location:** Project-local (`backend/data/`) — survives OS temp cleanup
- **Expiry:** 24 hours from last access

## Testing

```bash
python -m pytest tests/unit/test_session_store.py -v       # SQLite CRUD tests
python -m pytest tests/unit/test_session_persistence.py -v  # Integration tests
```

Test coverage:
- Create/get/update/delete sessions
- History persistence and retrieval
- Expired session cleanup
- Timestamp format consistency
- Cold start restoration from SQLite
- Stale row deletion when file is missing

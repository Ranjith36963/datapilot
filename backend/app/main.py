"""
DataPilot Backend — FastAPI application entry point.

Run with:
    cd backend && uvicorn app.main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before any config reads os.environ
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import analysis, charts, data, export, ws
from .models.responses import HealthResponse
from .services.analyst import session_manager
from .services.data_service import DataService

# ---------------------------------------------------------------------------
# Ensure the engine package is importable
# ---------------------------------------------------------------------------
_engine_path = str(Path(__file__).resolve().parents[2] / "engine")
if _engine_path not in sys.path:
    sys.path.insert(0, _engine_path)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("datapilot.backend")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    logger.info("DataPilot backend starting...")

    # Verify engine is importable
    try:
        import datapilot
        logger.info(f"Engine loaded: datapilot v{datapilot.__version__}")
    except ImportError as e:
        logger.error(f"Cannot import datapilot engine: {e}")

    yield

    # Shutdown: cleanup temp files
    logger.info("DataPilot backend shutting down...")
    data_service = DataService()
    data_service.cleanup_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DataPilot API",
    description=(
        "AI-powered data analysis API. Upload datasets, ask questions "
        "in natural language, and get insights powered by 81+ analysis skills."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow frontend origins
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------
app.include_router(data.router)
app.include_router(analysis.router)
app.include_router(charts.router)
app.include_router(export.router)
app.include_router(ws.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Health check endpoint."""
    try:
        import datapilot
        version = datapilot.__version__
    except ImportError:
        version = "unknown"

    return HealthResponse(status="ok", version=version)


@app.get("/api/sessions", tags=["system"])
async def list_sessions():
    """List active analysis sessions."""
    return {
        "status": "success",
        "count": session_manager.count,
        "sessions": session_manager.list_sessions(),
    }


@app.delete("/api/sessions/{session_id}", tags=["system"])
async def delete_session(session_id: str):
    """Delete an analysis session and its data."""
    removed = session_manager.remove_session(session_id)
    data_service = DataService()
    data_service.cleanup_session(session_id)

    if not removed:
        return {"status": "not_found", "message": f"Session '{session_id}' not found"}

    return {"status": "success", "message": f"Session '{session_id}' deleted"}

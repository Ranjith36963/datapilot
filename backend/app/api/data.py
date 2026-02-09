"""
Data API — upload, preview, and profile endpoints.
"""

import asyncio
import logging

from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile

from ..models.responses import (
    ColumnInfo,
    PreviewResponse,
    ProfileResponse,
    UploadResponse,
)
from ..services.analyst import session_manager
from ..services.data_service import DataService

logger = logging.getLogger("datapilot.api.data")
router = APIRouter(prefix="/api", tags=["data"])
data_service = DataService()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset (CSV, Excel, JSON, Parquet).

    Creates a new analysis session and returns a session_id.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        session_id, file_path = data_service.save_upload(file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create analyst session in a thread to avoid blocking the event loop
    try:
        analyst = await asyncio.to_thread(
            session_manager.create_session, session_id, str(file_path)
        )
    except Exception as e:
        data_service.cleanup_session(session_id)
        raise HTTPException(status_code=422, detail=f"Failed to load data: {e}")

    # Build column info
    from datapilot.core.router import build_data_context
    ctx = build_data_context(analyst.df)
    columns = [
        ColumnInfo(
            name=c["name"],
            dtype=c["dtype"],
            semantic_type=c["semantic_type"],
            n_unique=c["n_unique"],
            null_pct=c["null_pct"],
        )
        for c in ctx["columns"]
    ]

    # Preview
    preview_df = analyst.df.head(20)
    preview = preview_df.where(preview_df.notna(), None).to_dict(orient="records")

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        shape=list(analyst.shape),
        columns=columns,
        preview=preview,
    )


@router.get("/preview", response_model=PreviewResponse)
async def get_preview(
    rows: int = Query(default=20, ge=1, le=500),
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Preview the first N rows of the uploaded dataset."""
    analyst = session_manager.get_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    preview_df = analyst.df.head(rows)
    data = preview_df.where(preview_df.notna(), None).to_dict(orient="records")

    return PreviewResponse(
        shape=list(analyst.shape),
        columns=analyst.columns,
        data=data,
    )


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Get the full dataset profile."""
    analyst = session_manager.get_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        profile = await asyncio.to_thread(analyst.profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile failed: {e}")

    return ProfileResponse(profile=profile)

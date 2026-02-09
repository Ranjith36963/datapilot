"""
Export API — generate reports in PDF, DOCX, or PPTX.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse

from ..models.requests import ExportRequest
from ..models.responses import ExportResponse
from ..services.analyst import session_manager

logger = logging.getLogger("datapilot.api.export")
router = APIRouter(prefix="/api/export", tags=["export"])

ALLOWED_FORMATS = {"pdf", "docx", "pptx"}


@router.post("/{fmt}", response_model=ExportResponse)
async def export_report(
    fmt: str,
    body: ExportRequest,
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Export analysis results as a report.

    Supported formats: pdf, docx, pptx.
    """
    fmt = fmt.lower().strip()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: '{fmt}'. Allowed: {', '.join(sorted(ALLOWED_FORMATS))}",
        )

    analyst = session_manager.get_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build output path
    output_name = f"report_{uuid.uuid4().hex[:8]}.{fmt}"
    output_dir = Path("/tmp/datapilot/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / output_name)

    # Gather analysis results from history
    results = []
    if body.include_history:
        results = [
            r.execution.result for r in analyst.history
            if r.execution.result
        ]

    try:
        kwargs = {}
        if body.title:
            kwargs["title"] = body.title
        if body.subtitle:
            kwargs["subtitle"] = body.subtitle
        if body.brand_name:
            kwargs["brand_name"] = body.brand_name

        analyst.export(
            path=output_path,
            analysis_results=results,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    return ExportResponse(
        format=fmt,
        filename=output_name,
        download_url=f"/api/export/download/{output_name}",
    )


@router.get("/download/{filename}")
async def download_report(filename: str):
    """Download a generated report file."""
    file_path = Path("/tmp/datapilot/exports") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    ext = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    return FileResponse(
        path=str(file_path),
        media_type=media_types.get(ext, "application/octet-stream"),
        filename=filename,
    )

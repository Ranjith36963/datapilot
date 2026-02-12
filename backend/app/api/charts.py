"""
Charts API — create and suggest charts.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from ..models.requests import ChartRequest
from ..models.responses import ChartResponse, SuggestChartItem, SuggestChartResponse
from ..services.analyst import session_manager

logger = logging.getLogger("datapilot.api.charts")
router = APIRouter(prefix="/api/chart", tags=["charts"])


@router.post("/create", response_model=ChartResponse)
async def create_chart(
    body: ChartRequest,
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Create a chart from the uploaded dataset."""
    analyst = session_manager.get_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = analyst.chart(
            chart_type=body.chart_type,
            x=body.x,
            y=body.y,
            hue=body.hue,
            title=body.title,
        )
    except Exception as e:
        logger.error(f"Chart creation failed: {e}", exc_info=True)
        return ChartResponse(
            status="error",
            chart_type=body.chart_type,
            error=str(e),
        )

    # If the chart result contains a file path, read and base64-encode it
    image_b64 = None
    chart_path = result.get("chart_path") or result.get("path")
    if chart_path and Path(chart_path).exists():
        image_b64 = base64.b64encode(Path(chart_path).read_bytes()).decode("utf-8")

    # Generate a one-line AI insight from chart summary data
    insight = None
    chart_summary = result.get("chart_summary")
    if chart_summary and image_b64:
        try:
            insight = analyst.provider.generate_chart_insight(chart_summary)
        except Exception:
            pass  # Insight is optional, don't fail chart creation

    return ChartResponse(
        status=result.get("status", "success"),
        chart_type=body.chart_type,
        image_base64=image_b64,
        plotly_json=result.get("plotly_json"),
        insight=insight,
    )


@router.get("/suggest", response_model=SuggestChartResponse)
async def suggest_chart(
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Ask the LLM to suggest the best chart for this dataset."""
    analyst = session_manager.get_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = analyst.suggest_chart()
        items = [
            SuggestChartItem(
                chart_type=s.get("chart_type", "histogram"),
                x=s.get("x"),
                y=s.get("y"),
                hue=s.get("hue"),
                title=s.get("title", ""),
                reason=s.get("reason", ""),
            )
            for s in result.get("suggestions", [])
        ]
    except Exception as e:
        logger.warning(f"Chart suggestion failed: {e}")
        items = [
            SuggestChartItem(
                chart_type="histogram",
                x=None,
                y=None,
                title="Data Distribution",
                reason="Fallback suggestion due to an error.",
            )
        ]

    return SuggestChartResponse(suggestions=items)

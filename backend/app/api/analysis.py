"""
Analysis API — ask questions and run skills.
"""

import logging

from fastapi import APIRouter, Header, HTTPException, Query

from ..models.requests import AnalyzeRequest, AskRequest
from ..models.responses import AnalyzeResponse, AskResponse, HistoryEntry, HistoryResponse, NarrativeResponse
from ..services.analyst import session_manager

logger = logging.getLogger("datapilot.api.analysis")
router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    body: AskRequest,
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Ask a natural-language question about the dataset.

    Routes the question to the best analysis skill, executes it,
    and returns the result immediately. If narrate=true (default),
    narrative generation runs in the background — poll GET /api/narrative
    to retrieve it when ready.
    """
    analyst = await session_manager.get_or_restore_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build lightweight conversation context string
    conv_context = ""
    if body.conversation_context:
        parts = []
        for entry in body.conversation_context[-3:]:
            parts.append(f"Q: {entry.question}\nFindings: {entry.summary}")
        conv_context = "\n---\n".join(parts)

    try:
        result = analyst.ask(
            question=body.question,
            narrate=body.narrate,
            conversation_context=conv_context or None,
        )
    except Exception as e:
        logger.error(f"Ask failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Persist updated history to SQLite
    if session_manager._store:
        await session_manager.persist_history(x_session_id)

    return AskResponse(
        status=result.status,
        question=result.question,
        skill=result.skill_name,
        confidence=result.routing.confidence,
        reasoning=result.routing.reasoning,
        route_method=result.route_method,
        result=result.data,
        narrative=result.text,
        narrative_pending=False,
        key_points=result.key_points,
        suggestions=result.suggestions,
        code_snippet=result.code_snippet,
        columns_used=result.columns_used or [],
        elapsed_seconds=result.execution.elapsed_seconds,
        routing_ms=round(result.routing_ms, 1),
        execution_ms=round(result.execution_ms, 1),
        narration_ms=round(result.narration_ms, 1),
        error=result.execution.error,
    )


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Return analysis history for session restoration on frontend refresh."""
    # 1. Try in-memory analyst first (richer data)
    analyst = session_manager.get_session(x_session_id)
    if analyst and (analyst.history or getattr(analyst, "_restored_history", None)):
        entries = []
        # Merge restored history (from previous SQLite session) with new entries
        restored = getattr(analyst, "_restored_history", None) or []
        for h in restored:
            entries.append(HistoryEntry(**h))
        for entry in analyst.history:
            entries.append(HistoryEntry(
                question=entry.question,
                skill=entry.skill_name,
                narrative=entry.text[:500] if entry.text else None,
                key_points=entry.key_points[:5] if entry.key_points else [],
                confidence=entry.routing.confidence if entry.routing else 0.5,
                reasoning=(entry.routing.reasoning[:200]
                           if entry.routing and entry.routing.reasoning else ""),
                result=entry.data,
            ))
        return HistoryResponse(history=entries)

    # 2. Fall back to SQLite compact history
    if session_manager._store:
        session_data = await session_manager._store.get_session(x_session_id)
        if session_data and session_data.get("analysis_history"):
            entries = [
                HistoryEntry(**h)
                for h in session_data["analysis_history"]
            ]
            return HistoryResponse(history=entries)

    return HistoryResponse(history=[])


@router.get("/narrative", response_model=NarrativeResponse)
async def get_narrative(
    x_session_id: str = Header(..., alias="x-session-id"),
    index: int = Query(-1, description="History index (-1 = latest)"),
    timeout: float = Query(30.0, ge=0, le=60, description="Max seconds to wait"),
):
    """Poll for the narrative of a previous /ask result.

    Blocks up to `timeout` seconds for background narration to finish.
    Returns immediately if narrative is already available or was not requested.
    """
    analyst = await session_manager.get_or_restore_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    if not analyst.history:
        raise HTTPException(status_code=404, detail="No ask history yet")

    try:
        result = analyst.history[index]
    except IndexError:
        raise HTTPException(status_code=404, detail=f"History index {index} not found")

    result.wait_for_narrative(timeout=timeout)

    if result.narrative:
        return NarrativeResponse(
            ready=True,
            narrative=result.narrative.text,
            key_points=result.narrative.key_points,
            suggestions=result.narrative.suggestions,
            narration_ms=round(result.narration_ms, 1),
        )
    return NarrativeResponse(ready=False)


@router.post("/analyze", response_model=AnalyzeResponse)
async def run_analysis(
    body: AnalyzeRequest,
    x_session_id: str = Header(..., alias="x-session-id"),
):
    """Run a specific analysis skill directly.

    Bypasses LLM routing — specify the exact skill name and parameters.
    """
    analyst = await session_manager.get_or_restore_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        execution = analyst.executor.execute(
            skill_name=body.skill,
            df=analyst.df,
            parameters=body.params,
        )
    except Exception as e:
        logger.error(f"Analyze failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    return AnalyzeResponse(
        status=execution.status,
        skill=body.skill,
        result=execution.result,
        elapsed_seconds=execution.elapsed_seconds,
        error=execution.error,
    )

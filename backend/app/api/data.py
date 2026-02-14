"""
Data API — upload, preview, profile, fingerprint, and autopilot endpoints.
"""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, File, Header, HTTPException, Query, UploadFile

from ..models.responses import (
    AutopilotStatusResponse,
    ColumnInfo,
    FingerprintResponse,
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

    # Persist session to SQLite
    if session_manager._store:
        await session_manager.persist_new_session(session_id, str(file_path))

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
    analyst = await session_manager.get_or_restore_session(x_session_id)
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
    analyst = await session_manager.get_or_restore_session(x_session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        profile = await asyncio.to_thread(analyst.profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile failed: {e}")

    return ProfileResponse(profile=profile)


@router.post("/fingerprint/{session_id}", response_model=FingerprintResponse)
async def fingerprint_dataset_endpoint(
    session_id: str,
    background_tasks: BackgroundTasks,
):
    """Detect the domain of the dataset using LLM-driven understanding.

    Returns cached result if available, otherwise runs understand_dataset()
    and caches the result in SQLite. Kicks off autopilot in the background.
    """
    analyst = await session_manager.get_or_restore_session(session_id)
    if not analyst:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if fingerprint is already cached in SQLite
    if session_manager._store:
        session_data = await session_manager._store.get_session(session_id)
        if session_data and session_data.get("domain"):
            explanation = session_data.get("domain_explanation") or {}
            return FingerprintResponse(
                domain=session_data["domain"],
                domain_short=explanation.get("domain_short", "General"),
                confidence=session_data.get("domain_confidence", 0.0),
                target_column=explanation.get("target_column"),
                target_type=explanation.get("target_type"),
                key_observations=explanation.get("key_observations", []),
                suggested_questions=explanation.get("suggested_questions", []),
                data_quality_notes=explanation.get("data_quality_notes", []),
                provider_used=explanation.get("provider_used"),
            )

    # Perform LLM-driven understanding (blocking, run in thread)
    try:
        from datapilot.data.fingerprint import understand_dataset

        profile = await asyncio.to_thread(analyst.profile)

        # Get filename from session data
        filename = "unknown.csv"
        if session_manager._store:
            session_data = await session_manager._store.get_session(session_id)
            if session_data:
                filename = session_data.get("filename", filename)

        understanding = await asyncio.to_thread(
            understand_dataset,
            analyst.df,
            filename,
            profile,
            analyst.llm_provider,
        )

        if understanding is None:
            return FingerprintResponse(
                domain="general",
                domain_short="General",
                confidence=0.0,
            )

        # Cache all understanding fields in SQLite
        explanation_dict = {
            "domain_short": understanding.domain_short,
            "target_column": understanding.target_column,
            "target_type": understanding.target_type,
            "key_observations": understanding.key_observations,
            "suggested_questions": understanding.suggested_questions,
            "data_quality_notes": understanding.data_quality_notes,
            "provider_used": understanding.provider_used,
        }
        if session_manager._store:
            await session_manager._store.update_domain(
                session_id=session_id,
                domain=understanding.domain,
                confidence=understanding.confidence,
                explanation=explanation_dict,
            )

        # Kick off autopilot in background (only if not already running/complete)
        if session_manager._store:
            ap_session = await session_manager._store.get_session(session_id)
            existing_ap = ap_session.get("autopilot_status") if ap_session else None
            if existing_ap not in ("planning", "running", "complete"):
                background_tasks.add_task(
                    _run_autopilot_background, session_id, analyst
                )

        return FingerprintResponse(
            domain=understanding.domain,
            domain_short=understanding.domain_short,
            confidence=understanding.confidence,
            target_column=understanding.target_column,
            target_type=understanding.target_type,
            key_observations=understanding.key_observations,
            suggested_questions=understanding.suggested_questions,
            data_quality_notes=understanding.data_quality_notes,
            provider_used=understanding.provider_used,
        )

    except ImportError as e:
        logger.error(f"Failed to import fingerprint module: {e}")
        raise HTTPException(
            status_code=500,
            detail="Fingerprint module not available",
        )
    except Exception as e:
        logger.error(f"Fingerprinting failed for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Fingerprinting failed",
        )


async def _run_autopilot_background(session_id: str, analyst) -> None:
    """Background task: generate plan -> execute -> summarize.

    Updates SQLite autopilot_status at each phase.
    """
    try:
        from datapilot.core.autopilot import (
            generate_analysis_plan,
            generate_summary,
            get_available_skills_description,
            run_autopilot,
        )
        from datapilot.data.fingerprint import DatasetUnderstanding

        if not session_manager._store:
            return

        session_data = await session_manager._store.get_session(session_id)
        if not session_data or not session_data.get("domain"):
            return

        explanation = session_data.get("domain_explanation") or {}
        understanding = DatasetUnderstanding(
            domain=session_data["domain"],
            domain_short=explanation.get("domain_short", "General"),
            target_column=explanation.get("target_column"),
            target_type=explanation.get("target_type"),
            key_observations=explanation.get("key_observations", []),
            suggested_questions=explanation.get("suggested_questions", []),
            data_quality_notes=explanation.get("data_quality_notes", []),
            confidence=session_data.get("domain_confidence", 0.0),
            provider_used=explanation.get("provider_used", "unknown"),
        )

        # Phase 1: Planning
        await session_manager._store.update_autopilot(session_id, "planning")
        skills_desc = get_available_skills_description()
        plan = await generate_analysis_plan(
            understanding, skills_desc, analyst.llm_provider
        )
        if plan is None:
            await session_manager._store.update_autopilot(session_id, "failed")
            return

        # Phase 2: Running
        await session_manager._store.update_autopilot(session_id, "running")
        result = await run_autopilot(analyst, plan, understanding)

        # Phase 3: Summary
        summary = await generate_summary(
            understanding, result.results, analyst.llm_provider
        )
        result.summary = summary

        # Persist history (autopilot questions get added to analyst.history)
        await session_manager.persist_history(session_id)

        # Store complete results
        autopilot_data = {
            "plan_title": plan.title,
            "completed_steps": result.completed_steps,
            "total_steps": len(plan.steps),
            "skipped_steps": result.skipped_steps,
            "total_duration_seconds": result.total_duration_seconds,
            "summary": summary,
            "results": [
                {"step": r["step"], "status": r["status"]}
                for r in result.results
            ],
        }
        await session_manager._store.update_autopilot(
            session_id, "complete", autopilot_data
        )

    except Exception as e:
        logger.error(
            f"Autopilot background task failed for {session_id}: {e}"
        )
        try:
            await session_manager._store.update_autopilot(
                session_id, "failed"
            )
        except Exception:
            pass


@router.get("/autopilot/{session_id}", response_model=AutopilotStatusResponse)
async def get_autopilot_status(session_id: str):
    """Get auto-pilot analysis status for a session."""
    if not session_manager._store:
        return AutopilotStatusResponse(status="unavailable")

    session_data = await session_manager._store.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    status = session_data.get("autopilot_status") or "unavailable"
    results_data = session_data.get("autopilot_results")

    if results_data and isinstance(results_data, dict):
        return AutopilotStatusResponse(
            status=status,
            completed_steps=results_data.get("completed_steps"),
            total_steps=results_data.get("total_steps"),
            results=results_data.get("results"),
            summary=results_data.get("summary"),
        )

    return AutopilotStatusResponse(status=status)

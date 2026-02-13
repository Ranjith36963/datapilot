"""
WebSocket API — streaming chat interface.
"""

import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.analyst import session_manager

logger = logging.getLogger("datapilot.api.ws")
router = APIRouter(tags=["websocket"])


@router.websocket("/api/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming chat.

    Protocol:
        Client sends: {"type": "question", "content": "...", "session_id": "..."}
        Server sends: {"type": "status", "content": "routing..."} (progress updates)
        Server sends: {"type": "result", "data": {...}}  (final result)
        Server sends: {"type": "error", "content": "..."}  (on failure)
    """
    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid JSON",
                })
                continue

            session_id = message.get("session_id")
            content = message.get("content", "").strip()
            msg_type = message.get("type", "question")

            if not session_id:
                await websocket.send_json({
                    "type": "error",
                    "content": "Missing session_id",
                })
                continue

            analyst = await session_manager.get_or_restore_session(session_id)
            if not analyst:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Session '{session_id}' not found",
                })
                continue

            if msg_type == "question" and content:
                await _handle_question(websocket, analyst, content)
            else:
                await websocket.send_json({
                    "type": "error",
                    "content": "Send {type: 'question', content: '...', session_id: '...'}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


async def _handle_question(websocket: WebSocket, analyst, question: str):
    """Process a question through the Analyst pipeline with progress updates.

    Sends the skill result immediately, then streams narrative as a follow-up.
    """

    # Step 1: Routing
    await websocket.send_json({
        "type": "status",
        "content": "Routing question to best skill...",
    })

    t0 = time.perf_counter()
    try:
        routing = analyst.router.route(question, analyst.data_context)
    except Exception as e:
        await websocket.send_json({"type": "error", "content": f"Routing failed: {e}"})
        return
    t1 = time.perf_counter()
    routing_ms = (t1 - t0) * 1000

    await websocket.send_json({
        "type": "status",
        "content": f"Using skill: {routing.skill_name} (confidence: {routing.confidence:.0%}) [{routing_ms:.0f}ms]",
    })

    # Step 2: Execution
    await websocket.send_json({
        "type": "status",
        "content": f"Running {routing.skill_name}...",
    })

    try:
        execution = analyst.executor.execute(
            skill_name=routing.skill_name,
            df=analyst.df,
            parameters=routing.parameters,
        )
    except Exception as e:
        await websocket.send_json({"type": "error", "content": f"Execution failed: {e}"})
        return
    t2 = time.perf_counter()
    execution_ms = (t2 - t1) * 1000

    if execution.status == "error":
        await websocket.send_json({
            "type": "error",
            "content": execution.error or "Skill execution failed",
        })
        return

    # Send result immediately (no waiting for narration)
    response = {
        "type": "result",
        "data": {
            "question": question,
            "skill": routing.skill_name,
            "confidence": routing.confidence,
            "reasoning": routing.reasoning,
            "status": execution.status,
            "result": execution.result,
            "elapsed_seconds": round(execution.elapsed_seconds, 3),
            "routing_ms": round(routing_ms, 1),
            "execution_ms": round(execution_ms, 1),
        },
    }
    await websocket.send_json(response)

    # Step 3: Narrative (sent as separate follow-up message)
    await websocket.send_json({
        "type": "status",
        "content": "Generating insights...",
    })

    t3 = time.perf_counter()
    narrative = None
    try:
        narrative = analyst.provider.generate_narrative(
            analysis_result=execution.result,
            question=question,
        )
    except Exception as e:
        logger.warning(f"Narrative failed: {e}")
    t4 = time.perf_counter()
    narration_ms = (t4 - t3) * 1000

    if narrative:
        await websocket.send_json({
            "type": "narrative",
            "data": {
                "narrative": narrative.text,
                "key_points": narrative.key_points,
                "suggestions": narrative.suggestions,
                "narration_ms": round(narration_ms, 1),
            },
        })

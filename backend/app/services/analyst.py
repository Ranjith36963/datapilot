"""
Analyst service — two-tier session management.

Tier 1: In-memory dict for fast access (no DB hit on hot path)
Tier 2: SQLite via SessionStore for durability across restarts
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("datapilot.backend.analyst_service")

# Ensure engine is importable
_engine_path = str(Path(__file__).resolve().parents[3] / "engine")
if _engine_path not in sys.path:
    sys.path.insert(0, _engine_path)


class SessionManager:
    """Two-tier session manager: in-memory cache + SQLite persistence.

    Tier 1: In-memory dict for fast access (no DB hit on hot path)
    Tier 2: SQLite via SessionStore for durability across restarts
    """

    def __init__(self):
        self._sessions: Dict[str, Any] = {}  # session_id -> Analyst (hot cache)
        self._store = None  # SessionStore, set during app startup

    def set_store(self, store):
        """Attach the SQLite store (called during app lifespan startup)."""
        self._store = store

    def create_session(
        self,
        session_id: str,
        file_path: str,
        llm: Optional[str] = None,
    ) -> "Analyst":
        """Create a new Analyst session.

        Args:
            session_id: Unique session identifier.
            file_path: Path to the uploaded data file.
            llm: LLM provider name.

        Returns:
            The created Analyst instance.
        """
        from datapilot.core.analyst import Analyst

        analyst = Analyst(data=file_path, llm=llm, auto_profile=True)
        self._sessions[session_id] = analyst
        logger.info(
            f"Session {session_id} created: "
            f"{analyst.shape[0]} rows x {analyst.shape[1]} cols"
        )
        return analyst

    def get_session(self, session_id: str) -> Optional["Analyst"]:
        """Get an existing Analyst session (in-memory only)."""
        return self._sessions.get(session_id)

    async def get_or_restore_session(self, session_id: str) -> Optional["Analyst"]:
        """Get a session from cache, or restore from SQLite on cache miss."""
        # Fast path: in-memory hit
        analyst = self._sessions.get(session_id)
        if analyst is not None:
            return analyst
        # Cold path: try SQLite restoration
        if self._store:
            return await self._restore_from_db(session_id)
        return None

    async def _restore_from_db(self, session_id: str) -> Optional["Analyst"]:
        """Restore an Analyst from SQLite (cold start path)."""
        session_data = await self._store.get_session(session_id)
        if not session_data:
            return None
        file_path = session_data["file_path"]
        if not Path(file_path).exists():
            # File was deleted — remove stale DB entry
            logger.warning(
                f"Session {session_id} file missing: {file_path}"
            )
            await self._store.delete_session(session_id)
            return None
        try:
            from datapilot.core.analyst import Analyst

            analyst = await asyncio.to_thread(
                Analyst, data=file_path, auto_profile=True
            )
            # Store restored history as metadata for export
            history = session_data.get("analysis_history", [])
            analyst._restored_history = history
            self._sessions[session_id] = analyst
            logger.info(
                f"Session {session_id} restored from SQLite "
                f"({analyst.shape[0]} rows)"
            )
            return analyst
        except Exception as e:
            logger.warning(f"Failed to restore session {session_id}: {e}")
            await self._store.delete_session(session_id)
            return None

    async def persist_new_session(
        self, session_id: str, file_path: str
    ) -> None:
        """Persist a newly created session to SQLite.

        Enforces the single-project invariant: deletes all other sessions
        from SQLite before inserting the new one.
        """
        if not self._store:
            return
        analyst = self._sessions.get(session_id)
        if not analyst:
            return
        try:
            # Single-project invariant: remove old sessions from memory and SQLite
            old_ids = [sid for sid in self._sessions if sid != session_id]
            for sid in old_ids:
                del self._sessions[sid]
            await self._store.delete_all_except(session_id)

            filename = Path(file_path).name
            columns = list(analyst.columns)
            shape = {"rows": analyst.shape[0], "columns": analyst.shape[1]}
            await self._store.create_session(
                session_id=session_id,
                filename=filename,
                file_path=file_path,
                columns=columns,
                shape=shape,
            )
        except Exception as e:
            logger.warning(f"Failed to persist new session {session_id}: {e}")

    async def persist_history(self, session_id: str) -> None:
        """Persist compact analysis history to SQLite.

        Merges restored history (from previous SQLite session) with new
        engine history entries to avoid overwriting old entries.
        """
        if not self._store:
            return
        analyst = self._sessions.get(session_id)
        if not analyst:
            return
        try:
            # Start with restored history from previous session (if any)
            restored = getattr(analyst, "_restored_history", None) or []

            # Append new engine history entries
            new_entries = []
            for entry in analyst.history:
                new_entries.append({
                    "question": entry.question,
                    "skill": entry.skill_name,
                    "narrative": entry.text[:500] if entry.text else None,
                    "key_points": entry.key_points[:5] if entry.key_points else [],
                    "confidence": entry.routing.confidence if entry.routing else 0.5,
                    "reasoning": (entry.routing.reasoning[:200] if entry.routing and entry.routing.reasoning else ""),
                })

            compact = restored + new_entries
            await self._store.update_history(session_id, compact)
        except Exception as e:
            logger.warning(
                f"Failed to persist history for {session_id}: {e}"
            )

    def remove_session(self, session_id: str) -> bool:
        """Remove a session from cache and schedule SQLite deletion."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session {session_id} removed")
            # Schedule async DB deletion if store is attached
            if self._store:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._store.delete_session(session_id))
                except RuntimeError:
                    pass  # No running loop — skip DB cleanup
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active in-memory sessions."""
        sessions = []
        for sid, analyst in self._sessions.items():
            sessions.append({
                "session_id": sid,
                "shape": list(analyst.shape),
                "columns": analyst.columns,
                "llm_provider": type(analyst.provider).__name__,
                "history_count": len(analyst.history),
            })
        return sessions

    @property
    def count(self) -> int:
        return len(self._sessions)


# Singleton instance
session_manager = SessionManager()

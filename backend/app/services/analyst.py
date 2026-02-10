"""
Analyst service — session management wrapping the engine Analyst class.

Maintains an in-memory dict of {session_id: Analyst} instances.
"""

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
    """Manages Analyst sessions keyed by session_id."""

    def __init__(self):
        self._sessions: Dict[str, Any] = {}  # session_id -> Analyst

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
        """Get an existing Analyst session."""
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session {session_id} removed")
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
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

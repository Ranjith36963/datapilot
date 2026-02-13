"""
SQLite session persistence for DataPilot.

Provides durable session storage that survives backend restarts.
Uses WAL mode for safe concurrent reads under FastAPI's async model.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger("datapilot.backend.session_store")

# Consistent format matching SQLite's datetime('now') output
_SQLITE_DT_FMT = "%Y-%m-%d %H:%M:%S"


def _utcnow_str() -> str:
    """Return current UTC time formatted for SQLite string comparison."""
    return datetime.now(timezone.utc).strftime(_SQLITE_DT_FMT)

# Project-local DB path (survives OS temp cleanup)
_PROJECT_DATA = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DB_PATH = Path(
    os.environ.get(
        "DATAPILOT_DB_PATH",
        str(_PROJECT_DATA / "sessions.db"),
    )
)


class SessionStore:
    """Async SQLite session store with WAL mode."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init_db(self):
        """Initialize database connection and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT,
                columns_json TEXT,
                shape_json TEXT,
                analysis_history TEXT DEFAULT '[]',
                domain TEXT DEFAULT NULL,
                domain_confidence REAL DEFAULT NULL,
                domain_explanation TEXT DEFAULT NULL,
                autopilot_status TEXT DEFAULT NULL,
                autopilot_results TEXT DEFAULT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._db.commit()
        logger.info(f"SessionStore initialized: {self.db_path}")

    async def create_session(
        self,
        session_id: str,
        filename: str,
        file_path: str,
        columns: List[str],
        shape: Dict[str, int],
    ) -> None:
        """Insert a new session into the database."""
        if self._db is None:
            return
        try:
            now = _utcnow_str()
            await self._db.execute(
                """
                INSERT INTO sessions (session_id, filename, file_path, columns_json, shape_json, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    filename,
                    file_path,
                    json.dumps(columns),
                    json.dumps(shape),
                    now,
                    now,
                ),
            )
            await self._db.commit()
            logger.info(f"Session {session_id} persisted to SQLite")
        except Exception as e:
            logger.warning(f"Failed to persist session {session_id}: {e}")

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by ID, updating last_accessed."""
        if self._db is None:
            return None
        try:
            cursor = await self._db.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None

            # Update last_accessed
            now = _utcnow_str()
            await self._db.execute(
                "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                (now, session_id),
            )
            await self._db.commit()

            # Parse JSON fields
            result = dict(row)
            result["columns_json"] = json.loads(result["columns_json"] or "[]")
            result["shape_json"] = json.loads(result["shape_json"] or "{}")
            result["analysis_history"] = json.loads(
                result["analysis_history"] or "[]"
            )
            if result["domain_explanation"]:
                result["domain_explanation"] = json.loads(
                    result["domain_explanation"]
                )
            if result["autopilot_results"]:
                result["autopilot_results"] = json.loads(
                    result["autopilot_results"]
                )
            return result
        except Exception as e:
            logger.warning(f"Failed to get session {session_id}: {e}")
            return None

    async def update_history(
        self, session_id: str, history: List[Dict]
    ) -> None:
        """Update the analysis history for a session."""
        if self._db is None:
            return
        try:
            now = _utcnow_str()
            await self._db.execute(
                """
                UPDATE sessions
                SET analysis_history = ?, last_accessed = ?
                WHERE session_id = ?
                """,
                (json.dumps(history), now, session_id),
            )
            await self._db.commit()
        except Exception as e:
            logger.warning(f"Failed to update history for {session_id}: {e}")

    async def update_domain(
        self,
        session_id: str,
        domain: str,
        confidence: float,
        explanation: Any,
    ) -> None:
        """Update domain fingerprint fields for a session."""
        if self._db is None:
            return
        try:
            explanation_json = json.dumps(explanation) if explanation else None
            await self._db.execute(
                """
                UPDATE sessions
                SET domain = ?, domain_confidence = ?, domain_explanation = ?
                WHERE session_id = ?
                """,
                (domain, confidence, explanation_json, session_id),
            )
            await self._db.commit()
        except Exception as e:
            logger.warning(f"Failed to update domain for {session_id}: {e}")

    async def update_autopilot(
        self,
        session_id: str,
        status: str,
        results: Any = None,
    ) -> None:
        """Update autopilot status and results for a session."""
        if self._db is None:
            return
        try:
            results_json = json.dumps(results) if results is not None else None
            await self._db.execute(
                """
                UPDATE sessions
                SET autopilot_status = ?, autopilot_results = ?
                WHERE session_id = ?
                """,
                (status, results_json, session_id),
            )
            await self._db.commit()
        except Exception as e:
            logger.warning(
                f"Failed to update autopilot for {session_id}: {e}"
            )

    async def delete_all_except(self, keep_session_id: str) -> int:
        """Delete all sessions except the given one (single-project invariant)."""
        if self._db is None:
            return 0
        try:
            cursor = await self._db.execute(
                "DELETE FROM sessions WHERE session_id != ?",
                (keep_session_id,),
            )
            await self._db.commit()
            count = cursor.rowcount
            if count > 0:
                logger.info(f"Deleted {count} old session(s), keeping {keep_session_id}")
            return count
        except Exception as e:
            logger.warning(f"Failed to delete old sessions: {e}")
            return 0

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if a row was deleted."""
        if self._db is None:
            return False
        try:
            cursor = await self._db.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            await self._db.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.warning(f"Failed to delete session {session_id}: {e}")
            return False

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with summary info."""
        if self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                "SELECT session_id, filename, shape_json, created_at, last_accessed FROM sessions"
            )
            rows = await cursor.fetchall()
            results = []
            for row in rows:
                entry = dict(row)
                entry["shape_json"] = json.loads(entry["shape_json"] or "{}")
                results.append(entry)
            return results
        except Exception as e:
            logger.warning(f"Failed to list sessions: {e}")
            return []

    async def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours. Also deletes their files."""
        if self._db is None:
            return 0
        try:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            ).strftime(_SQLITE_DT_FMT)

            # Find expired sessions and their file paths
            cursor = await self._db.execute(
                "SELECT session_id, file_path FROM sessions WHERE last_accessed < ?",
                (cutoff,),
            )
            expired = await cursor.fetchall()

            if not expired:
                return 0

            # Delete files from disk
            for row in expired:
                file_path = row["file_path"]
                try:
                    p = Path(file_path)
                    if p.exists():
                        p.unlink()
                    # Also remove the parent session directory if empty
                    parent = p.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {e}")

            # Delete DB rows
            session_ids = [row["session_id"] for row in expired]
            placeholders = ",".join("?" * len(session_ids))
            await self._db.execute(
                f"DELETE FROM sessions WHERE session_id IN ({placeholders})",
                session_ids,
            )
            await self._db.commit()

            count = len(session_ids)
            logger.info(f"Cleaned up {count} expired sessions")
            return count
        except Exception as e:
            logger.warning(f"Failed to cleanup expired sessions: {e}")
            return 0

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

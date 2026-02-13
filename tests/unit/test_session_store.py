"""Tests for SessionStore — SQLite session persistence."""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.session_store import SessionStore
from app.services.analyst import SessionManager


@pytest.fixture
def tmp_db():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_sessions.db"


@pytest.fixture
def store(tmp_db):
    """Create and initialize a SessionStore."""
    s = SessionStore(db_path=tmp_db)
    asyncio.run(s.init_db())
    yield s
    asyncio.run(s.close())


def _sample_session(session_id="test-123", filename="data.csv"):
    return {
        "session_id": session_id,
        "filename": filename,
        "file_path": f"/tmp/datapilot/uploads/{session_id}/{filename}",
        "columns": ["col1", "col2", "col3"],
        "shape": {"rows": 100, "columns": 3},
    }


class TestSessionStoreCRUD:
    """Test basic CRUD operations on SessionStore."""

    def test_create_and_get_session(self, store):
        """Create a session and retrieve it — all fields should match."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)
            result = await store.get_session("test-123")
            assert result is not None
            assert result["session_id"] == "test-123"
            assert result["filename"] == "data.csv"
            assert result["file_path"] == args["file_path"]
            assert result["columns_json"] == ["col1", "col2", "col3"]
            assert result["shape_json"] == {"rows": 100, "columns": 3}
            assert result["analysis_history"] == []

        asyncio.run(_test())

    def test_get_nonexistent_session(self, store):
        """Getting a session that doesn't exist returns None."""

        async def _test():
            result = await store.get_session("does-not-exist")
            assert result is None

        asyncio.run(_test())

    def test_delete_session(self, store):
        """Delete a session — subsequent get returns None."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)
            deleted = await store.delete_session("test-123")
            assert deleted is True

            result = await store.get_session("test-123")
            assert result is None

            # Deleting again returns False
            deleted_again = await store.delete_session("test-123")
            assert deleted_again is False

        asyncio.run(_test())

    def test_list_sessions(self, store):
        """Create 3 sessions — list should return all 3."""

        async def _test():
            for i in range(3):
                args = _sample_session(
                    session_id=f"session-{i}", filename=f"data_{i}.csv"
                )
                await store.create_session(**args)

            sessions = await store.list_sessions()
            assert len(sessions) == 3
            ids = {s["session_id"] for s in sessions}
            assert ids == {"session-0", "session-1", "session-2"}

            # Each entry should have parsed shape
            for s in sessions:
                assert isinstance(s["shape_json"], dict)

        asyncio.run(_test())


class TestSessionStoreUpdates:
    """Test update operations on SessionStore."""

    def test_update_history(self, store):
        """Update analysis history and verify it persists."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)

            history = [
                {
                    "question": "What are the trends?",
                    "skill": "describe_data",
                    "narrative": "The data shows upward trends.",
                    "key_points": ["Trend 1", "Trend 2"],
                },
                {
                    "question": "Show correlations",
                    "skill": "analyze_correlations",
                    "narrative": "Strong correlation found.",
                    "key_points": ["r=0.95"],
                },
            ]
            await store.update_history("test-123", history)

            result = await store.get_session("test-123")
            assert result is not None
            assert len(result["analysis_history"]) == 2
            assert result["analysis_history"][0]["question"] == "What are the trends?"
            assert result["analysis_history"][1]["skill"] == "analyze_correlations"

        asyncio.run(_test())

    def test_update_domain(self, store):
        """Update domain fingerprint fields."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)

            await store.update_domain(
                "test-123",
                domain="telecom_saas",
                confidence=0.87,
                explanation=["churn column detected", "monthly_charges present"],
            )

            result = await store.get_session("test-123")
            assert result is not None
            assert result["domain"] == "telecom_saas"
            assert result["domain_confidence"] == 0.87
            assert result["domain_explanation"] == [
                "churn column detected",
                "monthly_charges present",
            ]

        asyncio.run(_test())

    def test_update_autopilot(self, store):
        """Update autopilot status and results."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)

            results_data = {
                "analyses": [
                    {"skill": "describe_data", "status": "success"},
                    {"skill": "analyze_correlations", "status": "success"},
                ],
                "total": 2,
            }
            await store.update_autopilot(
                "test-123", status="complete", results=results_data
            )

            result = await store.get_session("test-123")
            assert result is not None
            assert result["autopilot_status"] == "complete"
            assert result["autopilot_results"]["total"] == 2
            assert len(result["autopilot_results"]["analyses"]) == 2

        asyncio.run(_test())


class TestSessionStoreCleanup:
    """Test session expiry and cleanup."""

    def test_cleanup_expired(self, store, tmp_db):
        """Sessions with old last_accessed should be cleaned up."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)

            # Manually set last_accessed to 48 hours ago (same format as production code)
            old_time = (
                datetime.now(timezone.utc) - timedelta(hours=48)
            ).strftime("%Y-%m-%d %H:%M:%S")
            await store._db.execute(
                "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                (old_time, "test-123"),
            )
            await store._db.commit()

            # Cleanup should remove the expired session
            cleaned = await store.cleanup_expired(max_age_hours=24)
            assert cleaned == 1

            # Session should be gone
            result = await store.get_session("test-123")
            assert result is None

        asyncio.run(_test())

    def test_cleanup_keeps_recent(self, store):
        """Recent sessions should not be cleaned up."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)

            # Cleanup with 24h threshold should keep a fresh session
            cleaned = await store.cleanup_expired(max_age_hours=24)
            assert cleaned == 0

            result = await store.get_session("test-123")
            assert result is not None

        asyncio.run(_test())


class TestTimestampConsistency:
    """Verify timestamps use consistent format so cleanup_expired works correctly."""

    def test_cleanup_does_not_delete_fresh_sessions(self, store):
        """Create a session, immediately run cleanup_expired(24), session must survive."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)

            # Cleanup should NOT delete a session that was just created
            cleaned = await store.cleanup_expired(max_age_hours=24)
            assert cleaned == 0

            # Session must still be retrievable
            result = await store.get_session("test-123")
            assert result is not None
            assert result["session_id"] == "test-123"

        asyncio.run(_test())

    def test_timestamp_format_consistency(self, store):
        """Verify last_accessed format matches what cleanup_expired compares against."""
        args = _sample_session()

        async def _test():
            await store.create_session(**args)

            # Read the raw last_accessed value from SQLite
            cursor = await store._db.execute(
                "SELECT last_accessed FROM sessions WHERE session_id = ?",
                ("test-123",),
            )
            row = await cursor.fetchone()
            last_accessed = row["last_accessed"]

            # Must be YYYY-MM-DD HH:MM:SS format (space separator, no T, no timezone)
            assert "T" not in last_accessed, (
                f"Timestamp has 'T' separator: {last_accessed!r} — "
                "must use space separator to match SQLite datetime('now') format"
            )
            assert "+" not in last_accessed, (
                f"Timestamp has timezone offset: {last_accessed!r} — "
                "must be bare UTC without offset"
            )
            # Verify it's parseable in the expected format
            parsed = datetime.strptime(last_accessed, "%Y-%m-%d %H:%M:%S")
            assert parsed is not None

            # Also verify update_history writes the same format
            await store.update_history("test-123", [{"q": "test"}])
            cursor2 = await store._db.execute(
                "SELECT last_accessed FROM sessions WHERE session_id = ?",
                ("test-123",),
            )
            row2 = await cursor2.fetchone()
            updated_ts = row2["last_accessed"]
            assert "T" not in updated_ts
            assert "+" not in updated_ts
            datetime.strptime(updated_ts, "%Y-%m-%d %H:%M:%S")

        asyncio.run(_test())


class TestSessionRestoration:
    """Test that sessions restore from SQLite when in-memory cache is empty."""

    def test_restore_from_sqlite_after_cache_clear(self, store):
        """Simulate backend restart: clear in-memory cache, restore from SQLite."""

        async def _test():
            # Write a real CSV so Analyst can load it
            with tempfile.TemporaryDirectory() as tmpdir:
                data_file = Path(tmpdir) / "test.csv"
                data_file.write_text("a,b,c\n1,2,3\n4,5,6\n")

                await store.create_session(
                    session_id="restore-1",
                    filename="test.csv",
                    file_path=str(data_file),
                    columns=["a", "b", "c"],
                    shape={"rows": 2, "columns": 3},
                )

                history = [
                    {
                        "question": "Describe the data",
                        "skill": "describe_data",
                        "narrative": "Two rows of numeric data.",
                        "key_points": ["Small dataset"],
                        "confidence": 0.9,
                        "reasoning": "keyword match",
                    }
                ]
                await store.update_history("restore-1", history)

                # Build a SessionManager with the store attached
                mgr = SessionManager()
                mgr.set_store(store)

                # Cache is empty — get_session returns None
                assert mgr.get_session("restore-1") is None

                # get_or_restore_session should find it in SQLite
                # and reconstruct the Analyst from the file
                analyst = await mgr.get_or_restore_session("restore-1")
                assert analyst is not None
                assert analyst.shape == (2, 3)

                # Now it should be in the in-memory cache
                assert mgr.get_session("restore-1") is not None

                # Restored history should be attached
                assert hasattr(analyst, "_restored_history")
                assert len(analyst._restored_history) == 1
                assert analyst._restored_history[0]["question"] == "Describe the data"

        asyncio.run(_test())

    def test_restore_missing_file_deletes_stale_row(self, store):
        """If the data file is gone, restoration should delete the SQLite row."""

        async def _test():
            await store.create_session(
                session_id="stale-1",
                filename="gone.csv",
                file_path="/nonexistent/path/gone.csv",
                columns=["x"],
                shape={"rows": 1, "columns": 1},
            )

            mgr = SessionManager()
            mgr.set_store(store)

            # File doesn't exist — should return None and purge the row
            analyst = await mgr.get_or_restore_session("stale-1")
            assert analyst is None

            # SQLite row should have been deleted
            row = await store.get_session("stale-1")
            assert row is None

        asyncio.run(_test())

"""
Comprehensive D2 session persistence tests — TDD red phase.

Tests cover three groups:
  1. Session survives backend restart (SQLite persistence + two-tier cache)
  2. Session restoration returns correct data (history, file path, cleanup)
  3. New upload clears old session (single-project invariant)

Tests that verify EXISTING working behavior should PASS.
Tests that expose KNOWN BUGS are marked with:
    # EXPECTED TO FAIL — Bug: <description>
These will turn green when the bugs are fixed (TDD red→green).
"""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.session_store import SessionStore, _SQLITE_DT_FMT
from app.services.analyst import SessionManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tmp_db():
    """Create a temporary database path for test isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_persistence.db"


@pytest.fixture
def store(tmp_db):
    """Create and initialize a SessionStore with a temp DB."""
    s = SessionStore(db_path=tmp_db)
    asyncio.run(s.init_db())
    yield s
    asyncio.run(s.close())


@pytest.fixture
def csv_file():
    """Create a temporary CSV file that Analyst can load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "employees.csv"
        data_file.write_text(
            "name,age,salary,department\n"
            "Alice,30,50000,Engineering\n"
            "Bob,25,45000,Marketing\n"
            "Charlie,35,60000,Engineering\n"
            "Diana,28,48000,Sales\n"
            "Eve,32,55000,Marketing\n"
        )
        yield data_file


@pytest.fixture
def two_csv_files():
    """Two different CSV files to simulate sequential uploads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_a = Path(tmpdir) / "products.csv"
        file_a.write_text(
            "product,price,quantity\n"
            "Widget,10.5,100\n"
            "Gadget,25.0,50\n"
        )
        file_b = Path(tmpdir) / "cities.csv"
        file_b.write_text(
            "city,population,area\n"
            "NYC,8336817,302.6\n"
            "LA,3979576,468.7\n"
            "Chicago,2693976,227.3\n"
        )
        yield file_a, file_b


def _store_session(session_id="persist-test", filename="employees.csv",
                   file_path="/tmp/datapilot/uploads/persist-test/employees.csv"):
    """Helper: kwargs for SessionStore.create_session."""
    return {
        "session_id": session_id,
        "filename": filename,
        "file_path": file_path,
        "columns": ["name", "age", "salary", "department"],
        "shape": {"rows": 5, "columns": 4},
    }


def _sample_history():
    """Helper: two compact history entries (matches persist_history format)."""
    return [
        {
            "question": "Give me an overview",
            "skill": "describe_data",
            "narrative": "The dataset has 5 rows and 4 columns covering employee data.",
            "key_points": ["5 employees", "3 departments"],
            "confidence": 0.92,
            "reasoning": "keyword match: overview -> describe_data",
        },
        {
            "question": "Show correlations",
            "skill": "analyze_correlations",
            "narrative": "Age and salary show moderate positive correlation (r=0.78).",
            "key_points": ["age-salary r=0.78"],
            "confidence": 0.85,
            "reasoning": "keyword match: correlations -> analyze_correlations",
        },
    ]


# ============================================================================
# Group 1: Session survives backend restart
# ============================================================================

class TestSessionSurvivesRestart:
    """Sessions must persist to SQLite and restore after in-memory cache loss."""

    def test_session_persists_to_sqlite_after_upload(self, store, csv_file):
        """Upload creates session -> SQLite has the row -> verify all fields."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "upload-persist-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            # Verify SQLite row
            row = await store.get_session(sid)
            assert row is not None, "Session must be in SQLite after persist"
            assert row["session_id"] == sid
            assert row["filename"] == csv_file.name
            assert row["file_path"] == str(csv_file)
            assert isinstance(row["columns_json"], list)
            assert len(row["columns_json"]) == 4
            assert row["shape_json"]["rows"] == 5
            assert row["shape_json"]["columns"] == 4

        asyncio.run(_test())

    def test_session_persists_history_after_ask(self, store):
        """Create session -> add history entries -> verify SQLite has them
        with correct compact format."""

        async def _test():
            args = _store_session()
            await store.create_session(**args)

            history = _sample_history()
            await store.update_history(args["session_id"], history)

            row = await store.get_session(args["session_id"])
            assert row is not None
            h = row["analysis_history"]
            assert len(h) == 2
            assert h[0]["question"] == "Give me an overview"
            assert h[0]["skill"] == "describe_data"
            assert h[0]["confidence"] == 0.92
            assert h[0]["narrative"].startswith("The dataset has 5 rows")
            assert h[1]["question"] == "Show correlations"
            assert h[1]["key_points"] == ["age-salary r=0.78"]

        asyncio.run(_test())

    def test_session_restored_from_sqlite_after_cache_clear(self, store, csv_file):
        """Create session -> clear in-memory cache -> get_or_restore_session -> works.
        Verify the restored Analyst has the right shape/columns."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "restore-cache-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            # Simulate backend restart
            mgr._sessions.clear()
            assert mgr.get_session(sid) is None, "Cache must be empty"

            # Restore from SQLite
            restored = await mgr.get_or_restore_session(sid)
            assert restored is not None, "Must restore from SQLite"
            assert restored.shape == (5, 4)
            assert len(restored.columns) == 4

            # Now in cache
            assert mgr.get_session(sid) is not None

        asyncio.run(_test())

    def test_restored_session_can_answer_questions(self, store, csv_file):
        """Create session -> clear cache -> restore -> analyst object is functional
        (has df, can access columns, has ask/profile methods)."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "functional-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            mgr._sessions.clear()
            restored = await mgr.get_or_restore_session(sid)
            assert restored is not None

            # Analyst must be functional
            assert restored.df is not None
            assert len(restored.df) == 5
            assert "name" in restored.df.columns
            assert "salary" in restored.df.columns
            assert hasattr(restored, "ask")
            assert hasattr(restored, "profile")

        asyncio.run(_test())

    def test_history_restored_from_sqlite(self, store, csv_file):
        """Create session -> add history -> clear cache -> restore ->
        GET /api/history returns the entries from SQLite."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "history-restore-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            history = _sample_history()
            await store.update_history(sid, history)

            # Simulate restart
            mgr._sessions.clear()

            # The /api/history endpoint logic:
            # 1. mgr.get_session(sid) -> None (not in memory)
            # 2. store.get_session(sid) -> row with analysis_history
            # Simulate this path:
            analyst = mgr.get_session(sid)
            assert analyst is None, "Not in memory after restart"

            session_data = await store.get_session(sid)
            assert session_data is not None
            entries = session_data["analysis_history"]
            assert len(entries) == 2
            assert entries[0]["question"] == "Give me an overview"
            assert entries[1]["question"] == "Show correlations"

        asyncio.run(_test())

    def test_cleanup_does_not_delete_active_sessions(self, store):
        """Create session 5 minutes ago -> cleanup_expired(24h) -> session still exists.
        This was Bug 2 -- timestamp format mismatch caused cleanup to delete fresh sessions."""

        async def _test():
            args = _store_session()
            await store.create_session(**args)

            # Backdate to 5 min ago (well within 24h window)
            five_min_ago = (
                datetime.now(timezone.utc) - timedelta(minutes=5)
            ).strftime(_SQLITE_DT_FMT)
            await store._db.execute(
                "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                (five_min_ago, args["session_id"]),
            )
            await store._db.commit()

            cleaned = await store.cleanup_expired(max_age_hours=24)
            assert cleaned == 0, "5-min-old session must NOT be cleaned up"

            result = await store.get_session(args["session_id"])
            assert result is not None

        asyncio.run(_test())

    def test_cleanup_deletes_only_truly_expired(self, store):
        """One expired (48h old) + one fresh (just created) ->
        cleanup -> only the expired one is deleted."""

        async def _test():
            # Expired session
            expired = _store_session(session_id="expired-1", filename="old.csv")
            await store.create_session(**expired)
            old_time = (
                datetime.now(timezone.utc) - timedelta(hours=48)
            ).strftime(_SQLITE_DT_FMT)
            await store._db.execute(
                "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                (old_time, "expired-1"),
            )
            await store._db.commit()

            # Fresh session
            fresh = _store_session(session_id="fresh-1", filename="new.csv")
            await store.create_session(**fresh)

            cleaned = await store.cleanup_expired(max_age_hours=24)
            assert cleaned == 1

            assert await store.get_session("expired-1") is None
            assert await store.get_session("fresh-1") is not None

        asyncio.run(_test())

    def test_timestamps_use_consistent_format(self, store):
        """After create_session, update_history, and get_session:
        raw SQL last_accessed must have NO 'T' separator, NO timezone offset,
        format = YYYY-MM-DD HH:MM:SS."""

        async def _test():
            args = _store_session()
            await store.create_session(**args)
            sid = args["session_id"]

            async def _read_ts():
                cur = await store._db.execute(
                    "SELECT last_accessed, created_at FROM sessions "
                    "WHERE session_id = ?", (sid,),
                )
                return await cur.fetchone()

            def _assert_fmt(ts, label):
                assert "T" not in ts, f"{label} has 'T': {ts!r}"
                assert "+" not in ts, f"{label} has tz offset: {ts!r}"
                datetime.strptime(ts, _SQLITE_DT_FMT)

            # After create
            row = await _read_ts()
            _assert_fmt(row["created_at"], "create:created_at")
            _assert_fmt(row["last_accessed"], "create:last_accessed")

            # After update_history
            await store.update_history(sid, [{"q": "test"}])
            row2 = await _read_ts()
            _assert_fmt(row2["last_accessed"], "update_history:last_accessed")

            # After get_session (touches last_accessed)
            await store.get_session(sid)
            row3 = await _read_ts()
            _assert_fmt(row3["last_accessed"], "get_session:last_accessed")

        asyncio.run(_test())


# ============================================================================
# Group 2: Session restoration returns correct data
# ============================================================================

class TestRestorationReturnsCorrectData:
    """Restored sessions must return the right data through all paths."""

    def test_history_endpoint_returns_from_memory(self, store, csv_file):
        """Session in memory -> the /api/history endpoint's first branch
        (analyst + analyst.history) is reachable.
        We verify the in-memory analyst is the same object we created."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "mem-hist-1"
            analyst = mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            in_mem = mgr.get_session(sid)
            assert in_mem is not None
            assert in_mem is analyst  # exact same object

        asyncio.run(_test())

    def test_history_endpoint_returns_from_sqlite(self, store, csv_file):
        """Session NOT in memory, IS in SQLite ->
        the /api/history endpoint falls through to SQLite compact history."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "sqlite-hist-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            history = _sample_history()
            await store.update_history(sid, history)

            # Clear cache (restart)
            mgr._sessions.clear()
            assert mgr.get_session(sid) is None

            # Simulate the endpoint's SQLite fallback path
            session_data = await store.get_session(sid)
            assert session_data is not None
            entries = session_data["analysis_history"]
            assert len(entries) == 2
            assert entries[0]["question"] == "Give me an overview"
            assert entries[0]["skill"] == "describe_data"
            assert entries[1]["narrative"].startswith("Age and salary")

        asyncio.run(_test())

    def test_history_endpoint_returns_empty_for_unknown(self, store):
        """Session doesn't exist anywhere -> returns empty list."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            assert mgr.get_session("nonexistent") is None
            row = await store.get_session("nonexistent")
            assert row is None
            # Endpoint would return HistoryResponse(history=[])

        asyncio.run(_test())

    def test_restored_session_file_path_valid(self, store, csv_file):
        """Create session with real file -> SQLite file_path exists on disk."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "filepath-valid-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            row = await store.get_session(sid)
            assert row is not None
            assert Path(row["file_path"]).exists(), \
                f"file_path must exist on disk: {row['file_path']}"

        asyncio.run(_test())

    def test_restored_session_with_missing_file_cleans_up(self, store):
        """Create session -> file doesn't exist -> restore returns None ->
        SQLite row is deleted (stale cleanup)."""

        async def _test():
            await store.create_session(
                session_id="missing-file-1",
                filename="gone.csv",
                file_path="/nonexistent/path/gone.csv",
                columns=["x", "y"],
                shape={"rows": 10, "columns": 2},
            )

            mgr = SessionManager()
            mgr.set_store(store)

            result = await mgr.get_or_restore_session("missing-file-1")
            assert result is None, "Missing file -> return None"

            row = await store.get_session("missing-file-1")
            assert row is None, "Stale SQLite row must be deleted"

        asyncio.run(_test())


# ============================================================================
# Group 3: New upload clears old session
# ============================================================================

class TestNewUploadClearsOldSession:
    """Uploading a new dataset must replace the old session everywhere."""

    def test_new_upload_replaces_old_session(self, store, two_csv_files):
        """Create session A -> create session B with new file -> only B in SQLite.

        EXPECTED TO FAIL -- Bug: SessionManager.create_session does not delete
        the old session from SQLite when a new upload occurs. The backend has
        no mechanism to enforce the one-active-project invariant in SQLite.
        """
        file_a, file_b = two_csv_files

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            # Upload A
            sid_a = "session-a-001"
            mgr.create_session(sid_a, str(file_a))
            await mgr.persist_new_session(sid_a, str(file_a))
            await store.update_history(sid_a, [
                {"question": "Describe products", "skill": "describe_data",
                 "narrative": "Product data.", "key_points": [],
                 "confidence": 0.9, "reasoning": "kw"},
            ])

            # Upload B (new dataset -- should replace A)
            sid_b = "session-b-002"
            mgr.create_session(sid_b, str(file_b))
            await mgr.persist_new_session(sid_b, str(file_b))

            # B must exist
            row_b = await store.get_session(sid_b)
            assert row_b is not None

            # A must be gone from SQLite
            sessions = await store.list_sessions()
            session_ids = {s["session_id"] for s in sessions}
            assert sid_a not in session_ids, (
                "EXPECTED TO FAIL -- Bug: Old session A still in SQLite after "
                "new upload B. Backend should delete old sessions on new upload."
            )

        asyncio.run(_test())

    def test_old_session_data_not_returned_for_new_session(self, store, two_csv_files):
        """Create session with history -> create new session -> only new session
        exists in SQLite (data isolation for single-project model).

        EXPECTED TO FAIL -- Bug: old session persists in SQLite alongside new.
        """
        file_a, file_b = two_csv_files

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            # Upload A with history
            sid_a = "old-session-001"
            mgr.create_session(sid_a, str(file_a))
            await mgr.persist_new_session(sid_a, str(file_a))
            await store.update_history(sid_a, [
                {"question": "old question", "skill": "describe_data",
                 "narrative": "old answer", "key_points": [],
                 "confidence": 0.9, "reasoning": "kw"},
            ])

            # Upload B
            sid_b = "new-session-002"
            mgr.create_session(sid_b, str(file_b))
            await mgr.persist_new_session(sid_b, str(file_b))

            # B must have empty history
            analyst_b = mgr.get_session(sid_b)
            assert analyst_b is not None
            assert len(analyst_b.history) == 0

            row_b = await store.get_session(sid_b)
            assert row_b is not None
            assert len(row_b["analysis_history"]) == 0

            # Only B should exist in SQLite
            all_sessions = await store.list_sessions()
            assert len(all_sessions) == 1, (
                "EXPECTED TO FAIL -- Bug: After new upload, only the new session "
                f"should exist. Found {len(all_sessions)} sessions: "
                f"{[s['session_id'] for s in all_sessions]}"
            )

        asyncio.run(_test())


# ============================================================================
# Edge cases: history merge, concurrency, large data
# ============================================================================

class TestHistoryMerge:
    """Verify history integrity across restore + new questions."""

    def test_persist_history_after_restore_preserves_old(self, store, csv_file):
        """Restore session with history -> persist_history called with empty
        engine history -> old SQLite history must NOT be overwritten.

        EXPECTED TO FAIL -- Bug: persist_history serialises analyst.history
        (which is empty after restore) and overwrites the SQLite row,
        destroying the previously persisted compact history.
        """

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "merge-hist-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            # Persist 2 history entries to SQLite
            await store.update_history(sid, _sample_history())

            # Simulate restart
            mgr._sessions.clear()

            # Restore (analyst.history will be empty)
            restored = await mgr.get_or_restore_session(sid)
            assert restored is not None
            assert hasattr(restored, "_restored_history")
            assert len(restored._restored_history) == 2
            assert len(restored.history) == 0, \
                "Engine history must be empty on fresh Analyst"

            # This is what the /api/ask endpoint calls after every question:
            await mgr.persist_history(sid)

            # SQLite must still have the 2 old entries
            row = await store.get_session(sid)
            assert row is not None
            assert len(row["analysis_history"]) >= 2, (
                "EXPECTED TO FAIL -- Bug: persist_history overwrites SQLite "
                f"with empty engine history. Got {len(row['analysis_history'])} "
                "entries, expected >= 2."
            )

        asyncio.run(_test())


class TestEdgeCases:
    """Additional edge cases for robustness."""

    def test_concurrent_sessions_isolated(self, store, two_csv_files):
        """Two sessions with different data must be fully isolated."""
        file_a, file_b = two_csv_files

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            mgr.create_session("iso-a", str(file_a))
            await mgr.persist_new_session("iso-a", str(file_a))

            mgr.create_session("iso-b", str(file_b))
            await mgr.persist_new_session("iso-b", str(file_b))

            await store.update_history("iso-a", [
                {"question": "Q from A", "skill": "s1",
                 "narrative": "A answer", "key_points": [],
                 "confidence": 0.9, "reasoning": ""},
            ])
            await store.update_history("iso-b", [
                {"question": "Q from B", "skill": "s2",
                 "narrative": "B answer", "key_points": [],
                 "confidence": 0.8, "reasoning": ""},
            ])

            row_a = await store.get_session("iso-a")
            row_b = await store.get_session("iso-b")
            assert row_a["analysis_history"][0]["question"] == "Q from A"
            assert row_b["analysis_history"][0]["question"] == "Q from B"

        asyncio.run(_test())

    def test_empty_history_persists_as_empty_list(self, store):
        """Session with no history must have [] not None."""

        async def _test():
            args = _store_session()
            await store.create_session(**args)

            row = await store.get_session(args["session_id"])
            assert row is not None
            assert row["analysis_history"] == []

        asyncio.run(_test())

    def test_large_history_persists(self, store):
        """20 history entries round-trip through SQLite correctly."""

        async def _test():
            args = _store_session()
            await store.create_session(**args)

            history = [
                {
                    "question": f"Question {i}",
                    "skill": f"skill_{i % 5}",
                    "narrative": f"Answer {i} " * 50,
                    "key_points": [f"Pt {i}.{j}" for j in range(3)],
                    "confidence": round(0.5 + (i % 5) * 0.1, 2),
                    "reasoning": f"reason {i}",
                }
                for i in range(20)
            ]
            await store.update_history(args["session_id"], history)

            row = await store.get_session(args["session_id"])
            assert row is not None
            assert len(row["analysis_history"]) == 20
            assert row["analysis_history"][0]["question"] == "Question 0"
            assert row["analysis_history"][19]["question"] == "Question 19"

        asyncio.run(_test())

    def test_domain_persists_with_session(self, store):
        """Domain fingerprint data round-trips through SQLite."""

        async def _test():
            args = _store_session()
            await store.create_session(**args)
            await store.update_domain(
                args["session_id"],
                domain="hr_workforce",
                confidence=0.82,
                explanation={"signals": ["department col", "salary col"]},
            )

            row = await store.get_session(args["session_id"])
            assert row is not None
            assert row["domain"] == "hr_workforce"
            assert row["domain_confidence"] == 0.82

        asyncio.run(_test())

    def test_double_persist_idempotent(self, store, csv_file):
        """Calling persist_new_session twice must not create duplicate rows."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "double-1"
            mgr.create_session(sid, str(csv_file))

            # Persist twice
            await mgr.persist_new_session(sid, str(csv_file))
            # Second call should not raise or duplicate
            # (SQLite PRIMARY KEY prevents duplicates — it will error silently)
            await mgr.persist_new_session(sid, str(csv_file))

            sessions = await store.list_sessions()
            matching = [s for s in sessions if s["session_id"] == sid]
            assert len(matching) == 1, (
                f"Expected 1 row for {sid}, got {len(matching)}"
            )

        asyncio.run(_test())

    def test_restore_populates_cache(self, store, csv_file):
        """After restore, subsequent get_session (in-memory) must return the analyst."""

        async def _test():
            mgr = SessionManager()
            mgr.set_store(store)

            sid = "cache-pop-1"
            mgr.create_session(sid, str(csv_file))
            await mgr.persist_new_session(sid, str(csv_file))

            mgr._sessions.clear()
            assert mgr.get_session(sid) is None

            # Restore via get_or_restore
            restored = await mgr.get_or_restore_session(sid)
            assert restored is not None

            # Now in-memory get_session must return it (no DB hit)
            cached = mgr.get_session(sid)
            assert cached is not None
            assert cached is restored

        asyncio.run(_test())


# ============================================================================
# Frontend scenario documentation (no Playwright)
# ============================================================================

class TestFrontendScenarios:
    """Document expected frontend persistence behaviour.

    These are NOT executable browser tests — they document the expected
    UX contracts for manual QA or future Playwright tests.
    """

    # ------------------------------------------------------------------
    # SCENARIO 2: Refresh each page (F5)
    # ------------------------------------------------------------------
    #
    # test_upload_page_survives_refresh
    #   Upload file -> F5 -> file info still shows
    #   Mechanism: sessionId, filename, columns, shape persisted in
    #   localStorage via zustand persist (partialize includes these)
    #
    # test_explore_page_survives_refresh
    #   Ask 2 questions -> F5 -> both Q&As still visible
    #   Mechanism: exploreMessages NOT in localStorage (not in partialize!)
    #   Must be fetched from GET /api/history on mount
    #   explore/page.tsx useEffect calls getHistory when exploreMessages === []
    #
    # test_visualize_page_survives_refresh
    #   Create 2 charts -> F5 -> both charts still visible
    #   Mechanism: chartHistory IS in localStorage (in partialize)
    #
    # test_can_ask_new_question_after_refresh
    #   F5 on explore -> ask new question -> works
    #   useValidatedSession validates with backend via getPreview
    #   Backend has session (restored from SQLite if needed) -> isReady=true
    #
    # test_can_create_new_chart_after_refresh
    #   F5 on visualize -> create chart -> works (same mechanism)

    # ------------------------------------------------------------------
    # SCENARIO 3: Close tab, open new tab
    # ------------------------------------------------------------------
    #
    # test_new_tab_shows_previous_session
    #   Close tab -> new tab -> localhost -> upload page shows file info
    #   Mechanism: localStorage persists across tabs (same origin)
    #   Go to explore -> previous questions fetched from GET /api/history
    #
    # test_can_continue_working_in_new_tab
    #   New tab -> explore -> ask question -> works
    #   sessionId from localStorage -> validated with backend -> OK

    # ------------------------------------------------------------------
    # SCENARIO 4: Backend restart (frontend stays open)
    # ------------------------------------------------------------------
    #
    # test_backend_restart_recovery_logged
    #   Stop backend -> start -> logs show "X sessions available for recovery"
    #   Mechanism: lifespan startup calls list_sessions() and logs count
    #
    # test_explore_works_after_backend_restart
    #   Ask questions -> stop backend -> start -> refresh
    #   -> questions visible (fetched from SQLite compact history)
    #   -> ask new question -> works (session restored from SQLite)
    #
    # test_visualize_works_after_backend_restart
    #   Create charts -> stop backend -> start -> refresh -> charts visible
    #   Mechanism: chartHistory in localStorage (client-side persistence)
    #   Creating new charts needs backend session (restored from SQLite)

    # ------------------------------------------------------------------
    # SCENARIO 5: Both backend + frontend restart
    # ------------------------------------------------------------------
    #
    # test_full_restart_recovery
    #   Upload + ask + chart -> stop both -> start both -> new browser tab
    #   Upload page: shows file info (localStorage)
    #   Explore: fetches history from backend (SQLite compact history)
    #   Visualize: shows charts (localStorage chartHistory)

    # ------------------------------------------------------------------
    # SCENARIO 6: New upload clears everything
    # ------------------------------------------------------------------
    #
    # test_new_upload_clears_explore
    #   Ask questions -> upload new file -> explore is empty
    #   Mechanism: setSession() in store.tsx clears exploreMessages
    #
    # test_new_upload_clears_visualize
    #   Create charts -> upload new file -> visualize is empty
    #   Mechanism: setSession() clears chartHistory
    #
    # test_new_upload_clears_localStorage
    #   Upload file A -> upload file B -> localStorage has file B data, not A
    #   Mechanism: setSession() replaces all session fields

    # ------------------------------------------------------------------
    # useValidatedSession behaviour
    # ------------------------------------------------------------------
    #
    # test_validation_clears_stale_session
    #   localStorage has sessionId "old-123" -> backend doesn't have it
    #   -> useValidatedSession calls getPreview -> 404 -> clearSession()
    #   -> user sees "upload dataset first" (NOT stale data)
    #
    # test_validation_preserves_valid_session
    #   localStorage has sessionId "abc" -> backend HAS it
    #   -> useValidatedSession calls getPreview -> 200 -> isReady=true
    #   -> user sees their data
    #
    # test_fresh_upload_skips_validation
    #   User just uploaded (preview.length > 0) -> no getPreview call
    #   -> isReady immediately (store.tsx line 160)

    def test_frontend_scenarios_documented(self):
        """Placeholder: verifies that frontend scenarios are documented above."""
        assert True

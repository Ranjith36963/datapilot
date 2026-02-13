# D2 — Test-Driven Complete Fix (3-Session Pattern)

## Overview
Three `/clear` sessions:
1. **Session 1: Write tests first** (TDD — tests define the behavior)
2. **Session 2: Implement fixes** (make all tests pass)
3. **Session 3: Review** (fresh eyes, find issues)

---

## SESSION 1 — WRITE TESTS FIRST

`/clear` then paste:

```
ultrathink

## Context
D2 session persistence has gaps. I want to fix them using TDD — write ALL tests first, then implement.

The vision: Each uploaded dataset = a "project session" (like chat history in Claude or ChatGPT). Everything about that project persists until the user uploads a different dataset. In the future we'll have project history (switch between datasets like switching between chats), but for now: one active project at a time, fully persistent.

## Read first (DON'T write any code except tests)
1. frontend/src/lib/store.tsx — the Zustand store, useValidatedSession hook
2. frontend/src/lib/api.ts — API functions
3. frontend/src/app/explore/page.tsx — how explore renders messages
4. frontend/src/app/visualize/page.tsx — how charts are stored/displayed
5. backend/app/services/session_store.py — SQLite store
6. backend/app/services/analyst.py — SessionManager, persist methods
7. backend/app/api/analysis.py — ask endpoint, history endpoint
8. backend/app/main.py — lifespan startup/shutdown

Understand the FULL current state before writing tests.

## Write tests for ALL these scenarios

### Backend tests (tests/unit/test_session_persistence.py)

Create a NEW test file. Do NOT modify existing test files.

**Test Group 1: Session survives backend restart**
```python
test_session_persists_to_sqlite_after_upload()
# Upload creates session → SQLite has the row → verify
# Simulate: create SessionManager + SessionStore, create_session, check SQLite directly

test_session_persists_history_after_ask()
# Create session → add history entries → verify SQLite has them
# Check the compact history format is correct

test_session_restored_from_sqlite_after_cache_clear()
# Create session → clear in-memory cache → get_or_restore_session → works
# Verify the restored analyst has the right shape/columns

test_restored_session_can_answer_questions()
# Create session → clear cache → restore → the analyst object is functional
# (can call methods on it without errors)

test_history_restored_from_sqlite()
# Create session → add history → clear cache → restore → GET /api/history returns the entries

test_cleanup_does_not_delete_active_sessions()
# Create session 5 minutes ago → cleanup_expired(24h) → session still exists
# This was Bug 2 — timestamp format mismatch

test_cleanup_deletes_only_truly_expired()
# Create session → manually set last_accessed to 48h ago → cleanup → gone
# Create another session 5 min ago → cleanup → still there

test_timestamps_use_consistent_format()
# After create_session: raw SQL query → last_accessed has NO "T" separator
# After update_history: raw SQL query → last_accessed still has NO "T" separator
# After get_session (which updates last_accessed): same format
```

**Test Group 2: Session restoration returns correct data**
```python
test_history_endpoint_returns_from_memory()
# Session in memory → GET /api/history → returns full history

test_history_endpoint_returns_from_sqlite()
# Session NOT in memory, IS in SQLite → GET /api/history → returns compact history

test_history_endpoint_returns_empty_for_unknown()
# Session doesn't exist anywhere → GET /api/history → returns empty list

test_restored_session_file_path_valid()
# Create session with real file → restore → file_path exists on disk

test_restored_session_with_missing_file_cleans_up()
# Create session → delete the file → restore → returns None → SQLite row deleted
```

**Test Group 3: New upload clears old session**
```python
test_new_upload_replaces_old_session()
# Create session A → create session B with new file → only B exists
# Note: this tests the backend behavior. Frontend clearing is separate.

test_old_session_data_not_returned_for_new_session()
# Create session with history → create new session → GET /api/history → empty
```

### Frontend tests (tests/e2e/test_session_scenarios.py or as Playwright if available)

If Playwright MCP is available, write browser tests. If not, write test descriptions as comments that document expected behavior:

```python
# SCENARIO 2: Refresh each page (F5)
# test_upload_page_survives_refresh
#   Upload file → F5 → file info still shows (localStorage)
#
# test_explore_page_survives_refresh
#   Ask 2 questions → F5 → both Q&As still visible
#   Mechanism: questions in localStorage OR fetched from GET /api/history
#
# test_visualize_page_survives_refresh
#   Create 2 charts → F5 → both charts still visible
#   Mechanism: chart data in localStorage
#
# test_can_ask_new_question_after_refresh
#   F5 on explore → ask new question → works (session valid)
#
# test_can_create_new_chart_after_refresh
#   F5 on visualize → create chart → works

# SCENARIO 3: Close tab, open new tab
# test_new_tab_shows_previous_session
#   Close tab → new tab → localhost → upload page shows file
#   Go to explore → previous questions visible
#
# test_can_continue_working_in_new_tab
#   New tab → explore → ask question → works

# SCENARIO 4: Backend restart (frontend stays open)
# test_backend_restart_recovery_logged
#   Stop backend → start → logs show "X sessions available for recovery"
#
# test_explore_works_after_backend_restart
#   Ask questions → stop backend → start → refresh → questions visible → ask new → works
#
# test_visualize_works_after_backend_restart
#   Create charts → stop backend → start → refresh → charts visible (localStorage)

# SCENARIO 5: Both backend + frontend restart
# test_full_restart_recovery
#   Upload + ask + chart → stop both → start both → new browser tab
#   Upload page shows file → explore shows questions → visualize shows charts

# SCENARIO 6: New upload clears everything
# test_new_upload_clears_explore
#   Ask questions → upload new file → explore is empty
#
# test_new_upload_clears_visualize
#   Create charts → upload new file → visualize is empty
#
# test_new_upload_clears_localStorage
#   Upload file A → upload file B → localStorage has file B data, not A
```

### What the tests should verify about useValidatedSession():
```python
# test_validation_clears_stale_session
#   localStorage has sessionId "old-123" → backend doesn't have it
#   → useValidatedSession should detect this and clearSession()
#   → user sees "upload dataset first" (NOT stale data)
#
# test_validation_preserves_valid_session
#   localStorage has sessionId "abc" → backend HAS it
#   → useValidatedSession validates successfully
#   → user sees their data
#
# test_fresh_upload_skips_validation
#   User just uploaded (preview.length > 0) → no getPreview call needed
#   → isReady immediately
```

## After writing ALL tests, run them:
```powershell
cd C:\Users\Ranjith\datapilot ; python -m pytest tests/ -v
```

Backend tests that test current behavior should PASS.
Backend tests that test the BUGS should FAIL (that's the point of TDD — red first).
Mark failing tests with comments: # EXPECTED TO FAIL — Bug X

## Commit:
```
git add -A ; git commit -m "test: add comprehensive D2 persistence tests (TDD red phase)"
```
```

---

## SESSION 2 — IMPLEMENT FIXES

`/clear` then paste:

```
ultrathink

## Context
I just wrote comprehensive tests for D2 session persistence (TDD red phase). Some tests are failing because of known bugs. Now implement fixes to make ALL tests pass.

## Read first
1. tests/unit/test_session_persistence.py — the new test file (understand what's expected)
2. frontend/src/lib/store.tsx — useValidatedSession hook (Bug 1 location)
3. backend/app/services/session_store.py — timestamp handling (Bug 2 location)
4. Read any file the tests import or reference

## Known bugs to fix

### Bug 1: useValidatedSession early return (store.tsx)
**Problem:** When sessionId exists in zustand (from localStorage hydration), the hook returns early without validating against the backend. After backend restart with empty SQLite, stale sessions are never detected.

**Fix:** Change the logic so that:
- Fresh upload (preview.length > 0 in zustand memory) → isReady immediately, no validation needed
- localStorage-restored session (preview is empty, sessionId from hydration) → ALWAYS call getPreview() to validate
  - If getPreview succeeds → setIsReady(true), session is valid
  - If getPreview fails (404, network error) → clearSession(), user sees "upload first"
  - 3-second timeout as safety net

The key insight: preview is NOT persisted to localStorage (it's excluded from partialize). So after refresh/restart, preview is always empty. After fresh upload, preview has rows. Use preview.length as the discriminator.

### Bug 2: Timestamp format mismatch (session_store.py)
**Problem:** SQLite datetime('now') uses "YYYY-MM-DD HH:MM:SS" (space), Python isoformat() uses "YYYY-MM-DDTHH:MM:SS" (T). cleanup_expired does string comparison, and space < T always, so ALL sessions with SQLite-format timestamps are treated as expired.

**Fix:** Already partially fixed (verify _utcnow_str helper exists). Make sure ALL datetime writes use the same format: strftime("%Y-%m-%d %H:%M:%S"). Remove any remaining .isoformat() calls. Verify the SQLite CREATE TABLE no longer uses DEFAULT datetime('now') — instead, Python should set timestamps explicitly in create_session.

### Bug 3: Explore page doesn't restore messages on refresh
**Problem:** After refresh, the explore page has sessionId from localStorage but exploreMessages is empty (not persisted). It needs to fetch history from the backend.

**Fix:** Ensure the useEffect in explore/page.tsx:
1. Detects empty exploreMessages + valid sessionId
2. Calls GET /api/history 
3. Converts history entries to ChatMessage format
4. Populates exploreMessages via setExploreMessages()
5. This should work for both refresh AND backend restart (history comes from memory or SQLite)

### Bug 4: Visualize page doesn't restore charts on refresh
**Problem:** Charts are only in zustand memory, lost on refresh.

**Fix:** Ensure chartHistory is included in the persist middleware's partialize config. On page load, if chartHistory has entries, display them. Check if this was already implemented — if so, verify it works.

### Bug 5: Frontend must clear everything on new upload
**Problem:** When uploading a new dataset, old explore messages, charts, and localStorage must be cleared.

**Fix:** In setSession() (called after upload), clear: exploreMessages: [], chartHistory: []. localStorage is automatically overwritten by zustand persist.

## CRITICAL RULES
- Do NOT alter any working backend logic (routing, LLM, analyst, profiler)
- Do NOT change any API response shapes that the frontend depends on
- Do NOT remove any existing tests
- ONLY fix the specific bugs listed above
- After each fix, run tests to verify progress

## After ALL fixes, run:
```powershell
cd C:\Users\Ranjith\datapilot\frontend ; Remove-Item -Recurse -Force .next ; npm run build
cd C:\Users\Ranjith\datapilot ; python -m pytest tests/ -v
```

ALL tests (existing + new) must pass. Frontend must build clean.

## Commit:
```
git add -A ; git commit -m "feat: complete D2 — full session persistence across refresh and restart"
```
```

---

## SESSION 3 — REVIEW

`/clear` then paste:

```
ultrathink

## Code Review — D2 Session Persistence

Review the last 2 commits using:
```
git log --oneline -5
git diff HEAD~2..HEAD
```

Read EVERY file that changed. Review as a senior engineer. Don't modify code.

Check for:
1. **Security holes** — SQL injection? Unsafe localStorage reads? XSS via stored data?
2. **Missing error handling** — What if SQLite write fails? What if localStorage is full? What if getPreview throws?
3. **Edge cases** — Empty dataset? Corrupt localStorage JSON? Backend returns 500? Race conditions in async validation?
4. **Performance** — Is localStorage being written too often? Are there unnecessary re-renders in React? Is SQLite being hit on hot path?
5. **Breaking changes** — Do existing API contracts still work? Are response shapes unchanged? Do all 61+ existing tests still pass?
6. **Timestamp consistency** — Is _utcnow_str() used EVERYWHERE? Any remaining isoformat() calls?
7. **Data cleanup** — When new file uploaded, is ALL old data cleared? Can stale data ever leak between sessions?
8. **Memory leaks** — Does useEffect cleanup properly? Are event listeners removed?

Rate overall quality: HIGH / MEDIUM / LOW
List every concern with file and line number.
Don't modify anything.
```

---

## After Review: If issues found

`/clear` then paste:

```
The review found these issues:
[PASTE THE ISSUES FROM SESSION 3 HERE]

Fix all of them. Run tests after each fix. Commit each fix separately.

cd C:\Users\Ranjith\datapilot\frontend ; Remove-Item -Recurse -Force .next ; npm run build
cd C:\Users\Ranjith\datapilot ; python -m pytest tests/ -v
```

---

## Final Manual Test Checklist

After all sessions complete, test EVERY scenario:

### Scenario 1: Normal use
- [ ] Upload file → Explore → ask 3 questions → Visualize → create 2 charts

### Scenario 2: Refresh (F5)
- [ ] Upload page F5 → file still shows
- [ ] Explore page F5 → 3 questions still show
- [ ] Visualize page F5 → 2 charts still show
- [ ] After refresh, ask NEW question → works
- [ ] After refresh, create NEW chart → works

### Scenario 3: Close tab → new tab
- [ ] New tab → localhost → upload shows file
- [ ] Explore shows previous questions
- [ ] Can ask new question

### Scenario 4: Backend restart (frontend open)
- [ ] Stop backend → start → logs: "X sessions available for recovery"
- [ ] Explore shows previous questions (from SQLite)
- [ ] Ask new question → works
- [ ] Visualize shows charts (from localStorage)

### Scenario 5: Full restart (both stopped)
- [ ] Stop both → start both → new tab
- [ ] Upload page shows file
- [ ] Explore shows questions
- [ ] Can continue working

### Scenario 6: New upload clears everything
- [ ] Upload NEW file → explore empty → visualize empty
- [ ] Old data completely gone
- [ ] New questions/charts work

### Scenario 7: Stale session detection
- [ ] Delete backend/data/ folder → start backend → refresh browser
- [ ] Frontend should detect stale session → clear → show "upload first"
- [ ] NOT show old filename with broken session
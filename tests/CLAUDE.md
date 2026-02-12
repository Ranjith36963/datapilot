# DataPilot Tests

pytest test suite for the DataPilot engine and backend.

## Structure

```
tests/
├── unit/           Unit tests for individual functions
├── integration/    Integration tests for end-to-end flows
├── test_data/      Sample datasets for testing
└── conftest.py     Shared fixtures
```

## Conventions

- Test files named `test_<module>.py`
- Fixtures in `conftest.py` provide sample DataFrames
- Use `@pytest.mark.slow` for tests taking >5s
- Use `@pytest.mark.skipif` for optional dependency tests (spacy, prophet, etc.)
- Test data: `tests/test_data/` for small fixtures, `examples/` for full datasets

## Running

```bash
pytest tests/ -v                    # all tests
pytest tests/unit/ -v               # unit only
pytest tests/integration/ -v        # integration only
pytest tests/ -k "test_classify"    # specific test
pytest tests/ --tb=short            # short tracebacks
```

## Coverage

```bash
pytest tests/ --cov=engine/datapilot --cov-report=html
```

## Planned Phase 2 Test Files
- `tests/unit/test_gemini_provider.py` — Gemini mock tests
- `tests/unit/test_failover.py` — FailoverProvider logic tests
- `tests/unit/test_session_store.py` — SQLite persistence tests
- `tests/unit/test_fingerprint.py` — Domain detection tests
- `tests/unit/test_autopilot.py` — Auto-pilot recipe tests
- `tests/unit/test_domain_narratives.py` — Domain narrative tests

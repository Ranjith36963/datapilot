Run all linters on the DataPilot codebase.

1. Python: `ruff check engine/ backend/ tests/` and `ruff format --check engine/ backend/ tests/`
2. Type checking: `mypy engine/datapilot/ --ignore-missing-imports`
3. Frontend (if exists): `cd frontend && npm run lint`

Report all issues found, grouped by severity.

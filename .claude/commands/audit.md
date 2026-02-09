Run a code quality audit on the DataPilot codebase.

Check and report on:

1. **Test coverage**: `pytest tests/ --cov=engine/datapilot --cov-report=term-missing`
2. **Lint issues**: `ruff check engine/ backend/ --statistics`
3. **Type coverage**: `mypy engine/datapilot/ --ignore-missing-imports`
4. **Security**: Check for hardcoded secrets, unsafe deserialization, SQL injection
5. **Dead code**: Functions defined but never imported/called
6. **Missing docstrings**: Public functions without docstrings
7. **Import health**: Circular imports, unused imports

Output a scorecard with grades (A-F) for each category.

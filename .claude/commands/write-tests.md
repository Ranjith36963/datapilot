Write tests for: $ARGUMENTS

Follow existing test patterns from `tests/unit/`.

1. **Study patterns** — Read 1-2 existing test files in `tests/unit/` to match:
   - Import style and fixture patterns
   - Use of `MagicMock`, `@pytest.fixture`, `@pytest.mark`
   - Test class grouping (e.g., `TestPrimarySucceeds`, `TestEdgeCases`)
   - Naming convention: `test_<what>_<expected_outcome>`

2. **Write tests** — Cover three categories:
   - **Happy path**: Normal inputs produce expected outputs
   - **Edge cases**: Empty data, single row, missing columns, NaN values, boundary conditions
   - **Error cases**: Invalid inputs, provider failures, missing dependencies

3. **Implementation details**:
   - Place test file in `tests/unit/test_<module>.py`
   - Mock external dependencies (LLM providers, file I/O, network)
   - Use fixtures for reusable test data
   - Each test should be independent and fast
   - Assert specific values, not just truthiness

4. **Run tests** — Execute `python -m pytest tests/ -v --tb=short` and ensure all pass.

5. **Commit** — Stage only test files and commit with message describing what was tested.

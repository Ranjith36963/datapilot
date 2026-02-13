Review the current git diff for issues. Focus area: $ARGUMENTS

Run `git diff` (and `git diff --staged` if there are staged changes) then analyze:

1. **Security** — Injection vulnerabilities, hardcoded secrets, unsafe eval/exec, exposed API keys, OWASP top 10 issues.

2. **Correctness** — Logic errors, off-by-one, null/undefined access, type mismatches, missing await on async calls, wrong column names.

3. **Edge cases** — Empty datasets, missing columns, single-row data, NaN/None values, unicode in strings, very large inputs.

4. **Error handling** — Uncaught exceptions, silent failures, missing try/catch around LLM calls, missing deterministic fallbacks.

5. **Pattern violations** — Check against CLAUDE.md rules:
   - Skills return `{"status": "success|error", ...}` dicts, never raw DataFrames
   - LLM calls go through FailoverProvider, never direct provider calls
   - `is_bool_dtype` checked before `is_numeric_dtype`
   - Column names use spaces (not underscores) in suggestions
   - Response metadata includes `provider_used`

6. **Breaking changes** — API contract changes, removed exports, changed function signatures, missing backward compatibility.

Provide specific file:line feedback with severity (critical/warning/nit) and suggested fixes.

Do NOT modify any files. This is a read-only review.

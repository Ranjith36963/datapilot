Review the current git diff for code quality.

Run `git diff` (or `git diff --staged` if there are staged changes) and analyze:

1. **Correctness**: Logic errors, edge cases, off-by-one errors
2. **Security**: Injection vulnerabilities, hardcoded secrets, unsafe operations
3. **Performance**: N+1 queries, unnecessary copies, missing indexes
4. **Style**: Naming conventions, consistent patterns, readability
5. **Tests**: Are new functions tested? Are edge cases covered?
6. **Breaking changes**: API contract changes, removed exports, changed signatures

Provide specific line-by-line feedback with suggestions.

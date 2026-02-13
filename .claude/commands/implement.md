Implement the following task from docs/feature-plan.md: $ARGUMENTS

Steps:

1. **Read the plan** — Open `docs/feature-plan.md` and find the specified task. Understand its scope, the files involved, and how it connects to previous/next tasks.

2. **Implement** — Write only the code required for this task. Follow existing patterns in the codebase. Do not refactor surrounding code or add unrelated improvements.

3. **Test** — Run `python -m pytest tests/ -v` to verify nothing broke. If the plan specifies new tests for this task, write them first or alongside the implementation.

4. **Verify frontend** — If TSX files were changed, run `cd frontend && npm run build` to check for type errors.

5. **Update the plan** — In `docs/feature-plan.md`, check off the completed task by changing `- [ ]` to `- [x]`.

6. **Commit** — Stage only the files related to this task and commit with a message describing what was implemented.

Do NOT implement tasks other than the one specified. One task per invocation.

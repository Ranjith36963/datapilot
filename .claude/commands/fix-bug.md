When fixing a bug, follow this 3-step process strictly:

STEP 1 — REPRODUCE: Figure out how to reproduce the error. Don't fix anything. Understand the complete flow from frontend → backend → engine → LLM. Report the exact file:line where the bug originates.

STEP 2 — THINK: Use extended thinking. Analyze all possible causes. Propose two solutions. Pick the best one. Explain why. Don't implement yet.

STEP 3 — FIX + TEST: Implement the best solution. Run python -m pytest tests/ -v. If tests fail, fix. If tests pass, report what changed.

Do NOT skip steps. Do NOT combine steps.

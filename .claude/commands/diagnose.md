Diagnose the following bug WITHOUT modifying any code: $ARGUMENTS

Follow this process:

1. **Identify relevant files** — Based on the bug description, find the files involved using Glob and Grep. Start from the user-facing layer (frontend/backend API) and trace inward.

2. **Trace the execution flow** — Read each file along the path: Frontend → API endpoint → service layer → engine skill → LLM provider. Map the exact data transformations at each step.

3. **Find the root cause** — Look for:
   - Mismatched types or field names between layers
   - Missing null/empty checks
   - Incorrect parameter passing or filtering
   - LLM prompt issues (wrong template, missing context)
   - State management bugs (session store, cache)

4. **Report findings** — Provide:
   - The exact file:line where the bug originates
   - The execution flow that leads to the bug
   - Why it fails (with evidence from the code)
   - Suggested fix approach (but do NOT implement it)

Do NOT modify any files. This is a read-only diagnosis.

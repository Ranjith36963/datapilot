Plan the implementation of: $ARGUMENTS

Do NOT write any code. Only produce a plan document.

1. **Research phase** — Read the relevant parts of the codebase to understand:
   - Existing patterns and conventions
   - Files that will need changes
   - Dependencies and potential conflicts
   - How similar features were implemented

2. **Design phase** — Think hard about the best approach. Consider:
   - Where new code should live (engine skill, service, API, frontend)
   - How it integrates with existing architecture (request flow, LLM pipeline, session store)
   - Edge cases and error handling boundaries
   - What can break

3. **Write the plan** — Create `docs/feature-plan.md` with:
   - **Goal**: One-sentence summary
   - **Files to modify**: Each file with what changes and why
   - **New files**: If any, with their responsibility
   - **Dependencies**: External packages, API keys, config changes
   - **Task checklist**: Numbered steps in implementation order, each with:
     - [ ] Task description
     - Files: `path/to/file.py`
     - Details: What specifically to implement
   - **Testing strategy**: What to test, which patterns to follow from `tests/`
   - **Risks**: What could go wrong, what to watch for

Keep tasks small and independently testable. Order them so each builds on the last.

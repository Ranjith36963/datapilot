"""
Auto-pilot analysis — LLM-generated analysis plans and execution.

D3: AI Dataset Understanding + Auto-Pilot
- LLM generates custom analysis plans for any dataset
- Serial execution with rate limit protection
- Progressive result streaming
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from ..utils.helpers import setup_logging
from ..llm.prompts.base import build_skill_catalog, get_skill_names

logger = setup_logging("datapilot.autopilot")

# Safety rails
MAX_STEPS = 5
STEP_TIMEOUT_SECONDS = 30
DELAY_BETWEEN_STEPS = 1.0


@dataclass
class AnalysisStep:
    """A single step in an analysis plan."""
    skill: str                    # "analyze_correlations"
    question: str                 # "Which features correlate most with churn?"
    target_column: str | None     # "churn"
    priority: int                 # 1, 2, or 3


@dataclass
class AnalysisPlan:
    """LLM-generated analysis plan."""
    title: str                    # "Analysis of Telecom Churn Dataset"
    steps: list[AnalysisStep]     # ordered list of analysis steps
    provider_used: str            # "gemini" or "groq"


@dataclass
class AutopilotResult:
    """Complete result of an auto-pilot analysis run."""
    understanding: Any            # DatasetUnderstanding from fingerprint
    plan: AnalysisPlan
    results: list[dict]           # [{step, status, result}, ...]
    summary: str | None           # LLM-generated final summary
    total_duration_seconds: float
    completed_steps: int
    skipped_steps: int


# ---------------------------------------------------------------------------
# get_available_skills_description
# ---------------------------------------------------------------------------

def get_available_skills_description() -> str:
    """Build a description of available skills for the LLM prompt.

    Returns a human-readable skill catalog string like:
        - profile_data(param: type): Description
        - analyze_correlations(...): Description
        ...
    """
    return build_skill_catalog()


# ---------------------------------------------------------------------------
# generate_analysis_plan
# ---------------------------------------------------------------------------

async def generate_analysis_plan(
    understanding,
    available_skills: str,
    llm_provider,
) -> Optional[AnalysisPlan]:
    """LLM generates a custom analysis plan based on dataset understanding.

    Args:
        understanding: DatasetUnderstanding instance.
        available_skills: Skill catalog string.
        llm_provider: LLM provider with generate_plan() method.

    Returns:
        AnalysisPlan or None if generation/parsing fails.
    """
    try:
        prompt = (
            f"Dataset domain: {understanding.domain}\n"
            f"Target column: {understanding.target_column} ({understanding.target_type})\n"
            f"Key observations: {'; '.join(understanding.key_observations)}\n\n"
            f"Available analysis skills:\n{available_skills}\n\n"
            "Generate an analysis plan with up to 5 steps. "
            "Each step uses one of the available skills listed above.\n"
            "Respond ONLY with valid JSON:\n"
            '{"title": "<plan title>", "steps": ['
            '{"skill": "<skill_name>", "question": "<analysis question>", '
            '"target_column": "<column or null>", "priority": <1|2|3>}'
            ", ...]}"
        )

        response = llm_provider.generate_plan(prompt)

        if response is None:
            return None

        # Parse JSON (handle markdown code blocks)
        text = response.strip() if isinstance(response, str) else json.dumps(response)
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)

        if "steps" not in data or not isinstance(data["steps"], list):
            logger.warning("Plan JSON missing 'steps' array")
            return None

        # Validate skill names against the real registry
        valid_skills = set(get_skill_names())

        steps: list[AnalysisStep] = []
        for raw in data["steps"]:
            skill = raw.get("skill", "")
            if skill not in valid_skills:
                logger.debug(f"Filtering unknown skill: {skill}")
                continue

            priority = raw.get("priority", 2)
            if not isinstance(priority, int) or priority not in (1, 2, 3):
                priority = 2

            steps.append(AnalysisStep(
                skill=skill,
                question=raw.get("question", ""),
                target_column=raw.get("target_column"),
                priority=priority,
            ))

        # Cap at MAX_STEPS
        steps = steps[:MAX_STEPS]

        if not steps:
            logger.warning("No valid steps after filtering")
            return None

        provider_name = getattr(llm_provider, "name", "unknown")

        return AnalysisPlan(
            title=data.get("title", "Analysis Plan"),
            steps=steps,
            provider_used=provider_name,
        )

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse plan JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"generate_analysis_plan failed: {e}")
        return None


# ---------------------------------------------------------------------------
# run_autopilot
# ---------------------------------------------------------------------------

async def run_autopilot(
    analyst,
    plan: AnalysisPlan,
    understanding,
    on_step_complete=None,
) -> AutopilotResult:
    """Execute the analysis plan step by step.

    Serial execution with DELAY_BETWEEN_STEPS delay between steps.
    Each step calls analyst.ask(question).  Errors are caught per-step
    so one failure doesn't abort the whole run.

    Args:
        analyst: Analyst instance with ask() method.
        plan: The AnalysisPlan to execute.
        understanding: DatasetUnderstanding for context.
        on_step_complete: Optional async callback(step_number, total_steps, result_dict).

    Returns:
        AutopilotResult with all step results.
    """
    start_time = time.time()
    results: list[dict] = []
    completed = 0
    skipped = 0
    total_steps = len(plan.steps)

    for i, step in enumerate(plan.steps):
        step_result: dict

        try:
            result = analyst.ask(step.question)
            step_result = {
                "step": step.skill,
                "status": "complete",
                "result": result,
            }
            completed += 1
        except Exception as e:
            logger.warning(f"Step {i+1}/{total_steps} ({step.skill}) failed: {e}")
            step_result = {
                "step": step.skill,
                "status": "error",
                "error": str(e),
            }
            skipped += 1

        results.append(step_result)

        # Invoke callback
        if on_step_complete is not None:
            await on_step_complete(i + 1, total_steps, step_result)

        # Rate-limit delay (skip after the last step)
        if i < total_steps - 1:
            await asyncio.sleep(DELAY_BETWEEN_STEPS)

    duration = time.time() - start_time

    return AutopilotResult(
        understanding=understanding,
        plan=plan,
        results=results,
        summary=None,
        total_duration_seconds=duration,
        completed_steps=completed,
        skipped_steps=skipped,
    )


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------

async def generate_summary(
    understanding,
    results: list,
    llm_provider,
) -> Optional[str]:
    """LLM synthesizes all auto-pilot results into a coherent summary.

    Args:
        understanding: DatasetUnderstanding for context.
        results: List of step result dicts from run_autopilot.
        llm_provider: LLM provider with generate_summary() method.

    Returns:
        Summary text string, or None if LLM fails.
    """
    try:
        results_text = json.dumps(results, default=str)
        if len(results_text) > 4000:
            results_text = results_text[:4000] + "... (truncated)"

        prompt = (
            f"Dataset: {understanding.domain}\n"
            f"Target: {understanding.target_column}\n\n"
            f"Analysis results:\n{results_text}\n\n"
            "Provide a concise 2-4 sentence summary of the key findings."
        )

        summary = llm_provider.generate_summary(prompt)
        return summary

    except Exception as e:
        logger.warning(f"generate_summary failed: {e}")
        return None

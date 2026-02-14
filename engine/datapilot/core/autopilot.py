"""
Auto-pilot analysis — LLM-generated analysis plans and execution.

D3: AI Dataset Understanding + Auto-Pilot
- LLM generates custom analysis plans for any dataset
- Serial execution with rate limit protection
- Progressive result streaming
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..utils.helpers import setup_logging

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


def get_available_skills_description() -> str:
    """Build a description of available skills for the LLM prompt."""
    raise NotImplementedError("D3: get_available_skills_description not yet implemented")


async def generate_analysis_plan(understanding, available_skills: str, llm_provider) -> AnalysisPlan | None:
    """LLM generates a custom analysis plan based on dataset understanding."""
    raise NotImplementedError("D3: generate_analysis_plan not yet implemented")


async def run_autopilot(analyst, plan: AnalysisPlan, understanding, on_step_complete=None) -> AutopilotResult:
    """Execute the analysis plan step by step. Serial execution with 1s delay."""
    raise NotImplementedError("D3: run_autopilot not yet implemented")


async def generate_summary(understanding, results: list, llm_provider) -> str | None:
    """LLM synthesizes all auto-pilot results into a coherent summary."""
    raise NotImplementedError("D3: generate_summary not yet implemented")

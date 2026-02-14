"""
D3 AI Dataset Understanding + Auto-Pilot tests.

Tests cover:
  1. get_available_skills_description -- returns skill list for LLM prompts
  2. generate_analysis_plan -- LLM-generated analysis plan with validation
  3. run_autopilot -- step-by-step execution with error handling
  4. generate_summary -- LLM-generated final summary
  5. Full pipeline integration (mocked)
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from datapilot.core.autopilot import (
    get_available_skills_description,
    generate_analysis_plan,
    run_autopilot,
    generate_summary,
    AnalysisStep,
    AnalysisPlan,
    AutopilotResult,
)

# DatasetUnderstanding will be added to fingerprint.py as part of D3 stubs.
# Import it here so tests fail at import time if the dataclass is missing.
from datapilot.data.fingerprint import DatasetUnderstanding


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_understanding():
    """A DatasetUnderstanding instance for telecom churn."""
    return DatasetUnderstanding(
        domain="telecom customer churn",
        domain_short="Telecom",
        target_column="churn",
        target_type="classification",
        key_observations=["Churn rate is ~14.5%"],
        suggested_questions=["What drives churn?"],
        data_quality_notes=["No missing values"],
        confidence=0.9,
        provider_used="gemini",
    )


@pytest.fixture
def available_skills_desc():
    """The available skills description string (from get_available_skills_description)."""
    # This mirrors what the real function should return, used to pass to generate_analysis_plan.
    return (
        "- profile_data: Overview statistics, shape, types, quality metrics\n"
        "- describe_data: Detailed column-level statistics (mean, std, distribution)\n"
        "- analyze_correlations: Find correlations between numeric columns. Can target a specific column.\n"
        "- detect_outliers: Find anomalous records using statistical methods\n"
        "- classify: Train classification model to predict a binary/categorical target column\n"
        "- predict_numeric: Train regression model to predict a numeric target column\n"
        "- find_clusters: Discover natural groupings in the data using clustering\n"
        "- analyze_time_series: Analyze trends, seasonality in time-ordered data\n"
        "- run_hypothesis_test: Statistical hypothesis testing between groups\n"
        "- compare_groups: Compare metrics across categorical groups"
    )


@pytest.fixture
def valid_plan_json():
    """Valid LLM plan response with 5 steps."""
    return json.dumps({
        "title": "Analysis of Telecom Churn Dataset",
        "steps": [
            {"skill": "profile_data", "question": "What does the overall data look like?", "target_column": None, "priority": 1},
            {"skill": "analyze_correlations", "question": "Which features correlate with churn?", "target_column": "churn", "priority": 1},
            {"skill": "detect_outliers", "question": "Are there outlier customers?", "target_column": None, "priority": 2},
            {"skill": "classify", "question": "Can we predict churn?", "target_column": "churn", "priority": 1},
            {"skill": "find_clusters", "question": "Are there distinct customer segments?", "target_column": None, "priority": 2},
        ],
    })


@pytest.fixture
def oversized_plan_json():
    """LLM plan response with 8 steps -- should be trimmed to 5."""
    return json.dumps({
        "title": "Oversized Analysis Plan",
        "steps": [
            {"skill": "profile_data", "question": "Overview", "target_column": None, "priority": 1},
            {"skill": "analyze_correlations", "question": "Correlations", "target_column": "churn", "priority": 1},
            {"skill": "detect_outliers", "question": "Outliers", "target_column": None, "priority": 2},
            {"skill": "classify", "question": "Classification", "target_column": "churn", "priority": 1},
            {"skill": "find_clusters", "question": "Clusters", "target_column": None, "priority": 2},
            {"skill": "describe_data", "question": "Descriptions", "target_column": None, "priority": 3},
            {"skill": "predict_numeric", "question": "Prediction", "target_column": "tenure", "priority": 3},
            {"skill": "run_hypothesis_test", "question": "Hypothesis", "target_column": None, "priority": 3},
        ],
    })


@pytest.fixture
def plan_with_invalid_skills_json():
    """LLM plan response containing unknown skill names that should be filtered."""
    return json.dumps({
        "title": "Plan with Invalid Skills",
        "steps": [
            {"skill": "profile_data", "question": "Overview", "target_column": None, "priority": 1},
            {"skill": "nonexistent_skill", "question": "This should be filtered", "target_column": None, "priority": 1},
            {"skill": "analyze_correlations", "question": "Correlations", "target_column": "churn", "priority": 1},
            {"skill": "magic_analysis", "question": "This too should be filtered", "target_column": None, "priority": 2},
        ],
    })


@pytest.fixture
def mock_llm_provider(valid_plan_json):
    """Mock LLM provider with generate_plan and generate_summary methods."""
    provider = MagicMock()
    provider.generate_plan = MagicMock(return_value=valid_plan_json)
    provider.generate_summary = MagicMock(return_value="This telecom dataset reveals a 14.5% churn rate driven primarily by contract type and tenure.")
    return provider


@pytest.fixture
def mock_analyst():
    """Mock analyst with ask() returning a successful result dict."""
    analyst = MagicMock()
    analyst.ask = MagicMock(return_value={
        "status": "success",
        "narrative": "Test narrative",
        "key_points": ["point1"],
        "skill": "profile_data",
    })
    return analyst


@pytest.fixture
def sample_plan():
    """A pre-built AnalysisPlan for execution tests."""
    return AnalysisPlan(
        title="Analysis of Telecom Churn Dataset",
        steps=[
            AnalysisStep(skill="profile_data", question="What does the overall data look like?", target_column=None, priority=1),
            AnalysisStep(skill="analyze_correlations", question="Which features correlate with churn?", target_column="churn", priority=1),
            AnalysisStep(skill="detect_outliers", question="Are there outlier customers?", target_column=None, priority=2),
        ],
        provider_used="gemini",
    )


# ============================================================================
# Group 1: get_available_skills_description
# ============================================================================

class TestGetAvailableSkillsDescription:
    """Test the skill listing function used in LLM prompts."""

    def test_get_available_skills_returns_string(self):
        """get_available_skills_description returns a non-empty string."""
        result = get_available_skills_description()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_available_skills_contains_known_skills(self):
        """Result contains the core skill names: profile_data, analyze_correlations, detect_outliers."""
        result = get_available_skills_description()

        assert "profile_data" in result
        assert "analyze_correlations" in result
        assert "detect_outliers" in result


# ============================================================================
# Group 2: generate_analysis_plan (mocked LLM)
# ============================================================================

class TestGenerateAnalysisPlan:
    """Test LLM-driven analysis plan generation."""

    @pytest.mark.asyncio
    async def test_generate_plan_returns_analysis_plan(self, sample_understanding, available_skills_desc, mock_llm_provider):
        """generate_analysis_plan returns an AnalysisPlan instance."""
        plan = await generate_analysis_plan(sample_understanding, available_skills_desc, mock_llm_provider)

        assert isinstance(plan, AnalysisPlan)
        assert plan.title is not None
        assert len(plan.title) > 0

    @pytest.mark.asyncio
    async def test_generate_plan_has_valid_steps(self, sample_understanding, available_skills_desc, mock_llm_provider):
        """Each step in the plan has skill, question, and priority fields."""
        plan = await generate_analysis_plan(sample_understanding, available_skills_desc, mock_llm_provider)

        assert len(plan.steps) > 0
        for step in plan.steps:
            assert isinstance(step, AnalysisStep)
            assert isinstance(step.skill, str) and len(step.skill) > 0
            assert isinstance(step.question, str) and len(step.question) > 0
            assert isinstance(step.priority, int) and step.priority in (1, 2, 3)

    @pytest.mark.asyncio
    async def test_generate_plan_limits_to_5_steps(self, sample_understanding, available_skills_desc, oversized_plan_json):
        """When LLM returns 8 steps, plan is trimmed to max 5."""
        provider = MagicMock()
        provider.generate_plan = MagicMock(return_value=oversized_plan_json)

        plan = await generate_analysis_plan(sample_understanding, available_skills_desc, provider)

        assert isinstance(plan, AnalysisPlan)
        assert len(plan.steps) <= 5

    @pytest.mark.asyncio
    async def test_generate_plan_validates_skill_names(self, sample_understanding, available_skills_desc, plan_with_invalid_skills_json):
        """Unknown skill names are filtered out of the plan."""
        provider = MagicMock()
        provider.generate_plan = MagicMock(return_value=plan_with_invalid_skills_json)

        plan = await generate_analysis_plan(sample_understanding, available_skills_desc, provider)

        assert isinstance(plan, AnalysisPlan)
        skill_names = [step.skill for step in plan.steps]
        assert "nonexistent_skill" not in skill_names
        assert "magic_analysis" not in skill_names
        # Valid skills should remain
        assert "profile_data" in skill_names
        assert "analyze_correlations" in skill_names

    @pytest.mark.asyncio
    async def test_generate_plan_handles_invalid_json(self, sample_understanding, available_skills_desc):
        """Garbage LLM response results in None return."""
        provider = MagicMock()
        provider.generate_plan = MagicMock(return_value="This is not valid JSON at all {{{")

        plan = await generate_analysis_plan(sample_understanding, available_skills_desc, provider)

        assert plan is None

    @pytest.mark.asyncio
    async def test_generate_plan_both_fail_returns_none(self, sample_understanding, available_skills_desc):
        """When both LLM providers fail (exception), returns None."""
        provider = MagicMock()
        provider.generate_plan = MagicMock(side_effect=Exception("All providers down"))

        plan = await generate_analysis_plan(sample_understanding, available_skills_desc, provider)

        assert plan is None


# ============================================================================
# Group 3: run_autopilot (mocked analyst)
# ============================================================================

class TestRunAutopilot:
    """Test the autopilot execution engine."""

    @pytest.mark.asyncio
    async def test_run_autopilot_executes_all_steps(self, mock_analyst, sample_plan, sample_understanding):
        """All steps succeed, returns AutopilotResult with correct counts."""
        result = await run_autopilot(mock_analyst, sample_plan, sample_understanding)

        assert isinstance(result, AutopilotResult)
        assert result.completed_steps == len(sample_plan.steps)
        assert result.skipped_steps == 0
        assert len(result.results) == len(sample_plan.steps)
        for step_result in result.results:
            assert step_result["status"] == "complete"

    @pytest.mark.asyncio
    async def test_run_autopilot_handles_step_error(self, sample_plan, sample_understanding):
        """When one step raises an exception, it gets status 'error' but others continue."""
        analyst = MagicMock()

        call_count = 0
        def side_effect(question):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Skill execution failed")
            return {
                "status": "success",
                "narrative": "Test narrative",
                "key_points": ["point1"],
                "skill": "profile_data",
            }

        analyst.ask = MagicMock(side_effect=side_effect)

        result = await run_autopilot(analyst, sample_plan, sample_understanding)

        assert isinstance(result, AutopilotResult)
        assert len(result.results) == len(sample_plan.steps)

        # Second step should have error status
        assert result.results[1]["status"] == "error"

        # Other steps should have completed
        assert result.results[0]["status"] == "complete"
        assert result.results[2]["status"] == "complete"

    @pytest.mark.asyncio
    async def test_run_autopilot_calls_on_step_complete(self, mock_analyst, sample_plan, sample_understanding):
        """The on_step_complete callback is invoked for each completed step."""
        callback = AsyncMock()

        result = await run_autopilot(
            mock_analyst,
            sample_plan,
            sample_understanding,
            on_step_complete=callback,
        )

        assert callback.call_count == len(sample_plan.steps)
        # Verify callback was called with step index and total
        first_call_args = callback.call_args_list[0]
        # Should receive (step_number, total_steps, result)
        assert first_call_args[0][0] == 1  # first step number
        assert first_call_args[0][1] == len(sample_plan.steps)  # total steps


# ============================================================================
# Group 4: generate_summary
# ============================================================================

class TestGenerateSummary:
    """Test LLM-generated final summary."""

    @pytest.mark.asyncio
    async def test_generate_summary_returns_text(self, sample_understanding, mock_llm_provider):
        """generate_summary returns a non-empty summary string."""
        mock_results = [
            {"step": "profile_data", "status": "complete", "result": {"narrative": "Overview narrative"}},
            {"step": "analyze_correlations", "status": "complete", "result": {"narrative": "Correlation narrative"}},
        ]

        summary = await generate_summary(sample_understanding, mock_results, mock_llm_provider)

        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_generate_summary_handles_failure(self, sample_understanding):
        """When LLM fails, generate_summary returns None."""
        provider = MagicMock()
        provider.generate_summary = MagicMock(side_effect=Exception("LLM unavailable"))

        mock_results = [
            {"step": "profile_data", "status": "complete", "result": {"narrative": "Overview"}},
        ]

        summary = await generate_summary(sample_understanding, mock_results, provider)

        assert summary is None


# ============================================================================
# Group 5: Full Pipeline Integration (all mocked)
# ============================================================================

class TestFullPipeline:
    """End-to-end test: understanding -> plan -> execute -> summary."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(self, sample_understanding, available_skills_desc, mock_llm_provider, mock_analyst):
        """Full pipeline produces AutopilotResult with all fields populated."""
        # Step 1: Generate plan
        plan = await generate_analysis_plan(sample_understanding, available_skills_desc, mock_llm_provider)
        assert isinstance(plan, AnalysisPlan)
        assert len(plan.steps) > 0

        # Step 2: Execute plan
        result = await run_autopilot(mock_analyst, plan, sample_understanding)
        assert isinstance(result, AutopilotResult)
        assert result.completed_steps > 0
        assert result.understanding is sample_understanding
        assert result.plan is plan
        assert len(result.results) == len(plan.steps)

        # Step 3: Generate summary
        summary = await generate_summary(sample_understanding, result.results, mock_llm_provider)
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Verify AutopilotResult structure
        assert result.total_duration_seconds >= 0
        assert isinstance(result.results, list)
        assert result.skipped_steps >= 0

"""Tests for FailoverProvider — task-aware failover logic."""

import pytest
from unittest.mock import MagicMock

from datapilot.llm.failover import DEFAULT_TASK_ROUTING, FailoverProvider
from datapilot.llm.provider import NarrativeResult, RoutingResult


def _make_routing_result(skill="describe_data", confidence=0.9):
    return RoutingResult(
        skill_name=skill,
        parameters={},
        confidence=confidence,
        reasoning="test",
        route_method="llm",
    )


def _make_narrative_result(text="The dataset has 100 rows and 5 columns with interesting patterns."):
    return NarrativeResult(
        text=text,
        key_points=["100 rows"],
        suggestions=["Explore more"],
    )


@pytest.fixture
def mock_groq():
    provider = MagicMock()
    provider.route_question.return_value = _make_routing_result()
    provider.generate_narrative.return_value = _make_narrative_result()
    provider.suggest_chart.return_value = {"suggestions": [{"chart_type": "histogram"}]}
    provider.generate_chart_insight.return_value = "Age peaks at 35 years old."
    return provider


@pytest.fixture
def mock_gemini():
    provider = MagicMock()
    provider.route_question.return_value = _make_routing_result(skill="profile_data")
    provider.generate_narrative.return_value = _make_narrative_result("Gemini narrative with 200 rows analyzed.")
    provider.suggest_chart.return_value = {"suggestions": [{"chart_type": "scatter"}]}
    provider.generate_chart_insight.return_value = "Salary correlates strongly with experience."
    return provider


@pytest.fixture
def failover(mock_groq, mock_gemini):
    return FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})


class TestPrimarySucceeds:
    def test_primary_succeeds_secondary_not_called(self, failover, mock_groq, mock_gemini):
        """When primary succeeds, secondary provider is never called."""
        result = failover.route_question("describe data", {}, "catalog")

        assert result is not None
        assert result.skill_name == "describe_data"
        mock_groq.route_question.assert_called_once()
        mock_gemini.route_question.assert_not_called()

    def test_narrative_primary_is_gemini(self, failover, mock_groq, mock_gemini):
        """For narratives, Gemini is tried first per DEFAULT_TASK_ROUTING."""
        result = failover.generate_narrative({"data": "test"})

        assert result is not None
        assert "Gemini" in result.text
        mock_gemini.generate_narrative.assert_called_once()
        mock_groq.generate_narrative.assert_not_called()


class TestFailover:
    def test_primary_fails_secondary_succeeds(self, failover, mock_groq, mock_gemini):
        """When primary raises exception, secondary succeeds."""
        mock_groq.route_question.side_effect = Exception("Groq timeout")

        result = failover.route_question("describe data", {}, "catalog")

        assert result is not None
        assert result.skill_name == "profile_data"  # Gemini's response
        mock_groq.route_question.assert_called_once()
        mock_gemini.route_question.assert_called_once()

    def test_both_fail_returns_none(self, failover, mock_groq, mock_gemini):
        """When all providers fail, returns None."""
        mock_groq.route_question.side_effect = Exception("Groq down")
        mock_gemini.route_question.return_value = None

        result = failover.route_question("describe data", {}, "catalog")

        assert result is None


class TestTaskRouting:
    def test_routing_tries_groq_first(self):
        """route_question tries Groq first per DEFAULT_TASK_ROUTING."""
        assert DEFAULT_TASK_ROUTING["routing"][0] == "groq"

    def test_narrative_tries_gemini_first(self):
        """generate_narrative tries Gemini first per DEFAULT_TASK_ROUTING."""
        assert DEFAULT_TASK_ROUTING["narrative"][0] == "gemini"

    def test_chart_suggest_tries_gemini_first(self):
        """suggest_chart tries Gemini first per DEFAULT_TASK_ROUTING."""
        assert DEFAULT_TASK_ROUTING["chart_suggest"][0] == "gemini"

    def test_chart_insight_tries_groq_first(self):
        """generate_chart_insight tries Groq first per DEFAULT_TASK_ROUTING."""
        assert DEFAULT_TASK_ROUTING["chart_insight"][0] == "groq"


class TestEdgeCases:
    def test_provider_not_in_routing_still_tried(self):
        """A provider not listed in routing table still gets tried as fallback."""
        custom = MagicMock()
        custom.route_question.return_value = _make_routing_result(skill="custom_skill")

        fp = FailoverProvider(providers={"custom_provider": custom})
        result = fp.route_question("test", {}, "catalog")

        assert result is not None
        assert result.skill_name == "custom_skill"

    def test_single_provider_mode(self, mock_groq):
        """Works with only one provider for all tasks."""
        fp = FailoverProvider(providers={"groq": mock_groq})

        route = fp.route_question("test", {}, "catalog")
        assert route is not None

        narrative = fp.generate_narrative({"data": "test"})
        assert narrative is not None

        chart = fp.suggest_chart({"columns": []})
        assert chart["suggestions"]

        insight = fp.generate_chart_insight({"summary": "test"})
        assert len(insight) > 0

    def test_route_method_includes_provider_name(self, failover):
        """route_method metadata includes the provider name."""
        result = failover.route_question("describe data", {}, "catalog")

        assert result is not None
        assert result.route_method == "llm:groq"

    def test_narrative_failover_route_method(self, failover, mock_groq, mock_gemini):
        """When narrative primary (Gemini) fails, Groq result is returned."""
        mock_gemini.generate_narrative.side_effect = Exception("Gemini down")

        result = failover.generate_narrative({"data": "test"})

        assert result is not None
        assert result.text  # Groq's narrative
        mock_gemini.generate_narrative.assert_called_once()
        mock_groq.generate_narrative.assert_called_once()


# ============================================================================
# D3 method tests: understand_dataset, generate_plan, generate_summary
# ============================================================================

class TestD3UnderstandDataset:
    """Test FailoverProvider.understand_dataset() failover behavior."""

    def test_primary_succeeds(self, mock_groq, mock_gemini):
        """Gemini (primary for understand) called first, returns result."""
        mock_gemini.understand_dataset.return_value = {"domain": "telecom", "_provider_used": "gemini"}
        mock_groq.understand_dataset.return_value = {"domain": "telecom_groq"}

        fp = FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})
        result = fp.understand_dataset("snapshot text")

        assert result is not None
        assert result["domain"] == "telecom"
        mock_gemini.understand_dataset.assert_called_once()
        mock_groq.understand_dataset.assert_not_called()

    def test_failover_to_secondary(self, mock_groq, mock_gemini):
        """Gemini fails → Groq called and succeeds."""
        mock_gemini.understand_dataset.side_effect = Exception("Gemini down")
        mock_groq.understand_dataset.return_value = {"domain": "telecom_groq"}

        fp = FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})
        result = fp.understand_dataset("snapshot text")

        assert result is not None
        assert result["domain"] == "telecom_groq"
        mock_gemini.understand_dataset.assert_called_once()
        mock_groq.understand_dataset.assert_called_once()

    def test_both_fail_returns_none(self, mock_groq, mock_gemini):
        """Both providers fail → returns None."""
        mock_gemini.understand_dataset.side_effect = Exception("Gemini down")
        mock_groq.understand_dataset.return_value = None

        fp = FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})
        result = fp.understand_dataset("snapshot text")

        assert result is None


class TestD3GeneratePlan:
    """Test FailoverProvider.generate_plan() failover behavior."""

    def test_primary_succeeds(self, mock_groq, mock_gemini):
        """Gemini returns plan text → Groq not called."""
        mock_gemini.generate_plan.return_value = '{"title": "Plan", "steps": []}'
        mock_groq.generate_plan.return_value = '{"title": "Groq Plan", "steps": []}'

        fp = FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})
        result = fp.generate_plan("prompt text")

        assert result is not None
        assert "Plan" in result
        mock_gemini.generate_plan.assert_called_once()
        mock_groq.generate_plan.assert_not_called()

    def test_failover(self, mock_groq, mock_gemini):
        """Gemini fails → Groq returns plan."""
        mock_gemini.generate_plan.side_effect = Exception("Gemini down")
        mock_groq.generate_plan.return_value = '{"title": "Groq Plan", "steps": []}'

        fp = FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})
        result = fp.generate_plan("prompt text")

        assert result is not None
        assert "Groq" in result


class TestD3GenerateSummary:
    """Test FailoverProvider.generate_summary() failover behavior."""

    def test_primary_succeeds(self, mock_groq, mock_gemini):
        """Gemini returns summary → Groq not called."""
        mock_gemini.generate_summary.return_value = "The dataset reveals clear churn patterns driven by tenure."
        mock_groq.generate_summary.return_value = "Groq summary."

        fp = FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})
        result = fp.generate_summary("prompt text")

        assert result is not None
        assert "churn" in result.lower()
        mock_gemini.generate_summary.assert_called_once()
        mock_groq.generate_summary.assert_not_called()

    def test_both_fail_returns_none(self, mock_groq, mock_gemini):
        """Both providers fail → returns None."""
        mock_gemini.generate_summary.side_effect = Exception("Gemini down")
        mock_groq.generate_summary.return_value = None

        fp = FailoverProvider(providers={"groq": mock_groq, "gemini": mock_gemini})
        result = fp.generate_summary("prompt text")

        assert result is None

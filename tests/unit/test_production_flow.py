"""
Production-flow E2E tests — uses Analyst.ask() exactly like the backend API.

This test simulates the EXACT code path that runs when a user types a question
on the Explore page:

  Frontend POST /api/ask → analyst.ask(question) → Router → Executor → Narrative → Response

The LLM provider is a realistic mock that:
  - Returns sensible JSON params when skills call _extract_params_via_llm()
  - Returns narrative text when generate_narrative() is called
  - Generates code when smart_query calls generate_plan()

This tests the FULL chain, not individual components.
"""

import json
import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from datapilot.core.analyst import Analyst, AnalystResult
from datapilot.llm.provider import LLMProvider, NarrativeResult, RoutingResult

TEST_DATA = pathlib.Path(__file__).resolve().parent.parent / "test_data"


# ============================================================================
# Realistic LLM mock — behaves like a real Groq/Gemini provider
# ============================================================================

class RealisticMockLLM(LLMProvider):
    """Mock LLM that returns realistic responses based on prompt content.

    In production, _extract_params_via_llm() sends prompts like:
        "Dataset columns: Survived, Pclass, Name, Sex, Age...
         User question: How many survived vs died?
         Extract parameters for 'value_counts':
           column: str (column to count)
           top_n: int (optional)"

    This mock reads the skill name and question from the prompt and
    returns what a real LLM would return.
    """

    def __init__(self):
        self.call_log = []  # Track all calls for debugging

    def generate_plan(self, prompt: str) -> str:
        """Simulate LLM parameter extraction — the critical production path."""
        self.call_log.append(("generate_plan", prompt[:100]))

        # Parse skill name from prompt: "Extract parameters for 'skill_name':"
        import re
        skill_match = re.search(r"Extract parameters for '(\w+)'", prompt)
        skill = skill_match.group(1) if skill_match else ""

        q_match = re.search(r"User question:\s*(.+?)(?:\n|$)", prompt)
        question = q_match.group(1).strip().lower() if q_match else ""

        # Return realistic JSON based on skill + question
        if skill == "value_counts":
            if "survived" in question or "died" in question:
                return json.dumps({"column": "Survived", "top_n": None})
            if "class" in question or "pclass" in question:
                return json.dumps({"column": "Pclass", "top_n": None})
            if "embark" in question or "port" in question:
                return json.dumps({"column": "Embarked", "top_n": None})
            # Default: find first categorical-looking column from prompt
            return json.dumps({"column": "Survived", "top_n": None})

        if skill == "pivot_table":
            if "age" in question and "class" in question:
                return json.dumps({"values": "Age", "index": "Pclass", "aggfunc": "mean"})
            if "fare" in question and "class" in question:
                return json.dumps({"values": "Fare", "index": "Pclass", "aggfunc": "mean"})
            return json.dumps({"values": "Age", "index": "Pclass", "aggfunc": "mean"})

        if skill == "query_data":
            if "under 10" in question and "survived" in question:
                return json.dumps({"filter_expression": "Age < 10 and Survived == 1", "columns": None})
            if "female" in question:
                return json.dumps({"filter_expression": "Sex == 'female'", "columns": None})
            return json.dumps({"filter_expression": "Survived == 1", "columns": None})

        if skill == "top_n":
            if "fare" in question:
                return json.dumps({"column": "Fare", "n": 10, "ascending": False})
            if "age" in question:
                return json.dumps({"column": "Age", "n": 10, "ascending": False})
            return json.dumps({"column": "Fare", "n": 10, "ascending": False})

        if skill == "cross_tab":
            if "sex" in question and "survived" in question:
                return json.dumps({"row": "Sex", "col": "Survived", "values": None, "aggfunc": None})
            if "class" in question and "survived" in question:
                return json.dumps({"row": "Pclass", "col": "Survived", "values": None, "aggfunc": None})
            return json.dumps({"row": "Sex", "col": "Survived", "values": None, "aggfunc": None})

        # smart_query — return pandas code
        if "Write pandas code" in prompt:
            if "survived" in question and "rate" in question and "port" in question:
                return "result = df.groupby('Embarked')['Survived'].mean().reset_index()\nresult.columns = ['Port', 'Survival_Rate']"
            return "result = df.head(10)"

        # Default: return None (let deterministic fallback handle it)
        return None

    def route_question(self, question, data_context, skill_catalog) -> RoutingResult:
        return RoutingResult(
            skill_name="profile_data", parameters={},
            confidence=0.5, reasoning="mock", route_method="mock",
        )

    def generate_narrative(self, analysis_result, question=None,
                           skill_name=None, conversation_context=None) -> NarrativeResult:
        """Simulate LLM narrative generation."""
        self.call_log.append(("generate_narrative", skill_name))
        # Return a realistic narrative based on the actual result data
        status = analysis_result.get("status", "success")
        if status == "error":
            return NarrativeResult(
                text=f"The analysis encountered an issue: {analysis_result.get('message', 'unknown error')}",
                key_points=[], suggestions=[],
            )

        # Build narrative from actual data (like production LLM would)
        data = analysis_result.get("data", [])
        total = analysis_result.get("total_rows", len(data) if isinstance(data, list) else 0)

        text = f"Based on the analysis of your question"
        if question:
            text = f"To answer '{question}'"

        if skill_name == "value_counts":
            col = analysis_result.get("column", "")
            if isinstance(data, dict):
                items = list(data.items())[:3]
                details = ", ".join(f"{k}: {v}" for k, v in items)
                text += f", the distribution of {col} shows: {details}."
            else:
                text += f", I found {total} unique values in {col}."

        elif skill_name == "pivot_table":
            idx = analysis_result.get("index_column", "")
            vals = analysis_result.get("values_column", "")
            agg = analysis_result.get("aggfunc", "mean")
            if isinstance(data, list) and data:
                text += f", the {agg} of {vals} by {idx} shows {len(data)} groups."
            else:
                text += f", I computed {agg} of {vals} grouped by {idx}."

        elif skill_name == "classify":
            algo = analysis_result.get("algorithm", "unknown")
            metrics = analysis_result.get("metrics", {})
            acc = metrics.get("accuracy", 0)
            text += f", I trained a {algo} model with {acc*100:.1f}% accuracy."

        elif skill_name == "analyze_correlations":
            top = analysis_result.get("top_correlations", [])
            if top:
                best = top[0]
                text += (f", the strongest correlation is between "
                         f"{best.get('col1', '?')} and {best.get('col2', '?')} "
                         f"(r={best.get('correlation', 0):.3f}).")

        elif skill_name == "detect_outliers":
            n = analysis_result.get("n_outliers", 0)
            pct = analysis_result.get("outlier_pct", 0)
            text += f", I found {n} outliers ({pct}% of the data)."

        elif skill_name == "run_hypothesis_test":
            pval = analysis_result.get("pvalue", 0)
            sig = analysis_result.get("significant", False)
            text += f", the test returned p={pval:.4f}. "
            text += "The difference is statistically significant." if sig else "No significant difference found."

        elif skill_name == "cross_tab":
            row_col = analysis_result.get("row_column", "")
            col_col = analysis_result.get("col_column", "")
            text += f", the cross-tabulation of {row_col} by {col_col} shows {len(data) if isinstance(data, list) else 0} groups."

        elif skill_name == "top_n":
            n = analysis_result.get("n", 10)
            col = analysis_result.get("column", "")
            direction = analysis_result.get("direction", "top")
            text += f", here are the {direction} {n} records by {col}."

        elif skill_name == "query_data":
            text += f", I found {total} matching rows."

        else:
            text += f", the {skill_name} analysis is complete with {total} results."

        return NarrativeResult(
            text=text,
            key_points=["Analysis completed successfully"],
            suggestions=["Try another question about this data"],
        )

    def suggest_chart(self, data_context, analysis_result=None):
        return {"suggestions": []}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def titanic_df():
    return pd.read_csv(TEST_DATA / "Titanic-Dataset.csv")


@pytest.fixture(scope="module")
def mock_provider():
    return RealisticMockLLM()


@pytest.fixture(scope="module")
def analyst(titanic_df, mock_provider):
    """Create Analyst exactly like production — Analyst(data, llm=provider)."""
    return Analyst(data=titanic_df, llm=mock_provider, auto_profile=False)


# ============================================================================
# Tests — Production Flow via Analyst.ask()
# ============================================================================

class TestProductionFlow:
    """Each test calls analyst.ask() — the EXACT same code path as POST /api/ask."""

    def test_q01_survived_vs_died(self, analyst):
        """Q1: How many passengers survived vs died?"""
        result = analyst.ask("How many passengers survived vs died?")

        assert isinstance(result, AnalystResult)
        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("value_counts", "cross_tab", "query_data", "smart_query", "profile_data")
        assert result.narrative is not None, "No narrative generated"
        assert len(result.text) > 20, f"Narrative too short: '{result.text}'"
        assert result.routing_ms > 0
        assert result.execution_ms > 0

    def test_q02_avg_age_by_class(self, analyst):
        """Q2: What's the average age by passenger class?"""
        result = analyst.ask("What's the average age by passenger class?")

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("pivot_table", "compare_groups", "smart_query", "describe_data")
        assert result.narrative is not None
        # Check actual data
        data = result.data
        if result.skill_name == "pivot_table":
            assert data.get("values_column") in ("Age", "age")
            assert data.get("index_column") in ("Pclass", "pclass")
            records = data.get("data", [])
            assert len(records) >= 2, "Should have at least 2 passenger classes"

    def test_q03_young_survivors(self, analyst):
        """Q3: Show all passengers under 10 who survived."""
        result = analyst.ask("Show all passengers under 10 who survived.")

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("query_data", "smart_query", "profile_data")
        data = result.data
        if result.skill_name == "query_data" and data.get("query_description") != "all rows":
            total = data.get("total_rows", 0)
            assert total < 891, f"Filter should reduce rows, got {total}"
            assert total > 0, "Should find some young survivors"

    def test_q04_survival_rate_by_port(self, analyst):
        """Q4: What's the survival rate by embarkation port?"""
        result = analyst.ask("What's the survival rate by embarkation port?")

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("pivot_table", "cross_tab", "value_counts",
                                      "compare_groups", "smart_query")
        assert result.narrative is not None

    def test_q05_top_10_fare(self, analyst):
        """Q5: Show me the top 10 passengers by fare."""
        result = analyst.ask("Show me the top 10 passengers by fare.")

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("top_n", "smart_query")
        data = result.data
        if result.skill_name == "top_n":
            assert data.get("n") == 10
            assert data.get("column") in ("Fare", "fare")
            assert data.get("direction") == "top"
            records = data.get("data", [])
            assert len(records) == 10, f"Expected 10 records, got {len(records)}"
            # Verify sorted descending
            fares = [r.get("Fare", 0) for r in records]
            assert fares == sorted(fares, reverse=True), "Should be sorted descending"

    def test_q06_hypothesis_test_fare(self, analyst):
        """Q6: Is there a significant difference in fare between survivors and non-survivors?"""
        result = analyst.ask(
            "Is there a significant difference in fare between survivors and non-survivors?"
        )

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("run_hypothesis_test", "compare_groups", "smart_query")
        data = result.data
        if result.skill_name == "run_hypothesis_test":
            assert "pvalue" in data, f"Missing pvalue in result: {list(data.keys())}"
            assert isinstance(data["pvalue"], float)
            assert "significant" in data
            assert data.get("test") in ("t_test", "mann_whitney", "welch_t", "independent_t_test")

    def test_q07_correlations(self, analyst):
        """Q7: Show correlations between Age, Fare, SibSp, Parch, and Survived."""
        result = analyst.ask(
            "Show correlations between Age, Fare, SibSp, Parch, and Survived."
        )

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("analyze_correlations", "smart_query")
        data = result.data
        if result.skill_name == "analyze_correlations":
            top = data.get("top_correlations", [])
            assert len(top) > 0, "Should find at least one correlation pair"
            # Each pair has col1, col2, correlation
            first = top[0]
            assert "col1" in first
            assert "col2" in first
            assert "correlation" in first

    def test_q08_crosstab_sex_survived(self, analyst):
        """Q8: Create a cross-tab of Sex vs Survived."""
        result = analyst.ask("Create a cross-tab of Sex vs Survived.")

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("cross_tab", "smart_query", "value_counts")
        data = result.data
        if result.skill_name == "cross_tab":
            assert data.get("row_column") in ("Sex", "Survived")
            assert data.get("col_column") in ("Sex", "Survived")
            records = data.get("data", [])
            assert len(records) >= 2, "Should have male/female rows"

    def test_q09_fare_outliers(self, analyst):
        """Q9: Are there outliers in the Fare column?"""
        result = analyst.ask("Are there outliers in the Fare column?")

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("detect_outliers", "smart_query", "describe_data")
        data = result.data
        if result.skill_name == "detect_outliers":
            assert "n_outliers" in data or "outlier_count" in data
            n = data.get("n_outliers", data.get("outlier_count", 0))
            assert n > 0, "Titanic Fare should have outliers"

    def test_q10_predict_survival(self, analyst):
        """Q10: Build a model to predict survival."""
        result = analyst.ask("Build a model to predict survival.")

        assert result.status == "success", f"Failed: {result.execution.error}"
        assert result.skill_name in ("classify", "smart_query", "predict_numeric")
        data = result.data
        if result.skill_name == "classify":
            # Check model was actually trained
            assert "algorithm" in data, f"Missing algorithm: {list(data.keys())}"
            metrics = data.get("metrics", {})
            assert metrics.get("accuracy") is not None or data.get("accuracy") is not None
            fi = data.get("feature_importance", data.get("feature_importances", []))
            assert len(fi) > 0, "Should have feature importance"


# ============================================================================
# Summary test — verify the full chain outputs match API response format
# ============================================================================

class TestAPIResponseFormat:
    """Verify result matches what the backend sends to the frontend."""

    def test_result_has_all_api_fields(self, analyst):
        """The AnalystResult.to_dict() must contain all fields the frontend expects."""
        result = analyst.ask("How many passengers survived vs died?")
        d = result.to_dict()

        # These are the fields the AskResponse Pydantic model requires
        assert "question" in d
        assert "skill" in d
        assert "confidence" in d
        assert "reasoning" in d
        assert "route_method" in d
        assert "status" in d
        assert "routing_ms" in d
        assert "execution_ms" in d
        assert "narration_ms" in d

    def test_narrative_is_not_empty(self, analyst):
        """Production always generates a narrative (LLM or template)."""
        result = analyst.ask("Show me the top 10 passengers by fare.")
        assert result.narrative is not None
        assert len(result.text) > 10, f"Narrative too short: '{result.text}'"
        # Should have key_points
        assert isinstance(result.key_points, list)
        # Should have suggestions
        assert isinstance(result.suggestions, list)

"""Tests for GeminiProvider — all API calls mocked."""

import json
import sys
import pytest
from unittest.mock import MagicMock, patch


# Set up mock google.genai module before importing GeminiProvider
_mock_genai_module = MagicMock()
_mock_types_module = MagicMock()


@pytest.fixture(autouse=True)
def _patch_google_genai():
    """Mock google-genai SDK in sys.modules so GeminiProvider can import it."""
    mock_google = MagicMock()
    mock_google.genai = _mock_genai_module
    with patch.dict(sys.modules, {
        "google": mock_google,
        "google.genai": _mock_genai_module,
        "google.genai.types": _mock_types_module,
    }):
        yield


from datapilot.llm.gemini import GeminiProvider
from datapilot.llm.provider import NarrativeResult, RoutingResult


@pytest.fixture
def provider():
    """Create a GeminiProvider with mocked client."""
    with patch("datapilot.llm.gemini.Config") as mock_config:
        mock_config.GEMINI_API_KEY = "test-key-123"
        mock_config.GEMINI_MODEL = "gemini-2.0-flash"
        mock_client = MagicMock()
        _mock_genai_module.Client.return_value = mock_client
        p = GeminiProvider()
        p.client = mock_client
        yield p


@pytest.fixture
def data_context():
    return {
        "shape": "100 rows x 5 columns",
        "columns": [
            {"name": "age", "semantic_type": "numeric", "dtype": "int64", "n_unique": 50, "null_pct": 0.0},
            {"name": "salary", "semantic_type": "numeric", "dtype": "float64", "n_unique": 95, "null_pct": 0.02},
            {"name": "department", "semantic_type": "categorical", "dtype": "object", "n_unique": 5, "null_pct": 0.0},
        ],
        "n_rows": 100,
        "n_cols": 5,
    }


def _mock_response(text: str) -> MagicMock:
    """Create a mock Gemini response with .text attribute."""
    resp = MagicMock()
    resp.text = text
    return resp


class TestGeminiInit:
    def test_init_with_api_key(self):
        """Provider initializes when API key is set."""
        with patch("datapilot.llm.gemini.Config") as mock_config:
            mock_config.GEMINI_API_KEY = "test-key-123"
            mock_config.GEMINI_MODEL = "gemini-2.0-flash"
            mock_client = MagicMock()
            _mock_genai_module.Client.return_value = mock_client
            p = GeminiProvider()
            assert p.model == "gemini-2.0-flash"
            _mock_genai_module.Client.assert_called_with(api_key="test-key-123")

    def test_init_without_api_key_raises(self):
        """Provider raises ValueError when API key is missing."""
        with patch("datapilot.llm.gemini.Config") as mock_config:
            mock_config.GEMINI_API_KEY = None
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiProvider()


class TestRouteQuestion:
    def test_parses_valid_json(self, provider, data_context):
        """route_question parses valid JSON response into RoutingResult."""
        response_json = json.dumps({
            "skill": "describe_data",
            "params": {"columns": ["age"]},
            "confidence": 0.9,
            "reasoning": "User wants descriptive statistics",
        })
        provider.client.models.generate_content.return_value = _mock_response(response_json)

        result = provider.route_question("describe age", data_context, "skill catalog text")

        assert isinstance(result, RoutingResult)
        assert result.skill_name == "describe_data"
        assert result.confidence == 0.9
        assert result.parameters == {"columns": ["age"]}

    def test_api_error_returns_none(self, provider, data_context):
        """route_question returns None on API error."""
        provider.client.models.generate_content.side_effect = Exception("API timeout")

        result = provider.route_question("describe data", data_context, "catalog")

        assert result is None


class TestGenerateNarrative:
    def test_returns_narrative_result(self, provider):
        """generate_narrative returns NarrativeResult with text."""
        response_json = json.dumps({
            "text": "The dataset contains 100 rows and 5 columns. The mean age is 35.2 years with a standard deviation of 12.1.",
            "key_points": ["100 rows analyzed", "Mean age is 35.2"],
            "suggestions": ["Explore salary distribution"],
        })
        provider.client.models.generate_content.return_value = _mock_response(response_json)

        result = provider.generate_narrative(
            {"n_rows": 100, "n_cols": 5, "mean_age": 35.2},
            question="What does the data look like?",
            skill_name="profile_data",
        )

        assert isinstance(result, NarrativeResult)
        assert "100 rows" in result.text
        assert len(result.key_points) == 2

    def test_rejects_narrative_with_question_mark(self, provider):
        """generate_narrative rejects response containing '?' placeholder."""
        response_json = json.dumps({
            "text": "The dataset has ? rows and ? columns.",
            "key_points": [],
            "suggestions": [],
        })
        provider.client.models.generate_content.return_value = _mock_response(response_json)

        result = provider.generate_narrative({"n_rows": 100})

        assert result is None


class TestSuggestChart:
    def test_parses_json_array(self, provider, data_context):
        """suggest_chart parses JSON array of suggestions."""
        response_json = json.dumps([
            {
                "chart_type": "histogram",
                "x": "age",
                "y": None,
                "hue": None,
                "title": "Age Distribution",
                "reason": "Shows the spread of ages",
            },
            {
                "chart_type": "scatter",
                "x": "age",
                "y": "salary",
                "hue": "department",
                "title": "Age vs Salary",
                "reason": "Reveals salary trends by age",
            },
        ])
        provider.client.models.generate_content.return_value = _mock_response(response_json)

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        assert len(result["suggestions"]) == 2
        assert result["suggestions"][0]["chart_type"] == "histogram"
        assert result["suggestions"][1]["hue"] == "department"

    def test_api_error_returns_deterministic_fallback(self, provider, data_context):
        """suggest_chart returns deterministic fallback on API error."""
        provider.client.models.generate_content.side_effect = Exception("API error")

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        assert len(result["suggestions"]) >= 1
        assert result["suggestions"][0]["chart_type"] == "histogram"
        assert result["suggestions"][0]["x"] == "age"

"""Tests for chart suggestion flow.

Verifies:
  1. Groq suggest_chart returns {"suggestions": [...]} with valid items
  2. Null-string cleanup works correctly
  3. Invalid chart types fall back to histogram
  4. Fallback returns sensible defaults with actual column names
  5. Each suggestion includes a "reason" field
"""

import pytest
from unittest.mock import MagicMock, patch

from engine.datapilot.llm.groq import GroqProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_context():
    """Sample data context for chart suggestion tests."""
    return {
        "shape": "100 rows x 5 columns",
        "columns": [
            {"name": "age", "dtype": "int64", "semantic_type": "numeric", "n_unique": 40, "null_pct": 0.0},
            {"name": "income", "dtype": "float64", "semantic_type": "numeric", "n_unique": 100, "null_pct": 0.0},
            {"name": "score", "dtype": "float64", "semantic_type": "numeric", "n_unique": 100, "null_pct": 0.0},
            {"name": "category", "dtype": "object", "semantic_type": "categorical", "n_unique": 3, "null_pct": 0.0},
            {"name": "purchased", "dtype": "bool", "semantic_type": "boolean", "n_unique": 2, "null_pct": 0.0},
        ],
        "n_rows": 100,
        "n_cols": 5,
    }


@pytest.fixture
def mock_groq_response():
    """Create a mock Groq API response."""
    def _make(content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp
    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSuggestChartParsing:
    """Test that suggest_chart correctly parses Groq responses."""

    def test_valid_response(self, data_context, mock_groq_response):
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '[{"chart_type": "scatter", "x": "age", "y": "income", "hue": "category", '
            '"title": "Age vs Income", "reason": "Shows relationship between age and income"}]'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) >= 1
        first = result["suggestions"][0]
        assert first["chart_type"] == "scatter"
        assert first["x"] == "age"
        assert first["y"] == "income"
        assert first["hue"] == "category"
        assert first["title"] == "Age vs Income"

    def test_null_string_cleanup(self, data_context, mock_groq_response):
        """Groq sometimes returns "null" string instead of JSON null."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '[{"chart_type": "histogram", "x": "age", "y": "null", "hue": "null", '
            '"title": "Age Distribution", "reason": "See age spread"}]'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        first = result["suggestions"][0]
        assert first["chart_type"] == "histogram"
        assert first["x"] == "age"
        assert first["y"] is None
        assert first["hue"] is None

    def test_invalid_chart_type_fallback(self, data_context, mock_groq_response):
        """Unknown chart types should fall back to histogram."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '[{"chart_type": "sunburst", "x": "category", "y": null, "hue": null, '
            '"title": "Category Chart", "reason": "Explore categories"}]'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        assert result["suggestions"][0]["chart_type"] == "histogram"

    def test_code_block_wrapper(self, data_context, mock_groq_response):
        """Groq sometimes wraps response in ```json blocks."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '```json\n[{"chart_type": "bar", "x": "category", "y": "income", "hue": null, '
            '"title": "Income by Category", "reason": "Compare income across categories"}]\n```'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        first = result["suggestions"][0]
        assert first["chart_type"] == "bar"
        assert first["x"] == "category"
        assert first["y"] == "income"

    def test_multiple_suggestions(self, data_context, mock_groq_response):
        """Should handle multiple ranked suggestions."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '[{"chart_type": "scatter", "x": "age", "y": "income", "hue": null, '
            '"title": "Age vs Income", "reason": "Explore correlation"}, '
            '{"chart_type": "histogram", "x": "score", "y": null, "hue": null, '
            '"title": "Score Distribution", "reason": "See score spread"}, '
            '{"chart_type": "box", "x": "category", "y": "income", "hue": null, '
            '"title": "Income by Category", "reason": "Compare groups"}]'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        assert len(result["suggestions"]) == 3
        assert result["suggestions"][0]["chart_type"] == "scatter"
        assert result["suggestions"][1]["chart_type"] == "histogram"
        assert result["suggestions"][2]["chart_type"] == "box"

    def test_reason_field_present(self, data_context, mock_groq_response):
        """Each suggestion must include a reason field."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '[{"chart_type": "scatter", "x": "age", "y": "income", "hue": null, '
            '"title": "Age vs Income", "reason": "Shows relationship between age and income"}, '
            '{"chart_type": "bar", "x": "category", "y": "score", "hue": null, '
            '"title": "Score by Category"}]'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        for s in result["suggestions"]:
            assert "reason" in s
        # First has explicit reason
        assert result["suggestions"][0]["reason"] == "Shows relationship between age and income"
        # Second had no reason — should default to empty string
        assert result["suggestions"][1]["reason"] == ""


class TestSuggestChartFallback:
    """Test the fallback when Groq API fails."""

    def test_api_failure_fallback(self, data_context):
        """On API failure, should return suggestions list with fallback items."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) >= 1
        first = result["suggestions"][0]
        assert first["chart_type"] == "histogram"
        assert first["x"] == "age"  # First numeric column
        assert first["y"] is None
        assert "reason" in first

    def test_fallback_with_two_numeric_columns(self, data_context):
        """Fallback should include scatter plot when two numeric columns exist."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert "suggestions" in result
        assert len(result["suggestions"]) == 2
        assert result["suggestions"][1]["chart_type"] == "scatter"
        assert result["suggestions"][1]["x"] == "age"
        assert result["suggestions"][1]["y"] == "income"

    def test_fallback_empty_columns(self):
        """Fallback with no columns returns basic defaults."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        provider._client = mock_client

        result = provider.suggest_chart({"columns": [], "shape": "0 rows x 0 columns"})

        assert "suggestions" in result
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) >= 1
        assert result["suggestions"][0]["chart_type"] == "histogram"
        assert result["suggestions"][0]["x"] is None

"""Tests for chart suggestion flow.

Verifies:
  1. Groq suggest_chart returns valid chart_type, x, y, hue, title
  2. Null-string cleanup works correctly
  3. Invalid chart types fall back to histogram
  4. Fallback returns sensible defaults with first numeric column
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
            '{"chart_type": "scatter", "x": "age", "y": "income", "hue": "category", "title": "Age vs Income"}'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert result["chart_type"] == "scatter"
        assert result["x"] == "age"
        assert result["y"] == "income"
        assert result["hue"] == "category"
        assert result["title"] == "Age vs Income"

    def test_null_string_cleanup(self, data_context, mock_groq_response):
        """Groq sometimes returns "null" string instead of JSON null."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '{"chart_type": "histogram", "x": "age", "y": "null", "hue": "null", "title": "Age Distribution"}'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert result["chart_type"] == "histogram"
        assert result["x"] == "age"
        assert result["y"] is None
        assert result["hue"] is None

    def test_invalid_chart_type_fallback(self, data_context, mock_groq_response):
        """Unknown chart types should fall back to histogram."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '{"chart_type": "sunburst", "x": "category", "y": null, "hue": null, "title": "Category Chart"}'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert result["chart_type"] == "histogram"

    def test_code_block_wrapper(self, data_context, mock_groq_response):
        """Groq sometimes wraps response in ```json blocks."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_groq_response(
            '```json\n{"chart_type": "bar", "x": "category", "y": "income", "hue": null, "title": "Income by Category"}\n```'
        )
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert result["chart_type"] == "bar"
        assert result["x"] == "category"
        assert result["y"] == "income"


class TestSuggestChartFallback:
    """Test the fallback when Groq API fails."""

    def test_api_failure_fallback(self, data_context):
        """On API failure, should return histogram with first numeric column."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        provider._client = mock_client

        result = provider.suggest_chart(data_context)

        assert result["chart_type"] == "histogram"
        assert result["x"] == "age"  # First numeric column
        assert result["y"] is None
        assert "age" in result["title"].lower() or "distribution" in result["title"].lower()

    def test_fallback_empty_columns(self):
        """Fallback with no columns returns basic defaults."""
        provider = GroqProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        provider._client = mock_client

        result = provider.suggest_chart({"columns": [], "shape": "0 rows x 0 columns"})

        assert result["chart_type"] == "histogram"
        assert result["x"] is None

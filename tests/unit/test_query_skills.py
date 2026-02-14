"""
Hybrid Query Skills — RED phase tests.

Tests cover:
  1. _extract_params_via_llm — LLM-based parameter extraction
  2. query_data — filter/select rows via natural language
  3. pivot_table — grouped aggregation
  4. value_counts — frequency distribution
  5. top_n — ranking records
  6. cross_tab — cross-tabulation of two categoricals
  7. smart_query — LLM-generated pandas code in sandbox
  8. _validate_code — sandbox safety checks
  9. Routing integration — keyword routes for new skills
 10. Executor integration — question passing and code snippets
"""

import json
import pytest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from datapilot.analysis.query import (
    _extract_params_via_llm,
    query_data,
    pivot_table,
    value_counts,
    top_n,
    cross_tab,
    smart_query,
    _validate_code,
    _execute_safe,
)

from datapilot.core.router import _try_keyword_route
from datapilot.core.executor import Executor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_df():
    """20-row DataFrame with diverse column types."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "name": [f"Customer_{i}" for i in range(n)],
        "age": np.random.randint(18, 70, n),
        "city": np.random.choice(["New York", "Chicago", "Houston", "Phoenix"], n),
        "price": np.round(np.random.uniform(10.0, 500.0, n), 2),
        "category": np.random.choice(["A", "B", "C"], n),
        "churn": np.random.choice(["Yes", "No"], n),
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider with generate_plan() for param extraction."""
    provider = MagicMock()
    # Default: return valid JSON for query_data params
    provider.generate_plan = MagicMock(return_value=json.dumps({
        "filter_expression": "churn == 'Yes'",
        "columns": None,
    }))
    return provider


# ============================================================================
# Group 1: _extract_params_via_llm
# ============================================================================

class TestExtractParamsViaLLM:
    """Test LLM-based parameter extraction for query skills."""

    def test_extract_params_success(self, mock_llm_provider):
        """Valid LLM JSON response is parsed into a dict."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "age",
            "top_n": 10,
        })
        result = _extract_params_via_llm(
            "top 10 by age", mock_llm_provider, "value_counts",
            {"column": "str", "top_n": "int"}, ["age", "city"],
        )
        assert isinstance(result, dict)
        assert "column" in result

    def test_extract_params_llm_failure(self):
        """LLM exception returns None instead of crashing."""
        provider = MagicMock()
        provider.generate_plan.side_effect = Exception("API error")
        result = _extract_params_via_llm(
            "test", provider, "value_counts", {}, ["age"],
        )
        assert result is None

    def test_extract_params_invalid_json(self):
        """Garbage LLM response returns None."""
        provider = MagicMock()
        provider.generate_plan.return_value = "not json at all {{"
        result = _extract_params_via_llm(
            "test", provider, "value_counts", {}, ["age"],
        )
        assert result is None


# ============================================================================
# Group 2: query_data
# ============================================================================

class TestQueryData:
    """Test natural-language row filtering and column selection."""

    def test_filter_rows(self, sample_df, mock_llm_provider):
        """Rows matching the LLM filter expression are returned."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": "churn == 'Yes'",
            "columns": None,
        })
        result = query_data(sample_df, "Show rows where churn = Yes", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert isinstance(result["data"], list)
        assert result["total_rows"] > 0
        for row in result["data"]:
            assert row["churn"] == "Yes"

    def test_select_columns(self, sample_df, mock_llm_provider):
        """Only requested columns appear in the output records."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": None,
            "columns": ["name", "age"],
        })
        result = query_data(sample_df, "Show name and age", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        for row in result["data"]:
            assert set(row.keys()) == {"name", "age"}

    def test_empty_result(self, sample_df, mock_llm_provider):
        """Impossible filter returns success with zero rows."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": "age > 9999",
            "columns": None,
        })
        result = query_data(sample_df, "Show rows where age > 9999", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert result["total_rows"] == 0

    def test_llm_failure_uses_heuristic(self, sample_df):
        """With no LLM provider, function still returns without crashing."""
        result = query_data(sample_df, "Show rows where churn = Yes", llm_provider=None)
        assert result["status"] in ("success", "error")

    def test_invalid_expression_returns_error(self, sample_df, mock_llm_provider):
        """Invalid filter expression from LLM results in error status."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": ")))invalid(((",
            "columns": None,
        })
        result = query_data(sample_df, "Bad filter", llm_provider=mock_llm_provider)
        assert result["status"] == "error"


# ============================================================================
# Group 3: pivot_table
# ============================================================================

class TestPivotTable:
    """Test grouped aggregation via pivot tables."""

    def test_basic_pivot(self, sample_df, mock_llm_provider):
        """Simple mean aggregation by category returns success."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "values": "price",
            "index": "category",
            "aggfunc": "mean",
        })
        result = pivot_table(sample_df, "Average price by category", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert isinstance(result["data"], list)

    def test_multi_aggfunc(self, sample_df, mock_llm_provider):
        """Multiple aggregation functions return success."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "values": "price",
            "index": "category",
            "aggfunc": ["sum", "count"],
        })
        result = pivot_table(sample_df, "Sum and count of price by category", llm_provider=mock_llm_provider)
        assert result["status"] == "success"

    def test_missing_column_error(self, sample_df, mock_llm_provider):
        """Non-existent column in pivot returns error status."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "values": "nonexistent_col",
            "index": "category",
            "aggfunc": "mean",
        })
        result = pivot_table(sample_df, "Average nonexistent by category", llm_provider=mock_llm_provider)
        assert result["status"] == "error"

    def test_llm_failure_fallback(self, sample_df):
        """With no LLM provider, function returns without crashing."""
        result = pivot_table(sample_df, "Average price by category", llm_provider=None)
        assert result["status"] in ("success", "error")


# ============================================================================
# Group 4: value_counts
# ============================================================================

class TestValueCounts:
    """Test frequency distribution computation."""

    def test_basic_counts(self, sample_df, mock_llm_provider):
        """Counting a categorical column returns a dict of category->count."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "category",
        })
        result = value_counts(sample_df, "How many per category?", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert isinstance(result["data"], dict)
        # Keys should be the category values
        for key in result["data"]:
            assert key in ("A", "B", "C")

    def test_top_n_limit(self, sample_df, mock_llm_provider):
        """Top-N limit restricts the number of returned entries."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "city",
            "top_n": 2,
        })
        result = value_counts(sample_df, "Top 2 cities", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert len(result["data"]) <= 2

    def test_nonexistent_column(self, sample_df, mock_llm_provider):
        """Non-existent column returns error status."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "nonexistent_col",
        })
        result = value_counts(sample_df, "Count nonexistent", llm_provider=mock_llm_provider)
        assert result["status"] == "error"


# ============================================================================
# Group 5: top_n
# ============================================================================

class TestTopN:
    """Test ranking and top/bottom N retrieval."""

    def test_top_records(self, sample_df, mock_llm_provider):
        """Top 5 by price returns 5 records sorted descending."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "price",
            "n": 5,
            "ascending": False,
        })
        result = top_n(sample_df, "Top 5 by price", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert len(result["data"]) == 5
        # First record should have the highest price
        assert result["data"][0]["price"] >= result["data"][1]["price"]

    def test_bottom_records(self, sample_df, mock_llm_provider):
        """Bottom 3 by price returns 3 records sorted ascending."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "price",
            "n": 3,
            "ascending": True,
        })
        result = top_n(sample_df, "Bottom 3 by price", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert len(result["data"]) == 3
        # First record should have the lowest price
        assert result["data"][0]["price"] <= result["data"][1]["price"]

    def test_default_n(self, sample_df, mock_llm_provider):
        """When n is not specified, defaults to 10 (or len(df) if smaller)."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "price",
        })
        result = top_n(sample_df, "Top by price", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        expected_len = min(10, len(sample_df))
        assert len(result["data"]) == expected_len

    def test_invalid_column(self, sample_df, mock_llm_provider):
        """Non-existent column returns error status."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "nonexistent_col",
            "n": 5,
        })
        result = top_n(sample_df, "Top 5 by nonexistent", llm_provider=mock_llm_provider)
        assert result["status"] == "error"


# ============================================================================
# Group 6: cross_tab
# ============================================================================

class TestCrossTab:
    """Test cross-tabulation of two categorical columns."""

    def test_basic_crosstab(self, sample_df, mock_llm_provider):
        """Basic cross-tab of two categoricals returns a list of dicts."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "row": "category",
            "col": "churn",
        })
        result = cross_tab(sample_df, "Crosstab of category and churn", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert isinstance(result["data"], list)

    def test_with_values(self, sample_df, mock_llm_provider):
        """Cross-tab with values and aggfunc returns success."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "row": "category",
            "col": "churn",
            "values": "price",
            "aggfunc": "mean",
        })
        result = cross_tab(sample_df, "Average price by category and churn", llm_provider=mock_llm_provider)
        assert result["status"] == "success"

    def test_missing_column(self, sample_df, mock_llm_provider):
        """Non-existent column returns error status."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "row": "nonexistent_col",
            "col": "churn",
        })
        result = cross_tab(sample_df, "Crosstab of nonexistent and churn", llm_provider=mock_llm_provider)
        assert result["status"] == "error"

    def test_llm_failure_fallback(self, sample_df):
        """With no LLM provider, function returns without crashing."""
        result = cross_tab(sample_df, "Crosstab of category and churn", llm_provider=None)
        assert result["status"] in ("success", "error")


# ============================================================================
# Group 7: smart_query
# ============================================================================

class TestSmartQuery:
    """Test LLM-generated pandas code execution in sandbox."""

    def test_simple_code_executes(self, sample_df, mock_llm_provider):
        """Simple df.head() code returns 5 records."""
        mock_llm_provider.generate_plan.return_value = "result = df.head(5)"
        result = smart_query(sample_df, "Show first 5 rows", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert isinstance(result["data"], list)
        assert len(result["data"]) == 5

    def test_forbidden_import_rejected(self, sample_df, mock_llm_provider):
        """Code with forbidden imports is rejected."""
        mock_llm_provider.generate_plan.return_value = "import os\nresult = df"
        result = smart_query(sample_df, "Import os", llm_provider=mock_llm_provider)
        assert result["status"] == "error"
        msg = result.get("message", result.get("error", "")).lower()
        assert "import" in msg or "forbidden" in msg

    def test_forbidden_function_rejected(self, sample_df, mock_llm_provider):
        """Code calling eval() is rejected."""
        mock_llm_provider.generate_plan.return_value = "result = eval('1+1')"
        result = smart_query(sample_df, "Eval something", llm_provider=mock_llm_provider)
        assert result["status"] == "error"

    def test_max_lines_exceeded(self, sample_df, mock_llm_provider):
        """Code exceeding max line count is rejected."""
        lines = ["x = 1"] * 15
        lines.append("result = df")
        mock_llm_provider.generate_plan.return_value = "\n".join(lines)
        result = smart_query(sample_df, "Long code", llm_provider=mock_llm_provider)
        assert result["status"] == "error"
        msg = result.get("message", result.get("error", "")).lower()
        assert "line" in msg

    def test_output_validation(self, sample_df, mock_llm_provider):
        """Code returning non-serializable type is rejected."""
        mock_llm_provider.generate_plan.return_value = "result = set([1, 2, 3])"
        result = smart_query(sample_df, "Return a set", llm_provider=mock_llm_provider)
        assert result["status"] == "error"

    @pytest.mark.slow
    def test_timeout_handling(self, sample_df, mock_llm_provider):
        """Infinite loop is killed by timeout."""
        mock_llm_provider.generate_plan.return_value = "result = None\nwhile True: pass"
        result = smart_query(sample_df, "Infinite loop", llm_provider=mock_llm_provider)
        assert result["status"] == "error"
        msg = result.get("message", result.get("error", "")).lower()
        assert "timeout" in msg

    def test_no_llm_provider_error(self, sample_df):
        """smart_query without an LLM provider returns error."""
        result = smart_query(sample_df, "test", llm_provider=None)
        assert result["status"] == "error"


# ============================================================================
# Group 8: _validate_code
# ============================================================================

class TestValidateCode:
    """Test sandbox code validation rules."""

    def test_safe_code_passes(self):
        """Safe pandas code passes validation."""
        is_safe, reason = _validate_code("result = df.head(5)")
        assert is_safe is True
        assert reason == "OK"

    def test_import_blocked(self):
        """Import statements are blocked."""
        is_safe, reason = _validate_code("import os")
        assert is_safe is False
        assert "import" in reason.lower()

    def test_dunder_access_blocked(self):
        """Dunder attribute access is blocked."""
        is_safe, reason = _validate_code("x = df.__class__")
        assert is_safe is False
        assert "dunder" in reason.lower() or "__" in reason

    def test_smart_query_blocks_pd_read_csv(self):
        """pd.read_csv() is blocked by AST validator."""
        is_safe, reason = _validate_code('result = pd.read_csv("http://evil.com/data.csv")')
        assert is_safe is False
        assert "read_csv" in reason

    def test_smart_query_blocks_df_to_csv(self):
        """df.to_csv() is blocked by AST validator."""
        is_safe, reason = _validate_code('df.to_csv("/tmp/stolen.csv")')
        assert is_safe is False
        assert "to_csv" in reason

    def test_smart_query_blocks_pd_eval(self):
        """pd.eval() is blocked by AST validator."""
        is_safe, reason = _validate_code('result = pd.eval("1+1")')
        assert is_safe is False
        assert "eval" in reason

    def test_smart_query_blocks_read_html(self):
        """pd.read_html() is blocked by AST validator."""
        is_safe, reason = _validate_code('result = pd.read_html("http://evil.com")')
        assert is_safe is False
        assert "read_html" in reason


class TestSandboxProxyExecution:
    """Test that sandbox proxy objects block dangerous operations at runtime."""

    def test_smart_query_proxy_blocks_pd_read_csv(self, sample_df):
        """pd.read_csv() blocked even if AST check is bypassed — proxy raises AttributeError."""
        result = _execute_safe('result = pd.read_csv("test.csv")', sample_df)
        assert result["status"] == "error"
        assert "read_csv" in result["message"] or "not allowed" in result["message"]

    def test_smart_query_proxy_allows_safe_pd(self, sample_df):
        """Safe pd methods like pd.to_numeric work through the proxy."""
        result = _execute_safe('result = pd.to_numeric(df["age"])', sample_df)
        assert result["status"] == "success"

    def test_smart_query_proxy_blocks_np_eval(self, sample_df):
        """np methods not in allowlist are blocked by the proxy."""
        result = _execute_safe('result = np.loadtxt("test.txt")', sample_df)
        assert result["status"] == "error"
        assert "not allowed" in result["message"]


class TestFilterExpressionValidation:
    """Test that df.query() filter expressions reject function calls."""

    def test_smart_query_filter_no_function_calls(self, sample_df, mock_llm_provider):
        """df.query() rejects expressions with function calls."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": "__import__('os').system('rm -rf /')",
            "columns": None,
        })
        result = query_data(sample_df, "hack me", llm_provider=mock_llm_provider)
        assert result["status"] == "error"
        assert "function call" in result["message"].lower() or "not allowed" in result["message"].lower()


# ============================================================================
# Group 9: Routing integration — keyword routes for new skills
# ============================================================================

class TestRouting:
    """Test that keyword routing recognizes the new query skill patterns."""

    def test_query_data_keyword(self):
        """'Show me rows where ...' routes to query_data."""
        result = _try_keyword_route("Show me rows where price > 100")
        assert result is not None
        assert result.skill_name == "query_data"

    def test_pivot_keyword(self):
        """'Average ... by ...' routes to pivot_table."""
        result = _try_keyword_route("Average monthly charges by contract type")
        assert result is not None
        assert result.skill_name == "pivot_table"

    def test_value_counts_keyword(self):
        """'How many ... per ...' routes to value_counts."""
        result = _try_keyword_route("How many customers per state")
        assert result is not None
        assert result.skill_name == "value_counts"

    def test_top_n_keyword(self):
        """'Top 10 ... by ...' routes to top_n."""
        result = _try_keyword_route("Top 10 customers by monthly charges")
        assert result is not None
        assert result.skill_name == "top_n"

    def test_cross_tab_keyword(self):
        """'Crosstab of ...' routes to cross_tab."""
        result = _try_keyword_route("Crosstab of churn and contract type")
        assert result is not None
        assert result.skill_name == "cross_tab"


# ============================================================================
# Group 10: Executor integration
# ============================================================================

class TestExecutorIntegration:
    """Test that Executor correctly handles new-style skills with question param."""

    def test_question_passed_to_new_skill(self, sample_df):
        """New-style skills (accepting question param) receive the question."""
        executor = Executor()
        mock_func = MagicMock(return_value={"status": "success"})
        # Give mock_func a signature with (df, question, llm_provider)
        import inspect
        params = [
            inspect.Parameter("df", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("question", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("llm_provider", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
        ]
        mock_func.__signature__ = inspect.Signature(params)

        with patch("datapilot.core.executor.get_skill_function", return_value=mock_func):
            result = executor.execute("query_data", sample_df, parameters={"question": "test query"})
        assert result.status == "success"
        call_kwargs = mock_func.call_args[1]
        assert call_kwargs.get("question") == "test query"

    def test_question_not_passed_to_old_skill(self, sample_df):
        """Old-style skills (no question param) do not receive question kwarg."""
        executor = Executor()
        mock_func = MagicMock(return_value={"status": "success"})
        import inspect
        params = [
            inspect.Parameter("file_path", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("columns", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
        ]
        mock_func.__signature__ = inspect.Signature(params)

        with patch("datapilot.core.executor.get_skill_function", return_value=mock_func):
            result = executor.execute("old_skill", sample_df, parameters={"question": "test"})
        call_kwargs = mock_func.call_args[1]
        assert "question" not in call_kwargs

    def test_code_snippet_excludes_provider(self, sample_df):
        """Code snippet does not expose llm_provider internals."""
        executor = Executor()
        snippet = executor._build_code_snippet(
            "test_skill",
            {"df": sample_df, "question": "test", "llm_provider": "<provider>"},
        )
        assert "llm_provider" not in snippet

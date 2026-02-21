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
    _resolve_column,
    query_data,
    pivot_table,
    value_counts,
    top_n,
    cross_tab,
    smart_query,
    _validate_code,
    _execute_safe,
    _summarize_dataframe,
)

from datapilot.core.router import Router
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
    """Test that data query questions route via semantic embedding or smart_query.

    With semantic-first routing, data query skills (query_data, pivot_table,
    value_counts, top_n, cross_tab) are matched via embedding similarity.
    When no embedding match is found, smart_query handles data questions.
    """

    @pytest.fixture
    def data_context(self):
        return {
            "shape": "20 rows x 5 columns",
            "columns": [
                {"name": "price", "dtype": "float64", "semantic_type": "numeric", "n_unique": 20, "null_pct": 0},
                {"name": "monthly charges", "dtype": "float64", "semantic_type": "numeric", "n_unique": 18, "null_pct": 0},
                {"name": "contract type", "dtype": "object", "semantic_type": "categorical", "n_unique": 3, "null_pct": 0},
                {"name": "state", "dtype": "object", "semantic_type": "categorical", "n_unique": 10, "null_pct": 0},
                {"name": "churn", "dtype": "object", "semantic_type": "categorical", "n_unique": 2, "null_pct": 0},
            ],
            "n_rows": 20,
            "n_cols": 5,
        }

    def test_query_data_routes_semantically(self, data_context):
        """'Show me rows where ...' routes via semantic embedding or smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("Show me rows where price > 100", data_context)
        assert result.skill_name in ("query_data", "smart_query")

    def test_pivot_routes_semantically(self, data_context):
        """'Average ... by ...' routes via semantic embedding or smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("Average monthly charges by contract type", data_context)
        assert result.skill_name in ("pivot_table", "smart_query")

    def test_value_counts_routes_semantically(self, data_context):
        """'How many ... per ...' routes via semantic embedding or smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("How many customers per state", data_context)
        assert result.skill_name in ("value_counts", "smart_query")

    def test_top_n_routes_semantically(self, data_context):
        """'Top 10 ... by ...' routes via semantic embedding or smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("Top 10 customers by monthly charges", data_context)
        assert result.skill_name in ("top_n", "smart_query")

    def test_cross_tab_routes_semantically(self, data_context):
        """'Crosstab of ...' routes via semantic embedding or smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("Crosstab of churn and contract type", data_context)
        assert result.skill_name in ("cross_tab", "smart_query")


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


# ============================================================================
# Group 11: Router.route() end-to-end tests
# ============================================================================

class TestRouterRouteEndToEnd:
    """Test Router.route() with provider and without."""

    @pytest.fixture
    def data_context(self):
        return {
            "shape": "20 rows x 5 columns",
            "columns": [
                {"name": "age", "dtype": "int64", "semantic_type": "numeric", "n_unique": 15, "null_pct": 0},
                {"name": "city", "dtype": "object", "semantic_type": "categorical", "n_unique": 4, "null_pct": 0},
                {"name": "price", "dtype": "float64", "semantic_type": "numeric", "n_unique": 20, "null_pct": 0},
                {"name": "churn", "dtype": "bool", "semantic_type": "boolean", "n_unique": 2, "null_pct": 0},
            ],
            "n_rows": 20,
            "n_cols": 5,
        }

    def test_route_with_provider_returns_smart_query_or_semantic(self, data_context):
        """With an LLM provider, data questions route via embedding or smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("show me rows where price > 100", data_context)
        assert result.skill_name in ("query_data", "smart_query")

    def test_route_with_provider_matches_analytical_skills(self, data_context):
        """With an LLM provider, analytical questions match via semantic embedding."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("classify customers by churn", data_context)
        # Embedding model may match classify or find_clusters (semantically close)
        assert result.skill_name in ("classify", "find_clusters")
        assert result.route_method == "semantic_embedding"

    def test_route_with_provider_chart(self, data_context):
        """Chart questions route to create_chart via semantic embedding."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("draw a scatter plot of age vs price", data_context)
        assert result.skill_name == "create_chart"
        assert result.route_method == "semantic_embedding"

    def test_route_no_provider_semantic_match(self, data_context):
        """Without an LLM provider, semantic embedding still routes data questions."""
        router = Router(provider=None)
        result = router.route("show rows where price > 100", data_context)
        # Semantic embedding matches query_data, or falls to profile_data
        assert result.skill_name in ("query_data", "profile_data")

    def test_route_no_provider_pivot(self, data_context):
        """Without an LLM provider, 'average by' routes via semantic embedding."""
        router = Router(provider=None)
        result = router.route("average price by city", data_context)
        assert result.skill_name in ("pivot_table", "profile_data")

    def test_route_no_provider_unmatched_falls_to_profile(self, data_context):
        """Without an LLM provider, truly unmatched questions fall to profile_data."""
        router = Router(provider=None)
        router._semantic_attempted = True
        router._semantic_matcher = None
        result = router.route("xyzzy nonsense question", data_context)
        assert result.skill_name == "profile_data"
        assert result.route_method == "fallback"


# ============================================================================
# Group 12: Data query keyword fallback (no-LLM mode)
# ============================================================================

class TestSemanticDataQueryRouting:
    """Test semantic embedding routing for data query skills.

    With semantic-first routing, data query questions are matched via
    embedding similarity to SKILL_DESCRIPTIONS entries.
    """

    @pytest.fixture
    def data_context(self):
        return {
            "shape": "20 rows x 5 columns",
            "columns": [
                {"name": "price", "dtype": "float64", "semantic_type": "numeric", "n_unique": 20, "null_pct": 0},
                {"name": "monthly charges", "dtype": "float64", "semantic_type": "numeric", "n_unique": 18, "null_pct": 0},
                {"name": "churn", "dtype": "object", "semantic_type": "categorical", "n_unique": 2, "null_pct": 0},
            ],
            "n_rows": 20,
            "n_cols": 5,
        }

    def test_filter_routes_to_query_or_smart(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("show rows where price > 100", data_context)
        assert r.skill_name in ("query_data", "smart_query")

    def test_pivot_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("average monthly charges by contract type", data_context)
        assert r.skill_name in ("pivot_table", "smart_query")

    def test_value_counts_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("how many customers per state", data_context)
        assert r.skill_name in ("value_counts", "smart_query")

    def test_top_n_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("top 10 customers by monthly charges", data_context)
        assert r.skill_name in ("top_n", "smart_query")

    def test_cross_tab_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("crosstab of churn and contract type", data_context)
        assert r.skill_name in ("cross_tab", "smart_query")

    def test_classify_routes_correctly(self, data_context):
        """Analytical questions route to correct skill via embedding."""
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("classify customers by churn", data_context)
        # Embedding model may match classify or find_clusters (semantically close)
        assert r.skill_name in ("classify", "find_clusters")

    def test_generic_falls_to_smart_query(self, data_context):
        """Generic questions fall through to smart_query when LLM is available."""
        provider = MagicMock()
        router = Router(provider=provider)
        router._semantic_attempted = True
        router._semantic_matcher = None
        r = router.route("what is this dataset about", data_context)
        assert r.skill_name == "smart_query"


# ============================================================================
# Group 13: Sandbox builtin safety — type/print removed
# ============================================================================

class TestSandboxBuiltinSafety:
    """Test that removed builtins (type, print) no longer work in sandbox."""

    def test_type_not_in_sandbox(self, sample_df):
        """type() should fail — removed from _SAFE_BUILTINS."""
        result = _execute_safe("result = type(df).__name__", sample_df)
        assert result["status"] == "error"

    def test_print_is_noop_in_sandbox(self, sample_df):
        """print() is a no-op — does not crash but produces no output."""
        result = _execute_safe("print('hello')\nresult = 1", sample_df)
        assert result["status"] == "success"
        assert result["data"] == 1

    def test_isinstance_still_works(self, sample_df):
        """isinstance() should still work — kept in _SAFE_BUILTINS."""
        result = _execute_safe("result = isinstance(df.shape[0], int)", sample_df)
        assert result["status"] == "success"
        assert result["data"] is True

    def test_len_still_works(self, sample_df):
        """len() should still work in sandbox."""
        result = _execute_safe("result = len(df)", sample_df)
        assert result["status"] == "success"
        assert result["data"] == 20


# ============================================================================
# _summarize_dataframe tests
# ============================================================================

class TestSummarizeDataframe:
    """Test _summarize_dataframe produces correct aggregate stats."""

    def test_total_rows_matches(self, sample_df):
        summary = _summarize_dataframe(sample_df)
        assert summary["total_rows"] == len(sample_df)
        assert summary["total_columns"] == len(sample_df.columns)

    def test_numeric_columns_have_stats(self, sample_df):
        summary = _summarize_dataframe(sample_df)
        age_stats = summary["columns"]["age"]
        assert age_stats["type"] == "numeric"
        for key in ("mean", "std", "min", "max", "median", "count"):
            assert key in age_stats

    def test_categorical_columns_have_value_counts(self, sample_df):
        summary = _summarize_dataframe(sample_df)
        city_stats = summary["columns"]["city"]
        assert city_stats["type"] == "categorical"
        assert "value_counts" in city_stats
        assert "n_unique" in city_stats

    def test_bool_detected_before_numeric(self):
        df = pd.DataFrame({"flag": [True, False, True, True, False]})
        summary = _summarize_dataframe(df)
        assert summary["columns"]["flag"]["type"] == "boolean"
        assert summary["columns"]["flag"]["true_count"] == 3
        assert summary["columns"]["flag"]["false_count"] == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")})
        summary = _summarize_dataframe(df)
        assert summary["total_rows"] == 0
        assert summary["total_columns"] == 2

    def test_column_cap(self):
        df = pd.DataFrame({f"col_{i}": [1, 2, 3] for i in range(25)})
        summary = _summarize_dataframe(df, max_columns=10)
        assert len(summary["columns"]) <= 10
        assert "_columns_truncated" in summary

    def test_skips_id_columns(self):
        df = pd.DataFrame({
            "PassengerId": range(100),
            "Name": [f"Person_{i}" for i in range(100)],
            "Age": [25] * 100,
            "City": ["NYC"] * 100,
        })
        summary = _summarize_dataframe(df)
        assert "PassengerId" not in summary["columns"]
        assert "Name" not in summary["columns"]
        assert "Age" in summary["columns"]
        assert "_skipped_columns" in summary


# ============================================================================
# smart_query data_summary tests
# ============================================================================

class TestSmartQuerySummary:
    """Test that smart_query includes data_summary for accurate narration."""

    def test_includes_data_summary(self, sample_df):
        mock_llm = MagicMock()
        mock_llm.generate_plan.return_value = "result = df[df['age'] > 30]"
        result = smart_query(sample_df, "age above 30", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert "data_summary" in result
        assert result["data_summary"]["total_rows"] == result["total_rows"]

    def test_summary_has_column_stats(self, sample_df):
        mock_llm = MagicMock()
        mock_llm.generate_plan.return_value = "result = df[df['age'] > 30]"
        result = smart_query(sample_df, "age above 30", llm_provider=mock_llm)
        assert "columns" in result["data_summary"]
        assert "age" in result["data_summary"]["columns"]

    def test_no_raw_output_leaked(self, sample_df):
        mock_llm = MagicMock()
        mock_llm.generate_plan.return_value = "result = df.head(5)"
        result = smart_query(sample_df, "first 5", llm_provider=mock_llm)
        assert "_raw_output" not in result

    def test_scalar_result_no_summary(self, sample_df):
        mock_llm = MagicMock()
        mock_llm.generate_plan.return_value = "result = len(df)"
        result = smart_query(sample_df, "how many rows", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert "data_summary" not in result


# ============================================================================
# Group 14: Narrative verification (_verify_narrative)
# ============================================================================

from datapilot.core.analyst import _verify_narrative
from datapilot.llm.provider import NarrativeResult


class TestVerifyNarrative:
    """Test post-narrative verification of LLM-generated numbers."""

    def test_matching_numbers_accepted(self):
        """Narrative with numbers that appear in results passes verification."""
        narrative = NarrativeResult(
            text="The dataset has 100 rows and the mean age is 35.5.",
            key_points=[], suggestions=[],
        )
        result = {"total_rows": 100, "mean_age": 35.5}
        assert _verify_narrative(narrative, result) is True

    def test_hallucinated_numbers_rejected(self):
        """Narrative with made-up numbers is rejected."""
        narrative = NarrativeResult(
            text="There are 999 outliers with a score of 42.7 and 88 clusters.",
            key_points=[], suggestions=[],
        )
        result = {"total_rows": 10, "outlier_count": 2}
        assert _verify_narrative(narrative, result) is False

    def test_no_numbers_accepted(self):
        """Text-only narrative with no numbers passes (nothing to verify)."""
        narrative = NarrativeResult(
            text="The data shows interesting patterns across all categories.",
            key_points=[], suggestions=[],
        )
        result = {"some_key": "some_value"}
        assert _verify_narrative(narrative, result) is True

    def test_empty_narrative_rejected(self):
        """Empty narrative text is rejected."""
        narrative = NarrativeResult(text="", key_points=[], suggestions=[])
        result = {"total_rows": 10}
        assert _verify_narrative(narrative, result) is False

    def test_partial_match_above_threshold(self):
        """Narrative where >50% of numbers match passes."""
        narrative = NarrativeResult(
            text="Found 100 rows, 50 matched, and 25 were unique, with a random 999.",
            key_points=[], suggestions=[],
        )
        # 100, 50, 25 match; 999 doesn't → 75% match rate → pass
        result = {"total": 100, "matched": 50, "unique": 25}
        assert _verify_narrative(narrative, result) is True

    def test_nested_result_numbers_extracted(self):
        """Numbers in nested dicts/lists are found for matching."""
        narrative = NarrativeResult(
            text="The top correlation is 0.85 between age and price.",
            key_points=[], suggestions=[],
        )
        result = {
            "top_correlations": [
                {"col1": "age", "col2": "price", "correlation": 0.85}
            ]
        }
        assert _verify_narrative(narrative, result) is True

    def test_rounded_numbers_match(self):
        """Narrative rounds a float — verification still matches."""
        narrative = NarrativeResult(
            text="Average age is 35.",
            key_points=[], suggestions=[],
        )
        result = {"mean_age": 35.0042}
        assert _verify_narrative(narrative, result) is True


# ============================================================================
# Group 15: data_summary in query_data and top_n
# ============================================================================

class TestResolveColumn:
    """Test case-insensitive column name resolution."""

    def test_exact_match(self):
        assert _resolve_column("Fare", ["Fare", "Age"]) == "Fare"

    def test_case_insensitive(self):
        assert _resolve_column("fare", ["Fare", "Age"]) == "Fare"

    def test_partial_match(self):
        """'class' matches 'Pclass'."""
        assert _resolve_column("class", ["Pclass", "Fare"]) == "Pclass"

    def test_none_input(self):
        assert _resolve_column(None, ["Fare"]) is None

    def test_no_match(self):
        assert _resolve_column("nonexistent", ["Fare", "Age"]) is None

    def test_pivot_with_case_insensitive_columns(self, sample_df, mock_llm_provider):
        """pivot_table resolves 'price' -> 'price' and 'category' -> 'category'."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "values": "Price",  # Capital P — should resolve to 'price'
            "index": "Category",
            "aggfunc": "mean",
        })
        result = pivot_table(sample_df, "Average Price by Category", llm_provider=mock_llm_provider)
        assert result["status"] == "success"

    def test_pivot_null_values_falls_to_heuristic(self, sample_df, mock_llm_provider):
        """pivot_table with LLM returning null values falls back to heuristic."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "values": None,
            "index": "category",
            "aggfunc": "mean",
        })
        result = pivot_table(sample_df, "average price by category", llm_provider=mock_llm_provider)
        assert result["status"] == "success"


class TestQueryDataSummary:
    """Test that query_data includes data_summary for LLM narration."""

    def test_query_data_has_summary(self, sample_df, mock_llm_provider):
        """Filtered query_data result includes data_summary."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": "age > 30",
            "columns": None,
        })
        result = query_data(sample_df, "age above 30", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert result["total_rows"] > 0
        assert "data_summary" in result
        assert result["data_summary"]["total_rows"] == result["total_rows"]

    def test_query_data_empty_no_summary(self, sample_df, mock_llm_provider):
        """Empty filter result does not include data_summary."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": "age > 9999",
            "columns": None,
        })
        result = query_data(sample_df, "age above 9999", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert result["total_rows"] == 0
        assert "data_summary" not in result

    def test_query_data_summary_has_columns(self, sample_df, mock_llm_provider):
        """data_summary contains column-level statistics."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "filter_expression": "age > 30",
            "columns": None,
        })
        result = query_data(sample_df, "age above 30", llm_provider=mock_llm_provider)
        if result["total_rows"] > 0:
            assert "columns" in result["data_summary"]


class TestTopNSummary:
    """Test that top_n includes data_summary for LLM narration."""

    def test_top_n_has_summary(self, sample_df, mock_llm_provider):
        """Top N result includes data_summary."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "price",
            "n": 5,
            "ascending": False,
        })
        result = top_n(sample_df, "Top 5 by price", llm_provider=mock_llm_provider)
        assert result["status"] == "success"
        assert "data_summary" in result
        assert result["data_summary"]["total_rows"] == 5

    def test_top_n_summary_has_columns(self, sample_df, mock_llm_provider):
        """data_summary in top_n contains column stats."""
        mock_llm_provider.generate_plan.return_value = json.dumps({
            "column": "price",
            "n": 5,
            "ascending": False,
        })
        result = top_n(sample_df, "Top 5 by price", llm_provider=mock_llm_provider)
        assert "columns" in result["data_summary"]

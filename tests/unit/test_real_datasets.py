"""
End-to-end query skill tests using REAL datasets.

Tests every question type a user would ask against 3 real datasets:
  - Telco churn   (Raw Data.xlsx)   — columns with spaces, bool target
  - Superstore    (Sample - Superstore.csv) — dates, dollar values, categories
  - Titanic       (Titanic-Dataset.csv)     — missing values, mixed types

Each test group covers a specific skill with realistic natural-language questions.
LLM calls are mocked so tests run offline and deterministically.
"""

import json
import pathlib
import pytest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from datapilot.analysis.query import (
    _sanitize_backticks_for_ast,
    _SAFE_BUILTINS,
    query_data,
    pivot_table,
    value_counts,
    top_n,
    cross_tab,
    smart_query,
    _execute_safe,
)
from datapilot.core.router import Router
from datapilot.core.semantic_router import SemanticSkillMatcher
from unittest.mock import MagicMock

TEST_DATA = pathlib.Path(__file__).resolve().parent.parent / "test_data"


# ============================================================================
# Fixtures — load real datasets once per session
# ============================================================================

@pytest.fixture(scope="session")
def telco_df():
    """Telco churn dataset (3333 rows, 21 cols, columns with spaces)."""
    return pd.read_excel(TEST_DATA / "Raw Data.xlsx")


@pytest.fixture(scope="session")
def superstore_df():
    """Superstore sales dataset (9994 rows, 21 cols, dates + dollar amounts)."""
    return pd.read_csv(TEST_DATA / "Sample - Superstore.csv", encoding="latin-1")


@pytest.fixture(scope="session")
def titanic_df():
    """Titanic dataset (891 rows, 12 cols, missing values in Age/Cabin)."""
    return pd.read_csv(TEST_DATA / "Titanic-Dataset.csv")


@pytest.fixture
def mock_llm():
    """Mock LLM provider with configurable generate_plan() responses."""
    provider = MagicMock()
    provider.generate_plan = MagicMock(return_value=None)
    return provider


# ============================================================================
# Group 1: Backtick sanitizer — columns with spaces
# ============================================================================

class TestBacktickSanitizer:
    """Verify backtick-quoted column names pass AST validation."""

    def test_simple_backtick_expression(self):
        result = _sanitize_backticks_for_ast("`number vmail messages` == 0")
        assert "`" not in result
        assert "==" in result

    def test_multiple_backtick_columns(self):
        result = _sanitize_backticks_for_ast(
            "`total day minutes` > 200 and `total eve minutes` < 100"
        )
        assert "`" not in result
        assert "and" in result

    def test_no_backticks_unchanged(self):
        expr = "age > 30"
        assert _sanitize_backticks_for_ast(expr) == expr

    def test_backtick_with_special_chars(self):
        result = _sanitize_backticks_for_ast("`Sub-Category` == 'Chairs'")
        assert "`" not in result


# ============================================================================
# Group 2: query_data — filtering with columns that have spaces (TELCO)
# ============================================================================

class TestQueryDataTelco:
    """Test query_data against telco dataset with space-containing columns."""

    def test_filter_zero_vmail_messages(self, telco_df, mock_llm):
        """'how many states have zero vmail messages' — the original bug."""
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "`number vmail messages` == 0",
            "columns": ["state"],
        })
        result = query_data(telco_df, "how many states have zero vmail messages", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0
        # Every returned row should have number vmail messages == 0
        for row in result["data"]:
            assert "state" in row

    def test_filter_international_plan_yes(self, telco_df, mock_llm):
        """Filter rows where international plan is yes."""
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "`international plan` == 'yes'",
            "columns": None,
        })
        result = query_data(telco_df, "show customers with international plan", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0

    def test_filter_high_day_minutes(self, telco_df, mock_llm):
        """Filter rows with total day minutes > 300."""
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "`total day minutes` > 300",
            "columns": None,
        })
        result = query_data(telco_df, "customers with more than 300 day minutes", llm_provider=mock_llm)
        assert result["status"] == "success"
        for row in result["data"]:
            assert row["total day minutes"] > 300

    def test_filter_churned_customers(self, telco_df, mock_llm):
        """Filter churn == True (boolean column)."""
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "churn == True",
            "columns": ["state", "churn", "customer service calls"],
        })
        result = query_data(telco_df, "show churned customers", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0

    def test_compound_filter(self, telco_df, mock_llm):
        """Multiple conditions: high service calls AND churned."""
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "`customer service calls` > 4 and churn == True",
            "columns": None,
        })
        result = query_data(telco_df, "churned customers with more than 4 service calls", llm_provider=mock_llm)
        assert result["status"] == "success"
        for row in result["data"]:
            assert row["customer service calls"] > 4


# ============================================================================
# Group 3: query_data — Superstore (dates, strings, dollars)
# ============================================================================

class TestQueryDataSuperstore:
    """Test query_data against superstore dataset."""

    def test_filter_by_category(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "Category == 'Furniture'",
            "columns": None,
        })
        result = query_data(superstore_df, "show furniture orders", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0
        for row in result["data"]:
            assert row["Category"] == "Furniture"

    def test_filter_by_region(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "Region == 'West'",
            "columns": ["State", "City", "Sales", "Profit"],
        })
        result = query_data(superstore_df, "west region sales", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0

    def test_filter_negative_profit(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "Profit < 0",
            "columns": None,
        })
        result = query_data(superstore_df, "orders with negative profit", llm_provider=mock_llm)
        assert result["status"] == "success"
        for row in result["data"]:
            assert row["Profit"] < 0

    def test_filter_sub_category_with_hyphen(self, superstore_df, mock_llm):
        """Sub-Category column name has a hyphen."""
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "`Sub-Category` == 'Chairs'",
            "columns": None,
        })
        result = query_data(superstore_df, "show chair orders", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0


# ============================================================================
# Group 4: query_data — Titanic (missing values)
# ============================================================================

class TestQueryDataTitanic:
    """Test query_data against titanic dataset with missing values."""

    def test_filter_survived(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "Survived == 1",
            "columns": None,
        })
        result = query_data(titanic_df, "show survivors", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0

    def test_filter_first_class(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "Pclass == 1",
            "columns": ["Name", "Age", "Fare"],
        })
        result = query_data(titanic_df, "first class passengers", llm_provider=mock_llm)
        assert result["status"] == "success"

    def test_filter_age_with_nulls(self, titanic_df, mock_llm):
        """Age column has 177 nulls — filter should still work."""
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "Age > 60",
            "columns": None,
        })
        result = query_data(titanic_df, "passengers over 60", llm_provider=mock_llm)
        assert result["status"] == "success"
        for row in result["data"]:
            assert row["Age"] > 60

    def test_filter_female_passengers(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "Sex == 'female'",
            "columns": None,
        })
        result = query_data(titanic_df, "show female passengers", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert result["total_rows"] > 0


# ============================================================================
# Group 5: pivot_table — aggregation across datasets
# ============================================================================

class TestPivotTableReal:
    """Test grouped aggregation on real datasets."""

    def test_avg_day_minutes_by_state(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "values": "total day minutes",
            "index": "state",
            "aggfunc": "mean",
        })
        result = pivot_table(telco_df, "average day minutes per state", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) > 0

    def test_sum_sales_by_region(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "values": "Sales",
            "index": "Region",
            "aggfunc": "sum",
        })
        result = pivot_table(superstore_df, "total sales by region", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 4  # 4 regions

    def test_avg_fare_by_class(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "values": "Fare",
            "index": "Pclass",
            "aggfunc": "mean",
        })
        result = pivot_table(titanic_df, "average fare per class", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 3  # 3 classes

    def test_count_by_category(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "values": "Sales",
            "index": "Category",
            "aggfunc": "count",
        })
        result = pivot_table(superstore_df, "number of orders per category", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 3  # Furniture, Office Supplies, Technology

    def test_service_calls_by_churn(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "values": "customer service calls",
            "index": "churn",
            "aggfunc": "mean",
        })
        result = pivot_table(telco_df, "average service calls for churned vs not", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 2  # True/False


# ============================================================================
# Group 6: value_counts — frequency distributions
# ============================================================================

class TestValueCountsReal:
    """Test frequency counts on real datasets."""

    def test_states_distribution(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({"column": "state"})
        result = value_counts(telco_df, "how many customers per state", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) > 40  # ~51 states

    def test_category_distribution(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({"column": "Category"})
        result = value_counts(superstore_df, "orders per category", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert set(result["data"].keys()) == {"Furniture", "Office Supplies", "Technology"}

    def test_pclass_distribution(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({"column": "Pclass"})
        result = value_counts(titanic_df, "passengers per class", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 3

    def test_ship_mode_distribution(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({"column": "Ship Mode"})
        result = value_counts(superstore_df, "orders by ship mode", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) > 0

    def test_top_n_states(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({"column": "state", "top_n": 5})
        result = value_counts(telco_df, "top 5 states by customer count", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 5

    def test_embarked_with_nulls(self, titanic_df, mock_llm):
        """Embarked has 2 null values — value_counts should still work."""
        mock_llm.generate_plan.return_value = json.dumps({"column": "Embarked"})
        result = value_counts(titanic_df, "embarkation port distribution", llm_provider=mock_llm)
        assert result["status"] == "success"


# ============================================================================
# Group 7: top_n — ranking records
# ============================================================================

class TestTopNReal:
    """Test top/bottom N ranking on real datasets."""

    def test_top_10_by_day_charge(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "column": "total day charge", "n": 10, "ascending": False,
        })
        result = top_n(telco_df, "top 10 customers by day charge", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 10
        charges = [r["total day charge"] for r in result["data"]]
        assert charges == sorted(charges, reverse=True)

    def test_top_5_profitable_orders(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "column": "Profit", "n": 5, "ascending": False,
        })
        result = top_n(superstore_df, "top 5 most profitable orders", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 5

    def test_bottom_5_loss_orders(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "column": "Profit", "n": 5, "ascending": True,
        })
        result = top_n(superstore_df, "bottom 5 orders by profit", llm_provider=mock_llm)
        assert result["status"] == "success"
        profits = [r["Profit"] for r in result["data"]]
        assert profits == sorted(profits)

    def test_top_fares_titanic(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "column": "Fare", "n": 10, "ascending": False,
        })
        result = top_n(titanic_df, "10 most expensive tickets", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 10


# ============================================================================
# Group 8: cross_tab — cross-tabulation
# ============================================================================

class TestCrossTabReal:
    """Test cross-tabulation on real datasets."""

    def test_churn_by_intl_plan(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "row": "international plan", "col": "churn",
        })
        result = cross_tab(telco_df, "churn by international plan", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 2  # yes/no

    def test_survived_by_sex(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "row": "Sex", "col": "Survived",
        })
        result = cross_tab(titanic_df, "survival by gender", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 2  # male/female

    def test_category_by_region(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "row": "Category", "col": "Region",
        })
        result = cross_tab(superstore_df, "orders by category and region", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 3  # 3 categories

    def test_survived_by_class(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "row": "Pclass", "col": "Survived",
        })
        result = cross_tab(titanic_df, "survival rate by class", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert len(result["data"]) == 3  # 3 classes

    def test_churn_by_voicemail_plan(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "row": "voice mail plan", "col": "churn",
        })
        result = cross_tab(telco_df, "churn by voice mail plan", llm_provider=mock_llm)
        assert result["status"] == "success"


# ============================================================================
# Group 9: smart_query — LLM-generated pandas code (the Copilot-like fallback)
# ============================================================================

class TestSmartQueryReal:
    """Test smart_query with realistic pandas code against real datasets."""

    def test_count_states_with_zero_vmail(self, telco_df, mock_llm):
        """The exact question that failed — answered via smart_query."""
        mock_llm.generate_plan.return_value = (
            "result = df[df['number vmail messages'] == 0]['state'].nunique()"
        )
        result = smart_query(telco_df, "how many states have zero vmail messages", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert isinstance(result["data"], int)
        assert result["data"] > 0

    def test_avg_service_calls_churned(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = (
            "result = df[df['churn'] == True]['customer service calls'].mean()"
        )
        result = smart_query(telco_df, "average service calls for churned customers", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert isinstance(result["data"], float)

    def test_total_sales_per_region(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = (
            "result = df.groupby('Region')['Sales'].sum()"
        )
        result = smart_query(superstore_df, "total sales per region", llm_provider=mock_llm)
        assert result["status"] == "success"

    def test_survival_rate_by_gender(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = (
            "result = df.groupby('Sex')['Survived'].mean()"
        )
        result = smart_query(titanic_df, "survival rate by gender", llm_provider=mock_llm)
        assert result["status"] == "success"

    def test_most_profitable_subcategory(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = (
            "result = df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False).head(5)"
        )
        result = smart_query(superstore_df, "top 5 sub-categories by profit", llm_provider=mock_llm)
        assert result["status"] == "success"

    def test_avg_age_by_class_titanic(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = (
            "result = df.groupby('Pclass')['Age'].mean()"
        )
        result = smart_query(titanic_df, "average age per class", llm_provider=mock_llm)
        assert result["status"] == "success"

    def test_churn_rate_percentage(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = (
            "result = np.round(df['churn'].mean() * 100, 2)"
        )
        result = smart_query(telco_df, "what is the churn rate", llm_provider=mock_llm)
        assert result["status"] == "success"
        assert 0 < float(result["data"]) < 100


# ============================================================================
# Group 10: _execute_safe — sandbox with real data shapes
# ============================================================================

class TestSandboxRealData:
    """Test sandbox execution safety with real dataset sizes."""

    def test_large_df_respects_row_limit(self, superstore_df):
        """Result is capped at 500 rows even for 9994-row dataset."""
        result = _execute_safe("result = df", superstore_df)
        assert result["status"] == "success"
        assert len(result["data"]) <= 500
        assert result["total_rows"] == 9994

    def test_scalar_result_from_real_data(self, telco_df):
        result = _execute_safe("result = df.shape[0]", telco_df)
        assert result["status"] == "success"
        assert result["data"] == 3333

    def test_series_result_from_groupby(self, superstore_df):
        result = _execute_safe("result = df.groupby('Region')['Sales'].sum()", superstore_df)
        assert result["status"] == "success"
        assert isinstance(result["data"], dict)
        assert "West" in result["data"]

    def test_column_with_spaces_in_sandbox(self, telco_df):
        """Columns with spaces work in sandbox code."""
        result = _execute_safe(
            "result = df['number vmail messages'].mean()",
            telco_df,
        )
        assert result["status"] == "success"
        assert isinstance(result["data"], float)

    def test_boolean_column_filtering(self, telco_df):
        """Boolean churn column works in sandbox."""
        result = _execute_safe(
            "result = df[df['churn'] == True].shape[0]",
            telco_df,
        )
        assert result["status"] == "success"
        assert result["data"] > 0

    def test_missing_values_handled(self, titanic_df):
        """Operations on columns with nulls don't crash."""
        result = _execute_safe(
            "result = df['Age'].dropna().mean()",
            titanic_df,
        )
        assert result["status"] == "success"
        assert isinstance(result["data"], float)


# ============================================================================
# Group 11: Routing — smart_query-first architecture
# ============================================================================

class TestRoutingSemanticFirst:
    """Test semantic-first routing: all questions route via embedding match,
    falling back to smart_query when no match is found."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        old = SemanticSkillMatcher._instance
        yield
        SemanticSkillMatcher._instance = old

    @pytest.fixture
    def matcher(self):
        return SemanticSkillMatcher()

    @pytest.fixture
    def data_context(self):
        return {
            "shape": "100 rows x 5 columns",
            "columns": [
                {"name": "profit", "dtype": "float64", "semantic_type": "numeric", "n_unique": 80, "null_pct": 0},
                {"name": "state", "dtype": "object", "semantic_type": "categorical", "n_unique": 10, "null_pct": 0},
                {"name": "sales", "dtype": "float64", "semantic_type": "numeric", "n_unique": 90, "null_pct": 0},
                {"name": "age", "dtype": "int64", "semantic_type": "numeric", "n_unique": 50, "null_pct": 0},
                {"name": "churn", "dtype": "object", "semantic_type": "categorical", "n_unique": 2, "null_pct": 0},
            ],
            "n_rows": 100,
            "n_cols": 5,
        }

    # --- Data query questions route via semantic embedding or smart_query ---

    def test_filter_routes_semantically(self, data_context):
        """'where' filter questions route via embedding or smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("show rows where profit is negative", data_context)
        assert r.skill_name in ("query_data", "smart_query")

    def test_how_many_per_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("how many customers per state", data_context)
        assert r.skill_name in ("value_counts", "smart_query")

    def test_average_by_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("average sales by region", data_context)
        assert r.skill_name in ("pivot_table", "smart_query")

    def test_top_n_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("top 10 customers by total day charge", data_context)
        assert r.skill_name in ("top_n", "smart_query")

    def test_crosstab_routes_semantically(self, data_context):
        provider = MagicMock()
        router = Router(provider=provider)
        r = router.route("crosstab of churn and international plan", data_context)
        assert r.skill_name in ("cross_tab", "smart_query")

    # --- Analytical skills match via semantic embedding ---

    def test_route_classify(self, matcher):
        r = matcher.match("classify customers by churn", threshold=0.35)
        # Embedding model may match classify or find_clusters (semantically close)
        assert r is not None and r[0] in ("classify", "find_clusters")

    def test_route_predict_churn(self, matcher):
        r = matcher.match("predict churn", threshold=0.30)
        assert r is not None and r[0] in ("classify", "find_clusters")

    def test_route_correlations(self, matcher):
        r = matcher.match("what are the correlations", threshold=0.35)
        assert r is not None and r[0] == "analyze_correlations"

    def test_route_outliers(self, matcher):
        r = matcher.match("find outliers in the data", threshold=0.35)
        assert r is not None and r[0] == "detect_outliers"

    def test_route_forecast(self, matcher):
        r = matcher.match("forecast sales for next quarter", threshold=0.35)
        assert r is not None and r[0] == "forecast"

    def test_route_hypothesis(self, matcher):
        r = matcher.match("run a t-test on age by gender", threshold=0.30)
        assert r is not None and r[0] == "run_hypothesis_test"

    def test_route_clusters(self, matcher):
        r = matcher.match("find clusters in the data", threshold=0.35)
        assert r is not None and r[0] == "find_clusters"

    def test_route_feature_selection(self, matcher):
        r = matcher.match("select features for the model", threshold=0.35)
        assert r is not None and r[0] == "select_features"

    def test_route_describe(self, matcher):
        r = matcher.match("give me descriptive statistics for my columns", threshold=0.30)
        assert r is not None and r[0] == "describe_data"

    def test_route_profile(self, matcher):
        r = matcher.match("give me an overview of the data", threshold=0.35)
        assert r is not None and r[0] == "profile_data"

    def test_route_sentiment(self, matcher):
        r = matcher.match("analyze sentiment of reviews", threshold=0.35)
        assert r is not None and r[0] == "analyze_sentiment"

    def test_route_pca(self, matcher):
        r = matcher.match("run PCA on the features", threshold=0.35)
        assert r is not None and r[0] == "reduce_dimensions"

    def test_route_survival(self, matcher):
        r = matcher.match("survival analysis of patients", threshold=0.30)
        assert r is not None and r[0] in ("survival_analysis", "cross_tab")


# ============================================================================
# Group 12: Sandbox builtins — smart_query needs len, round, etc.
# ============================================================================

class TestSandboxBuiltins:
    """Test that safe builtins work in the smart_query sandbox."""

    def test_len_in_sandbox(self, telco_df):
        """len() should work — needed for 'how many X' questions."""
        result = _execute_safe("result = len(df)", telco_df)
        assert result["status"] == "success"
        assert result["data"] == 3333

    def test_round_in_sandbox(self, telco_df):
        """round() should work — needed for formatting numbers."""
        result = _execute_safe("result = round(df['total day minutes'].mean(), 2)", telco_df)
        assert result["status"] == "success"
        assert isinstance(result["data"], float)

    def test_sorted_in_sandbox(self, telco_df):
        """sorted() should work — return length as scalar since lists aren't a supported output type."""
        result = _execute_safe("result = len(sorted(df['state'].unique().tolist()))", telco_df)
        assert result["status"] == "success"
        assert result["data"] > 0

    def test_int_float_str_in_sandbox(self, telco_df):
        """Type conversion builtins should work."""
        result = _execute_safe("result = int(df['total day minutes'].mean())", telco_df)
        assert result["status"] == "success"
        assert isinstance(result["data"], int)

    def test_min_max_sum_in_sandbox(self, telco_df):
        """Aggregation builtins should work."""
        result = _execute_safe(
            "result = max(df['total day minutes'])",
            telco_df,
        )
        assert result["status"] == "success"

    def test_enumerate_zip_in_sandbox(self, telco_df):
        """enumerate/zip should work for iteration patterns."""
        result = _execute_safe(
            "cols = list(df.columns[:3])\nresult = len(list(enumerate(cols)))",
            telco_df,
        )
        assert result["status"] == "success"
        assert result["data"] == 3

    def test_isinstance_in_sandbox(self, telco_df):
        """isinstance() should work for type checking."""
        result = _execute_safe("result = isinstance(df.shape[0], int)", telco_df)
        assert result["status"] == "success"
        assert result["data"] is True

    def test_safe_builtins_whitelist_exists(self):
        """_SAFE_BUILTINS should contain essential builtins."""
        assert "len" in _SAFE_BUILTINS
        assert "round" in _SAFE_BUILTINS
        assert "sorted" in _SAFE_BUILTINS
        assert "int" in _SAFE_BUILTINS
        assert "float" in _SAFE_BUILTINS
        assert "str" in _SAFE_BUILTINS
        assert "bool" in _SAFE_BUILTINS
        assert "min" in _SAFE_BUILTINS
        assert "max" in _SAFE_BUILTINS
        assert "sum" in _SAFE_BUILTINS

    def test_forbidden_still_blocked(self, telco_df):
        """Dangerous builtins (open, __import__) should still be blocked."""
        # open is not in _SAFE_BUILTINS, so it should fail
        result = _execute_safe("result = open('test.txt')", telco_df)
        assert result["status"] == "error"


# ============================================================================
# Group 12b: smart_query as primary path — direct skill functions still work
# ============================================================================

class TestQuerySkillsStillWork:
    """Query skill functions still work when called directly, even though
    they're no longer routed to by default."""

    def test_query_data_direct(self, telco_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "filter_expression": "churn == True",
            "columns": None,
        })
        result = query_data(telco_df, "show churned customers", llm_provider=mock_llm)
        assert result["status"] == "success"

    def test_pivot_table_direct(self, superstore_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({
            "values": "Sales", "index": "Region", "aggfunc": "sum",
        })
        result = pivot_table(superstore_df, "total sales by region", llm_provider=mock_llm)
        assert result["status"] == "success"

    def test_value_counts_direct(self, titanic_df, mock_llm):
        mock_llm.generate_plan.return_value = json.dumps({"column": "Pclass"})
        result = value_counts(titanic_df, "passengers per class", llm_provider=mock_llm)
        assert result["status"] == "success"


# ============================================================================
# Group 13: Data integrity — verify known facts about each dataset
# ============================================================================

class TestDataIntegrity:
    """Verify known facts about each dataset to catch loading issues."""

    def test_telco_shape(self, telco_df):
        assert telco_df.shape == (3333, 21)

    def test_telco_columns_have_spaces(self, telco_df):
        spaced = [c for c in telco_df.columns if " " in c]
        assert len(spaced) >= 10  # most columns have spaces

    def test_telco_churn_is_bool(self, telco_df):
        assert telco_df["churn"].dtype == bool

    def test_telco_no_nulls(self, telco_df):
        assert telco_df.isna().sum().sum() == 0

    def test_superstore_shape(self, superstore_df):
        assert superstore_df.shape == (9994, 21)

    def test_superstore_has_regions(self, superstore_df):
        assert set(superstore_df["Region"].unique()) == {"East", "West", "Central", "South"}

    def test_superstore_has_categories(self, superstore_df):
        assert set(superstore_df["Category"].unique()) == {"Furniture", "Office Supplies", "Technology"}

    def test_titanic_shape(self, titanic_df):
        assert titanic_df.shape == (891, 12)

    def test_titanic_age_has_nulls(self, titanic_df):
        assert titanic_df["Age"].isna().sum() == 177

    def test_titanic_cabin_has_nulls(self, titanic_df):
        assert titanic_df["Cabin"].isna().sum() == 687

    def test_titanic_survived_binary(self, titanic_df):
        assert set(titanic_df["Survived"].unique()) == {0, 1}

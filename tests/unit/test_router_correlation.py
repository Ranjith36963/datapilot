"""Tests for correlation routing and skill registry fixes.

Verifies:
  1. Explicit correlation phrases route to analyze_correlations via embedding
  2. Ambiguous phrases gracefully fall to smart_query (valid fallback)
  3. End-to-end Router routes correlation questions correctly
  4. correlation_matrix (raw DataFrame helper) is excluded from skill catalog
"""

import pytest
from unittest.mock import MagicMock

from engine.datapilot.core.router import Router
from engine.datapilot.core.semantic_router import SemanticSkillMatcher, get_semantic_matcher
from engine.datapilot.llm.prompts import (
    build_skill_registry,
    build_skill_catalog,
    get_skill_names,
    _SKILL_REGISTRY,
    _CATALOG_CACHE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_registry_cache():
    """Reset the module-level registry/catalog caches between tests."""
    import engine.datapilot.llm.prompts as prompts_mod
    old_registry = prompts_mod._SKILL_REGISTRY
    old_cache = prompts_mod._CATALOG_CACHE
    prompts_mod._SKILL_REGISTRY = []
    prompts_mod._CATALOG_CACHE = None
    yield
    prompts_mod._SKILL_REGISTRY = old_registry
    prompts_mod._CATALOG_CACHE = old_cache


@pytest.fixture
def data_context():
    return {
        "shape": "100 rows x 4 columns",
        "columns": [
            {"name": "age", "dtype": "int64", "semantic_type": "numeric", "n_unique": 50, "null_pct": 0},
            {"name": "income", "dtype": "float64", "semantic_type": "numeric", "n_unique": 80, "null_pct": 0},
            {"name": "gender", "dtype": "object", "semantic_type": "categorical", "n_unique": 2, "null_pct": 0},
            {"name": "churn", "dtype": "object", "semantic_type": "categorical", "n_unique": 2, "null_pct": 0},
        ],
        "n_rows": 100,
        "n_cols": 4,
    }


@pytest.fixture
def router():
    """Router with a mock LLM provider (enables smart_query fallback)."""
    mock_provider = MagicMock()
    return Router(provider=mock_provider)


# ---------------------------------------------------------------------------
# 1. Explicit correlation phrases — must hit analyze_correlations
# ---------------------------------------------------------------------------

class TestExplicitCorrelationRouting:
    """Phrases with strong correlation signal should route via embedding."""

    def test_correlation_matrix(self, router, data_context):
        result = router.route("Show me the correlation matrix for all columns", data_context)
        assert result.skill_name == "analyze_correlations"
        assert result.route_method == "semantic_embedding"

    def test_how_correlated(self, router, data_context):
        result = router.route("How correlated are age and income in this dataset?", data_context)
        assert result.skill_name == "analyze_correlations"
        assert result.route_method == "semantic_embedding"


# ---------------------------------------------------------------------------
# 2. Ambiguous phrases — embedding OR smart_query are both valid
# ---------------------------------------------------------------------------

class TestAmbiguousCorrelationRouting:
    """Short/ambiguous 'relate' phrases may fall to smart_query — that's OK.

    The embedding model can't always distinguish 'relate' from 30+ skills.
    smart_query (LLM-generated pandas) is designed to catch exactly this.
    """

    VALID_SKILLS = {"analyze_correlations", "smart_query"}

    def test_correlation_between(self, router, data_context):
        result = router.route("What is the correlation between age and income?", data_context)
        assert result.skill_name in self.VALID_SKILLS

    def test_relate_to(self, router, data_context):
        result = router.route(
            "How does age relate to income?", data_context
        )
        assert result.skill_name in self.VALID_SKILLS

    def test_related_to(self, router, data_context):
        result = router.route(
            "How is tenure related to monthly charges?", data_context
        )
        assert result.skill_name in self.VALID_SKILLS

    def test_relationship_between(self, router, data_context):
        result = router.route(
            "What is the relationship between income and spending?", data_context
        )
        assert result.skill_name in self.VALID_SKILLS

    def test_correlate_verb(self, router, data_context):
        result = router.route("Correlate age with income", data_context)
        assert result.skill_name in self.VALID_SKILLS

    def test_association(self, router, data_context):
        result = router.route(
            "Is there an association between gender and churn?", data_context
        )
        assert result.skill_name in self.VALID_SKILLS


# ---------------------------------------------------------------------------
# 3. Skill registry — correlation_matrix excluded
# ---------------------------------------------------------------------------

class TestSkillRegistryExclusion:
    """correlation_matrix should NOT appear in the skill catalog."""

    def test_correlation_matrix_not_in_registry(self):
        registry = build_skill_registry()
        names = [name for name, *_ in registry]
        assert "correlation_matrix" not in names

    def test_correlation_matrix_not_in_catalog_text(self):
        catalog = build_skill_catalog()
        assert "correlation_matrix" not in catalog

    def test_correlation_matrix_not_in_skill_names(self):
        names = get_skill_names()
        assert "correlation_matrix" not in names

    def test_analyze_correlations_IS_in_registry(self):
        names = get_skill_names()
        assert "analyze_correlations" in names

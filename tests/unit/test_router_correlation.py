"""Tests for correlation routing and skill registry fixes.

Verifies:
  1. Keyword router matches "relate" variants -> analyze_correlations
  2. Existing correlation keywords still work
  3. correlation_matrix (raw DataFrame helper) is excluded from skill catalog
"""

import pytest

from engine.datapilot.core.router import _try_keyword_route
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


# ---------------------------------------------------------------------------
# 1. Keyword router — "relate" variants
# ---------------------------------------------------------------------------

class TestRelateKeywordRouting:
    """The word 'relate' (and variants) should route to analyze_correlations."""

    def test_relate_to(self):
        result = _try_keyword_route(
            "How does number vmail messages relate to customer service calls?"
        )
        assert result is not None
        assert result.skill_name == "analyze_correlations"

    def test_related_to(self):
        result = _try_keyword_route(
            "How is tenure related to monthly charges?"
        )
        assert result is not None
        assert result.skill_name == "analyze_correlations"

    def test_relate_standalone(self):
        result = _try_keyword_route(
            "Do these columns relate at all?"
        )
        assert result is not None
        assert result.skill_name == "analyze_correlations"


# ---------------------------------------------------------------------------
# 2. Existing correlation keywords still work
# ---------------------------------------------------------------------------

class TestExistingCorrelationKeywords:
    """Regression: existing patterns must continue routing correctly."""

    def test_correlation_between(self):
        result = _try_keyword_route(
            "What is the correlation between X and Y?"
        )
        assert result is not None
        assert result.skill_name == "analyze_correlations"

    def test_relationship_between(self):
        result = _try_keyword_route(
            "What is the relationship between income and spending?"
        )
        assert result is not None
        assert result.skill_name == "analyze_correlations"

    def test_correlate(self):
        result = _try_keyword_route("Correlate age with income")
        assert result is not None
        assert result.skill_name == "analyze_correlations"

    def test_association(self):
        result = _try_keyword_route(
            "Is there an association between gender and churn?"
        )
        assert result is not None
        assert result.skill_name == "analyze_correlations"


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

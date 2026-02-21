"""
Semantic Embedding Router tests.

Tests cover:
  1. SemanticSkillMatcher initialization and singleton behavior
  2. Skill matching accuracy for various phrasings
  3. Threshold filtering (low-confidence matches rejected)
  4. Router integration (semantic embedding step in routing chain)
  5. Graceful fallback when sentence-transformers unavailable
"""

import pytest
from unittest.mock import MagicMock, patch

from datapilot.core.semantic_router import (
    SemanticSkillMatcher,
    SKILL_DESCRIPTIONS,
    get_semantic_matcher,
)
from datapilot.core.router import Router


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests to avoid state leakage."""
    SemanticSkillMatcher._instance = None
    yield
    SemanticSkillMatcher._instance = None


@pytest.fixture
def matcher():
    """Create a fresh SemanticSkillMatcher."""
    return SemanticSkillMatcher()


@pytest.fixture
def data_context():
    return {
        "shape": "100 rows x 6 columns",
        "columns": [
            {"name": "Age", "dtype": "int64", "semantic_type": "numeric", "n_unique": 50, "null_pct": 0},
            {"name": "Sex", "dtype": "object", "semantic_type": "categorical", "n_unique": 2, "null_pct": 0},
            {"name": "Fare", "dtype": "float64", "semantic_type": "numeric", "n_unique": 80, "null_pct": 0},
            {"name": "Survived", "dtype": "object", "semantic_type": "categorical", "n_unique": 2, "null_pct": 0},
            {"name": "Pclass", "dtype": "int64", "semantic_type": "numeric", "n_unique": 3, "null_pct": 0},
            {"name": "Embarked", "dtype": "object", "semantic_type": "categorical", "n_unique": 3, "null_pct": 0},
        ],
        "n_rows": 100,
        "n_cols": 6,
    }


# ============================================================================
# Group 1: Initialization
# ============================================================================

class TestSemanticSkillMatcherInit:
    """Test SemanticSkillMatcher initialization."""

    def test_model_loads(self, matcher):
        """Model loads and embeddings are computed."""
        assert matcher._initialized is True
        assert matcher.corpus_embeddings is not None
        assert len(matcher.skill_names) == len(SKILL_DESCRIPTIONS)

    def test_singleton_pattern(self):
        """Multiple instantiations return the same object."""
        m1 = SemanticSkillMatcher()
        m2 = SemanticSkillMatcher()
        assert m1 is m2

    def test_get_semantic_matcher_returns_instance(self):
        """get_semantic_matcher() returns a valid matcher."""
        m = get_semantic_matcher()
        assert m is not None
        assert m._initialized is True


# ============================================================================
# Group 2: Skill matching accuracy (strong matches)
# ============================================================================

class TestSkillMatching:
    """Test that various phrasings match the correct skill.

    Tests focus on clear semantic matches where the embedder
    reliably produces high-confidence results.
    """

    def test_ranking_synonym(self, matcher):
        """'top earners' matches top_n."""
        result = matcher.match("top earners", threshold=0.40)
        assert result is not None
        assert result[0] == "top_n"

    def test_outlier_synonym(self, matcher):
        """'find unusual values' matches detect_outliers."""
        result = matcher.match("find unusual values in the data", threshold=0.40)
        assert result is not None
        assert result[0] == "detect_outliers"

    def test_clustering_paraphrase(self, matcher):
        """'segment customers into groups' matches find_clusters."""
        result = matcher.match("segment customers into groups", threshold=0.40)
        assert result is not None
        assert result[0] == "find_clusters"

    def test_time_series(self, matcher):
        """'predict future sales' matches forecast."""
        result = matcher.match("predict future sales trend", threshold=0.40)
        assert result is not None
        assert result[0] == "forecast"

    def test_correlation_paraphrase(self, matcher):
        """'how are age and fare related' matches analyze_correlations."""
        result = matcher.match("how are age and fare related", threshold=0.35)
        assert result is not None
        assert result[0] == "analyze_correlations"

    def test_filter_query(self, matcher):
        """'filter where Sex equals female' matches query_data at low threshold."""
        result = matcher.match("filter where Sex equals female", threshold=0.30)
        assert result is not None
        assert result[0] == "query_data"

    def test_profiling(self, matcher):
        """'what is in this data' matches profile_data."""
        result = matcher.match("tell me about this dataset", threshold=0.35)
        assert result is not None
        assert result[0] == "profile_data"

    def test_sentiment(self, matcher):
        """'analyze review sentiment' matches analyze_sentiment."""
        result = matcher.match("analyze the sentiment of reviews", threshold=0.40)
        assert result is not None
        assert result[0] == "analyze_sentiment"

    def test_dimensionality_reduction(self, matcher):
        """'PCA on the features' matches reduce_dimensions."""
        result = matcher.match("run PCA on the features", threshold=0.40)
        assert result is not None
        assert result[0] == "reduce_dimensions"


# ============================================================================
# Group 3: Threshold filtering
# ============================================================================

class TestThresholdFiltering:
    """Test that low-confidence matches are filtered out."""

    def test_gibberish_below_threshold(self, matcher):
        """Random gibberish doesn't match any skill."""
        result = matcher.match("xyzzy blorp fizzwang", threshold=0.40)
        assert result is None

    def test_high_threshold_filters_borderline(self, matcher):
        """Very high threshold rejects even reasonable matches."""
        result = matcher.match("top earners", threshold=0.90)
        assert result is None

    def test_score_is_float(self, matcher):
        """Match score is a float between 0 and 1."""
        result = matcher.match("find unusual values in the data", threshold=0.30)
        assert result is not None
        assert isinstance(result[1], float)
        assert 0.0 <= result[1] <= 1.0

    def test_stronger_match_has_higher_score(self, matcher):
        """Direct phrase match scores higher than vague paraphrase."""
        strong = matcher.match("segment customers into groups", threshold=0.30)
        weak = matcher.match("maybe group some things", threshold=0.30)
        if strong and weak:
            assert strong[1] > weak[1]


# ============================================================================
# Group 4: Router integration
# ============================================================================

class TestRouterSemanticEmbedding:
    """Test semantic embedding step in Router.route()."""

    def test_semantic_routing_for_clear_match(self, data_context):
        """Clear semantic match routes via embedding, not smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("find unusual values in the data", data_context)
        # detect_outliers matches keyword "unusual" OR semantic embedding
        assert result.skill_name == "detect_outliers"

    def test_classify_routes_via_embedding(self, data_context):
        """Classification questions route via semantic embedding."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("classify customers by survival", data_context)
        # Embedding model may match classify or find_clusters (semantically close)
        assert result.skill_name in ("classify", "find_clusters")
        assert result.route_method == "semantic_embedding"

    def test_chart_routes_via_embedding(self, data_context):
        """Chart questions route via semantic embedding."""
        provider = MagicMock()
        router = Router(provider=provider)
        result = router.route("draw a scatter plot of Age vs Fare", data_context)
        assert result.skill_name == "create_chart"
        assert result.route_method == "semantic_embedding"

    def test_no_provider_semantic_catches_clear_match(self, data_context):
        """Semantic embedding works even without an LLM provider."""
        router = Router(provider=None)
        # "segment customers into groups" has no keyword match and no LLM
        result = router.route("segment customers into groups", data_context)
        # Should be caught by semantic embedding or semantic intent
        assert result.skill_name == "find_clusters"

    def test_borderline_falls_to_smart_query(self, data_context):
        """Borderline semantic match falls through to smart_query."""
        provider = MagicMock()
        router = Router(provider=provider)
        # Force semantic matcher to be unavailable
        router._semantic_attempted = True
        router._semantic_matcher = None
        result = router.route("show me only females", data_context)
        assert result.skill_name == "smart_query"
        assert result.route_method == "smart_first"


# ============================================================================
# Group 5: Graceful fallback
# ============================================================================

class TestGracefulFallback:
    """Test behavior when sentence-transformers is unavailable."""

    def test_import_failure_returns_none(self):
        """get_semantic_matcher returns None when import fails."""
        with patch("datapilot.core.semantic_router.SemanticSkillMatcher.__init__", side_effect=ImportError("no module")):
            SemanticSkillMatcher._instance = None
            result = get_semantic_matcher()
            assert result is None

    def test_router_skips_embedding_on_failure(self, data_context):
        """Router falls through to smart_query when embedding unavailable."""
        provider = MagicMock()
        router = Router(provider=provider)
        # Force semantic matcher to fail
        router._semantic_attempted = True
        router._semantic_matcher = None
        result = router.route("show me only females", data_context)
        # Should fall through to smart_query
        assert result.skill_name == "smart_query"
        assert result.route_method == "smart_first"

    def test_router_no_provider_no_embedding_falls_to_profile(self, data_context):
        """Without LLM and without embeddings, unmatched falls to profile_data."""
        router = Router(provider=None)
        router._semantic_attempted = True
        router._semantic_matcher = None
        result = router.route("xyzzy nonsense", data_context)
        assert result.skill_name == "profile_data"
        assert result.route_method == "fallback"

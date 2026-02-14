"""
Tests for dataset fingerprinting module (D3) — LLM-first approach.

Tests cover:
  1. build_data_snapshot — compact text snapshot of a DataFrame for LLM context
  2. understand_dataset — LLM-driven dataset understanding (mocked providers)
  3. fingerprint_dataset — backward-compatible wrapper around understand_dataset

This is TDD RED phase — all tests should FAIL because the implementations
are stubs.  Agent 3 will make them pass.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from datapilot.data.fingerprint import (
    build_data_snapshot,
    understand_dataset,
    fingerprint_dataset,
    DatasetUnderstanding,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _create_mock_profile(df: pd.DataFrame) -> dict:
    """Create a mock profile dict for testing."""
    profile = {"columns": {}}
    for col in df.columns:
        dtype = str(df[col].dtype)
        profile["columns"][col] = {
            "dtype": dtype,
            "semantic_type": "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical",
        }
    return profile


VALID_LLM_RESPONSE = {
    "domain": "telecom customer churn",
    "domain_short": "Telecom",
    "target_column": "churn",
    "target_type": "classification",
    "key_observations": [
        "Churn rate is ~14.5%",
        "Monthly charges range from $18 to $118",
    ],
    "suggested_questions": ["What is the churn rate by contract type?"],
    "data_quality_notes": ["No missing values detected"],
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_telecom_df():
    """Telecom churn DataFrame with a mix of numeric, categorical, and boolean columns."""
    return pd.DataFrame({
        "customer_id": [f"CUST-{i:04d}" for i in range(1, 21)],
        "gender": ["Male", "Female"] * 10,
        "tenure": [1, 12, 24, 36, 48, 60, 72, 6, 18, 30, 42, 54, 66, 3, 9, 15, 33, 45, 57, 69],
        "monthly_charges": [
            29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 89.10, 29.75, 56.20, 80.85,
            95.75, 18.25, 108.15, 77.40, 62.15, 45.25, 88.90, 71.35, 103.50, 55.00,
        ],
        "total_charges": [
            29.85, 683.40, 1292.40, 1523.80, 3394.60, 5978.10, 6414.00, 178.50,
            1011.60, 2425.50, 4020.75, 985.50, 7138.35, 232.20, 559.35, 678.75,
            2933.70, 3210.75, 5899.50, 3795.00,
        ],
        "churn": [False, False, True, False, True, False, False, True, False, False,
                  True, False, False, True, False, False, False, True, False, True],
    })


@pytest.fixture
def sample_profile(sample_telecom_df):
    """Profile dict derived from sample_telecom_df."""
    return _create_mock_profile(sample_telecom_df)


@pytest.fixture
def mock_llm_provider():
    """MagicMock LLM provider with an understand_dataset method returning valid JSON."""
    provider = MagicMock()
    provider.understand_dataset.return_value = VALID_LLM_RESPONSE
    return provider


# ============================================================================
# Group 1: build_data_snapshot
# ============================================================================

class TestBuildDataSnapshot:
    """Tests for the build_data_snapshot helper that creates an LLM-ready text summary."""

    def test_build_snapshot_includes_filename_and_shape(self, sample_telecom_df):
        """Snapshot string must contain the filename and a 'rows x columns' shape indicator."""
        snapshot = build_data_snapshot(sample_telecom_df, filename="telecom_churn.csv")

        assert "telecom_churn.csv" in snapshot
        # Should mention 20 rows and 6 columns (the shape)
        assert "20" in snapshot
        assert "6" in snapshot

    def test_build_snapshot_includes_numeric_column_stats(self, sample_telecom_df):
        """Snapshot must contain min/max/mean (or similar summary stats) for numeric columns."""
        snapshot = build_data_snapshot(sample_telecom_df, filename="telecom_churn.csv")

        # tenure, monthly_charges, total_charges are numeric — expect some stats
        snapshot_lower = snapshot.lower()
        # At least one stat keyword should be present
        assert any(kw in snapshot_lower for kw in ["min", "max", "mean", "avg", "average"])
        # The column name should appear
        assert "monthly_charges" in snapshot_lower or "monthly charges" in snapshot_lower

    def test_build_snapshot_includes_categorical_top_values(self, sample_telecom_df):
        """Snapshot must list top values for categorical columns (e.g., gender)."""
        snapshot = build_data_snapshot(sample_telecom_df, filename="telecom_churn.csv")

        # gender is categorical with Male/Female
        assert "Male" in snapshot or "Female" in snapshot

    def test_build_snapshot_includes_sample_rows(self, sample_telecom_df):
        """Snapshot must contain some sample row data from the DataFrame."""
        snapshot = build_data_snapshot(sample_telecom_df, filename="telecom_churn.csv")

        # Should contain at least one customer_id value as a sample
        assert "CUST-" in snapshot

    def test_build_snapshot_truncates_at_50_columns(self):
        """DataFrames with >50 columns should show first 30 and indicate remaining count."""
        wide_df = pd.DataFrame(
            {f"col_{i}": range(5) for i in range(60)}
        )
        snapshot = build_data_snapshot(wide_df, filename="wide.csv")

        # Should mention some form of "30 more" or "and 30 more"
        assert "30 more" in snapshot.lower() or "30 additional" in snapshot.lower()
        # The first 30 columns should be present
        assert "col_0" in snapshot
        assert "col_29" in snapshot
        # Column 59 should NOT be individually listed
        assert "col_59" not in snapshot

    def test_build_snapshot_redacts_pii_columns(self):
        """Columns named email, phone, or ssn should have their values replaced with [REDACTED]."""
        pii_df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "email": ["alice@example.com", "bob@example.com"],
            "phone": ["555-0100", "555-0200"],
            "ssn": ["123-45-6789", "987-65-4321"],
            "age": [30, 25],
        })
        snapshot = build_data_snapshot(pii_df, filename="users.csv")

        # Actual values must NOT appear
        assert "alice@example.com" not in snapshot
        assert "555-0100" not in snapshot
        assert "123-45-6789" not in snapshot
        # Redaction marker must appear
        assert "[REDACTED]" in snapshot

    def test_build_snapshot_handles_empty_dataframe(self):
        """An empty DataFrame should produce a valid snapshot string, not crash."""
        empty_df = pd.DataFrame()
        snapshot = build_data_snapshot(empty_df, filename="empty.csv")

        assert isinstance(snapshot, str)
        assert len(snapshot) > 0
        assert "empty.csv" in snapshot

    def test_build_snapshot_under_2000_tokens(self, sample_telecom_df):
        """Snapshot should be under ~2000 tokens; using 8000 chars as a rough proxy."""
        snapshot = build_data_snapshot(sample_telecom_df, filename="telecom_churn.csv")

        assert len(snapshot) < 8000


# ============================================================================
# Group 2: understand_dataset (mocked LLM)
# ============================================================================

class TestUnderstandDataset:
    """Tests for understand_dataset which sends the snapshot to an LLM provider."""

    def test_understand_returns_dataclass(self, sample_telecom_df, sample_profile, mock_llm_provider):
        """understand_dataset must return a DatasetUnderstanding instance."""
        result = understand_dataset(
            df=sample_telecom_df,
            filename="telecom_churn.csv",
            profile=sample_profile,
            llm_provider=mock_llm_provider,
        )

        assert isinstance(result, DatasetUnderstanding)

    def test_understand_parses_valid_json(self, sample_telecom_df, sample_profile, mock_llm_provider):
        """When the LLM returns valid JSON, all DatasetUnderstanding fields are populated."""
        result = understand_dataset(
            df=sample_telecom_df,
            filename="telecom_churn.csv",
            profile=sample_profile,
            llm_provider=mock_llm_provider,
        )

        assert result is not None
        assert result.domain == "telecom customer churn"
        assert result.domain_short == "Telecom"
        assert result.target_column == "churn"
        assert result.target_type == "classification"
        assert len(result.key_observations) == 2
        assert len(result.suggested_questions) >= 1
        assert len(result.data_quality_notes) >= 1

    def test_understand_handles_invalid_json(self, sample_telecom_df, sample_profile):
        """When the LLM returns unparseable garbage, understand_dataset returns None."""
        bad_provider = MagicMock()
        bad_provider.understand_dataset.return_value = "This is not JSON at all, just random text."

        result = understand_dataset(
            df=sample_telecom_df,
            filename="telecom_churn.csv",
            profile=sample_profile,
            llm_provider=bad_provider,
        )

        assert result is None

    def test_understand_fallback_to_groq(self, sample_telecom_df, sample_profile):
        """If the primary provider raises, the fallback provider should succeed."""
        # Primary provider raises an exception
        failing_provider = MagicMock()
        failing_provider.understand_dataset.side_effect = Exception("Gemini API down")

        # Fallback provider returns valid data
        fallback_provider = MagicMock()
        fallback_provider.understand_dataset.return_value = VALID_LLM_RESPONSE

        result = understand_dataset(
            df=sample_telecom_df,
            filename="telecom_churn.csv",
            profile=sample_profile,
            llm_provider=failing_provider,
            fallback_provider=fallback_provider,
        )

        assert result is not None
        assert isinstance(result, DatasetUnderstanding)
        assert result.domain == "telecom customer churn"

    def test_understand_both_fail_returns_none(self, sample_telecom_df, sample_profile):
        """When both primary and fallback providers fail, result is None."""
        failing_primary = MagicMock()
        failing_primary.understand_dataset.side_effect = Exception("Gemini down")

        failing_fallback = MagicMock()
        failing_fallback.understand_dataset.side_effect = Exception("Groq down")

        result = understand_dataset(
            df=sample_telecom_df,
            filename="telecom_churn.csv",
            profile=sample_profile,
            llm_provider=failing_primary,
            fallback_provider=failing_fallback,
        )

        assert result is None

    def test_understand_sets_provider_used(self, sample_telecom_df, sample_profile, mock_llm_provider):
        """The provider_used field must reflect which provider actually answered."""
        mock_llm_provider.name = "gemini"

        result = understand_dataset(
            df=sample_telecom_df,
            filename="telecom_churn.csv",
            profile=sample_profile,
            llm_provider=mock_llm_provider,
        )

        assert result is not None
        assert result.provider_used == "gemini"


# ============================================================================
# Group 3: Backward-compat wrapper
# ============================================================================

class TestFingerprintDatasetWrapper:
    """Test that fingerprint_dataset() still works as a backward-compatible wrapper."""

    def test_fingerprint_dataset_wrapper_calls_understand(
        self, sample_telecom_df, sample_profile, mock_llm_provider
    ):
        """fingerprint_dataset() must call understand_dataset() internally and return a dict with status=success."""
        result = fingerprint_dataset(
            df=sample_telecom_df,
            filename="telecom_churn.csv",
            profile=sample_profile,
            llm_provider=mock_llm_provider,
        )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert "domain" in result
        # Should carry through the LLM understanding
        assert result["domain"] == "telecom customer churn" or result.get("domain_short") == "Telecom"

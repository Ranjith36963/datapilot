"""
Tests for D3 fingerprint — LLM-first dataset understanding.

Covers:
  1. build_data_snapshot() — basic, PII, empty, wide, bool columns
  2. understand_dataset() — success, failover, both fail, invalid JSON, missing domain
  3. fingerprint_dataset() — backward-compatible wrapper
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from datapilot.data.fingerprint import (
    DatasetUnderstanding,
    build_data_snapshot,
    understand_dataset,
    fingerprint_dataset,
    _is_pii_column,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_df():
    """Basic DataFrame for snapshot tests."""
    return pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "Salary": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
        "Active": [True, False, True, True, False],
    })


@pytest.fixture
def mock_provider_success():
    """Mock provider that returns valid understanding JSON."""
    provider = MagicMock()
    provider.understand_dataset.return_value = {
        "domain": "human resources employee analytics",
        "domain_short": "HR",
        "target_column": "Active",
        "target_type": "classification",
        "key_observations": ["Average age is 35"],
        "suggested_questions": ["What factors predict attrition?"],
        "data_quality_notes": ["No missing values"],
    }
    provider.name = "gemini"
    return provider


@pytest.fixture
def mock_provider_fail():
    """Mock provider that returns None."""
    provider = MagicMock()
    provider.understand_dataset.return_value = None
    provider.name = "groq"
    return provider


# ============================================================================
# Group 1: build_data_snapshot
# ============================================================================

class TestBuildDataSnapshot:
    def test_basic_snapshot(self, sample_df):
        """Snapshot contains filename, shape, and column info."""
        snapshot = build_data_snapshot(sample_df, "employees.csv")
        assert "employees.csv" in snapshot
        assert "5 rows x 4 columns" in snapshot
        assert "Age" in snapshot
        assert "Salary" in snapshot

    def test_pii_redaction(self):
        """Columns with PII names show [REDACTED]."""
        df = pd.DataFrame({
            "email": ["a@b.com", "c@d.com"],
            "ssn": ["123-45-6789", "987-65-4321"],
            "score": [85, 90],
        })
        snapshot = build_data_snapshot(df, "users.csv")
        assert "[REDACTED]" in snapshot
        # score should NOT be redacted
        assert "score" in snapshot

    def test_empty_dataframe(self):
        """Empty DataFrame returns special message."""
        df = pd.DataFrame()
        snapshot = build_data_snapshot(df, "empty.csv")
        assert "Empty dataset" in snapshot

    def test_wide_dataset_truncation(self):
        """Datasets with >50 columns truncate to 30."""
        data = {f"col_{i}": [1, 2, 3] for i in range(60)}
        df = pd.DataFrame(data)
        snapshot = build_data_snapshot(df, "wide.csv")
        assert "and 30 more columns" in snapshot

    def test_bool_columns_show_value_counts(self, sample_df):
        """Bool dtype columns show value_counts, not min/max/mean."""
        snapshot = build_data_snapshot(sample_df, "test.csv")
        # Active is bool — should show values dict, not min/max
        # Find the Active line
        for line in snapshot.split("\n"):
            if "Active" in line and "bool" in line:
                assert "min:" not in line
                assert "values:" in line or "True" in line
                break

    def test_numeric_columns_show_stats(self, sample_df):
        """Numeric columns show min, max, mean."""
        snapshot = build_data_snapshot(sample_df, "test.csv")
        for line in snapshot.split("\n"):
            if "Age" in line and "int" in line:
                assert "min:" in line
                assert "max:" in line
                assert "mean:" in line
                break

    def test_categorical_columns_show_top_values(self, sample_df):
        """Object columns show top values."""
        snapshot = build_data_snapshot(sample_df, "test.csv")
        for line in snapshot.split("\n"):
            if "Name" in line and "object" in line:
                assert "top values:" in line
                break

    def test_missing_values_shown(self):
        """Missing values section appears when nulls exist."""
        df = pd.DataFrame({
            "A": [1, 2, None, 4, 5],
            "B": ["x", None, "z", None, "w"],
        })
        snapshot = build_data_snapshot(df, "nulls.csv")
        assert "Missing values:" in snapshot

    def test_no_missing_values_section_when_clean(self, sample_df):
        """Missing values section absent when no nulls."""
        snapshot = build_data_snapshot(sample_df, "clean.csv")
        assert "Missing values:" not in snapshot

    def test_profile_correlations_shown(self):
        """Profile dict with correlations adds notable correlations section."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        profile = {"correlations": {"A_vs_B": 0.95, "C_vs_D": 0.1}}
        snapshot = build_data_snapshot(df, "test.csv", profile=profile)
        assert "Notable correlations:" in snapshot
        assert "A_vs_B" in snapshot
        # Low correlation (0.1) should not appear
        assert "C_vs_D" not in snapshot


# ============================================================================
# Group 2: _is_pii_column
# ============================================================================

class TestIsPiiColumn:
    def test_email_is_pii(self):
        assert _is_pii_column("email") is True

    def test_ssn_is_pii(self):
        assert _is_pii_column("SSN") is True

    def test_phone_number_is_pii(self):
        assert _is_pii_column("phone_number") is True

    def test_age_is_not_pii(self):
        assert _is_pii_column("Age") is False

    def test_salary_is_not_pii(self):
        assert _is_pii_column("Salary") is False


# ============================================================================
# Group 3: understand_dataset
# ============================================================================

class TestUnderstandDataset:
    def test_success(self, sample_df, mock_provider_success):
        """Valid provider response → DatasetUnderstanding returned."""
        result = understand_dataset(
            sample_df, "employees.csv", {}, mock_provider_success
        )
        assert result is not None
        assert isinstance(result, DatasetUnderstanding)
        assert result.domain == "human resources employee analytics"
        assert result.domain_short == "HR"
        assert result.target_column == "Active"
        assert result.confidence == 0.9  # default when not in response

    def test_primary_fails_fallback_succeeds(
        self, sample_df, mock_provider_fail, mock_provider_success
    ):
        """Primary returns None → fallback provider used."""
        result = understand_dataset(
            sample_df, "employees.csv", {},
            mock_provider_fail, mock_provider_success,
        )
        assert result is not None
        assert result.domain_short == "HR"

    def test_both_fail(self, sample_df, mock_provider_fail):
        """Both providers fail → returns None."""
        fail2 = MagicMock()
        fail2.understand_dataset.return_value = None
        result = understand_dataset(
            sample_df, "employees.csv", {},
            mock_provider_fail, fail2,
        )
        assert result is None

    def test_invalid_json_string(self, sample_df):
        """Provider returns unparseable string → returns None."""
        provider = MagicMock()
        provider.understand_dataset.return_value = "this is not json at all"
        result = understand_dataset(sample_df, "test.csv", {}, provider)
        assert result is None

    def test_missing_domain_field(self, sample_df):
        """Provider returns JSON without 'domain' → returns None."""
        provider = MagicMock()
        provider.understand_dataset.return_value = {
            "domain_short": "Test",
            "key_observations": [],
        }
        result = understand_dataset(sample_df, "test.csv", {}, provider)
        assert result is None

    def test_provider_raises_exception(self, sample_df):
        """Provider raises exception → returns None, doesn't crash."""
        provider = MagicMock()
        provider.understand_dataset.side_effect = RuntimeError("API down")
        result = understand_dataset(sample_df, "test.csv", {}, provider)
        assert result is None


# ============================================================================
# Group 4: fingerprint_dataset (backward-compatible wrapper)
# ============================================================================

class TestFingerprintDataset:
    def test_no_provider_returns_general(self, sample_df):
        """No LLM provider → returns general domain with 0 confidence."""
        result = fingerprint_dataset(sample_df, "test.csv", {})
        assert result["status"] == "success"
        assert result["domain"] == "general"
        assert result["confidence"] == 0.0

    def test_with_provider_returns_flat_dict(self, sample_df, mock_provider_success):
        """With provider → returns flat dict with domain info."""
        result = fingerprint_dataset(
            sample_df, "test.csv", {}, mock_provider_success
        )
        assert result["status"] == "success"
        assert result["domain"] == "human resources employee analytics"
        assert result["domain_short"] == "HR"
        assert "target_column" in result

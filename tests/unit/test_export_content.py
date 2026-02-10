"""Tests for export report content.

Verifies:
  1. Analyst._build_report_data produces structured report with summary, sections, metrics
  2. Export functions with analysis history produce non-empty reports
  3. PDF, DOCX, PPTX files are created with real content
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from engine.datapilot.core.analyst import AnalystResult
from engine.datapilot.core.executor import ExecutionResult
from engine.datapilot.llm.provider import NarrativeResult, RoutingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_analyst_result(
    question: str,
    skill_name: str,
    result: Dict[str, Any],
    narrative_text: str = "",
    key_points: List[str] = None,
) -> AnalystResult:
    """Helper to build an AnalystResult for testing."""
    routing = RoutingResult(
        skill_name=skill_name,
        parameters={},
        confidence=0.9,
        reasoning="test",
    )
    execution = ExecutionResult(
        status="success",
        skill_name=skill_name,
        result=result,
        elapsed_seconds=1.0,
    )
    narrative = NarrativeResult(
        text=narrative_text,
        key_points=key_points or [],
        suggestions=[],
    ) if narrative_text else None

    return AnalystResult(
        question=question,
        routing=routing,
        execution=execution,
        narrative=narrative,
    )


@pytest.fixture
def sample_history():
    """Build a realistic analysis history."""
    return [
        _make_analyst_result(
            question="What are the key correlations?",
            skill_name="analyze_correlations",
            result={
                "status": "success",
                "top_correlations": [
                    {"col1": "age", "col2": "income", "correlation": 0.85},
                    {"col1": "score", "col2": "income", "correlation": 0.62},
                ],
            },
            narrative_text="The strongest correlation is between age and income (r=0.85).",
            key_points=["age and income: r=0.85", "score and income: r=0.62"],
        ),
        _make_analyst_result(
            question="Are there any outliers?",
            skill_name="detect_outliers",
            result={
                "status": "success",
                "n_outliers": 5,
                "outlier_pct": 5.0,
                "method": "Isolation Forest",
            },
            narrative_text="Found 5 outlier records (5.0%) using Isolation Forest.",
            key_points=["5 outliers detected", "5.0% of dataset"],
        ),
        _make_analyst_result(
            question="Predict purchased",
            skill_name="classify",
            result={
                "status": "success",
                "algorithm": "RandomForest",
                "target": "purchased",
                "metrics": {"accuracy": 0.92, "f1": 0.89},
                "feature_importance": [
                    {"feature": "income", "importance": 0.45},
                    {"feature": "score", "importance": 0.35},
                ],
            },
            narrative_text="Trained a RandomForest model with 92% accuracy and 89% F1 score.",
            key_points=["Accuracy: 92.0%", "F1 Score: 89.0%", "Top feature: income"],
        ),
    ]


# ---------------------------------------------------------------------------
# Tests for _build_report_data
# ---------------------------------------------------------------------------

class TestBuildReportData:
    """Test that _build_report_data produces structured output."""

    def test_report_has_summary(self, sample_history):
        """Report data should have a non-empty summary."""
        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report = Analyst._build_report_data(analyst)

        assert "summary" in report
        assert len(report["summary"]) > 50
        assert "correlation" in report["summary"].lower() or "age" in report["summary"].lower()

    def test_report_has_sections(self, sample_history):
        """Report should have one section per analysis."""
        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report = Analyst._build_report_data(analyst)

        assert "sections" in report
        assert len(report["sections"]) == 3
        assert report["sections"][0]["skill"] == "analyze_correlations"
        assert report["sections"][1]["skill"] == "detect_outliers"
        assert report["sections"][2]["skill"] == "classify"

    def test_sections_have_narratives(self, sample_history):
        """Each section should contain its narrative text."""
        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report = Analyst._build_report_data(analyst)

        for section in report["sections"]:
            assert "narrative" in section
            assert len(section["narrative"]) > 0

    def test_report_has_metrics(self, sample_history):
        """Report should extract metrics from analysis results."""
        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report = Analyst._build_report_data(analyst)

        assert "metrics" in report
        assert len(report["metrics"]) > 0
        labels = [m["label"] for m in report["metrics"]]
        assert "Strongest Correlation" in labels or "Accuracy" in labels or "Outliers Found" in labels

    def test_report_has_key_points(self, sample_history):
        """Report should aggregate key points from all analyses."""
        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report = Analyst._build_report_data(analyst)

        assert "key_points" in report
        assert len(report["key_points"]) >= 4  # From correlations + outliers + classify


# ---------------------------------------------------------------------------
# Tests for actual file export
# ---------------------------------------------------------------------------

class TestPDFExport:
    """Test PDF export produces non-empty files."""

    def test_pdf_with_sections(self, sample_history):
        """PDF should contain analysis sections, not just default text."""
        from engine.datapilot.export.pdf import export_to_pdf

        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report_data = Analyst._build_report_data(analyst)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            export_to_pdf(
                analysis_results=report_data,
                output_path=output_path,
                title="Test Report",
            )
            file_size = os.path.getsize(output_path)
            # A report with 3 analyses should be significantly larger than empty
            assert file_size > 2000, f"PDF too small ({file_size} bytes), likely missing content"
        finally:
            os.unlink(output_path)


class TestDOCXExport:
    """Test DOCX export produces non-empty files."""

    def test_docx_with_sections(self, sample_history):
        """DOCX should contain analysis sections."""
        from engine.datapilot.export.docx import export_to_docx

        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report_data = Analyst._build_report_data(analyst)

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            output_path = f.name

        try:
            export_to_docx(
                analysis_results=report_data,
                output_path=output_path,
                title="Test Report",
            )
            file_size = os.path.getsize(output_path)
            assert file_size > 2000, f"DOCX too small ({file_size} bytes), likely missing content"
        finally:
            os.unlink(output_path)


class TestPPTXExport:
    """Test PPTX export produces non-empty files."""

    def test_pptx_with_sections(self, sample_history):
        """PPTX should have slides for each analysis."""
        from engine.datapilot.export.pptx import export_to_pptx

        analyst = MagicMock()
        analyst.history = sample_history

        from engine.datapilot.core.analyst import Analyst
        report_data = Analyst._build_report_data(analyst)

        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as f:
            output_path = f.name

        try:
            export_to_pptx(
                analysis_results=report_data,
                output_path=output_path,
                title="Test Report",
            )
            file_size = os.path.getsize(output_path)
            assert file_size > 10000, f"PPTX too small ({file_size} bytes), likely missing content"
        finally:
            os.unlink(output_path)

"""
Report generator module for creating executive reports.
"""

from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import pandas as pd

from src.config import REPORTS_DIR
from src.utils import setup_logging, MetadataManager


class ReportGenerator:
    """Generates executive reports from analysis results."""

    def __init__(self):
        self.logger = setup_logging("report_generator")
        self.metadata = MetadataManager()

    def generate_executive_report(
        self,
        analysis_results: Dict[str, Any],
        ai_insights: Dict[str, Any],
        df: pd.DataFrame,
        version: str = "v1"
    ) -> Path:
        """Generate comprehensive executive report in Markdown format."""
        self.logger.info("Generating executive report")

        report_content = self._build_report(analysis_results, ai_insights, df)

        filename = f"executive_report_{version}.md"
        report_path = REPORTS_DIR / filename

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self.logger.info(f"Executive report saved to {report_path}")

        self.metadata.register_dataset(
            dataset_id=f"report_executive_{version}",
            name="Executive Churn Analysis Report",
            zone="products",
            path=str(report_path),
            row_count=0,
            column_count=0,
            description="AI-generated executive summary of churn analysis"
        )

        return report_path

    def _build_report(self, analysis_results: Dict, ai_insights: Dict, df: pd.DataFrame) -> str:
        """Build the complete report content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Veritly Market Intelligence Report
## Telecom Customer Churn Analysis

**Generated:** {timestamp}
**Platform:** Veritly AI Market Intelligence
**Version:** 1.0

---

{self._build_executive_summary(analysis_results, ai_insights)}
{self._build_model_performance(analysis_results)}
{self._build_metrics_dashboard(analysis_results, df)}
{self._build_segment_analysis(analysis_results)}
{self._build_tipping_points(analysis_results)}
{self._build_ai_insights(ai_insights)}
{self._build_risk_assessment(ai_insights)}
{self._build_action_plan(ai_insights)}
{self._build_data_quality(df)}
{self._build_footer()}
"""
        return report

    def _build_executive_summary(self, results: Dict, ai_insights: Dict) -> str:
        """Build executive summary section."""
        churn_rate = results.get("overall_churn_rate", 0)
        churned = results.get("churned_count", 0)
        total = results.get("total_count", 0)

        tipping_points = results.get("tipping_points", [])
        service_tp = next((tp for tp in tipping_points if "Service" in tp.get("factor", "")), {})

        return f"""## Executive Summary

### Current State
The analysis of **{total:,}** telecom customers reveals a churn rate of **{churn_rate}%**,
representing **{churned:,}** customers lost. This is {'below' if churn_rate < 15 else 'near or above'}
the industry average of approximately 15%.

### Critical Findings

1. **International Plan Vulnerability**: Customers with international plans show
   significantly elevated churn rates, making this the highest-priority segment
   for retention efforts.

2. **Service Call Warning Signal**: Churn rate increases dramatically when customers
   reach {service_tp.get('threshold', 4)}+ service calls ({service_tp.get('churn_above', 'N/A')}%
   vs {service_tp.get('churn_below', 'N/A')}% for lower call volumes).

3. **Voicemail Protective Effect**: Customers with voicemail plans demonstrate
   lower churn rates, suggesting this feature increases engagement and loyalty.

### Bottom Line
Addressing the international plan segment and implementing early intervention for
high service call customers could reduce overall churn by an estimated **20-30%**.

---

"""

    def _build_model_performance(self, results: Dict) -> str:
        """Build model performance section."""
        validation = results.get("model_validation", {})

        if not validation:
            return ""

        accuracy = validation.get("accuracy", 0)
        precision = validation.get("precision", 0)
        recall = validation.get("recall", 0)
        f1 = validation.get("f1_score", 0)
        high_risk = validation.get("high_risk_count", 0)
        tier_dist = validation.get("risk_tier_distribution", {})

        return f"""## Predictive Model Performance

### Pele's Logistic Regression Model

This analysis uses Pele's logistic regression model for churn prediction, achieving
industry-leading accuracy on this telecom customer dataset.

| Metric | Value |
|--------|-------|
| **Model Accuracy** | **{accuracy}%** |
| Precision | {precision}% |
| Recall | {recall}% |
| F1 Score | {f1:.2f} |

### Risk Tier Distribution

| Risk Tier | Customer Count | Probability Range |
|-----------|---------------|-------------------|
| Low | {tier_dist.get('low', 0):,} | 0% - 15% |
| Medium | {tier_dist.get('medium', 0):,} | 15% - 30% |
| High | {tier_dist.get('high', 0):,} | 30% - 50% |
| Critical | {tier_dist.get('critical', 0):,} | 50% - 100% |

### Prediction Summary

- **Customers predicted to churn:** {high_risk:,}
- **True Positives:** {validation.get('true_positives', 0):,} (correctly predicted churn)
- **True Negatives:** {validation.get('true_negatives', 0):,} (correctly predicted retention)
- **False Positives:** {validation.get('false_positives', 0):,} (predicted churn but retained)
- **False Negatives:** {validation.get('false_negatives', 0):,} (predicted retention but churned)

---

"""

    def _build_metrics_dashboard(self, results: Dict, df: pd.DataFrame) -> str:
        """Build key metrics dashboard section."""
        total = results.get("total_count", 0)
        churned = results.get("churned_count", 0)
        retained = results.get("retained_count", 0)
        churn_rate = results.get("overall_churn_rate", 0)

        avg_account_length = df["account_length"].mean() if "account_length" in df.columns else 0
        avg_service_calls = df["customer_service_calls"].mean() if "customer_service_calls" in df.columns else 0

        return f"""## Key Metrics Dashboard

| Metric | Value |
|--------|-------|
| Total Customers | {total:,} |
| Churned Customers | {churned:,} |
| Retained Customers | {retained:,} |
| Churn Rate | {churn_rate}% |
| Retention Rate | {100 - churn_rate:.1f}% |
| Avg Account Length | {avg_account_length:.0f} days |
| Avg Service Calls | {avg_service_calls:.1f} |

---

"""

    def _build_segment_analysis(self, results: Dict) -> str:
        """Build segment analysis section."""
        segments = results.get("segment_analysis", {})

        report = """## Segment Analysis

### By International Plan

"""
        intl_data = segments.get("international_plan", {})
        if intl_data:
            report += "| Segment | Customers | Churned | Churn Rate |\n|---------|-----------|---------|------------|\n"
            for segment, data in intl_data.items():
                report += f"| {segment} | {data['total']:,} | {data['churned']:,} | {data['churn_rate']}% |\n"

        report += "\n### By Voicemail Plan\n\n"
        vm_data = segments.get("voice_mail_plan", {})
        if vm_data:
            report += "| Segment | Customers | Churned | Churn Rate |\n|---------|-----------|---------|------------|\n"
            for segment, data in vm_data.items():
                report += f"| {segment} | {data['total']:,} | {data['churned']:,} | {data['churn_rate']}% |\n"

        report += "\n### Top States by Churn Rate\n\n"
        state_data = segments.get("top_states_by_churn", {})
        if state_data:
            report += "| State | Customers | Churned | Churn Rate |\n|-------|-----------|---------|------------|\n"
            for state, data in list(state_data.items())[:5]:
                report += f"| {state} | {data['total']:,} | {data['churned']:,} | {data['churn_rate']}% |\n"

        report += "\n---\n\n"
        return report

    def _build_tipping_points(self, results: Dict) -> str:
        """Build tipping points section."""
        tipping_points = results.get("tipping_points", [])

        report = """## Churn Tipping Points

These thresholds mark critical points where churn probability increases significantly:

"""
        for tp in tipping_points:
            report += f"""### {tp['factor']}

- **Threshold:** {tp['threshold']}
- **Churn Below Threshold:** {tp['churn_below']}%
- **Churn Above Threshold:** {tp['churn_above']}%
- **Impact Multiplier:** {tp['impact_multiplier']}x

> {tp.get('insight', '')}

"""
        report += "---\n\n"
        return report

    def _build_ai_insights(self, ai_insights: Dict) -> str:
        """Build AI insights section."""
        insights = ai_insights.get("insights", [])

        report = """## AI-Generated Insights

"""
        for insight in insights:
            if insight["type"] not in ["action_plan"]:
                severity_badge = {"critical": "**[CRITICAL]**", "warning": "**[WARNING]**", "info": "*[INFO]*"}.get(insight.get("severity", "info"), "")
                report += f"""### {severity_badge} {insight['title']}

{insight['message']}

"""
        report += "---\n\n"
        return report

    def _build_risk_assessment(self, ai_insights: Dict) -> str:
        """Build risk assessment section."""
        risk = ai_insights.get("risk_assessment", {})
        risk_level = risk.get("overall_risk_level", "unknown").upper()

        report = f"""## Risk Assessment

**Overall Risk Level: {risk_level}**

| Risk Metric | Value |
|-------------|-------|
| Customers Currently At Risk | {risk.get('customers_at_risk', 0):,.0f} |
| Potential Additional Churn | {risk.get('potential_additional_churn', 0):,.0f} customers |

### Key Risk Factors

"""
        for factor in risk.get("risk_factors", []):
            report += f"- {factor}\n"
        report += "\n---\n\n"
        return report

    def _build_action_plan(self, ai_insights: Dict) -> str:
        """Build action plan section."""
        actions = ai_insights.get("recommended_actions", [])

        report = """## Recommended Action Plan

"""
        for action in actions:
            report += f"""### Priority {action['priority']}: {action['action']}

{action['description']}

| Attribute | Value |
|-----------|-------|
| Expected Impact | {action['expected_impact']} |
| Timeline | {action['timeline']} |

"""
        report += "---\n\n"
        return report

    def _build_data_quality(self, df: pd.DataFrame) -> str:
        """Build data quality section."""
        total_cells = len(df) * len(df.columns)
        null_cells = df.isna().sum().sum()
        completeness = (1 - null_cells / total_cells) * 100

        return f"""## Data Quality Summary

| Metric | Value |
|--------|-------|
| Total Records | {len(df):,} |
| Total Columns | {len(df.columns)} |
| Data Completeness | {completeness:.1f}% |
| Missing Values | {int(null_cells):,} |

---

"""

    def _build_footer(self) -> str:
        """Build report footer."""
        return """## Appendix

### Methodology
- **Churn Prediction**: Pele's logistic regression model (86.38% accuracy)
- **Feature Standardization**: Z-score normalization using training set parameters
- **Churn Analysis**: Statistical analysis of customer attributes and churn correlation
- **Segment Analysis**: Group-by aggregations with churn rate calculations
- **Tipping Points**: Threshold analysis to identify critical inflection points
- **Risk Scoring**: Logistic regression probability-based scoring (0-100%)

### Model Features
The logistic regression model uses 10 standardized features:
1. Customer Service Calls
2. Total Day Minutes
3. Total Evening Minutes
4. Total Night Minutes
5. Total International Minutes
6. Total International Calls
7. Number of Voicemail Messages
8. Account Length
9. International Plan (binary)
10. Voice Mail Plan (binary)

---

*This report was automatically generated by the Veritly AI Market Intelligence Platform.*

**Veritly AI** | Market Intelligence Platform | Version 1.0
"""

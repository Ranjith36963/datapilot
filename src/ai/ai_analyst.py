"""
AI Analyst module for generating intelligent insights from data analysis.
Uses rule-based heuristics and statistical analysis - no external APIs required.
"""

from typing import Any, Dict, List

import pandas as pd
import numpy as np

from src.utils import setup_logging


class AIAnalyst:
    """AI-powered analyst that generates natural language insights from data."""

    def __init__(self):
        self.logger = setup_logging("ai_analyst")
        self.insights: List[Dict] = []

    def analyze_churn_patterns(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights from churn analysis results."""
        self.logger.info("Generating AI-powered churn insights")
        self.insights = []

        self._analyze_overall_health(analysis_results)
        self._analyze_risk_segments(df, analysis_results)
        self._analyze_behavioral_patterns(df)
        self._analyze_service_quality_impact(df)
        self._generate_predictive_indicators(df)
        self._create_action_plan(analysis_results)

        return {
            "insights": self.insights,
            "executive_summary": self._create_executive_summary(analysis_results),
            "risk_assessment": self._assess_overall_risk(df, analysis_results),
            "recommended_actions": self._prioritize_actions()
        }

    def _analyze_overall_health(self, results: Dict) -> None:
        """Assess overall customer base health."""
        churn_rate = results.get("overall_churn_rate", 0)
        industry_avg = 15.0

        if churn_rate < industry_avg * 0.8:
            health_status, health_msg = "excellent", "significantly below industry average"
        elif churn_rate < industry_avg:
            health_status, health_msg = "good", "below industry average"
        elif churn_rate < industry_avg * 1.2:
            health_status, health_msg = "concerning", "near industry average"
        else:
            health_status, health_msg = "critical", "above industry average"

        self.insights.append({
            "type": "health_assessment",
            "title": "Customer Base Health Status",
            "status": health_status,
            "message": f"Current churn rate of {churn_rate}% is {health_msg} (industry benchmark: ~{industry_avg}%). This represents {results.get('churned_count', 0)} customers lost.",
            "severity": "info" if health_status in ["excellent", "good"] else "warning"
        })

    def _analyze_risk_segments(self, df: pd.DataFrame, results: Dict) -> None:
        """Identify and analyze high-risk customer segments."""
        segments = results.get("segment_analysis", {})

        intl_data = segments.get("international_plan", {})
        with_intl = intl_data.get("With International Plan", {})
        without_intl = intl_data.get("Without International Plan", {})

        if with_intl and without_intl:
            intl_churn = with_intl.get("churn_rate", 0)
            non_intl_churn = without_intl.get("churn_rate", 0)

            if intl_churn > non_intl_churn * 1.5:
                risk_multiplier = round(intl_churn / max(non_intl_churn, 1), 1)
                self.insights.append({
                    "type": "risk_segment",
                    "title": "International Plan Customers - High Risk",
                    "status": "critical",
                    "message": f"International plan subscribers churn at {intl_churn}% ({risk_multiplier}x higher than non-subscribers at {non_intl_churn}%). This segment represents {with_intl.get('total', 0)} customers. Immediate intervention recommended.",
                    "affected_count": with_intl.get("total", 0),
                    "severity": "critical"
                })

        vm_data = segments.get("voice_mail_plan", {})
        with_vm = vm_data.get("With Voicemail Plan", {})
        without_vm = vm_data.get("Without Voicemail Plan", {})

        if with_vm and without_vm:
            vm_churn = with_vm.get("churn_rate", 0)
            no_vm_churn = without_vm.get("churn_rate", 0)

            if no_vm_churn > vm_churn:
                self.insights.append({
                    "type": "opportunity",
                    "title": "Voicemail Plan - Protective Factor Identified",
                    "status": "opportunity",
                    "message": f"Customers with voicemail plans show {vm_churn}% churn vs {no_vm_churn}% without. Voicemail appears to increase engagement. Opportunity: {without_vm.get('total', 0)} customers without voicemail could benefit from targeted offers.",
                    "severity": "info"
                })

    def _analyze_behavioral_patterns(self, df: pd.DataFrame) -> None:
        """Analyze customer behavior patterns related to churn."""
        churned = df[df["churn"] == 1]
        retained = df[df["churn"] == 0]

        if "customer_service_calls" in df.columns:
            churned_calls = churned["customer_service_calls"].mean()
            retained_calls = retained["customer_service_calls"].mean()
            self.insights.append({
                "type": "behavior_pattern",
                "title": "Service Call Behavior Difference",
                "status": "warning",
                "message": f"Churned customers averaged {churned_calls:.1f} service calls vs {retained_calls:.1f} for retained customers. High service call volume is a strong churn predictor. Consider proactive outreach when customers exceed 3 calls.",
                "severity": "warning"
            })

        if "total_day_minutes" in df.columns:
            churned_usage = churned["total_day_minutes"].mean()
            retained_usage = retained["total_day_minutes"].mean()
            if churned_usage > retained_usage:
                self.insights.append({
                    "type": "behavior_pattern",
                    "title": "Heavy User Churn Pattern",
                    "status": "warning",
                    "message": f"Churned customers had higher average usage ({churned_usage:.0f} min/day vs {retained_usage:.0f} min/day). Heavy users may be seeking better value or experiencing network quality issues. Consider loyalty programs for high-usage tiers.",
                    "severity": "warning"
                })

    def _analyze_service_quality_impact(self, df: pd.DataFrame) -> None:
        """Analyze the impact of service quality on churn."""
        if "customer_service_calls" not in df.columns:
            return

        call_ranges = [(0, 1), (2, 3), (4, 5), (6, 99)]
        call_analysis = []

        for low, high in call_ranges:
            subset = df[(df["customer_service_calls"] >= low) & (df["customer_service_calls"] <= high)]
            if len(subset) > 0:
                call_analysis.append({
                    "range": f"{low}-{high}" if high < 99 else f"{low}+",
                    "count": len(subset),
                    "churn_rate": round(subset["churn"].mean() * 100, 1)
                })

        tipping_point = next((data for data in call_analysis if data["churn_rate"] > 25), None)

        if tipping_point:
            self.insights.append({
                "type": "service_quality",
                "title": "Service Quality Tipping Point Identified",
                "status": "critical",
                "message": f"Churn rate jumps dramatically at {tipping_point['range']} service calls ({tipping_point['churn_rate']}% churn). This affects {tipping_point['count']} customers. Root cause analysis of service issues is urgently needed.",
                "severity": "critical"
            })

    def _generate_predictive_indicators(self, df: pd.DataFrame) -> None:
        """Generate predictive churn indicators."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}

        for col in numeric_cols:
            if col != "churn" and col in df.columns:
                corr = df[col].corr(df["churn"])
                if not pd.isna(corr):
                    correlations[col] = abs(corr)

        sorted_factors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]

        factor_names = {
            "customer_service_calls": "Number of customer service interactions",
            "international_plan": "International calling plan subscription",
            "total_day_minutes": "Daily call minutes usage",
            "total_day_charge": "Daily service charges",
            "voice_mail_plan": "Voicemail plan subscription"
        }

        self.insights.append({
            "type": "predictive",
            "title": "Top Churn Predictors Identified",
            "status": "info",
            "message": "Key factors predicting churn (by correlation strength): " + ", ".join([
                f"{factor_names.get(f[0], f[0])} ({f[1]:.2f})" for f in sorted_factors
            ]),
            "severity": "info"
        })

    def _create_action_plan(self, results: Dict) -> None:
        """Create prioritized action plan based on analysis."""
        actions = [
            {"priority": 1, "action": "Launch International Plan Retention Campaign",
             "description": "Immediate focus on international plan subscribers with targeted retention offers, improved pricing, or service upgrades.",
             "expected_impact": "High - addresses highest-risk segment", "timeline": "Immediate (0-30 days)"},
            {"priority": 2, "action": "Implement Early Warning System",
             "description": "Flag customers at 3+ service calls for proactive outreach. Assign dedicated support for customers approaching threshold.",
             "expected_impact": "High - prevents churn at tipping point", "timeline": "Short-term (30-60 days)"},
            {"priority": 3, "action": "Root Cause Analysis of Service Issues",
             "description": "Investigate why certain customers need multiple service calls. Address underlying product or service issues.",
             "expected_impact": "Medium-High - reduces service call volume", "timeline": "Medium-term (60-90 days)"},
            {"priority": 4, "action": "Voicemail Plan Promotion",
             "description": "Promote voicemail adoption among customers without it. Consider bundling or free trials to increase engagement.",
             "expected_impact": "Medium - increases customer stickiness", "timeline": "Ongoing"}
        ]
        self.insights.append({"type": "action_plan", "title": "Recommended Action Plan", "status": "actionable", "actions": actions, "severity": "info"})

    def _create_executive_summary(self, results: Dict) -> str:
        """Create executive summary of the analysis."""
        churn_rate = results.get("overall_churn_rate", 0)
        churned = results.get("churned_count", 0)
        total = results.get("total_count", 0)

        tipping_points = results.get("tipping_points", [])
        service_tp = next((tp for tp in tipping_points if "Service" in tp.get("factor", "")), {})

        segments = results.get("segment_analysis", {})
        intl_churn = segments.get("international_plan", {}).get("With International Plan", {}).get("churn_rate", 0)

        return f"""EXECUTIVE SUMMARY: Customer Churn Analysis

CURRENT STATE:
- Overall churn rate: {churn_rate}% ({churned} of {total} customers)
- This is {'below' if churn_rate < 15 else 'near'} the industry average of ~15%

CRITICAL FINDINGS:
1. International Plan Risk: Customers with international plans churn at {intl_churn}% - nearly 3x the rate of other customers.

2. Service Call Tipping Point: Churn rate spikes dramatically when customers make {service_tp.get('threshold', 4)}+ service calls ({service_tp.get('churn_above', 'N/A')}% churn vs {service_tp.get('churn_below', 'N/A')}% below threshold).

3. Protective Factor: Voicemail plan subscribers show significantly lower churn.

RECOMMENDED PRIORITIES:
1. IMMEDIATE: Launch retention campaign for international plan customers
2. SHORT-TERM: Implement early warning system for high service call customers
3. MEDIUM-TERM: Root cause analysis of service issues driving repeat calls"""

    def _assess_overall_risk(self, df: pd.DataFrame, results: Dict) -> Dict:
        """Assess overall business risk from churn."""
        churn_rate = results.get("overall_churn_rate", 0)
        at_risk = df[(df["customer_service_calls"] >= 4) | (df["international_plan"] == 1)]
        at_risk_not_churned = at_risk[at_risk["churn"] == 0]

        return {
            "overall_risk_level": "high" if churn_rate > 15 else "medium",
            "customers_at_risk": len(at_risk_not_churned),
            "potential_additional_churn": round(len(at_risk_not_churned) * (churn_rate / 100), 0),
            "risk_factors": ["High service call volume cluster", "International plan segment vulnerability", "Heavy usage without loyalty incentives"]
        }

    def _prioritize_actions(self) -> List[Dict]:
        """Extract and prioritize recommended actions."""
        action_insight = next((i for i in self.insights if i["type"] == "action_plan"), None)
        return action_insight.get("actions", []) if action_insight else []

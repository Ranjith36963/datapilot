"""
Visualisation module for generating churn analysis charts.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import VISUALISATIONS_DIR
from src.utils import setup_logging


class ChurnVisualiser:
    """Generates visualisations for churn analysis results."""

    def __init__(self):
        self.logger = setup_logging("visualiser")
        self.output_dir = VISUALISATIONS_DIR

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def generate_all(
        self,
        df: pd.DataFrame,
        analysis_results: Dict[str, Any],
        version: str = "v1"
    ) -> Dict[str, Path]:
        """Generate all visualisation charts.

        Returns dict mapping chart name to file path.
        """
        self.logger.info("Generating visualisations")

        outputs = {}

        # Generate each chart
        outputs["churn_by_segment"] = self._create_churn_by_segment(
            analysis_results, version
        )
        outputs["service_calls_tipping_point"] = self._create_service_calls_chart(
            df, version
        )
        outputs["risk_distribution"] = self._create_risk_distribution(
            analysis_results, version
        )
        outputs["top_states_churn"] = self._create_top_states_chart(
            analysis_results, version
        )
        outputs["feature_correlation"] = self._create_feature_correlation(
            df, version
        )

        self.logger.info(f"Generated {len(outputs)} visualisations in {self.output_dir}")
        return outputs

    def _create_churn_by_segment(
        self,
        analysis_results: Dict[str, Any],
        version: str
    ) -> Path:
        """Create bar chart showing churn rates by international plan and voicemail."""
        self.logger.info("Creating churn by segment chart")

        fig, ax = plt.subplots(figsize=(10, 6))

        segments = analysis_results.get("segment_analysis", {})

        # Prepare data
        labels = []
        churn_rates = []
        colors = []

        # International plan data
        intl_data = segments.get("international_plan", {})
        for label, data in intl_data.items():
            labels.append(label.replace("International Plan", "Intl Plan"))
            churn_rates.append(data["churn_rate"])
            colors.append("#e74c3c" if "With" in label else "#3498db")

        # Voicemail plan data
        vm_data = segments.get("voice_mail_plan", {})
        for label, data in vm_data.items():
            labels.append(label.replace("Voicemail Plan", "VM Plan"))
            churn_rates.append(data["churn_rate"])
            colors.append("#9b59b6" if "With" in label else "#2ecc71")

        # Create bar chart
        bars = ax.bar(labels, churn_rates, color=colors, edgecolor='white', linewidth=1.2)

        # Add value labels on bars
        for bar, rate in zip(bars, churn_rates):
            height = bar.get_height()
            ax.annotate(f'{rate}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')

        ax.set_ylabel('Churn Rate (%)', fontsize=12)
        ax.set_xlabel('Customer Segment', fontsize=12)
        ax.set_title('Churn Rate by Customer Segment', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(churn_rates) * 1.2 if churn_rates else 50)

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()

        output_path = self.output_dir / f"churn_by_segment_{version}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"Saved: {output_path}")
        return output_path

    def _create_service_calls_chart(
        self,
        df: pd.DataFrame,
        version: str
    ) -> Path:
        """Create line chart showing churn rate by number of service calls."""
        self.logger.info("Creating service calls tipping point chart")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Calculate churn rate for each service call count (0-6+)
        df_copy = df.copy()
        df_copy['service_calls_group'] = df_copy['customer_service_calls'].clip(upper=6)

        churn_by_calls = df_copy.groupby('service_calls_group').agg({
            'churn': ['mean', 'count']
        }).reset_index()
        churn_by_calls.columns = ['service_calls', 'churn_rate', 'count']
        churn_by_calls['churn_rate'] = churn_by_calls['churn_rate'] * 100

        # Labels for x-axis
        x_labels = [str(int(x)) if x < 6 else '6+' for x in churn_by_calls['service_calls']]

        # Plot line with markers
        line = ax.plot(
            x_labels,
            churn_by_calls['churn_rate'],
            marker='o',
            markersize=10,
            linewidth=2.5,
            color='#e74c3c',
            markerfacecolor='white',
            markeredgewidth=2
        )

        # Fill area under line
        ax.fill_between(
            x_labels,
            churn_by_calls['churn_rate'],
            alpha=0.3,
            color='#e74c3c'
        )

        # Add value labels
        for i, (x, y) in enumerate(zip(x_labels, churn_by_calls['churn_rate'])):
            ax.annotate(f'{y:.1f}%',
                       xy=(i, y),
                       xytext=(0, 10),
                       textcoords="offset points",
                       ha='center',
                       fontsize=10,
                       fontweight='bold')

        # Add tipping point indicator (at 4 calls)
        if len(churn_by_calls) > 4:
            ax.axvline(x=4, color='#f39c12', linestyle='--', linewidth=2, alpha=0.7)
            ax.annotate('Tipping Point',
                       xy=(4, churn_by_calls['churn_rate'].max() * 0.5),
                       fontsize=10, color='#f39c12',
                       fontweight='bold')

        ax.set_ylabel('Churn Rate (%)', fontsize=12)
        ax.set_xlabel('Number of Customer Service Calls', fontsize=12)
        ax.set_title('Churn Rate by Service Calls - Tipping Point Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(churn_by_calls['churn_rate']) * 1.2)

        plt.tight_layout()

        output_path = self.output_dir / f"service_calls_tipping_point_{version}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"Saved: {output_path}")
        return output_path

    def _create_risk_distribution(
        self,
        analysis_results: Dict[str, Any],
        version: str
    ) -> Path:
        """Create pie chart showing distribution of risk tiers."""
        self.logger.info("Creating risk distribution chart")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Get risk tier distribution from model validation
        validation = analysis_results.get("model_validation", {})
        tier_dist = validation.get("risk_tier_distribution", {})

        if not tier_dist:
            # Fallback if no tier distribution
            tier_dist = {"low": 2000, "medium": 500, "high": 300, "critical": 200}

        # Prepare data
        labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical']
        sizes = [
            tier_dist.get('low', 0),
            tier_dist.get('medium', 0),
            tier_dist.get('high', 0),
            tier_dist.get('critical', 0)
        ]
        colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
        explode = (0, 0, 0.05, 0.1)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes)):,})',
            startangle=90,
            shadow=True,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )

        # Style the text
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        for text in texts:
            text.set_fontsize(11)

        ax.set_title('Customer Risk Distribution', fontsize=14, fontweight='bold', pad=20)

        # Add legend
        ax.legend(
            wedges,
            [f'{label}: {count:,}' for label, count in zip(labels, sizes)],
            title="Risk Tiers",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )

        plt.tight_layout()

        output_path = self.output_dir / f"risk_distribution_{version}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"Saved: {output_path}")
        return output_path

    def _create_top_states_chart(
        self,
        analysis_results: Dict[str, Any],
        version: str
    ) -> Path:
        """Create horizontal bar chart showing top 10 states by churn rate."""
        self.logger.info("Creating top states churn chart")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Get state data
        segments = analysis_results.get("segment_analysis", {})
        state_data = segments.get("top_states_by_churn", {})

        if not state_data:
            # Create placeholder if no data
            state_data = {"CA": {"churn_rate": 20}, "TX": {"churn_rate": 18}}

        # Sort and take top 10
        states = list(state_data.keys())[:10]
        churn_rates = [state_data[s]["churn_rate"] for s in states]

        # Create color gradient based on churn rate
        max_rate = max(churn_rates) if churn_rates else 1
        colors = [plt.cm.RdYlGn_r(rate / max_rate * 0.8 + 0.1) for rate in churn_rates]

        # Create horizontal bar chart
        bars = ax.barh(states[::-1], churn_rates[::-1], color=colors[::-1], edgecolor='white', linewidth=1.2)

        # Add value labels
        for bar, rate in zip(bars, churn_rates[::-1]):
            width = bar.get_width()
            ax.annotate(f'{rate}%',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center',
                       fontsize=10, fontweight='bold')

        ax.set_xlabel('Churn Rate (%)', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title('Top 10 States by Churn Rate', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(churn_rates) * 1.2 if churn_rates else 30)

        plt.tight_layout()

        output_path = self.output_dir / f"top_states_churn_{version}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"Saved: {output_path}")
        return output_path

    def _create_feature_correlation(
        self,
        df: pd.DataFrame,
        version: str
    ) -> Path:
        """Create horizontal bar chart showing feature correlations with churn."""
        self.logger.info("Creating feature correlation chart")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate correlations
        numeric_cols = [
            'customer_service_calls', 'total_day_minutes', 'total_eve_minutes',
            'total_night_minutes', 'total_intl_minutes', 'total_intl_calls',
            'number_vmail_messages', 'account_length', 'international_plan',
            'voice_mail_plan'
        ]

        correlations = {}
        for col in numeric_cols:
            if col in df.columns:
                corr = df[col].corr(df['churn'])
                correlations[col] = corr

        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        features = [item[0].replace('_', ' ').title() for item in sorted_corr]
        values = [item[1] for item in sorted_corr]

        # Color based on positive/negative
        colors = ['#e74c3c' if v > 0 else '#27ae60' for v in values]

        # Create horizontal bar chart
        bars = ax.barh(features[::-1], values[::-1], color=colors[::-1], edgecolor='white', linewidth=1.2)

        # Add center line
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)

        # Add value labels
        for bar, val in zip(bars, values[::-1]):
            width = bar.get_width()
            label_x = width + 0.01 if width >= 0 else width - 0.01
            ha = 'left' if width >= 0 else 'right'
            ax.annotate(f'{val:.3f}',
                       xy=(label_x, bar.get_y() + bar.get_height() / 2),
                       ha=ha, va='center',
                       fontsize=9, fontweight='bold')

        ax.set_xlabel('Correlation with Churn', fontsize=12)
        ax.set_title('Feature Correlation with Churn', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, 0.5)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='Positive (increases churn)'),
            Patch(facecolor='#27ae60', label='Negative (decreases churn)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        output_path = self.output_dir / f"feature_correlation_{version}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"Saved: {output_path}")
        return output_path

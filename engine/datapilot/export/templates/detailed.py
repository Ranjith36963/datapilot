"""
Detailed report template configuration.

Defines the structure and sections for comprehensive technical reports.
"""



DETAILED_TEMPLATE = {
    "sections": [
        {
            "id": "title",
            "type": "title_page",
            "fields": ["title", "subtitle", "date", "brand_name"],
        },
        {
            "id": "executive_summary",
            "type": "text",
            "heading": "Executive Summary",
        },
        {
            "id": "data_overview",
            "type": "table",
            "heading": "Data Overview",
            "fields": ["rows", "columns", "quality_score", "missing_pct"],
        },
        {
            "id": "data_quality",
            "type": "checks_table",
            "heading": "Data Quality Assessment",
        },
        {
            "id": "descriptive_stats",
            "type": "stats_table",
            "heading": "Descriptive Statistics",
        },
        {
            "id": "correlations",
            "type": "heatmap",
            "heading": "Correlation Analysis",
        },
        {
            "id": "model_performance",
            "type": "metrics_grid",
            "heading": "Model Performance",
        },
        {
            "id": "feature_importance",
            "type": "bar_chart",
            "heading": "Feature Importance",
            "max_items": 15,
        },
        {
            "id": "visualisations",
            "type": "charts",
            "heading": "Visualisations",
        },
        {
            "id": "findings",
            "type": "findings_list",
            "heading": "Detailed Findings",
        },
        {
            "id": "recommendations",
            "type": "recommendations_list",
            "heading": "Recommendations",
        },
        {
            "id": "methodology",
            "type": "text",
            "heading": "Methodology",
        },
    ],
    "style": {
        "primary_color": "#2c3e50",
        "secondary_color": "#3498db",
        "accent_color": "#e74c3c",
        "success_color": "#27ae60",
        "warning_color": "#f1c40f",
        "font_family": "Helvetica",
        "title_size": 24,
        "heading_size": 14,
        "body_size": 10,
    },
}


def get_detailed_template() -> dict:
    """Return the detailed report template configuration."""
    return DETAILED_TEMPLATE.copy()

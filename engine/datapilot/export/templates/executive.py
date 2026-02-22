"""
Executive report template configuration.

Defines the structure and sections for executive-level reports.
"""



EXECUTIVE_TEMPLATE = {
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
            "fields": ["summary_text", "key_metrics"],
        },
        {
            "id": "key_metrics",
            "type": "metrics_grid",
            "heading": "Key Metrics",
            "max_items": 4,
        },
        {
            "id": "key_findings",
            "type": "bullet_list",
            "heading": "Key Findings",
            "max_items": 5,
        },
        {
            "id": "visualisations",
            "type": "charts",
            "heading": "Analysis Highlights",
            "max_items": 3,
        },
        {
            "id": "recommendations",
            "type": "bullet_list",
            "heading": "Recommendations",
            "max_items": 5,
        },
    ],
    "style": {
        "primary_color": "#3498db",
        "secondary_color": "#2c3e50",
        "accent_color": "#e74c3c",
        "font_family": "Helvetica",
        "title_size": 24,
        "heading_size": 16,
        "body_size": 11,
    },
}


def get_executive_template() -> dict:
    """Return the executive report template configuration."""
    return EXECUTIVE_TEMPLATE.copy()

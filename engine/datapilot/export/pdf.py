"""
PDF export — generate PDF reports from analysis results.

Uses reportlab. All text is configurable via parameters (no hardcoded brand/domain).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd


def export_to_pdf(
    analysis_results: Dict[str, Any],
    output_path: str,
    title: str = "Data Analysis Report",
    subtitle: str = "Comprehensive Analysis",
    brand_name: str = "DataPilot",
    visualisation_paths: Optional[Dict[str, Path]] = None,
    metrics: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Export analysis results to PDF format using reportlab.

    Args:
        analysis_results: Dict with analysis output (summary, sections, key_points, metrics).
        output_path: Where to save the PDF.
        title: Report title.
        subtitle: Report subtitle.
        brand_name: Brand name for footer.
        visualisation_paths: Dict of {name: path} for chart images.
        metrics: List of {"label": ..., "value": ...} dicts for key metrics table.

    Returns:
        The output file path.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, ListFlowable, ListItem,
    )
    from reportlab.lib.enums import TA_CENTER

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(p),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Title2',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
    ))
    styles.add(ParagraphStyle(
        name='SectionBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        leading=14,
    ))

    story = []

    # Title page
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(title, styles['Title2']))
    story.append(Paragraph(subtitle, styles['Subtitle']))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles['Subtitle'],
    ))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    story.append(Spacer(1, 12))

    summary_text = analysis_results.get("summary", "Analysis complete. See details below.")
    if isinstance(summary_text, str):
        story.append(Paragraph(summary_text, styles['SectionBody']))
    story.append(Spacer(1, 24))

    # Key Metrics Table
    report_metrics = metrics or analysis_results.get("metrics")
    if report_metrics:
        story.append(Paragraph("Key Metrics", styles['Heading2']))
        story.append(Spacer(1, 12))

        metrics_data = [["Metric", "Value"]]
        for m in report_metrics:
            metrics_data.append([m.get("label", ""), str(m.get("value", ""))])

        metrics_table = Table(metrics_data, colWidths=[3 * inch, 2 * inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 24))

    # Analysis Sections
    sections = analysis_results.get("sections", [])
    if sections:
        story.append(PageBreak())
        story.append(Paragraph("Detailed Analysis", styles['Heading1']))
        story.append(Spacer(1, 12))

        for section in sections:
            heading = section.get("heading", "Analysis")
            story.append(Paragraph(heading, styles['Heading2']))
            story.append(Spacer(1, 6))

            question = section.get("question")
            if question:
                story.append(Paragraph(
                    f"<i>Question: {question}</i>",
                    styles['SectionBody'],
                ))
                story.append(Spacer(1, 4))

            narrative = section.get("narrative", "")
            if narrative:
                story.append(Paragraph(narrative, styles['SectionBody']))
                story.append(Spacer(1, 6))

            key_points = section.get("key_points", [])
            if key_points:
                items = [
                    ListItem(Paragraph(point, styles['SectionBody']))
                    for point in key_points
                ]
                story.append(ListFlowable(items, bulletType='bullet', start=''))
                story.append(Spacer(1, 12))

            story.append(Spacer(1, 12))

    # Visualisations
    if visualisation_paths:
        story.append(PageBreak())
        story.append(Paragraph("Analysis Visualisations", styles['Heading1']))
        story.append(Spacer(1, 12))

        for chart_name, chart_path in visualisation_paths.items():
            cp = Path(chart_path) if not isinstance(chart_path, Path) else chart_path
            if cp.exists():
                story.append(Paragraph(
                    chart_name.replace('_', ' ').title(),
                    styles['Heading3'],
                ))
                story.append(Spacer(1, 6))
                img = Image(str(cp), width=5.5 * inch, height=3.5 * inch)
                story.append(img)
                story.append(Spacer(1, 24))

    # Footer
    story.append(Spacer(1, inch))
    story.append(Paragraph(
        f"Generated by {brand_name}",
        styles['Subtitle'],
    ))

    doc.build(story)
    return str(p)

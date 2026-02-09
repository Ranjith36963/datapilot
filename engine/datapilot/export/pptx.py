"""
PPTX export — generate PowerPoint presentations from analysis results.

Uses python-pptx. All text is configurable via parameters.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd


def export_to_pptx(
    analysis_results: Dict[str, Any],
    output_path: str,
    title: str = "Data Analysis Report",
    subtitle: str = "Comprehensive Analysis",
    brand_name: str = "DataPilot",
    visualisation_paths: Optional[Dict[str, Path]] = None,
    metrics: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Export analysis results to PowerPoint presentation.

    Args:
        analysis_results: Dict with analysis output.
        output_path: Where to save the PPTX.
        title: Report title.
        subtitle: Report subtitle.
        brand_name: Brand name for closing slide.
        visualisation_paths: Dict of {name: path} for chart images.
        metrics: List of {"label": ..., "value": ...} dicts for key metrics.

    Returns:
        The output file path.
    """
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide_layout = prs.slide_layouts[6]  # Blank

    # --- Title Slide ---
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.333), Inches(1.5))
    tf = title_box.text_frame
    pg = tf.paragraphs[0]
    pg.text = title
    pg.font.size = Pt(44)
    pg.font.bold = True
    pg.font.color.rgb = RGBColor(44, 62, 80)
    pg.alignment = PP_ALIGN.CENTER

    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4), Inches(12.333), Inches(1))
    tf = subtitle_box.text_frame
    pg = tf.paragraphs[0]
    pg.text = subtitle
    pg.font.size = Pt(28)
    pg.font.color.rgb = RGBColor(127, 140, 141)
    pg.alignment = PP_ALIGN.CENTER

    date_box = slide.shapes.add_textbox(Inches(0.5), Inches(5), Inches(12.333), Inches(0.5))
    tf = date_box.text_frame
    pg = tf.paragraphs[0]
    pg.text = datetime.now().strftime('%Y-%m-%d')
    pg.font.size = Pt(18)
    pg.font.color.rgb = RGBColor(149, 165, 166)
    pg.alignment = PP_ALIGN.CENTER

    # --- Metrics Slide ---
    if metrics:
        slide = prs.slides.add_slide(slide_layout)

        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
        tf = title_box.text_frame
        pg = tf.paragraphs[0]
        pg.text = "Key Metrics"
        pg.font.size = Pt(36)
        pg.font.bold = True
        pg.font.color.rgb = RGBColor(44, 62, 80)

        colors_list = [
            RGBColor(52, 152, 219),
            RGBColor(231, 76, 60),
            RGBColor(230, 126, 34),
            RGBColor(46, 204, 113),
        ]

        for i, m in enumerate(metrics[:4]):
            left = Inches(0.5 + i * 3.2)
            top = Inches(1.5)
            color = colors_list[i % len(colors_list)]

            shape = slide.shapes.add_shape(1, left, top, Inches(2.9), Inches(1.8))
            shape.fill.solid()
            shape.fill.fore_color.rgb = color
            shape.line.fill.background()

            tf = shape.text_frame
            tf.clear()
            pg = tf.paragraphs[0]
            pg.text = str(m.get("value", ""))
            pg.font.size = Pt(40)
            pg.font.bold = True
            pg.font.color.rgb = RGBColor(255, 255, 255)
            pg.alignment = PP_ALIGN.CENTER

            pg = tf.add_paragraph()
            pg.text = m.get("label", "")
            pg.font.size = Pt(16)
            pg.font.color.rgb = RGBColor(255, 255, 255)
            pg.alignment = PP_ALIGN.CENTER

    # --- Visualisation Slides ---
    if visualisation_paths:
        for chart_name, chart_path in visualisation_paths.items():
            cp = Path(chart_path) if not isinstance(chart_path, Path) else chart_path
            if cp.exists():
                slide = prs.slides.add_slide(slide_layout)

                title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.8))
                tf = title_box.text_frame
                pg = tf.paragraphs[0]
                pg.text = chart_name.replace('_', ' ').title()
                pg.font.size = Pt(32)
                pg.font.bold = True
                pg.font.color.rgb = RGBColor(44, 62, 80)

                slide.shapes.add_picture(
                    str(cp),
                    Inches(1.5), Inches(1.3),
                    width=Inches(10), height=Inches(5.5),
                )

    # --- Closing Slide ---
    slide = prs.slides.add_slide(slide_layout)

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(3), Inches(12.333), Inches(1.5))
    tf = title_box.text_frame
    pg = tf.paragraphs[0]
    pg.text = brand_name
    pg.font.size = Pt(44)
    pg.font.bold = True
    pg.font.color.rgb = RGBColor(44, 62, 80)
    pg.alignment = PP_ALIGN.CENTER

    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(12.333), Inches(1))
    tf = subtitle_box.text_frame
    pg = tf.paragraphs[0]
    pg.text = "AI-Powered Data Analysis"
    pg.font.size = Pt(24)
    pg.font.color.rgb = RGBColor(127, 140, 141)
    pg.alignment = PP_ALIGN.CENTER

    prs.save(str(p))
    return str(p)

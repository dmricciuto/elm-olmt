#!/usr/bin/env python3
"""Assemble standard SPRUCE diagnostic figures into a shareable PDF report."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage


PAGE_SIZE = landscape(letter)
FIGURES = (
    (
        "McFarlane peat profile and radiocarbon comparison",
        "mcfarlane2018/spruce_mcfarlane_profile_comparison.png",
        "Depth-resolved peat carbon density, bulk C:N, Delta14C, and cumulative peat carbon. "
        "Hollow and hummock are shown separately and as their area-weighted bog mean; the fen is excluded.",
    ),
    (
        "Hollow water table",
        "hollow_water_table_vs_observations.png",
        "Daily hollow water-table height is -ZWT + H2OSFC/1000. Gray periods contain soil ice and "
        "are excluded from the reported R2, RMSE, mean, and bias statistics.",
    ),
    (
        "Tree aboveground NPP",
        "carbon_budget_agnpp_tree_timeseries.png",
        "Annual tree aboveground NPP for the six standard non-eCO2 enclosures. Model values are on a "
        "hummock+hollow area basis.",
    ),
    (
        "Shrub aboveground NPP",
        "carbon_budget_agnpp_shrub_timeseries.png",
        "Annual shrub aboveground NPP for the six standard non-eCO2 enclosures.",
    ),
    (
        "Woody belowground NPP",
        "carbon_budget_bgnpp_woody_timeseries.png",
        "Annual combined tree and shrub belowground NPP.",
    ),
    (
        "Moss NPP",
        "carbon_budget_npp_moss_timeseries.png",
        "Annual moss NPP. The modeled value uses peatland moss PFT 18 and excludes the auxiliary fen.",
    ),
    (
        "Heterotrophic respiration",
        "carbon_budget_hr_timeseries.png",
        "Annual heterotrophic respiration, area-weighted across hollow and hummock.",
    ),
    (
        "Carbon-budget treatment response",
        "carbon_budget_temperature_response.png",
        "Observed and modeled intercepts and slopes fitted across T0, T2.25, T4.50, T6.75, and T9. "
        "TAMB is shown in time series but excluded from these regressions.",
    ),
    (
        "Porewater DOM, acetate, and methane profiles",
        "porewater_dom_acetate_ch4_profiles.png",
        "Late-growing-season treatment means compared with 2013 SPRUCE profile constraints. Model "
        "profiles are hummock+hollow weighted means; the fen is excluded.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-prefix", required=True)
    parser.add_argument(
        "--title", default="ELM-Peatlands-Microbe: SPRUCE treatment diagnostics"
    )
    parser.add_argument(
        "--c14-note",
        default=(
            "C14 is prognostic from cold start with a constant pre-bomb atmospheric signature; "
            "the available post-2015 fossil-CO2 tracer stream was not used."
        ),
    )
    return parser.parse_args()


def _scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _header_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CCD5DB"))
    canvas.line(0.48 * inch, 0.40 * inch, 10.52 * inch, 0.40 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#53636D"))
    canvas.drawString(0.50 * inch, 0.22 * inch, "ELM-Peatlands-Microbe SPRUCE diagnostics")
    canvas.drawRightString(10.50 * inch, 0.22 * inch, f"Page {document.page}")
    canvas.restoreState()


def build_report(arguments: argparse.Namespace) -> None:
    missing = [relative for _, relative, _ in FIGURES if not (arguments.diagnostics_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing diagnostic figures: " + ", ".join(missing))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=25, leading=29, textColor=colors.HexColor("#173B4F"),
        alignment=TA_LEFT, spaceAfter=14,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=12, leading=17,
        textColor=colors.HexColor("#40545F"), spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, leading=19, textColor=colors.HexColor("#173B4F"),
        spaceAfter=8,
    )
    caption_style = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontSize=8.5, leading=11,
        textColor=colors.HexColor("#3E4D55"), alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=10.5, leading=15,
        textColor=colors.HexColor("#263238"), spaceAfter=8,
    )

    document = SimpleDocTemplate(
        str(arguments.output), pagesize=PAGE_SIZE,
        leftMargin=0.55 * inch, rightMargin=0.55 * inch,
        topMargin=0.55 * inch, bottomMargin=0.55 * inch,
        title=arguments.title,
        author="ELM-Peatlands-Microbe team",
        subject="SPRUCE treatment model diagnostics",
    )
    story = [
        Spacer(1, 0.55 * inch),
        Paragraph(arguments.title, title_style),
        Paragraph(
            "A reproducible evaluation package for peat hydrology, vegetation productivity, "
            "heterotrophic respiration, peat carbon and radiocarbon, and microbial porewater chemistry.",
            subtitle_style,
        ),
        Spacer(1, 0.15 * inch),
    ]
    metadata = [
        ["Run prefix", arguments.case_prefix],
        ["Report generated", date.today().isoformat()],
        ["Spinup design", "153-year accelerated decomposition + 153-year final spinup"],
        ["Historical simulation", "1850-2015"],
        ["Treatments", "TAMB, T0, T+2.25, T+4.5, T+6.75, T+9 C; no elevated CO2"],
        ["Soil heating", "T0-referenced control at 2 m; enabled from 2015-08-15 in warmed cases"],
        ["Microbial settings", "0.8 m decomposition e-folding; 0.5 anoxic DOM-solubilization fraction"],
        ["Radiocarbon", arguments.c14_note],
    ]
    table = Table(metadata, colWidths=[1.65 * inch, 7.9 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F0F3")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#263238")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C8CF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 0.20 * inch),
            Paragraph(
                "Interpret the diagnostics jointly. Matching methane flux alone is insufficient: "
                "water-table dynamics, plant productivity, peat stocks, DOM delivery, acetate, and "
                "porewater methane constrain different portions of the coupled system.",
                body_style,
            ),
            PageBreak(),
        ]
    )

    for index, (heading, relative, caption) in enumerate(FIGURES):
        path = arguments.diagnostics_dir / relative
        # Leave explicit room for the heading, caption, and footer.  A 6.55-inch
        # image can force a short caption onto its own page in landscape letter.
        image = _scaled_image(path, 9.95 * inch, 6.10 * inch)
        story.append(
            KeepTogether(
                [
                    Paragraph(heading, section_style),
                    image,
                    Spacer(1, 0.07 * inch),
                    Paragraph(caption, caption_style),
                ]
            )
        )
        if index != len(FIGURES) - 1:
            story.append(PageBreak())

    document.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()

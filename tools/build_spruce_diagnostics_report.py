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
        "Heterotrophic-derived CO2 efflux",
        "carbon_budget_hr_timeseries.png",
        "Annual heterotrophic-derived CO2 efflux (ELM HR plus microbe-methane CO2 surface "
        "flux), area-weighted across hollow and hummock; CH4 and root respiration are excluded.",
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
SOIL_TEMPERATURE_FIGURE = (
    "Seasonal soil temperature: model and observations",
    "soil_temperature_model_vs_observations.png",
    "Seasonal hollow-column temperatures compared with SPRUCE.032 v2 plot probes "
    "(DOI 10.3334/CDIAC/spruce.032). Solid lines are observations; dashed "
    "lines are ELM. The auxiliary fen is excluded.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-prefix", required=True)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Mark synthetic-year output as a workflow check, not scientific validation.",
    )
    parser.add_argument("--soil-temperature", action="store_true",
                        help="Include the generated seasonal TSOI model/observation page.")
    parser.add_argument("--soil-temperature-model-years")
    parser.add_argument("--soil-temperature-observation-years")
    parser.add_argument(
        "--title", default="ELM-Peatlands-Microbe: SPRUCE treatment diagnostics"
    )
    parser.add_argument(
        "--c14-note",
        default="Check case settings and atmospheric forcing before interpreting C14.",
    )
    parser.add_argument("--decomp-depth-efolding", type=float)
    parser.add_argument(
        "--dom-solubilization-anoxic-fraction", type=float
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
    figures = list(FIGURES)
    if arguments.soil_temperature:
        figures.insert(2, SOIL_TEMPERATURE_FIGURE)
    missing = [relative for _, relative, _ in figures
               if not (arguments.diagnostics_dir / relative).is_file()]
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
            ("Workflow smoke test using synthetic model years. Most pages are model-only; "
             "the soil-temperature page aligns a synthetic model year with its observational "
             "forcing analog solely to check the comparison workflow. No scientific skill is inferred."
             if arguments.smoke_test and arguments.soil_temperature else
             "Workflow smoke test using synthetic model years. This PDF verifies that "
             "all treatment diagnostics can be generated; it is not an observation comparison."
             if arguments.smoke_test else
             "A reproducible evaluation package for peat hydrology, vegetation productivity, "
             "heterotrophic respiration, peat carbon and radiocarbon, and microbial porewater chemistry."),
            subtitle_style,
        ),
        Spacer(1, 0.15 * inch),
    ]
    microbial_settings = []
    if arguments.decomp_depth_efolding is not None:
        microbial_settings.append(
            f"{arguments.decomp_depth_efolding:.3g} m peat decomposition e-folding"
        )
    if arguments.dom_solubilization_anoxic_fraction is not None:
        microbial_settings.append(
            f"{arguments.dom_solubilization_anoxic_fraction:.3g} anoxic "
            "DOM-solubilization fraction"
        )
    metadata = [
        ["Run prefix", arguments.case_prefix],
        ["Report generated", date.today().isoformat()],
        ["Mode", "Synthetic-year workflow smoke test" if arguments.smoke_test
         else "Scientific diagnostic report"],
        ["Treatments", "TAMB, T0, T+2.25, T+4.5, T+6.75, T+9 C; no elevated CO2"],
        [
            "Microbial settings",
            "; ".join(microbial_settings)
            if microbial_settings else "Not found in the case parameter file",
        ],
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

    for index, (heading, relative, caption) in enumerate(figures):
        if arguments.smoke_test:
            if index == 0:
                heading = "Peat profile workflow check"
                caption = (
                    "Synthetic-year T0 restart peat stocks; observed core points are contextual "
                    "only. C14 is not enabled, and no scientific skill is inferred."
                )
            elif relative == SOIL_TEMPERATURE_FIGURE[1]:
                caption = (
                    "Workflow comparison only: synthetic model year(s) "
                    f"{arguments.soil_temperature_model_years} are aligned to observed "
                    f"{arguments.soil_temperature_observation_years} plot temperatures "
                    "(SPRUCE.032 v2) through the CPL-bypass forcing cycle. The short "
                    "spinup is not scientifically equilibrated; this is not a skill assessment."
                )
            else:
                caption = (
                    "Synthetic-year workflow check; model output only. Observation-based "
                    "skill metrics and scientific interpretation are not valid."
                )
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
        if index != len(figures) - 1:
            story.append(PageBreak())

    document.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()

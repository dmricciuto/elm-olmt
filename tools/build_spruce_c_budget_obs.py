#!/usr/bin/env python3
"""Build OLMT calibration observations from the SPRUCE C Budget workbook."""

import argparse
import math
import re
from pathlib import Path

import pandas as pd


TREATMENT_PLOTS = {
    "TAMB": "P07",
    "T0.00": "P06",
    "T2.25": "P20",
    "T4.50": "P13",
    "T6.75": "P08",
    "T9.00": "P17",
}

TREATMENT_OFFSETS = {
    "TAMB": 0.0,
    "T0.00": 0.0,
    "T2.25": 2.25,
    "T4.50": 4.5,
    "T6.75": 6.75,
    "T9.00": 9.0,
}

SOURCE_COLUMNS = {
    "tree": "ANPP Tree (~48%C)",
    "shrub": "ANPP Shrub (~50%C)",
    "sphagnum": "NPP Sphag.",
    "bnpp": "BNPP Tree & Shrub",
    "rhco2": "RHCO2",
}

PFT_COMPONENTS = {
    "AGNPP_pftgroup_tree": {
        "source": "tree",
        "pft_group": "tree",
        "pft_indices": "3,5",
        "notes": "measured tree aboveground NPP",
    },
    "AGNPP_pftgroup_shrub": {
        "source": "shrub",
        "pft_group": "shrub",
        "pft_indices": "14",
        "notes": "measured shrub aboveground NPP",
    },
    "BGNPP_pftgroup_woody": {
        "source": "bnpp",
        "pft_group": "woody",
        "pft_indices": "3,5,14",
        "notes": "measured combined tree and shrub belowground NPP",
    },
    "NPP_pftgroup_moss": {
        "source": "sphagnum",
        "pft_group": "moss",
        "pft_indices": "18",
        "notes": "measured sphagnum NPP",
    },
}


def numeric(value):
    if pd.isna(value):
        return math.nan
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.upper() == "ND":
            return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.6g}"


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "SPRUCE C Budget Summary 19JAN2025_natalie.xlsx"
    default_output = repo_root / "data" / "spruce" / "spruce_c_budget_ambient_annual.csv"

    parser = argparse.ArgumentParser(
        description="Extract ambient-CO2 annual total and PFT-level NPP and HR observations for OLMT."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--site", default="US-SPR")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--error-fraction", type=float, default=0.15)
    parser.add_argument("--error-floor", type=float, default=1.0)
    return parser.parse_args()


def build_rows(args):
    df = pd.read_excel(args.input, sheet_name="Data By Year", header=4)
    year_column = df.columns[0]
    plot_to_treatment = {plot: treatment for treatment, plot in TREATMENT_PLOTS.items()}
    current_year = None
    rows = []

    for _, row in df.iterrows():
        year_label = row.get(year_column)
        if isinstance(year_label, (int, float)) and not pd.isna(year_label):
            current_year = int(year_label)
        elif isinstance(year_label, str):
            match = re.fullmatch(r"(20\d{2})", year_label.strip())
            if match:
                current_year = int(match.group(1))

        plot = row.get("Plot")
        if pd.isna(plot):
            continue
        plot = str(plot).strip()
        if plot not in plot_to_treatment:
            continue
        if current_year is None or not (args.start_year <= current_year <= args.end_year):
            continue

        values = {key: numeric(row.get(column)) for key, column in SOURCE_COLUMNS.items()}
        treatment = plot_to_treatment[plot]
        common = {
            "site": args.site,
            "treatment": treatment,
            "plot": plot,
            "co2": row.get("CO2"),
            "temperature_offset_C": TREATMENT_OFFSETS[treatment],
            "year": current_year,
            "units": "gC m-2 y-1",
            "source_file": args.input.name,
            "anpp_tree": values["tree"],
            "anpp_shrub": values["shrub"],
            "npp_sphagnum": values["sphagnum"],
            "bnpp_tree_shrub": values["bnpp"],
            "rhco2": values["rhco2"],
        }

        npp_parts = [values["tree"], values["shrub"], values["sphagnum"], values["bnpp"]]
        if all(not pd.isna(value) for value in npp_parts):
            obs = sum(npp_parts)
            rows.append(
                {
                    **common,
                    "model_var": "NPP",
                    "obs": obs,
                    "obs_err": max(abs(obs) * args.error_fraction, args.error_floor),
                    "source_variable": "total_npp",
                    "source_components": "+".join(
                        SOURCE_COLUMNS[key]
                        for key in ["tree", "shrub", "sphagnum", "bnpp"]
                    ),
                    "notes": "sum of tree ANPP, shrub ANPP, sphagnum NPP, and tree/shrub BNPP",
                }
            )

        for model_var, component in PFT_COMPONENTS.items():
            source = component["source"]
            component_obs = values[source]
            if pd.isna(component_obs):
                continue
            rows.append(
                {
                    **common,
                    "model_var": model_var,
                    "obs": component_obs,
                    "obs_err": max(abs(component_obs) * args.error_fraction, args.error_floor),
                    "source_variable": SOURCE_COLUMNS[source],
                    "source_components": SOURCE_COLUMNS[source],
                    "pft_group": component["pft_group"],
                    "pft_indices": component["pft_indices"],
                    "notes": component["notes"],
                }
            )

        if not pd.isna(values["rhco2"]):
            obs = -values["rhco2"]
            rows.append(
                {
                    **common,
                    "model_var": "HR",
                    "obs": obs,
                    "obs_err": max(abs(obs) * args.error_fraction, args.error_floor),
                    "source_variable": SOURCE_COLUMNS["rhco2"],
                    "source_components": SOURCE_COLUMNS["rhco2"],
                    "notes": "RHCO2 sign flipped so positive values match model HR",
                }
            )

    return rows


def main():
    args = parse_args()
    rows = build_rows(args)
    columns = [
        "site",
        "treatment",
        "plot",
        "co2",
        "temperature_offset_C",
        "year",
        "model_var",
        "obs",
        "obs_err",
        "units",
        "source_variable",
        "source_components",
        "source_file",
        "pft_group",
        "pft_indices",
        "anpp_tree",
        "anpp_shrub",
        "npp_sphagnum",
        "bnpp_tree_shrub",
        "rhco2",
        "notes",
    ]
    out = pd.DataFrame(rows, columns=columns)
    numeric_columns = [
        "obs",
        "obs_err",
        "temperature_offset_C",
        "anpp_tree",
        "anpp_shrub",
        "npp_sphagnum",
        "bnpp_tree_shrub",
        "rhco2",
    ]
    for column in numeric_columns:
        out[column] = out[column].map(fmt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} observations to {args.output}")
    print(out.groupby(["treatment", "model_var"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()

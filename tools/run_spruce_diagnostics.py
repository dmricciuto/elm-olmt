#!/usr/bin/env python3
"""Generate reusable SPRUCE treatment diagnostic products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olmt_diagnostics.spruce import (  # noqa: E402
    compare_hollow_water_tables,
    water_table_metrics,
    write_water_table_products,
)
from olmt_diagnostics.spruce_carbon import (  # noqa: E402
    compare_carbon_budget,
    temperature_response,
    write_carbon_budget_products,
)
from olmt_diagnostics.spruce_profiles import (  # noqa: E402
    read_profile_observations,
    read_treatment_profiles,
    write_profile_products,
)
from olmt_diagnostics.plot_spruce_mcfarlane_profiles import (  # noqa: E402
    make_figure as make_mcfarlane_figure,
    read_model_profiles as read_mcfarlane_model_profiles,
    read_observations as read_mcfarlane_observations,
    write_summary as write_mcfarlane_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/output/e3sm_run"))
    parser.add_argument("--case-prefix", required=True)
    parser.add_argument(
        "--obs-file",
        type=Path,
        default=Path("data/spruce/spruce079/spruce_daily_water_table_processed_20250408.nc"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument(
        "--missing-h2osfc", choices=["error", "zero"], default="error",
        help="Use 'zero' only for legacy output that did not archive H2OSFC_col.",
    )
    parser.add_argument(
        "--soilice-threshold-kg-m2", type=float, default=0.1,
        help="Shade days when hollow-column SOILICE exceeds this inventory.",
    )
    parser.add_argument(
        "--include-carbon-budget", action="store_true",
        help="Generate AGNPP, BGNPP, moss NPP, HR, and response diagnostics.",
    )
    parser.add_argument(
        "--include-profiles", action="store_true",
        help="Generate bog-mean DOM, acetate, and methane profile diagnostics.",
    )
    parser.add_argument(
        "--include-mcfarlane", action="store_true",
        help="Generate the four-panel McFarlane peat/C14 comparison.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Generate water table, NPP/HR, porewater-profile, and McFarlane products.",
    )
    parser.add_argument(
        "--budget-obs-file", type=Path,
        default=Path("data/spruce/spruce_c_budget_ambient_annual.csv"),
    )
    parser.add_argument(
        "--profile-obs-dir", type=Path,
        default=Path("data/spruce/porewater_profiles"),
    )
    parser.add_argument("--profile-start-year", type=int, default=2019)
    parser.add_argument("--profile-end-year", type=int, default=2023)
    parser.add_argument("--profile-doy-start", type=int, default=121)
    parser.add_argument("--profile-doy-end", type=int, default=260)
    parser.add_argument("--profile-observation-year", type=int, default=2013)
    parser.add_argument(
        "--mcfarlane-state", type=Path,
        help=(
            "History file for the peat/C14 comparison. By default the latest "
            "PFT/column-resolved T0 history file is used."
        ),
    )
    parser.add_argument(
        "--mcfarlane-obs-file", type=Path,
        default=Path("data/spruce/mcfarlane2018/Peat_Characteristics_T0_20180425.csv"),
    )
    parser.add_argument(
        "--mcfarlane-radiocarbon-note",
        default="Confirm atmospheric C14 forcing before quantitative interpretation",
    )
    return parser.parse_args()


def _latest_mcfarlane_state(run_root: Path, case_prefix: str) -> Path:
    candidate_dirs = (
        run_root / f"{case_prefix}_T0.00" / "run",
        run_root / f"{case_prefix}_TAMB" / "run",
        run_root / case_prefix / "run",
    )
    for run_dir in candidate_dirs:
        history = sorted(run_dir.glob("*.elm.h1.*.nc"))
        if history:
            return history[-1]
    raise FileNotFoundError(
        "No column/PFT-resolved ELM h1 history file found for McFarlane "
        f"diagnostics under {run_root} for {case_prefix}"
    )


def main() -> None:
    args = parse_args()
    written: dict[str, str] = {}
    comparison = compare_hollow_water_tables(
        args.run_root,
        args.case_prefix,
        args.obs_file,
        args.start_year,
        args.end_year,
        args.missing_h2osfc,
        args.soilice_threshold_kg_m2,
    )
    paths = write_water_table_products(
        comparison,
        args.output_dir,
        source={
            "run_root": str(args.run_root),
            "case_prefix": args.case_prefix,
            "observation_file": str(args.obs_file),
            "start_year": args.start_year,
            "end_year": args.end_year,
            "missing_h2osfc_policy": args.missing_h2osfc,
            "soilice_threshold_kg_m2": args.soilice_threshold_kg_m2,
        },
    )
    print(water_table_metrics(comparison).to_string(index=False))
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
        written[f"water_table_{name}"] = str(path)

    if args.include_carbon_budget or args.full:
        carbon = compare_carbon_budget(
            args.run_root,
            args.case_prefix,
            args.budget_obs_file,
            args.start_year or 2016,
            args.end_year or 2023,
        )
        carbon_paths = write_carbon_budget_products(
            carbon,
            args.output_dir,
            source={
                "run_root": str(args.run_root),
                "case_prefix": args.case_prefix,
                "observation_file": str(args.budget_obs_file),
                "start_year": args.start_year or 2016,
                "end_year": args.end_year or 2023,
            },
        )
        print(temperature_response(carbon).to_string(index=False))
        for name, path in carbon_paths.items():
            print(f"Wrote carbon {name}: {path}")
            written[f"carbon_{name}"] = str(path)

    if args.include_profiles or args.full:
        profiles = read_treatment_profiles(
            args.run_root,
            args.case_prefix,
            args.profile_start_year,
            args.profile_end_year,
            args.profile_doy_start,
            args.profile_doy_end,
        )
        profile_observations = read_profile_observations(
            args.profile_obs_dir, args.profile_observation_year
        )
        profile_paths = write_profile_products(
            profiles,
            profile_observations,
            args.output_dir,
            args.profile_start_year,
            args.profile_end_year,
            args.profile_doy_start,
            args.profile_doy_end,
            args.profile_observation_year,
            source={
                "run_root": str(args.run_root),
                "case_prefix": args.case_prefix,
                "observation_directory": str(args.profile_obs_dir),
            },
        )
        for name, path in profile_paths.items():
            print(f"Wrote profile {name}: {path}")
            written[f"profile_{name}"] = str(path)

    if args.include_mcfarlane or args.full:
        state = args.mcfarlane_state or _latest_mcfarlane_state(
            args.run_root, args.case_prefix
        )
        model, state_file = read_mcfarlane_model_profiles(str(state))
        observations = read_mcfarlane_observations(args.mcfarlane_obs_file)
        mcfarlane_dir = args.output_dir / "mcfarlane2018"
        mcfarlane_dir.mkdir(parents=True, exist_ok=True)
        figure = mcfarlane_dir / "spruce_mcfarlane_profile_comparison.png"
        summary = mcfarlane_dir / "spruce_mcfarlane_profile_summary.csv"
        make_mcfarlane_figure(
            model,
            observations,
            figure,
            "SPRUCE peat profiles: ELM versus McFarlane et al. (2018)",
            args.mcfarlane_radiocarbon_note,
        )
        write_mcfarlane_summary(model, summary, state_file)
        print(f"Wrote McFarlane figure: {figure}")
        print(f"Wrote McFarlane summary: {summary}")
        written["mcfarlane_figure"] = str(figure)
        written["mcfarlane_summary"] = str(summary)

    manifest = args.output_dir / "spruce_diagnostics_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "diagnostic": "standard SPRUCE treatment diagnostics",
                "run_root": str(args.run_root),
                "case_prefix": args.case_prefix,
                "products": written,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote package manifest: {manifest}")


if __name__ == "__main__":
    main()

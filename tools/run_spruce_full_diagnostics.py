#!/usr/bin/env python3
"""Run the standard SPRUCE diagnostics and assemble the PDF report.

This is intentionally a small orchestration wrapper so OLMT, batch jobs, and
Docker can use the same post-treatment command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from netCDF4 import Dataset


def _case_microbe_parameters(run_root: Path, case_prefix: str) -> dict[str, float]:
    """Read report settings from the actual T0/TAMB case parameter copy."""
    names = (
        "decomp_depth_efolding",
        "dom_solubilization_anoxic_fraction",
    )
    for treatment in ("T0.00", "TAMB"):
        path = run_root / f"{case_prefix}_{treatment}" / "run" / "clm_params.nc"
        if not path.is_file():
            continue
        with Dataset(path) as parameters:
            return {
                name: float(parameters.variables[name][...].item())
                for name in names
                if name in parameters.variables
            }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case-prefix", required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--profile-start-year", type=int)
    parser.add_argument("--profile-end-year", type=int)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Label synthetic-year diagnostics as workflow checks, not validation.",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    diagnostics = repo / "tools" / "run_spruce_diagnostics.py"
    report = repo / "tools" / "build_spruce_diagnostics_report.py"
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python, str(diagnostics),
        "--run-root", str(args.run_root),
        "--case-prefix", args.case_prefix,
        "--output-dir", str(args.diagnostics_dir),
        "--full",
    ]
    if args.start_year is not None:
        command += ["--start-year", str(args.start_year)]
    if args.end_year is not None:
        command += ["--end-year", str(args.end_year)]
    if args.profile_start_year is not None:
        command += ["--profile-start-year", str(args.profile_start_year)]
    if args.profile_end_year is not None:
        command += ["--profile-end-year", str(args.profile_end_year)]
    if args.smoke_test:
        command.append("--smoke-test")
    subprocess.run(command, cwd=repo, check=True)

    report_path = args.diagnostics_dir / "standard_spruce_diagnostics.pdf"
    report_command = [
        args.python, str(report),
        "--diagnostics-dir", str(args.diagnostics_dir),
        "--output", str(report_path),
        "--case-prefix", args.case_prefix,
    ]
    if args.smoke_test:
        report_command += [
            "--smoke-test", "--c14-note",
            "C14 is not enabled in this synthetic-year workflow test.",
        ]
    for name, value in _case_microbe_parameters(args.run_root, args.case_prefix).items():
        report_command += ["--" + name.replace("_", "-"), str(value)]
    subprocess.run(report_command, cwd=repo, check=True)
    print(f"Wrote standard SPRUCE diagnostics: {report_path}")


if __name__ == "__main__":
    main()

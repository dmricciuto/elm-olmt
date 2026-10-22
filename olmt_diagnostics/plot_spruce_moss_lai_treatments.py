#!/usr/bin/env python3
"""Plot overlaid SPRUCE treatment trajectories of moss leaf area index."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd


CASES = ["TAMB", "T0.00", "T2.25", "T4.50", "T6.75", "T9.00"]
LABELS = {
    "TAMB": "Ambient",
    "T0.00": "T0",
    "T2.25": "T+2.25 °C",
    "T4.50": "T+4.50 °C",
    "T6.75": "T+6.75 °C",
    "T9.00": "T+9.00 °C",
}
COLORS = {
    "TAMB": "#777777",
    "T0.00": "#1a1a1a",
    "T2.25": "#2c7bb6",
    "T4.50": "#abd9e9",
    "T6.75": "#fdae61",
    "T9.00": "#d7191c",
}
LINESTYLES = {"TAMB": "--", "T0.00": "-"}
MOSS_PFT = 18
BOG_TOPOUNITS = (2, 3)  # hollow and hummock; auxiliary fen is excluded


def dates(dataset: Dataset) -> pd.DatetimeIndex:
    time = dataset.variables["time"]
    values = num2date(
        time[:],
        time.units,
        getattr(time, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    return pd.DatetimeIndex(pd.Timestamp(str(value)[:10]) for value in values)


def history_files(run_dir: Path, start_year: int, end_year: int) -> list[Path]:
    """Find the history stream containing PFT-resolved TLAI."""
    candidates = sorted(run_dir.glob("*.elm.h*.*.nc"))
    stream = None
    for path in candidates:
        with Dataset(path) as dataset:
            variable = dataset.variables.get("TLAI")
            if variable is not None and {"time", "pft"}.issubset(variable.dimensions):
                stream = path.name.split(".elm.", 1)[1].split(".", 1)[0]
                break
    if stream is None:
        raise RuntimeError(f"No PFT-resolved TLAI history stream in {run_dir}")

    selected = []
    for path in sorted(run_dir.glob(f"*.elm.{stream}.*.nc")):
        with Dataset(path) as dataset:
            file_dates = dates(dataset)
        if np.any((file_dates.year >= start_year) & (file_dates.year <= end_year)):
            selected.append(path)
    if not selected:
        raise RuntimeError(
            f"No {stream} files spanning {start_year}-{end_year} in {run_dir}"
        )
    return selected


def read_case(run_dir: Path, case: str, start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for path in history_files(run_dir, start_year, end_year):
        with Dataset(path) as dataset:
            file_dates = dates(dataset)
            selected_time = (
                (file_dates.year >= start_year) & (file_dates.year <= end_year)
            )
            if not selected_time.any():
                continue

            pft_type = np.asarray(dataset.variables["pfts1d_itype_veg"][:], dtype=int)
            topounit = np.asarray(dataset.variables["pfts1d_topounit"][:], dtype=int)
            active = np.asarray(dataset.variables["pfts1d_active"][:], dtype=int)
            grid_weight = np.asarray(
                dataset.variables["pfts1d_wtgcell"][:], dtype=float
            )
            moss = np.where(
                (pft_type == MOSS_PFT)
                & np.isin(topounit, BOG_TOPOUNITS)
                & (active == 1)
                & (grid_weight > 0.0)
            )[0]
            if len(moss) != 2:
                raise RuntimeError(
                    f"Expected one active moss PFT in hollow and hummock; found "
                    f"indices {moss.tolist()} in {path}"
                )

            weights = grid_weight[moss]
            weights = weights / weights.sum()
            lai = np.ma.filled(dataset.variables["TLAI"][:, moss], np.nan).astype(float)
            valid = np.isfinite(lai)
            effective_weights = valid * weights[np.newaxis, :]
            denominator = effective_weights.sum(axis=1)
            weighted_lai = np.divide(
                np.nansum(lai * weights[np.newaxis, :], axis=1),
                denominator,
                out=np.full(lai.shape[0], np.nan),
                where=denominator > 0.0,
            )
            frames.append(
                pd.DataFrame(
                    {
                        "date": file_dates[selected_time],
                        "case": case,
                        "treatment": LABELS[case],
                        "moss_lai": weighted_lai[selected_time],
                        "hollow_weight": weights[0],
                        "hummock_weight": weights[1],
                    }
                )
            )
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "case"])
        .sort_values("date")
    )


def plot(data: pd.DataFrame, output: Path, start_year: int, end_year: int) -> None:
    fig, axis = plt.subplots(figsize=(13, 6.6))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16, top=0.89)
    for case in CASES:
        subset = data[data["case"] == case]
        axis.plot(
            subset["date"],
            subset["moss_lai"],
            color=COLORS[case],
            linestyle=LINESTYLES.get(case, "-"),
            linewidth=1.65 if case in {"TAMB", "T0.00"} else 1.45,
            label=LABELS[case],
            zorder=3 if case == "T9.00" else 2,
        )

    treatment_start = pd.Timestamp("2015-08-15")
    axis.axvline(treatment_start, color="0.35", linewidth=1.0, linestyle=":")
    axis.text(
        treatment_start + pd.Timedelta(days=35),
        0.97,
        "Warming begins\n15 Aug 2015",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        color="0.25",
        fontsize=9,
    )

    axis.set_xlim(pd.Timestamp(f"{start_year}-01-01"), pd.Timestamp(f"{end_year}-12-31"))
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Moss total projected LAI (m² m⁻²)")
    axis.set_xlabel("Year", labelpad=8)
    axis.set_title(
        "SPRUCE moss LAI response to whole-ecosystem warming",
        fontsize=15,
        pad=12,
    )
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(True, color="0.88", linewidth=0.7)
    axis.legend(ncol=3, frameon=False, loc="upper right")
    fig.text(
        0.5,
        0.025,
        "Daily TLAI; hollow/hummock area-weighted mean, auxiliary fen excluded.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.3",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--case-prefix", required=True)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for case in CASES:
        run_dir = args.run_root / f"{args.case_prefix}_{case}" / "run"
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        frames.append(read_case(run_dir, case, args.start_year, args.end_year))
    data = pd.concat(frames, ignore_index=True)
    plot(data, args.output, args.start_year, args.end_year)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()

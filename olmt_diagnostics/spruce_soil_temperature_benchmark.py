#!/usr/bin/env python3
"""Summarize SPRUCE.032 v2 half-hourly peat temperatures by season.

The A/B/C depth probes are referenced to the hollow surface. This script
does not gap-fill observations or mix the separate hummock probes into them.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Non-eCO2 treatments in config_files/SPRUCE_microbe_20260924.cfg.
TREATMENT_PLOTS = {
    "TAMB": 7,
    "T0.00": 6,
    "T2.25": 20,
    "T4.50": 13,
    "T6.75": 8,
    "T9.00": 17,
}
DEPTH_CM = (0, 5, 10, 20, 30, 40, 50, 100, 200)
REPLICATES = "ABC"
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5),
           "JJA": (6, 7, 8), "SON": (9, 10, 11)}
SEASON_NAMES = tuple(SEASONS)
COLORS = {
    "TAMB": "#333333", "T0.00": "#4878a8", "T2.25": "#45a0a2",
    "T4.50": "#6aab64", "T6.75": "#dd9650", "T9.00": "#c45050",
}


def _probe_columns() -> list[str]:
    return [f"TS_{depth}_{rep}{index}" for depth, index in
            zip(DEPTH_CM, range(1, len(DEPTH_CM) + 1)) for rep in REPLICATES]


def _expected_halfhours(year: int, season: str) -> int:
    return 48 * sum(calendar.monthrange(year, month)[1]
                    for month in SEASONS[season])


def summarize_plot(path: Path, treatment: str, plot: int, start_year: int,
                   end_year: int, min_coverage: float = 0.75,
                   chunk_size: int = 50000) -> pd.DataFrame:
    """Average qualified probe seasons, then replicates, then years equally."""
    probes = _probe_columns()
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(probes + ["TIMESTAMP"]) - header
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    sums = np.zeros((end_year - start_year + 1, 4, len(probes)), dtype=float)
    counts = np.zeros_like(sums, dtype=np.int64)
    for chunk in pd.read_csv(path, usecols=["TIMESTAMP", *probes],
                             na_values=["NA", "-9999", -9999],
                             chunksize=chunk_size):
        dates = pd.to_datetime(chunk["TIMESTAMP"], errors="coerce")
        in_range = dates.dt.year.between(start_year, end_year)
        if not in_range.any():
            continue
        dates = dates.loc[in_range]
        values = chunk.loc[in_range, probes].apply(
            pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # Wide bounds only discard impossible sensor codes/spikes, not cold peat.
        values[(values < -50.0) | (values > 60.0)] = np.nan
        years = dates.dt.year.to_numpy() - start_year
        months = dates.dt.month.to_numpy()
        for year_index in np.unique(years):
            year_mask = years == year_index
            for season_index, season in enumerate(SEASON_NAMES):
                mask = year_mask & np.isin(months, SEASONS[season])
                if mask.any():
                    selected = values[mask]
                    sums[year_index, season_index] += np.nansum(selected, axis=0)
                    counts[year_index, season_index] += np.isfinite(selected).sum(axis=0)

    rows = []
    for season_index, season in enumerate(SEASON_NAMES):
        for depth_index, depth in enumerate(DEPTH_CM):
            year_means = []
            accepted_probes = 0
            valid_halfhours = 0
            expected_halfhours = 0
            for year_index, year in enumerate(range(start_year, end_year + 1)):
                probe_indices = slice(depth_index * 3, depth_index * 3 + 3)
                probe_counts = counts[year_index, season_index, probe_indices]
                expected = _expected_halfhours(year, season)
                qualified = probe_counts >= min_coverage * expected
                if not qualified.any():
                    continue
                probe_means = (sums[year_index, season_index, probe_indices]
                               [qualified] / probe_counts[qualified])
                year_means.append(float(np.mean(probe_means)))
                accepted_probes += int(qualified.sum())
                valid_halfhours += int(probe_counts[qualified].sum())
                expected_halfhours += int(qualified.sum()) * expected
            rows.append({
                "treatment": treatment, "plot": plot, "season": season,
                "depth_m": depth / 100.0,
                "temperature_C": float(np.mean(year_means)) if year_means else np.nan,
                "n_years": len(year_means),
                "qualified_probe_years": accepted_probes,
                "possible_probe_years": 3 * (end_year - start_year + 1),
                "valid_halfhours": valid_halfhours,
                "qualified_expected_halfhours": expected_halfhours,
            })
    return pd.DataFrame(rows)


def plot_profiles(summary: pd.DataFrame, output: Path, years: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2), sharex=True, sharey=True)
    for axis, season in zip(axes.flat, SEASON_NAMES):
        for treatment in TREATMENT_PLOTS:
            subset = summary[(summary.season == season) &
                             (summary.treatment == treatment)].sort_values("depth_m")
            axis.plot(subset.temperature_C, subset.depth_m, color=COLORS[treatment],
                      marker="o", markersize=3.5, linewidth=1.9,
                      label=treatment)
        axis.set_title(season)
        axis.grid(alpha=0.24)
        axis.set_ylim(2.05, -0.05)
        axis.set_yticks([0, 0.1, 0.2, 0.4, 0.6, 1, 1.5, 2])
    for axis in axes[:, 0]:
        axis.set_ylabel("Depth below hollow surface (m)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Observed peat temperature (°C)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(f"SPRUCE seasonal peat temperature · {years}\n"
                 "Non-eCO₂ plots; depth-labeled A/B/C probes; no gap filling")
    fig.tight_layout(rect=(0, 0.065, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path,
                        help="Recompute the summary from raw CSVs; omit to replot an existing summary")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    if args.end_year < args.start_year or not 0 < args.min_coverage <= 1:
        parser.error("Specify an ascending year interval and 0 < coverage <= 1")
    if args.raw_dir is None:
        summary = pd.read_csv(args.summary)
    else:
        frames = []
        for treatment, plot in TREATMENT_PLOTS.items():
            path = args.raw_dir / f"WEW_PLOT_{plot:02d}_Complete_Environ_20260702.csv"
            print(f"Reading {treatment}: {path}", flush=True)
            frames.append(summarize_plot(path, treatment, plot, args.start_year,
                                         args.end_year, args.min_coverage))
        summary = pd.concat(frames, ignore_index=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary, index=False, float_format="%.6f")
    plot_profiles(summary, args.figure, f"{args.start_year}–{args.end_year}")
    print(f"{'Wrote' if args.raw_dir else 'Read'} {args.summary}; wrote {args.figure}")
    print(summary.groupby("treatment")[["n_years", "qualified_probe_years"]].min())


if __name__ == "__main__":
    main()

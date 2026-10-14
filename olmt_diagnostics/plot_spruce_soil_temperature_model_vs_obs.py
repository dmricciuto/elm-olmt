#!/usr/bin/env python3
"""Compare seasonal hollow TSOI with depth-matched SPRUCE.032 observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd

if __package__:
    from .spruce_soil_temperature_benchmark import COLORS, SEASONS, TREATMENT_PLOTS
else:
    from spruce_soil_temperature_benchmark import COLORS, SEASONS, TREATMENT_PLOTS


def _hollow_column(dataset: Dataset) -> int:
    topo = np.asarray(dataset.variables["cols1d_topounit"][:], dtype=int)
    weight = np.asarray(dataset.variables["cols1d_wtgcell"][:], dtype=float)
    matches = np.flatnonzero((topo == 2) & (weight > 0))
    if len(matches) != 1:
        raise ValueError(f"Expected one active hollow column, found {len(matches)}")
    return int(matches[0])


def summarize_model(run_root: Path, case_prefix: str, start_year: int,
                    end_year: int) -> pd.DataFrame:
    """Average daily temperatures within each season, then years equally."""
    rows = []
    for treatment in TREATMENT_PLOTS:
        case = f"{case_prefix}_{treatment}"
        files = sorted((run_root / case / "run").glob(f"{case}.elm.h1.*.nc"))
        if not files:
            raise FileNotFoundError(f"No daily column history files for {case}")
        totals: dict[tuple[int, str], np.ndarray] = {}
        counts: dict[tuple[int, str], np.ndarray] = {}
        depths = None
        for filename in files:
            with Dataset(filename) as dataset:
                if "TSOI" not in dataset.variables:
                    raise ValueError(
                        f"Daily TSOI_col is absent from {filename}; rerun with "
                        "'TSOI_col' in the OLMT postprocessing variables"
                    )
                variable = dataset.variables["TSOI"]
                if variable.dimensions != ("time", "levgrnd", "column"):
                    raise ValueError(f"Unexpected TSOI dimensions: {variable.dimensions}")
                if variable.units.strip().lower() not in ("k", "kelvin"):
                    raise ValueError(f"Expected TSOI in kelvin: {filename}")
                current_depths = np.asarray(dataset.variables["levgrnd"][:], dtype=float)
                if depths is None:
                    depths = current_depths
                elif not np.allclose(depths, current_depths):
                    raise ValueError("Soil layer depths differ between model histories")
                hollow = _hollow_column(dataset)
                values = np.ma.filled(variable[:, :, hollow], np.nan).astype(float) - 273.15
                time = dataset.variables["time"]
                dates = num2date(time[:], time.units,
                                calendar=getattr(time, "calendar", "noleap"))
                years = np.fromiter((date.year for date in dates), dtype=int)
                months = np.fromiter((date.month for date in dates), dtype=int)
                for year in range(start_year, end_year + 1):
                    for season, season_months in SEASONS.items():
                        selected = values[(years == year) &
                                          np.isin(months, season_months)]
                        if not len(selected):
                            continue
                        key = (year, season)
                        totals.setdefault(key, np.zeros(len(depths)))
                        counts.setdefault(key, np.zeros(len(depths), dtype=int))
                        totals[key] += np.nansum(selected, axis=0)
                        counts[key] += np.isfinite(selected).sum(axis=0)
        assert depths is not None
        for season in SEASONS:
            for layer, depth in enumerate(depths):
                means = [totals[year, season][layer] / counts[year, season][layer]
                         for year in range(start_year, end_year + 1)
                         if (year, season) in counts and counts[year, season][layer] > 0]
                rows.append({
                    "treatment": treatment, "season": season,
                    "depth_m": float(depth),
                    "temperature_C": float(np.mean(means)) if means else np.nan,
                    "n_years": len(means),
                })
    result = pd.DataFrame(rows)
    if result.temperature_C.isna().any():
        raise ValueError("A requested model season has no daily TSOI data")
    return result


def plot_comparison(observed: pd.DataFrame, modeled: pd.DataFrame,
                    output: Path, model_years: str, observation_years: str,
                    smoke_test: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.7), sharex=True, sharey=True)
    for axis, season in zip(axes.flat, SEASONS):
        for treatment in TREATMENT_PLOTS:
            obs = observed[(observed.season == season) &
                           (observed.treatment == treatment)].sort_values("depth_m")
            model = modeled[(modeled.season == season) &
                            (modeled.treatment == treatment)].sort_values("depth_m")
            if obs.empty or model.empty:
                raise ValueError(f"Missing {season} profile for {treatment}")
            axis.plot(obs.temperature_C, obs.depth_m, color=COLORS[treatment],
                      linewidth=1.7, marker="o", markersize=2.8)
            axis.plot(model.temperature_C, model.depth_m, color=COLORS[treatment],
                      linewidth=1.8, linestyle="--")
        axis.set_title(season)
        axis.set_ylim(2.05, -0.05)
        axis.set_yticks([0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.0])
        axis.grid(alpha=0.23)
    for axis in axes[:, 0]:
        axis.set_ylabel("Depth below hollow surface (m)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Peat temperature (°C)")
    treatment_handles = [Line2D([0], [0], color=COLORS[name], linewidth=2,
                                label=name) for name in TREATMENT_PLOTS]
    fig.legend(handles=treatment_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.legend(handles=[
        Line2D([0], [0], color="0.25", marker="o", linewidth=1.8,
               label=f"SPRUCE observations ({observation_years})"),
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=1.8,
               label=f"ELM hollow ({model_years})"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.055), frameon=False)
    title = "SPRUCE seasonal soil temperature: hollow profile comparison"
    if smoke_test:
        title += "\nSynthetic model year aligned to observed forcing year - workflow check only"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.11, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--case-prefix", required=True)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--model-start-year", required=True, type=int)
    parser.add_argument("--model-end-year", required=True, type=int)
    parser.add_argument("--observation-years", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    observed = pd.read_csv(args.observations)
    modeled = summarize_model(args.run_root, args.case_prefix,
                              args.model_start_year, args.model_end_year)
    model_csv = args.output.with_name("soil_temperature_model_seasonal_profiles.csv")
    model_csv.parent.mkdir(parents=True, exist_ok=True)
    modeled.to_csv(model_csv, index=False, float_format="%.6f")
    model_years = (str(args.model_start_year) if args.model_start_year == args.model_end_year
                   else f"{args.model_start_year}-{args.model_end_year}")
    plot_comparison(observed, modeled, args.output, model_years,
                    args.observation_years, args.smoke_test)
    print(f"Wrote {args.output} and {model_csv}")


if __name__ == "__main__":
    main()

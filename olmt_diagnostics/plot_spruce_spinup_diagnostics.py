#!/usr/bin/env python3
"""Plot porewater profiles and PFT NPP from one SPRUCE spinup case."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd

from .spruce_profiles import (
    BOG_TOPOUNITS,
    CARBON_ATOMIC_MASS_G_MOL,
    PROFILE_DEFINITIONS,
    WATER_DENSITY_KG_M3,
    _bog_columns_and_weights,
    _layer_thickness,
    _time_layer_column,
    _weighted_mean,
    read_profile_observations,
)


PFTS = {
    "PFT 3: needleleaf evergreen tree": 3,
    "PFT 5: needleleaf deciduous tree": 5,
    "PFT 14: deciduous shrub": 14,
    "PFT 18: Sphagnum": 18,
}
PFT_COLORS = {
    "PFT 3: needleleaf evergreen tree": "#1b7837",
    "PFT 5: needleleaf deciduous tree": "#5aae61",
    "PFT 14: deciduous shrub": "#d95f02",
    "PFT 18: Sphagnum": "#7570b3",
    "Total": "#222222",
}
SECONDS_PER_DAY = 86400.0


def history_files(run_dir: Path, variable: str, dimension: str) -> list[Path]:
    """Return the highest-resolution stream carrying an indexed variable."""
    streams: dict[str, list[Path]] = {}
    for path in sorted(run_dir.glob("*.elm.h*.*.nc")):
        with Dataset(path) as dataset:
            item = dataset.variables.get(variable)
            if item is not None and {"time", dimension}.issubset(item.dimensions):
                stream = path.name.split(".elm.", 1)[1].split(".", 1)[0]
                streams.setdefault(stream, []).append(path)
    if not streams:
        raise RuntimeError(
            f"No {dimension}-resolved {variable} in {run_dir}"
        )
    record_counts = {}
    for stream, paths in streams.items():
        record_counts[stream] = 0
        for path in paths:
            with Dataset(path) as dataset:
                record_counts[stream] += len(dataset.dimensions["time"])
    stream = max(record_counts, key=record_counts.get)
    return streams[stream]


def date_fields(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return model year and day of year without Pandas' year-1677 limit."""
    variable = dataset.variables["time"]
    dates = num2date(
        variable[:], variable.units, getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=True,
    )
    return (
        np.asarray([value.year for value in dates], dtype=int),
        np.asarray([value.dayofyr for value in dates], dtype=int),
    )


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--profile-start-year", type=int, required=True)
    parser.add_argument("--profile-end-year", type=int, required=True)
    parser.add_argument("--doy-start", type=int, default=121)
    parser.add_argument("--doy-end", type=int, default=260)
    parser.add_argument(
        "--profile-observations",
        type=Path,
        default=repository / "data/spruce/porewater_profiles",
    )
    parser.add_argument("--observation-year", type=int, default=2013)
    return parser.parse_args()


def read_profiles(arguments: argparse.Namespace) -> pd.DataFrame:
    chunks = {name: [] for name in PROFILE_DEFINITIONS}
    depth = None
    for path in history_files(arguments.run_dir, "MM_DOM_POREWATER_C", "column"):
        with Dataset(path) as dataset:
            years, days = date_fields(dataset)
            keep = (
                (years >= arguments.profile_start_year)
                & (years <= arguments.profile_end_year)
                & (days >= arguments.doy_start)
                & (days <= arguments.doy_end)
            )
            if not np.any(keep):
                continue
            file_depth = np.asarray(dataset.variables["levdcmp"][:], dtype=float)
            if depth is None:
                depth = file_depth
            elif not np.allclose(depth, file_depth):
                raise RuntimeError(
                    f"Layer depths change between files in {arguments.run_dir}"
                )
            columns, weights = _bog_columns_and_weights(dataset)
            doc = _time_layer_column(dataset.variables["MM_DOM_POREWATER_C"])[
                keep, :, :
            ][:, :, columns]
            ch4 = _time_layer_column(dataset.variables["MM_CH4_POREWATER"])[
                keep, :, :
            ][:, :, columns]
            acetate_c = _time_layer_column(dataset.variables["MM_ACETATE_C_SAT"])[
                keep, :, :
            ][:, :, columns]
            liquid = _time_layer_column(dataset.variables["SOILLIQ"])[
                keep, : file_depth.size, :
            ][:, :, columns]
            liquid_fraction = liquid / (
                WATER_DENSITY_KG_M3
                * _layer_thickness(file_depth)[None, :, None]
            )
            acetate = np.divide(
                acetate_c,
                liquid_fraction * (2.0 * CARBON_ATOMIC_MASS_G_MOL),
                out=np.full_like(acetate_c, np.nan),
                where=liquid_fraction > 1.0e-12,
            )
            chunks["doc"].append(_weighted_mean(doc, weights))
            chunks["acetate"].append(_weighted_mean(acetate, weights))
            chunks["ch4"].append(_weighted_mean(ch4, weights))
    if depth is None or any(not values for values in chunks.values()):
        raise RuntimeError("No porewater samples selected for the requested period")

    rows = []
    for name, values in chunks.items():
        samples = np.concatenate(values, axis=0)
        for level, layer_depth in enumerate(depth):
            layer = samples[:, level]
            rows.append(
                {
                    "variable": name,
                    "depth_m": layer_depth,
                    "mean": np.nanmean(layer),
                    "p05": np.nanpercentile(layer, 5),
                    "p95": np.nanpercentile(layer, 95),
                    "n": int(np.isfinite(layer).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_profiles(
    model: pd.DataFrame,
    observations: pd.DataFrame,
    output: Path,
    arguments: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.2), sharey=True)
    for axis, (name, definition) in zip(axes, PROFILE_DEFINITIONS.items()):
        selected = model[model["variable"] == name].sort_values("depth_m")
        observed = observations[observations["variable"] == name]
        axis.fill_betweenx(
            selected["depth_m"], selected["p05"], selected["p95"],
            color="#3182bd", alpha=0.18, label="ELM 5–95% range",
        )
        axis.plot(
            selected["mean"], selected["depth_m"], "o-",
            color="#08519c", lw=2.0, ms=3.5, label="ELM mean",
        )
        axis.errorbar(
            observed["observed"], observed["depth_m"],
            xerr=observed["uncertainty"], fmt="o", ms=4.5,
            markerfacecolor="white", markeredgecolor="0.2", color="0.45",
            ecolor="0.65", alpha=0.8, capsize=2,
            label=f"Observed ({arguments.observation_year})",
        )
        axis.set_xlabel(definition["label"])
        axis.set_xlim(left=0.0)
        axis.set_ylim(2.5, 0.0)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Depth below local peat surface (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("SPRUCE final-spinup porewater profiles")
    fig.text(
        0.5, 0.045,
        f"ELM years {arguments.profile_start_year}–{arguments.profile_end_year}, "
        f"DOY {arguments.doy_start}–{arguments.doy_end}; "
        "hummock+hollow weighted mean, fen excluded.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def read_npp(arguments: argparse.Namespace) -> pd.DataFrame:
    annual = {
        (year, label): {"total": 0.0, "days": 0}
        for year in range(arguments.start_year, arguments.end_year + 1)
        for label in (*PFTS, "Total")
    }
    for path in history_files(arguments.run_dir, "NPP", "pft"):
        with Dataset(path) as dataset:
            years, _ = date_fields(dataset)
            variable = dataset.variables["NPP"]
            values = np.ma.filled(variable[:], np.nan).astype(float)
            values = np.moveaxis(
                values,
                (variable.dimensions.index("time"), variable.dimensions.index("pft")),
                (0, 1),
            )
            active = np.asarray(dataset.variables["pfts1d_active"][:], dtype=int)
            topounit = np.asarray(dataset.variables["pfts1d_topounit"][:], dtype=int)
            pft_type = np.asarray(dataset.variables["pfts1d_itype_veg"][:], dtype=int)
            weights = np.asarray(dataset.variables["pfts1d_wtgcell"][:], dtype=float)
            bog = (active == 1) & np.isin(topounit, BOG_TOPOUNITS) & (weights > 0.0)
            bog_area = weights[bog].sum()
            if bog_area <= 0.0:
                raise RuntimeError(f"No active bog PFTs in {path}")
            for label, indices in {
                **{label: (pft,) for label, pft in PFTS.items()},
                "Total": tuple(PFTS.values()),
            }.items():
                selected = bog & np.isin(pft_type, indices)
                flux = np.nansum(values[:, selected] * weights[selected], axis=1) / bog_area
                for year in range(arguments.start_year, arguments.end_year + 1):
                    keep = years == year
                    if np.any(keep):
                        annual[(year, label)]["total"] += float(
                            np.nansum(flux[keep]) * SECONDS_PER_DAY
                        )
                        annual[(year, label)]["days"] += int(np.count_nonzero(keep))
    return pd.DataFrame(
        {
            "year": year,
            "pft": label,
            "model_gC_m2_yr": values["total"],
            "model_days": values["days"],
        }
        for (year, label), values in annual.items()
        if values["days"] > 0
    )


def plot_npp(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for label in (*PFTS, "Total"):
        selected = frame[frame["pft"] == label].sort_values("year")
        axes[0].plot(
            selected["year"], selected["model_gC_m2_yr"], "o-",
            lw=2.2 if label == "Total" else 1.6,
            ms=4, color=PFT_COLORS[label], label=label,
        )
    axes[0].set(
        xlabel="Final-spinup model year",
        ylabel="NPP (g C m$^{-2}$ bog yr$^{-1}$)",
        title="Annual NPP",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    final_years = sorted(frame["year"].unique())[-3:]
    means = (
        frame[frame["year"].isin(final_years)]
        .groupby("pft", as_index=False)["model_gC_m2_yr"].mean()
        .set_index("pft")
    )
    labels = [*PFTS, "Total"]
    values = [means.loc[label, "model_gC_m2_yr"] for label in labels]
    axes[1].bar(
        np.arange(len(labels)), values,
        color=[PFT_COLORS[label] for label in labels], alpha=0.88,
    )
    axes[1].set_xticks(
        np.arange(len(labels)), ["PFT 3", "PFT 5", "Shrub", "Moss", "Total"],
        rotation=25, ha="right",
    )
    axes[1].set(
        ylabel="NPP (g C m$^{-2}$ bog yr$^{-1}$)",
        title=f"Mean of final {len(final_years)} years",
    )
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("SPRUCE bog vegetation productivity (fen excluded)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = read_profiles(arguments)
    observations = read_profile_observations(
        arguments.profile_observations, arguments.observation_year
    )
    profiles.to_csv(arguments.output_dir / "porewater_profiles_model.csv", index=False)
    observations.to_csv(
        arguments.output_dir / "porewater_profiles_observations.csv", index=False
    )
    plot_profiles(
        profiles, observations,
        arguments.output_dir / "porewater_dom_acetate_ch4_profiles.png",
        arguments,
    )

    npp = read_npp(arguments)
    npp.to_csv(arguments.output_dir / "pft_npp_annual.csv", index=False)
    plot_npp(npp, arguments.output_dir / "pft_npp_diagnostics.png")

    print(arguments.output_dir / "porewater_dom_acetate_ch4_profiles.png")
    print(arguments.output_dir / "pft_npp_diagnostics.png")


if __name__ == "__main__":
    main()

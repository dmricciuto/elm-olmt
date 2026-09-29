#!/usr/bin/env python3
"""Summarize dynamic peat density and burial throughflow through a spinup."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from .plot_spruce_mcfarlane_profiles import COLORS, TOPOUNITS, read_model_profiles


SECONDS_PER_YEAR = 365.0 * 86400.0
PLOT_INTERFACE_DEPTHS = (0.045, 0.166, 0.493, 0.829, 1.383, 2.296)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history_glob")
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _year(path: str) -> int:
    match = re.search(r"\.elm\.h3\.(\d{4})-", path)
    if match is None:
        raise ValueError(f"Cannot parse model year from {path}")
    return int(match.group(1))


def main() -> None:
    arguments = parse_args()
    files = sorted(glob.glob(arguments.history_glob), key=_year)
    files = [path for path in files if _year(path) <= 207]
    if not files:
        raise FileNotFoundError(arguments.history_glob)

    model, _ = read_model_profiles(str(arguments.restart))
    layer_depth = np.asarray(model["hollow"]["depth_m"], dtype=float)
    thickness = np.asarray(model["hollow"]["thickness_m"], dtype=float)
    lower_interface_depth = np.cumsum(thickness)
    selected_levels = sorted(
        {
            int(np.nanargmin(np.abs(lower_interface_depth - requested)))
            for requested in PLOT_INTERFACE_DEPTHS
        }
    )

    rows: list[dict[str, float | int | str]] = []
    for path in files:
        year = _year(path)
        with netCDF4.Dataset(path) as dataset:
            for label in TOPOUNITS:
                column = int(model[label]["column"])
                flux = np.ma.asarray(
                    dataset.variables["PEAT_C_BURIAL_FLUX"][:, :, column]
                )
                density = np.ma.asarray(
                    dataset.variables["PEAT_C_DENSITY"][:, :, column]
                )
                target = np.ma.asarray(
                    dataset.variables["PEAT_C_TARGET_DENSITY"][:, :, column]
                )
                mean_flux = np.ma.mean(flux, axis=0).filled(np.nan) * SECONDS_PER_YEAR
                mean_density = np.ma.mean(density, axis=0).filled(np.nan) / 1000.0
                mean_target = np.ma.mean(target, axis=0).filled(np.nan) / 1000.0
                active_fraction = np.ma.mean(flux > 1.0e-12, axis=0).filled(np.nan)
                flux_variability = np.ma.std(flux, axis=0).filled(np.nan) * SECONDS_PER_YEAR
                for level in range(len(layer_depth)):
                    rows.append(
                        {
                            "year": year,
                            "topounit": label,
                            "level": level + 1,
                            "layer_depth_m": float(layer_depth[level]),
                            "lower_interface_depth_m": float(lower_interface_depth[level]),
                            "mean_burial_flux_gC_m2_yr": float(mean_flux[level]),
                            "sd_daily_burial_flux_gC_m2_yr": float(flux_variability[level]),
                            "fraction_days_active": float(active_fraction[level]),
                            "density_kgC_m3": float(mean_density[level]),
                            "target_density_kgC_m3": float(mean_target[level]),
                            "density_minus_target_kgC_m3": float(
                                mean_density[level] - mean_target[level]
                            ),
                        }
                    )

    frame = pd.DataFrame(rows)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output_dir / "peat_throughflow_evolution.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 9.0), sharex=True)
    line_colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(selected_levels)))
    for axis, label in zip(axes, TOPOUNITS):
        subset = frame[frame["topounit"] == label]
        for color, level in zip(line_colors, selected_levels):
            values = subset[subset["level"] == level + 1]
            axis.plot(
                values["year"],
                values["mean_burial_flux_gC_m2_yr"],
                color=color,
                lw=1.5,
                label=f"{lower_interface_depth[level]:.2f} m",
            )
        axis.set_ylabel("Annual mean burial flux\n(g C m$^{-2}$ yr$^{-1}$)")
        axis.set_title(label.capitalize())
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, frameon=False, title="Interface depth")
    axes[-1].set_xlabel("Accelerated-decomposition spinup year")
    figure.suptitle("SPRUCE local peat burial throughflow evolution")
    figure.tight_layout()
    figure.savefig(
        arguments.output_dir / "peat_throughflow_evolution.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 9.0), sharex=True)
    line_colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(selected_levels)))
    for axis, label in zip(axes, TOPOUNITS):
        subset = frame[(frame["topounit"] == label) & (frame["year"] >= 50)]
        for color, level in zip(line_colors, selected_levels):
            values = subset[subset["level"] == level + 1]
            axis.plot(
                values["year"],
                values["mean_burial_flux_gC_m2_yr"],
                color=color,
                lw=1.5,
                label=f"{lower_interface_depth[level]:.2f} m",
            )
        axis.set_ylabel("Annual mean burial flux\n(g C m$^{-2}$ yr$^{-1}$)")
        axis.set_title(label.capitalize())
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, frameon=False, title="Interface depth")
    axes[-1].set_xlabel("Accelerated-decomposition spinup year")
    figure.suptitle("SPRUCE local peat burial throughflow after AD year 50")
    figure.tight_layout()
    figure.savefig(
        arguments.output_dir / "peat_throughflow_evolution_after50.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    plot_levels = [
        int(np.nanargmin(np.abs(layer_depth - depth)))
        for depth in (0.12, 0.37, 0.62, 1.04, 1.73)
    ]
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 9.0), sharex=True)
    line_colors = plt.cm.plasma(np.linspace(0.08, 0.9, len(plot_levels)))
    for axis, label in zip(axes, TOPOUNITS):
        subset = frame[frame["topounit"] == label]
        for color, level in zip(line_colors, plot_levels):
            values = subset[subset["level"] == level + 1]
            axis.plot(
                values["year"],
                values["density_minus_target_kgC_m3"],
                color=color,
                lw=1.5,
                label=f"{layer_depth[level]:.2f} m",
            )
        axis.axhline(0.0, color="black", lw=0.9, ls="--")
        axis.set_ylabel("Density minus target\n(kg C m$^{-3}$)")
        axis.set_title(label.capitalize())
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, frameon=False, title="Layer center")
    axes[-1].set_xlabel("Accelerated-decomposition spinup year")
    figure.suptitle("SPRUCE peat-density convergence")
    figure.tight_layout()
    figure.savefig(
        arguments.output_dir / "peat_density_target_evolution.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    final = frame[frame["year"].between(199, 207)]
    summary = (
        final.groupby(["topounit", "level"], as_index=False)
        .agg(
            layer_depth_m=("layer_depth_m", "first"),
            lower_interface_depth_m=("lower_interface_depth_m", "first"),
            mean_burial_flux_gC_m2_yr=("mean_burial_flux_gC_m2_yr", "mean"),
            year_to_year_sd_gC_m2_yr=("mean_burial_flux_gC_m2_yr", "std"),
            mean_fraction_days_active=("fraction_days_active", "mean"),
            mean_density_minus_target_kgC_m3=(
                "density_minus_target_kgC_m3",
                "mean",
            ),
        )
    )
    summary.to_csv(arguments.output_dir / "peat_throughflow_final9yr_summary.csv", index=False)

    print(arguments.output_dir / "peat_throughflow_evolution.png")
    print(arguments.output_dir / "peat_throughflow_evolution_after50.png")
    print(arguments.output_dir / "peat_density_target_evolution.png")
    print(arguments.output_dir / "peat_throughflow_final9yr_summary.csv")


if __name__ == "__main__":
    main()

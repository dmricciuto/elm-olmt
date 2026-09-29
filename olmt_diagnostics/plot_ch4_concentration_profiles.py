#!/usr/bin/env python3
"""Plot hollow and hummock porewater CH4 profiles from an OLMT history file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-depth", type=float, default=3.0)
    args = parser.parse_args()

    with nc.Dataset(args.history_file) as dataset:
        depth = np.asarray(dataset.variables["levdcmp"][:], dtype=float)
        concentration = np.ma.filled(
            dataset.variables["MM_CH4_POREWATER"][:], np.nan
        ).astype(float)
        soil_ice = np.ma.filled(dataset.variables["SOILICE"][:], np.nan).astype(float)
        time = dataset.variables["time"]
        dates = nc.num2date(time[:], time.units, getattr(time, "calendar", "standard"))

    months = np.asarray([date.month for date in dates])
    active_season = (months >= 4) & (months <= 11)
    columns = ((1, "Hollow", "tab:blue"), (2, "Hummock", "tab:orange"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=False)
    for axis, max_depth, title in zip(
        axes, (args.max_depth, min(0.7, args.max_depth)), ("Full peat profile", "Upper peat"),
    ):
        keep = depth <= max_depth
        for column, label, color in columns:
            values = concentration[:, :, column]
            annual_mean = np.nanmean(values, axis=0)
            annual_p05 = np.nanpercentile(values, 5, axis=0)
            annual_p95 = np.nanpercentile(values, 95, axis=0)

            thawed_active_mean = np.full(depth.shape, np.nan)
            for layer in range(depth.size):
                valid = active_season & (soil_ice[:, layer, column] < 0.01)
                if np.any(valid):
                    thawed_active_mean[layer] = np.nanmean(values[valid, layer])

            axis.fill_betweenx(
                depth[keep], annual_p05[keep], annual_p95[keep], color=color, alpha=0.16
            )
            axis.plot(annual_mean[keep], depth[keep], color=color, lw=2.4, label=f"{label}: annual mean")
            axis.plot(
                thawed_active_mean[keep], depth[keep], color=color, lw=2.0, ls="--",
                label=f"{label}: Apr–Nov, thawed",
            )

        axis.invert_yaxis()
        axis.set_ylim(max_depth, 0.0)
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.25)
        axis.set_title(title)
        axis.set_xlabel(r"Porewater CH$_4$ (mmol L$^{-1}$)")

    for axis in axes:
        axis.set_ylabel("Depth below local surface (m)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("SPRUCE mature-restart porewater methane profiles")
    fig.text(
        0.5, 0.015,
        "Solid: annual mean; shading: daily 5–95%; dashed: ice-free April–November mean",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()

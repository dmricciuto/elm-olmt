#!/usr/bin/env python3
"""Plot last-year SPRUCE oxygen-stress profiles from indexed ELM history."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


TOPOUNITS = {1: "Fen", 2: "Hollow", 3: "Hummock"}
COLORS = {2: "#2878b5", 3: "#d95319"}


def profile(data: np.ma.MaskedArray, col: int, nlev: int) -> tuple[np.ndarray, ...]:
    values = np.ma.filled(data[:, :nlev, col], np.nan)
    return (
        np.nanmean(values, axis=0),
        np.nanpercentile(values, 5, axis=0),
        np.nanpercentile(values, 95, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--hummock-offset", type=float, default=0.15,
                        help="hummock surface elevation above hollow (m)")
    args = parser.parse_args()

    with netCDF4.Dataset(args.history_file) as ds:
        depth = np.asarray(ds.variables["levdcmp"][:])
        nlev = int(np.count_nonzero(depth <= 3.0))
        depth = depth[:nlev]
        active = np.asarray(ds.variables["cols1d_active"][:]).astype(bool)
        topo = np.asarray(ds.variables["cols1d_topounit"][:]).astype(int)
        columns = {
            t: int(np.flatnonzero(active & (topo == t))[0]) for t in (2, 3)
        }
        diagnostic_names = [
            "O_SCALAR",
            "MM_O2_STRESS_UNSAT",
            "MM_O2_STRESS_SAT",
            "MM_UNSAT_WT_SAT_FRAC",
        ]
        diagnostic_names.extend(
            name for name in ("MM_O2_STRESS_ABOVE_WT", "MM_O2_STRESS_BELOW_WT")
            if name in ds.variables
        )
        data = {
            name: ds.variables[name][:]
            for name in diagnostic_names
        }
        zwt = np.ma.filled(ds.variables["ZWT"][:], np.nan)

    zwt_stats = {
        t: (float(np.nanmean(zwt[:, col])), float(np.nanmin(zwt[:, col])),
            float(np.nanmax(zwt[:, col])))
        for t, col in columns.items()
    }

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 5.4))
    panels = (
        ("O_SCALAR", "O scalar: local depth"),
        ("O_SCALAR", "O scalar: common elevation"),
        ("MM_O2_STRESS_UNSAT", "Mapped non-inundated O stress"),
        ("MM_UNSAT_WT_SAT_FRAC", "Layer fraction below water table"),
    )
    for panel_index, (ax, (variable, title)) in enumerate(zip(axes, panels)):
        for t, col in columns.items():
            mean, low, high = profile(data[variable], col, nlev)
            color = COLORS[t]
            plot_depth = depth.copy()
            if panel_index == 1 and t == 3:
                plot_depth = plot_depth - args.hummock_offset
            ax.fill_betweenx(plot_depth, low, high, color=color, alpha=0.14)
            ax.plot(mean, plot_depth, lw=2.2, color=color, label=TOPOUNITS[t])
        ax.set_xlim(0.0, 1.02)
        ax.set_xlabel("Annual mean (shading: daily 5–95%)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.22)
        ax.invert_yaxis()
    for ax in (axes[0], axes[2], axes[3]):
        ax.set_ylim(0.60, 0.0)
    axes[1].set_ylim(0.50, -0.18)
    for t in columns:
        axes[0].axhline(zwt_stats[t][0], color=COLORS[t], ls="--", lw=1.0, alpha=0.65)
        common_zwt = zwt_stats[t][0] - (args.hummock_offset if t == 3 else 0.0)
        axes[1].axhline(common_zwt, color=COLORS[t], ls="--", lw=1.0, alpha=0.65)
    axes[0].set_ylabel("Depth below local surface (m)")
    axes[1].set_ylabel("Depth below hollow-surface datum (m)")
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].legend(loc="lower left", frameon=False)

    stress_axis = axes[2]
    if "MM_O2_STRESS_ABOVE_WT" in data:
        for t, col in columns.items():
            for variable, linestyle, suffix in (
                ("MM_O2_STRESS_ABOVE_WT", ":", "above-WT endpoint"),
                ("MM_O2_STRESS_BELOW_WT", "--", "below-WT endpoint"),
            ):
                mean = np.nanmean(
                    np.ma.filled(data[variable][:, :nlev, col], np.nan), axis=0
                )
                stress_axis.plot(mean, depth, lw=1.35, color=COLORS[t], ls=linestyle,
                                 label=f"{TOPOUNITS[t]} {suffix}")
        stress_axis.legend(loc="lower left", frameon=False, fontsize=7)

    zwt_text = []
    for t in columns:
        mean_zwt, min_zwt, max_zwt = zwt_stats[t]
        zwt_text.append(
            f"{TOPOUNITS[t]} ZWT: {mean_zwt:.2f} m "
            f"({min_zwt:.2f} to {max_zwt:.2f})"
        )
    fig.text(0.5, 0.015, "Last simulation year; " + "; ".join(zwt_text), ha="center", fontsize=9)
    fig.suptitle("SPRUCE final-year oxygen limitation with water-table-resolved mapping", y=0.985)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")

    print("topounit,depth_m,o_scalar_mean,o_scalar_p05,o_scalar_p95,unsat_stress_mean,sat_stress_mean,below_wt_fraction")
    for t, col in columns.items():
        o_mean, o_low, o_high = profile(data["O_SCALAR"], col, nlev)
        u_mean, _, _ = profile(data["MM_O2_STRESS_UNSAT"], col, nlev)
        s_mean, _, _ = profile(data["MM_O2_STRESS_SAT"], col, nlev)
        f_mean, _, _ = profile(data["MM_UNSAT_WT_SAT_FRAC"], col, nlev)
        for row in zip(depth, o_mean, o_low, o_high, u_mean, s_mean, f_mean):
            print(f"{TOPOUNITS[t]}," + ",".join(f"{value:.6g}" for value in row))


if __name__ == "__main__":
    main()

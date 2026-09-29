#!/usr/bin/env python3
"""Plot annual vascular-PFT productivity and biomass for SPRUCE bog units."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


PFTS = {
    3: "Evergreen tree",
    5: "Deciduous tree",
    14: "Deciduous shrub",
}
TOPOS = {2: "Hollow", 3: "Hummock"}
COLORS = {"Hollow": "#2678b2", "Hummock": "#d95f02"}


def history_file(run_dir: Path, year: int) -> Path:
    matches = sorted(run_dir.glob(f"*.elm.h3.{year:04d}-*.nc"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one h3 file for year {year}, found {matches}")
    return matches[0]


def read_annual(run_dir: Path, years):
    result = {
        (pft, topo): {"GPP": [], "NPP": [], "TOTVEGC": []}
        for pft in PFTS for topo in TOPOS
    }
    for year in years:
        with Dataset(history_file(run_dir, year)) as ds:
            pft_type = np.asarray(ds["pfts1d_itype_veg"][:])
            pft_topo = np.asarray(ds["pfts1d_topounit"][:])
            active = np.asarray(ds["pfts1d_active"][:])
            for pft in PFTS:
                for topo in TOPOS:
                    indices = np.where((pft_type == pft) & (pft_topo == topo) &
                                       (active == 1))[0]
                    if len(indices) != 1:
                        raise RuntimeError(
                            f"Could not uniquely map PFT {pft}, topo {topo}: {indices}"
                        )
                    index = int(indices[0])
                    result[pft, topo]["GPP"].append(
                        float(np.sum(ds["GPP"][:, index])) * 86400.0
                    )
                    result[pft, topo]["NPP"].append(
                        float(np.sum(ds["NPP"][:, index])) * 86400.0
                    )
                    result[pft, topo]["TOTVEGC"].append(
                        float(np.mean(ds["TOTVEGC"][:, index]))
                    )
    return result


def plot(years, data, output: Path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True,
                             constrained_layout=True)
    variables = (
        ("GPP", "Annual GPP\n(gC m$^{-2}$ yr$^{-1}$)"),
        ("NPP", "Annual NPP\n(gC m$^{-2}$ yr$^{-1}$)"),
        ("TOTVEGC", "Mean live vegetation C\n(gC m$^{-2}$)"),
    )
    for column, (pft, title) in enumerate(PFTS.items()):
        axes[0, column].set_title(title)
        for row, (variable, ylabel) in enumerate(variables):
            ax = axes[row, column]
            for topo, topo_name in TOPOS.items():
                ax.plot(years, data[pft, topo][variable], marker="o", lw=1.7,
                        color=COLORS[topo_name], label=topo_name)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Ad-spinup model year")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("SPRUCE vascular PFTs: final seven ad-spinup years")
    fig.savefig(output, dpi=180)


def write_csv(years, data, output: Path):
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "pft", "topounit", "GPP_gC_m2_yr",
                         "NPP_gC_m2_yr", "TOTVEGC_gC_m2"])
        for year_index, year in enumerate(years):
            for pft, pft_name in PFTS.items():
                for topo, topo_name in TOPOS.items():
                    values = data[pft, topo]
                    writer.writerow([
                        year, pft_name, topo_name,
                        values["GPP"][year_index],
                        values["NPP"][year_index],
                        values["TOTVEGC"][year_index],
                    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start-year", type=int, default=44)
    parser.add_argument("--end-year", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(args.start_year, args.end_year + 1))
    data = read_annual(args.run_dir, years)
    stem = f"vascular_pft_productivity_years_{args.start_year}_{args.end_year}"
    plot(years, data, args.output_dir / f"{stem}.png")
    write_csv(years, data, args.output_dir / f"{stem}.csv")


if __name__ == "__main__":
    main()

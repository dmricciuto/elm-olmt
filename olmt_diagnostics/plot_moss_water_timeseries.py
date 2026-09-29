#!/usr/bin/env python3
"""Plot daily prognostic Sphagnum-water diagnostics from an OLMT run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


TOPOS = {2: "Hollow", 3: "Hummock"}
MOSS_PFT = 18
MOSS_CARBON_FRACTION_DRY_MASS = 0.5  # g C per g dry moss
WATER_CAPACITY_RATIO = 8.0  # g H2O per g dry moss
WATER_DRAINAGE_THRESHOLD = 8.5  # g H2O per g dry moss


def history_file(run_dir: Path, tape: str, year: int) -> Path:
    matches = sorted(run_dir.glob(f"*.elm.{tape}.{year:04d}-*.nc"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {tape} file for year {year}, found {matches}")
    return matches[0]


def read_series(run_dir: Path, years: range):
    data = {name: [] for name in (
        "rain", "snow", "storage_hollow", "storage_hummock",
        "carbon_hollow", "carbon_hummock", "ratio_hollow", "ratio_hummock",
        "potential_hollow", "potential_hummock", "exchange_hollow",
        "exchange_hummock", "evap_hollow", "evap_hummock", "npp_hollow",
        "npp_hummock", "gpp_hollow", "gpp_hummock", "wth_hollow", "wth_hummock",
    )}
    model_year = []
    day_of_year = []

    for year in years:
        with Dataset(history_file(run_dir, "h2", year)) as grid:
            if "RAIN" in grid.variables and "SNOW" in grid.variables:
                rain = np.asarray(grid["RAIN"][:], dtype=float).squeeze() * 86400.0
                snow = np.asarray(grid["SNOW"][:], dtype=float).squeeze() * 86400.0
            else:
                # Precipitation is optional in focused smoke-test histories.
                # Preserve the daily axis so water-state-only plots still run.
                rain = np.zeros(len(grid.dimensions["time"]), dtype=float)
                snow = np.zeros_like(rain)
        with Dataset(history_file(run_dir, "h3", year)) as sub:
            pft_type = np.asarray(sub["pfts1d_itype_veg"][:])
            pft_topo = np.asarray(sub["pfts1d_topounit"][:])
            pft_active = np.asarray(sub["pfts1d_active"][:])
            col_topo = np.asarray(sub["cols1d_topounit"][:])
            col_active = np.asarray(sub["cols1d_active"][:])

            nt = len(sub.dimensions["time"])
            model_year.extend([year] * nt)
            day_of_year.extend(range(1, nt + 1))
            data["rain"].extend(rain)
            data["snow"].extend(snow)

            for topo, label in TOPOS.items():
                key = label.lower()
                pidx = np.where((pft_type == MOSS_PFT) & (pft_topo == topo) &
                                (pft_active == 1))[0]
                cidx = np.where((col_topo == topo) & (col_active == 1))[0]
                if len(pidx) != 1 or len(cidx) != 1:
                    raise RuntimeError(f"Could not uniquely map {label}: pft={pidx}, col={cidx}")
                p, c = int(pidx[0]), int(cidx[0])
                storage = np.asarray(sub["H2O_MOSS_STORAGE"][:, p], dtype=float)
                carbon = np.asarray(sub["TOTVEGC"][:, p], dtype=float)
                dry_mass = carbon * 1.e-3 / MOSS_CARBON_FRACTION_DRY_MASS
                ratio = np.divide(storage, dry_mass,
                                  out=np.zeros_like(storage), where=carbon > 0.0)
                data[f"storage_{key}"].extend(storage)
                data[f"carbon_{key}"].extend(carbon)
                data[f"ratio_{key}"].extend(ratio)
                data[f"potential_{key}"].extend(
                    np.asarray(sub["MOSS_WATER_POTENTIAL"][:, p], dtype=float))
                data[f"exchange_{key}"].extend(
                    np.asarray(sub["QFLX_MOSS_SOIL"][:, p], dtype=float) * 86400.0)
                data[f"evap_{key}"].extend(
                    np.asarray(sub["QVEGT"][:, p], dtype=float) * 86400.0)
                data[f"npp_{key}"].extend(
                    np.asarray(sub["NPP"][:, p], dtype=float) * 86400.0)
                data[f"gpp_{key}"].extend(
                    np.asarray(sub["GPP"][:, p], dtype=float) * 86400.0)
                zwt = np.asarray(sub["ZWT"][:, c], dtype=float)
                h2osfc = np.asarray(sub["H2OSFC"][:, c], dtype=float)
                data[f"wth_{key}"].extend(-zwt + h2osfc / 1000.0)

    return (np.asarray(model_year), np.asarray(day_of_year),
            {name: np.asarray(values) for name, values in data.items()})


def forcing_year(model_year: int) -> int:
    return 2015 + (model_year - 1) % 9


def plot(model_year, day_of_year, data, years: range, output: Path):
    x = np.arange(len(model_year), dtype=float)
    fig, axes = plt.subplots(6, 1, figsize=(15, 16), sharex=True,
                             constrained_layout=True)
    colors = {"hollow": "#2678b2", "hummock": "#d95f02"}

    axes[0].bar(x, data["rain"], width=1.0, color="#4c9ed9", label="Rain")
    axes[0].bar(x, data["snow"], width=1.0, bottom=data["rain"],
                color="#b6d7ea", label="Snow")
    axes[0].set_ylabel("Precip.\n(mm d$^{-1}$)")
    axes[0].legend(ncol=2, loc="upper right")

    for key, color in colors.items():
        axes[1].plot(x, data[f"storage_{key}"], color=color, lw=0.9,
                     label=f"{key.title()} store")
        capacity = (data[f"carbon_{key}"] * 1.e-3 /
                    MOSS_CARBON_FRACTION_DRY_MASS * WATER_DRAINAGE_THRESHOLD)
        axes[1].plot(x, capacity, color=color, lw=0.9, ls="--",
                     label=f"{key.title()} drainage threshold")
    axes[1].set_ylabel("Moss water\n(kg m$^{-2}$)")
    axes[1].legend(ncol=2, loc="upper right")

    for key, color in colors.items():
        axes[2].plot(x, data[f"ratio_{key}"], color=color, lw=0.9,
                     label=key.title())
    axes[2].axhline(0.5, color="0.45", ls=":", lw=1, label="dry threshold")
    axes[2].axhline(8.0, color="0.45", ls="--", lw=1, label="capacity")
    axes[2].set_ylabel("Water / dry moss\n(g g$^{-1}$)")
    axes[2].legend(ncol=4, loc="upper right")

    for key, color in colors.items():
        axes[3].plot(x, data[f"exchange_{key}"], color=color, lw=0.75,
                     label=f"{key.title()} soil→moss")
        axes[3].plot(x, -data[f"evap_{key}"], color=color, lw=0.7, ls="--",
                     label=f"−{key.title()} evap.")
    axes[3].axhline(0.0, color="0.3", lw=0.7)
    axes[3].set_ylabel("Water flux\n(mm d$^{-1}$)")
    axes[3].legend(ncol=2, loc="lower left")

    for key, color in colors.items():
        axes[4].plot(x, data[f"npp_{key}"], color=color, lw=0.9,
                     label=key.title())
    axes[4].axhline(0.0, color="0.3", lw=0.7)
    axes[4].set_ylabel("Moss NPP\n(gC m$^{-2}$ d$^{-1}$)")
    axes[4].legend(loc="upper right")

    for key, color in colors.items():
        axes[5].plot(x, data[f"wth_{key}"], color=color, lw=0.9,
                     label=key.title())
    axes[5].axhline(0.0, color="0.3", lw=0.7)
    axes[5].set_ylabel("Water-table height\n(m above surface)")
    axes[5].legend(loc="lower left")

    ticks, labels = [], []
    for year in years:
        idx = int(np.where((model_year == year) & (day_of_year == 1))[0][0])
        ticks.append(idx)
        labels.append(f"Model {year}\nforcing {forcing_year(year)}")
        if forcing_year(year) in (2018, 2019):
            stop = idx + int(np.sum(model_year == year))
            for ax in axes:
                ax.axvspan(idx, stop, color="#d9a441", alpha=0.12, lw=0)
    axes[-1].set_xticks(ticks, labels)
    axes[-1].set_xlim(0, len(x) - 1)
    fig.suptitle("SPRUCE prognostic Sphagnum water: final seven ad-spinup years\n"
                 "shading marks 2018–2019 forcing (dry-period context)", fontsize=15)
    fig.savefig(output, dpi=180)


def write_csv(model_year, day_of_year, data, output: Path):
    names = list(data)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_year", "forcing_year", "day_of_year", *names])
        for i in range(len(model_year)):
            writer.writerow([model_year[i], forcing_year(int(model_year[i])),
                             day_of_year[i], *(data[name][i] for name in names)])


def plot_annual_trajectory(run_dir: Path, end_year: int, output: Path):
    years = np.arange(1, end_year + 1)
    carbon = {key: [] for key in ("hollow", "hummock")}
    npp = {key: [] for key in ("hollow", "hummock")}
    dry_fraction = {key: [] for key in ("hollow", "hummock")}
    pft_indices = None
    for year in years:
        with Dataset(history_file(run_dir, "h3", int(year))) as ds:
            if pft_indices is None:
                pft_type = np.asarray(ds["pfts1d_itype_veg"][:])
                pft_topo = np.asarray(ds["pfts1d_topounit"][:])
                pft_active = np.asarray(ds["pfts1d_active"][:])
                pft_indices = {
                    label.lower(): int(np.where(
                        (pft_type == MOSS_PFT) & (pft_topo == topo) &
                        (pft_active == 1))[0][0])
                    for topo, label in TOPOS.items()
                }
            for key, p in pft_indices.items():
                c = np.asarray(ds["TOTVEGC"][:, p], dtype=float)
                store = np.asarray(ds["H2O_MOSS_STORAGE"][:, p], dtype=float)
                dry_mass = c * 1.e-3 / MOSS_CARBON_FRACTION_DRY_MASS
                ratio = np.divide(store, dry_mass, out=np.zeros_like(store), where=c > 0)
                carbon[key].append(float(np.mean(c)))
                npp[key].append(float(np.sum(ds["NPP"][:, p]) * 86400.0))
                dry_fraction[key].append(float(np.mean(ratio < 0.5)))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             constrained_layout=True)
    colors = {"hollow": "#2678b2", "hummock": "#d95f02"}
    for key, color in colors.items():
        axes[0].semilogy(years, np.maximum(carbon[key], 1.e-6), color=color,
                         label=key.title())
        axes[1].plot(years, npp[key], color=color, label=key.title())
        axes[2].plot(years, dry_fraction[key], color=color, label=key.title())
    axes[0].set_ylabel("Mean live moss C\n(gC m$^{-2}$; log scale)")
    axes[1].set_ylabel("Annual moss NPP\n(gC m$^{-2}$ yr$^{-1}$)")
    axes[2].set_ylabel("Fraction of days\nwater ratio < 0.5")
    axes[2].set_xlabel("Ad-spinup model year")
    axes[2].set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("SPRUCE Sphagnum trajectory from 10 gC m$^{-2}$ seed")
    fig.savefig(output, dpi=180)


def plot_storage_ratio(model_year, day_of_year, data, output: Path):
    """Show the conserved water store per unit living moss dry mass."""
    x = np.arange(len(model_year), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             constrained_layout=True)
    styles = {
        "hollow": ("#2678b2", "-"),
        "hummock": ("#d95f02", "--"),
    }
    for key, (color, linestyle) in styles.items():
        for ax in axes:
            ax.plot(x, data[f"ratio_{key}"], color=color, ls=linestyle,
                    lw=1.2, label=key.title())
    for ax in axes:
        ax.axhline(0.5, color="0.35", ls=":", lw=1.0,
                   label="dry-response bound (0.5)")
        ax.axhline(8.0, color="0.35", ls="-.", lw=1.0,
                   label="physiological saturation (8)")
        ax.axhline(WATER_DRAINAGE_THRESHOLD, color="0.55", ls="--", lw=0.9,
                   label="drainage threshold (8.5)")
        ax.grid(alpha=0.2)
    axes[0].set_ylim(-0.15, 8.9)
    axes[0].set_ylabel("Water / dry moss\n(g H$_2$O g$^{-1}$ dry mass)")
    axes[0].legend(ncol=2, loc="upper right")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1.e-4, 10.0)
    axes[1].set_ylabel("Water / dry moss\n(log scale)")
    unique_years = np.unique(model_year)
    if len(unique_years) == 1:
        axes[1].set_xlabel("Day of model year")
        ticks = np.linspace(0, len(x) - 1, 7, dtype=int)
        labels = [str(int(day_of_year[i])) for i in ticks]
    else:
        ticks = [int(np.where((model_year == year) & (day_of_year == 1))[0][0])
                 for year in unique_years]
        labels = [f"{int(year)}\n({forcing_year(int(year))})"
                  for year in unique_years]
        axes[1].set_xlabel("Model year (forcing year)")
        for ax in axes:
            for tick in ticks[1:]:
                ax.axvline(tick, color="0.75", lw=0.6, alpha=0.7)
    axes[1].set_xticks(ticks, labels)
    fig.suptitle("SPRUCE prognostic moss storage ratio")
    fig.savefig(output, dpi=180)


def plot_collapse_budget(model_year, day_of_year, data, output: Path,
                         first_day: int = 45, last_day: int = 70):
    """Plot the daily internal-store budget around the initial dry-down."""
    select = ((model_year == model_year[0]) & (day_of_year >= first_day) &
              (day_of_year <= last_day))
    doy = day_of_year[select]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True,
                             constrained_layout=True)
    colors = {"hollow": "#2678b2", "hummock": "#d95f02"}
    for column, key in enumerate(("hollow", "hummock")):
        color = colors[key]
        storage = data[f"storage_{key}"][select]
        ratio = data[f"ratio_{key}"][select]
        soil = data[f"exchange_{key}"][select]
        evap = data[f"evap_{key}"][select]
        net = soil - evap

        axes[0, column].plot(doy, storage, color=color, lw=1.5,
                             label="Internal storage")
        axes[0, column].set_title(key.title())
        axes[0, column].set_ylabel("Storage (mm)")
        ratio_axis = axes[0, column].twinx()
        ratio_axis.plot(doy, ratio, color="0.25", ls="--", lw=1.0,
                        label="Water / dry mass")
        ratio_axis.axhline(0.5, color="0.55", ls=":", lw=0.9)
        ratio_axis.set_ylabel("g H$_2$O g$^{-1}$ dry mass")

        axes[1, column].plot(doy, soil, color="#2ca02c", lw=1.4,
                             label="Soil→moss exchange")
        axes[1, column].plot(doy, -evap, color="#9467bd", lw=1.4,
                             label="− canopy moss evaporation")
        axes[1, column].plot(doy, net, color="black", ls="--", lw=1.1,
                             label="Net diagnosed tendency")
        axes[1, column].axhline(0.0, color="0.4", lw=0.7)
        axes[1, column].set_ylabel("Budget term (mm d$^{-1}$)")
        axes[1, column].legend(loc="lower left", fontsize=8)

        predicted = storage[0] + np.r_[0.0, np.cumsum(net[:-1])]
        axes[2, column].plot(doy, storage, color=color, lw=1.6,
                             label="Daily-mean stored water")
        axes[2, column].plot(doy, predicted, color="black", ls="--", lw=1.2,
                             label="Integrated exchange − evaporation")
        axes[2, column].set_ylabel("Storage (mm)")
        axes[2, column].set_xlabel("Day of model year")
        axes[2, column].legend(loc="upper right", fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("SPRUCE prognostic moss-water budget during initial collapse\n"
                 "positive exchange is upper soil → moss")
    fig.savefig(output, dpi=180)


def plot_productivity(run_dir: Path, model_year, day_of_year, data,
                      output: Path):
    """Plot daily, cumulative, and annual NPP for the active moss PFTs."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    colors = {"hollow": "#2678b2", "hummock": "#d95f02"}
    for key, color in colors.items():
        axes[0].plot(day_of_year, data[f"npp_{key}"], color=color, lw=1.1,
                     label=key.title())
        axes[1].plot(day_of_year, np.cumsum(data[f"npp_{key}"]),
                     color=color, lw=1.4, label=key.title())
    axes[0].axhline(0.0, color="0.4", lw=0.7)
    axes[0].set_ylabel("Daily moss NPP\n(gC m$^{-2}$ d$^{-1}$)")
    axes[0].legend()
    axes[1].set_ylabel("Cumulative moss NPP\n(gC m$^{-2}$)")
    axes[1].legend()

    npp = {key: float(np.sum(data[f"npp_{key}"])) for key in colors}
    positions = np.arange(2)
    bars = axes[2].bar(positions, [npp["hollow"], npp["hummock"]],
                       0.55, color=[colors["hollow"], colors["hummock"]])
    axes[2].set_xticks(positions, ["Hollow", "Hummock"])
    axes[2].set_ylabel("Annual moss NPP\n(gC m$^{-2}$ yr$^{-1}$)")
    axes[2].bar_label(bars, fmt="%.1f")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].set_xlabel("Day of model year")
    fig.suptitle("SPRUCE moss NPP with retained internal water")
    fig.savefig(output, dpi=180)


def plot_npp_vs_internal_water(day_of_year, data, output: Path):
    """Plot realized daily moss NPP against internal dry-mass water content."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True,
                             constrained_layout=True)
    scatter = None
    for ax, key in zip(axes, ("hollow", "hummock")):
        water = data[f"ratio_{key}"]
        npp = data[f"npp_{key}"]
        valid = np.isfinite(water) & np.isfinite(npp)
        scatter = ax.scatter(water[valid], npp[valid], c=day_of_year[valid],
                             cmap="viridis", s=18, alpha=0.65,
                             edgecolors="none", label="Daily values")

        edges = np.arange(0.5, 9.01, 0.5)
        centers, means, lower, upper = [], [], [], []
        for left, right in zip(edges[:-1], edges[1:]):
            selected = valid & (water >= left) & (water < right)
            if np.count_nonzero(selected) >= 3:
                values = npp[selected]
                centers.append(0.5 * (left + right))
                means.append(np.mean(values))
                lower.append(np.quantile(values, 0.25))
                upper.append(np.quantile(values, 0.75))
        centers = np.asarray(centers)
        means = np.asarray(means)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        ax.plot(centers, means, color="black", lw=2.0, marker="o", ms=4,
                label="0.5-unit bin mean")
        ax.fill_between(centers, lower, upper, color="black", alpha=0.12,
                        label="interquartile range")
        ax.axvline(0.5, color="0.45", ls=":", lw=1.0,
                   label="dry-response bound")
        ax.axvline(8.0, color="0.45", ls="-.", lw=1.0,
                   label="physiological saturation")
        ax.axhline(0.0, color="0.4", lw=0.7)
        ax.set_title(key.title())
        ax.set_xlabel("Internal water (g H$_2$O g$^{-1}$ dry moss)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Daily moss NPP (gC m$^{-2}$ d$^{-1}$)")
    axes[1].legend(loc="upper left", fontsize=8)
    colorbar = fig.colorbar(scatter, ax=axes, label="Day of model year")
    colorbar.ax.tick_params(labelsize=9)
    fig.suptitle("Realized moss NPP as a function of internal water\n"
                 "color shows seasonal timing; black line is the binned daily mean")
    fig.savefig(output, dpi=180)


def plot_multiyear_productivity(model_year, day_of_year, data, years: range,
                                output: Path):
    """Plot daily and annual moss GPP/NPP over a selected year range."""
    x = np.arange(len(model_year), dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    colors = {"hollow": "#2678b2", "hummock": "#d95f02"}
    for key, color in colors.items():
        axes[0].plot(x, data[f"gpp_{key}"], color=color, lw=0.75,
                     label=key.title())
        axes[1].plot(x, data[f"npp_{key}"], color=color, lw=0.75,
                     label=key.title())
    axes[0].set_ylabel("Daily moss GPP\n(gC m$^{-2}$ d$^{-1}$)")
    axes[1].set_ylabel("Daily moss NPP\n(gC m$^{-2}$ d$^{-1}$)")
    axes[1].axhline(0.0, color="0.4", lw=0.7)
    axes[0].legend(ncol=2)
    axes[1].legend(ncol=2)

    year_values = np.asarray(list(years))
    for key, color in colors.items():
        annual_gpp = [np.sum(data[f"gpp_{key}"][model_year == year])
                      for year in year_values]
        annual_npp = [np.sum(data[f"npp_{key}"][model_year == year])
                      for year in year_values]
        axes[2].plot(year_values, annual_gpp, color=color, lw=1.8, marker="o",
                     label=f"{key.title()} GPP")
        axes[2].plot(year_values, annual_npp, color=color, lw=1.8, marker="s",
                     ls="--", label=f"{key.title()} NPP")
    axes[2].set_ylabel("Annual productivity\n(gC m$^{-2}$ yr$^{-1}$)")
    axes[2].set_xlabel("Ad-spinup model year")
    axes[2].legend(ncol=2)

    ticks, labels = [], []
    for year in years:
        idx = int(np.where((model_year == year) & (day_of_year == 1))[0][0])
        ticks.append(idx)
        labels.append(f"{year}\n({forcing_year(year)})")
    for ax in axes[:2]:
        ax.set_xticks(ticks, labels)
        ax.set_xlim(0, len(x) - 1)
    axes[1].set_xlabel("Model year (forcing year)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("SPRUCE moss productivity with retained internal water")
    fig.savefig(output, dpi=180)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start-year", type=int, default=44)
    parser.add_argument("--end-year", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    years = range(args.start_year, args.end_year + 1)
    model_year, day_of_year, data = read_series(args.run_dir, years)
    stem = f"moss_water_daily_years_{args.start_year}_{args.end_year}"
    plot(model_year, day_of_year, data, years, args.output_dir / f"{stem}.png")
    write_csv(model_year, day_of_year, data, args.output_dir / f"{stem}.csv")
    plot_storage_ratio(
        model_year, day_of_year, data,
        args.output_dir / f"moss_storage_ratio_years_{args.start_year}_{args.end_year}.png",
    )
    plot_multiyear_productivity(
        model_year, day_of_year, data, years,
        args.output_dir / f"moss_productivity_years_{args.start_year}_{args.end_year}.png",
    )
    plot_npp_vs_internal_water(
        day_of_year, data,
        args.output_dir /
        f"moss_npp_vs_internal_water_years_{args.start_year}_{args.end_year}.png",
    )
    if args.start_year == 1:
        plot_collapse_budget(
            model_year, day_of_year, data,
            args.output_dir / "moss_storage_budget_collapse_days_45_70.png",
        )
        if args.end_year == 1:
            plot_productivity(
                args.run_dir, model_year, day_of_year, data,
                args.output_dir / "moss_npp_year_1.png",
            )
    plot_annual_trajectory(
        args.run_dir, args.end_year,
        args.output_dir / f"moss_annual_trajectory_years_1_{args.end_year}.png",
    )


if __name__ == "__main__":
    main()

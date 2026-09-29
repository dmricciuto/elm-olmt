#!/usr/bin/env python3
"""Compare an ELM SPRUCE peat profile with McFarlane et al. (2018).

The first three panels compare depth-resolved peat carbon density, C:N, and
bulk-peat Delta14C. The fourth compares cumulative peat C with the paper's
site-level stock and depth estimate. For accelerated-decomposition spinup,
the radiocarbon panel is explicitly labeled as diagnostic because model time
and decay are accelerated and no historical atmospheric bomb curve is used.
"""

from __future__ import annotations

import argparse
import glob
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd


C14_REFERENCE_RATIO = 1.0e-12
CARBON_POOLS = (
    "CWDC_vr",
    "LITR1C_vr",
    "LITR2C_vr",
    "LITR3C_vr",
    "SOIL1C_vr",
    "SOIL2C_vr",
    "SOIL3C_vr",
    "SOIL4C_vr",
    "DOMC_vr",
    "BACTERIAC_vr",
    "FUNGIC_vr",
)
ROOT_C_POOLS = ("FROOTC", "LIVECROOTC", "DEADCROOTC")
# The SPRUCE three-topounit surface uses 1=fen, 2=hollow, 3=hummock.
# McFarlane et al. excluded raised-hummock C from the reported site stock; the
# profile panels retain hollow and hummock separately and intentionally omit fen.
TOPOUNITS = {"hollow": 2, "hummock": 3}
COLORS = {"hollow": "#2b6cb0", "hummock": "#c53030", "bog mean": "#202020"}
# McFarlane et al. (2018) assign 0 cm to the hollow surface.  The standard
# three-topounit SPRUCE case places the hummock surface 0.15 m above that
# datum.  Retain local layer depths for inventories, but use this common
# coordinate for all profile/observation comparisons.
SURFACE_HEIGHT_ABOVE_HOLLOW_M = {"hollow": 0.0, "hummock": 0.15}


def _as_float(variable, time_index: int | None = None) -> np.ndarray:
    values = variable[:] if time_index is None else variable[time_index, ...]
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)


def _final_history_file(specification: str) -> Path:
    matches = sorted(Path(path) for path in glob.glob(specification))
    if not matches:
        path = Path(specification)
        if path.is_file():
            return path
        raise FileNotFoundError(f"No history file matched {specification!r}")
    return matches[-1]


def _find_column(dataset: netCDF4.Dataset, topounit: int) -> int:
    topounit_name = _variable_name(dataset, "cols1d_topounit") or _variable_name(
        dataset, "cols1d_topounit_index"
    )
    if topounit_name is None:
        raise KeyError(
            "Column-to-topounit mapping is required; refusing to interpret "
            "one-based topounit IDs as zero-based column indices"
        )
    ids = _as_float(dataset.variables[topounit_name]).astype(int)
    candidates = np.flatnonzero(ids == topounit)
    landunit_name = _variable_name(dataset, "cols1d_itype_lunit") or _variable_name(
        dataset, "cols1d_ityplun"
    )
    if landunit_name is not None:
        landunit_type = _as_float(dataset.variables[landunit_name]).astype(int)
        natural = candidates[landunit_type[candidates] == 1]
        if natural.size:
            candidates = natural
    if not candidates.size:
        raise ValueError(f"No natural-vegetation column for topounit {topounit}")
    weight_name = _variable_name(dataset, "cols1d_wtgcell") or _variable_name(
        dataset, "cols1d_wtxy"
    )
    if weight_name is not None:
        weights = _as_float(dataset.variables[weight_name])
        return int(candidates[np.nanargmax(weights[candidates])])
    return int(candidates[0])


def _variable_name(dataset: netCDF4.Dataset, requested: str) -> str | None:
    """Resolve a history-style field name against history or restart naming."""
    if requested in dataset.variables:
        return requested
    by_lower = {name.lower(): name for name in dataset.variables}
    return by_lower.get(requested.lower())


def _profile(dataset: netCDF4.Dataset, name: str, column: int) -> np.ndarray:
    resolved = _variable_name(dataset, name)
    if resolved is None:
        raise KeyError(name)
    variable = dataset.variables[resolved]
    values = _as_float(variable, -1 if "time" in variable.dimensions else None)
    dimensions = tuple(dim for dim in variable.dimensions if dim != "time")
    if "column" not in dimensions:
        return np.squeeze(values)
    axis = dimensions.index("column")
    return np.asarray(np.take(values, column, axis=axis), dtype=float).squeeze()


def _profile_timeseries(dataset: netCDF4.Dataset, name: str, column: int) -> np.ndarray:
    resolved = _variable_name(dataset, name)
    if resolved is None:
        raise KeyError(name)
    variable = dataset.variables[resolved]
    dimensions = list(variable.dimensions)
    values = _as_float(variable)
    if "column" in dimensions:
        column_axis = dimensions.index("column")
        values = np.take(values, column, axis=column_axis)
        dimensions.pop(column_axis)
    if "time" not in dimensions:
        return np.asarray(values, dtype=float)[None, ...]
    time_axis = dimensions.index("time")
    return np.asarray(np.moveaxis(values, time_axis, 0), dtype=float)


def _sum_profiles(dataset: netCDF4.Dataset, names: tuple[str, ...], column: int) -> np.ndarray:
    available = [name for name in names if _variable_name(dataset, name) is not None]
    if not available:
        raise KeyError(f"None of the requested profile variables are present: {names}")
    profiles = [_profile(dataset, name, column) for name in available]
    return np.nansum(np.stack(profiles), axis=0)


def _sum_profile_timeseries(
    dataset: netCDF4.Dataset, names: tuple[str, ...], column: int
) -> np.ndarray:
    available = [name for name in names if _variable_name(dataset, name) is not None]
    profiles = [_profile_timeseries(dataset, name, column) for name in available]
    return np.nansum(np.stack(profiles), axis=0)


def _nitrogen_name(carbon_name: str) -> str:
    return carbon_name.replace("C_vr", "N_vr")


def _c14_names(carbon_name: str) -> tuple[str, str]:
    # ELM history fields have used the prefix form, while restart state uses
    # names such as soil1c_14_vr.
    return "C14_" + carbon_name, carbon_name.replace("C_vr", "C_14_vr")


def _companion_history_file(state_file: Path) -> Path | None:
    """Find a history file carrying geometry and topounit/PFT mappings."""
    for tape in ("h1", "h0"):
        for candidate in sorted(state_file.parent.glob(f"*.elm.{tape}.*.nc")):
            with netCDF4.Dataset(candidate) as dataset:
                required_mappings = {
                    "cols1d_topounit",
                    "cols1d_itype_lunit",
                    "pfts1d_topounit",
                    "pfts1d_wttopounit",
                }
                has_geometry = (
                    {"ZSOI", "DZSOI"}.issubset(dataset.variables)
                    or "levdcmp" in dataset.variables
                )
                if required_mappings.issubset(dataset.variables) and has_geometry:
                    return candidate
    return None


def _root_carbon_by_topounit(
    state: netCDF4.Dataset, geometry: netCDF4.Dataset, topounit: int
) -> float:
    """Return root C per unit topounit area from layer-unresolved PFT state."""
    topo_name = _variable_name(geometry, "pfts1d_topounit") or _variable_name(
        geometry, "pfts1d_topounit_index"
    )
    weight_name = _variable_name(geometry, "pfts1d_wttopounit")
    if topo_name is None or weight_name is None:
        return float("nan")
    topo = _as_float(geometry.variables[topo_name]).astype(int)
    weights = _as_float(geometry.variables[weight_name])
    selected = topo == topounit
    total = 0.0
    found = False
    for requested in ROOT_C_POOLS:
        name = _variable_name(state, requested)
        if name is None or "pft" not in state.variables[name].dimensions:
            continue
        values = _as_float(state.variables[name], -1 if "time" in state.variables[name].dimensions else None)
        total += float(np.nansum(values[selected] * weights[selected]))
        found = True
    return total if found else float("nan")


def read_model_profiles(history_specification: str) -> tuple[dict, Path]:
    history_file = _final_history_file(history_specification)
    result: dict[str, dict[str, np.ndarray | float]] = {}
    with ExitStack() as stack:
        dataset = stack.enter_context(netCDF4.Dataset(history_file))
        geometry_file = _companion_history_file(history_file)
        geometry = (
            stack.enter_context(netCDF4.Dataset(geometry_file))
            if geometry_file is not None
            else dataset
        )
        available_carbon_pools = tuple(
            name for name in CARBON_POOLS if _variable_name(dataset, name) is not None
        )
        missing_carbon_pools = tuple(
            name for name in CARBON_POOLS if _variable_name(dataset, name) is None
        )
        available_nitrogen_pools = tuple(
            name
            for name in (_nitrogen_name(pool) for pool in CARBON_POOLS)
            if _variable_name(dataset, name) is not None
        )
        result["available_carbon_pools"] = available_carbon_pools
        result["missing_carbon_pools"] = missing_carbon_pools
        result["nitrogen_profiles_available"] = bool(available_nitrogen_pools)
        available_c14_pools = tuple(
            candidate
            for pool in CARBON_POOLS
            for candidate in _c14_names(pool)
            if _variable_name(dataset, candidate) is not None
        )
        result["radiocarbon_profiles_available"] = bool(available_c14_pools)
        result["root_carbon_available"] = any(
            _variable_name(dataset, name) is not None for name in ROOT_C_POOLS
        )

        for label, topounit in TOPOUNITS.items():
            column = _find_column(geometry, topounit)
            carbon = _sum_profiles(dataset, available_carbon_pools, column)
            nitrogen = (
                _sum_profiles(dataset, available_nitrogen_pools, column)
                if available_nitrogen_pools
                else np.full_like(carbon, np.nan)
            )
            c14 = (
                _sum_profiles(dataset, available_c14_pools, column)
                if available_c14_pools
                else np.full_like(carbon, np.nan)
            )

            if "ZSOI" in geometry.variables:
                depth = np.abs(_profile(geometry, "ZSOI", column))
            else:
                depth = np.abs(_as_float(geometry.variables["levdcmp"]))
            if "DZSOI" in geometry.variables:
                thickness = _profile(geometry, "DZSOI", column)
            else:
                interfaces = np.r_[0.0, 0.5 * (depth[1:] + depth[:-1])]
                thickness = np.diff(np.r_[interfaces, depth[-1] + (depth[-1] - interfaces[-1])])

            count = min(len(carbon), len(depth), len(thickness))
            carbon = carbon[:count]
            nitrogen = nitrogen[:count]
            c14 = c14[:count]
            depth = depth[:count]
            thickness = thickness[:count]
            cn = np.divide(
                carbon,
                nitrogen,
                out=np.full(count, np.nan),
                where=np.isfinite(nitrogen) & (nitrogen > 0.0) & (carbon > 1.0),
            )
            delta14c = 1000.0 * (
                np.divide(
                    c14,
                    carbon * C14_REFERENCE_RATIO,
                    out=np.full(count, np.nan),
                    where=np.isfinite(carbon) & (carbon > 1.0),
                )
                - 1.0
            )
            cumulative_carbon = np.cumsum(np.nan_to_num(carbon) * thickness) / 1000.0
            carbon_timeseries = _sum_profile_timeseries(dataset, CARBON_POOLS, column)[:, :count]
            c14_timeseries = (
                _sum_profile_timeseries(dataset, available_c14_pools, column)[:, :count]
                if available_c14_pools
                else np.full_like(carbon_timeseries, np.nan)
            )
            layer_top = np.maximum(0.0, depth - 0.5 * thickness)
            layer_bottom = depth + 0.5 * thickness
            surface_offset = SURFACE_HEIGHT_ABOVE_HOLLOW_M[label]
            depth_below_hollow = depth - surface_offset
            layer_top_below_hollow = layer_top - surface_offset
            layer_bottom_below_hollow = layer_bottom - surface_offset
            overlap_to_2p25m = np.maximum(
                0.0,
                np.minimum(layer_bottom_below_hollow, 2.25)
                - np.maximum(layer_top_below_hollow, 0.0),
            )
            stock_timeseries = np.sum(
                np.nan_to_num(carbon_timeseries) * overlap_to_2p25m[None, :], axis=1
            ) / 1000.0
            cumulative_carbon = np.cumsum(
                np.nan_to_num(carbon)
                * np.maximum(
                    0.0,
                    layer_bottom_below_hollow
                    - np.maximum(layer_top_below_hollow, 0.0),
                )
            ) / 1000.0
            root_c = _root_carbon_by_topounit(dataset, geometry, topounit)
            root_timeseries = np.full(len(stock_timeseries), root_c)
            delta14c_timeseries = 1000.0 * (
                np.divide(
                    c14_timeseries,
                    carbon_timeseries * C14_REFERENCE_RATIO,
                    out=np.full_like(c14_timeseries, np.nan),
                    where=np.isfinite(carbon_timeseries) & (carbon_timeseries > 0.0),
                )
                - 1.0
            )
            result[label] = {
                "column": column,
                "depth_m": depth,
                "depth_below_hollow_m": depth_below_hollow,
                "thickness_m": thickness,
                "layer_bottom_m": layer_bottom,
                "layer_bottom_below_hollow_m": layer_bottom_below_hollow,
                "carbon_g_m3": carbon,
                "nitrogen_g_m3": nitrogen,
                "cn": cn,
                "delta14c_permil": delta14c,
                "cumulative_carbon_kg_m2": cumulative_carbon,
                "carbon_stock_to_2p25m_kg_m2": float(stock_timeseries[-1]),
                "stock_to_2p25m_timeseries_kg_m2": stock_timeseries,
                "root_carbon_kg_m2": root_c / 1000.0,
                "stock_to_2p25m_plus_roots_timeseries_kg_m2": (
                    stock_timeseries + root_timeseries / 1000.0
                ),
                "delta14c_timeseries_permil": delta14c_timeseries,
            }

        weights = {}
        for label, topounit in TOPOUNITS.items():
            column = int(result[label]["column"])
            weight_name = _variable_name(
                geometry, "cols1d_wtgcell"
            ) or _variable_name(geometry, "cols1d_wtxy")
            weights[label] = (
                float(_as_float(geometry.variables[weight_name])[column])
                if weight_name is not None
                else 1.0
            )
        total_weight = sum(weights.values())
        result["weights"] = {label: value / total_weight for label, value in weights.items()}
    return result, history_file


def read_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=[1], encoding="latin1", na_values=[-9999, "-9999"])
    selected = frame[
        (frame["Vegetation"] == "T")
        | ((frame["Vegetation"] == "N") & frame["Plot"].isin([4, 5]))
    ].copy()
    sample_columns = [
        "Plot",
        "Vegetation",
        "Topography",
        "Depth_increment",
        "Top_core_depth",
        "Bottom_core_depth",
        "Mid_core_depth",
    ]
    value_columns = ["Bulk_density", "Carbon", "C_N_ratio", "Delta_14C_LLNL"]
    samples = selected.groupby(sample_columns, dropna=False)[value_columns].mean().reset_index()
    samples["profile"] = (
        samples["Plot"].astype(str)
        + "-"
        + samples["Vegetation"]
        + "-"
        + samples["Topography"]
    )

    # McFarlane et al. (2018) explicitly define 0 cm as the hollow surface.
    # Positive source-data depths are raised hummock peat; negative values are
    # below the hollow. Do not re-zero each core to its local surface.
    samples["depth_m"] = -samples["Mid_core_depth"] / 100.0
    samples["carbon_g_m3"] = (
        samples["Bulk_density"] * samples["Carbon"] * 1.0e4
    )
    return samples[(samples["depth_m"] >= -0.5) & (samples["depth_m"] <= 3.2)]


def binned_observations(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    edges = np.r_[np.arange(-0.40, 1.01, 0.10), np.arange(1.25, 3.26, 0.25)]
    bins = pd.cut(frame["depth_m"], edges, include_lowest=True)
    grouped = frame.assign(depth_bin=bins).groupby("depth_bin", observed=True)
    summary = grouped.agg(
        depth_m=("depth_m", "mean"),
        mean=(variable, "mean"),
        sd=(variable, "std"),
        n=(variable, "count"),
    )
    return summary[summary["n"] > 0].reset_index(drop=True)


def _bog_mean(model: dict, variable: str) -> tuple[np.ndarray, np.ndarray]:
    reference_depth = np.asarray(model["hollow"]["depth_below_hollow_m"])
    mean = np.zeros_like(reference_depth, dtype=float)
    for label in TOPOUNITS:
        depth = np.asarray(model[label]["depth_below_hollow_m"])
        values = np.asarray(model[label][variable])
        mean += model["weights"][label] * np.interp(reference_depth, depth, values)
    return reference_depth, mean


def make_figure(
    model: dict,
    observations: pd.DataFrame,
    output: Path,
    title: str,
    radiocarbon_note: str = "Model radiocarbon interpretation depends on atmospheric forcing",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), sharey=True)
    panels = (
        ("carbon_g_m3", "carbon_g_m3", "Peat C density (kg C m$^{-3}$)", 1.0e-3),
        ("cn", "C_N_ratio", "Bulk peat C:N (mass ratio)", 1.0),
        ("delta14c_permil", "Delta_14C_LLNL", r"Bulk peat $\Delta^{14}$C (‰)", 1.0),
    )
    for axis, (model_name, observed_name, xlabel, scale) in zip(axes.flat[:3], panels):
        observed = binned_observations(observations, observed_name)
        axis.errorbar(
            observed["mean"] * scale,
            observed["depth_m"],
            xerr=observed["sd"] * scale,
            fmt="o",
            ms=4,
            color="#555555",
            ecolor="#aaaaaa",
            capsize=2,
            label="2012 cores: mean ± SD",
            zorder=4,
        )
        for label in TOPOUNITS:
            axis.plot(
                np.asarray(model[label][model_name]) * scale,
                model[label]["depth_below_hollow_m"],
                lw=1.8,
                color=COLORS[label],
                label=f"ELM {label}",
            )
        depth, mean = _bog_mean(model, model_name)
        axis.plot(mean * scale, depth, "--", lw=2.2, color=COLORS["bog mean"], label="ELM bog mean")
        axis.set_xlabel(xlabel)
        axis.axhline(0.30, color="#777777", lw=0.9, ls=":")
        axis.grid(alpha=0.2)

    axes.flat[2].set_title(
        radiocarbon_note,
        loc="left",
        fontsize=8,
        pad=7,
    )
    axes.flat[0].set_title(
        "ELM profile: "
        + ("litter + CWD + SOM + DOM + microbes" if not model["missing_carbon_pools"]
           else "litter + SOM + DOM + microbes; CWD was not saved")
        + "\nRoots lack a layer-resolved history field",
        loc="left",
        fontsize=8,
        pad=7,
    )
    if not model["nitrogen_profiles_available"]:
        axes.flat[1].set_title(
            "ELM N profiles were not saved in this run",
            loc="left",
            fontsize=9,
            color="#8B1E1E",
            pad=7,
        )

    stock_axis = axes.flat[3]
    for label in TOPOUNITS:
        stock_axis.plot(
            np.r_[0.0, model[label]["cumulative_carbon_kg_m2"]],
            np.r_[
                -SURFACE_HEIGHT_ABOVE_HOLLOW_M[label],
                model[label]["layer_bottom_below_hollow_m"],
            ],
            lw=1.8,
            color=COLORS[label],
            label=f"ELM {label}",
        )
        stock_at_2p25 = model[label]["carbon_stock_to_2p25m_kg_m2"]
        stock_axis.plot(
            stock_at_2p25 + model[label]["root_carbon_kg_m2"],
            2.25,
            marker="D",
            ms=5,
            color=COLORS[label],
            linestyle="none",
            label=f"ELM {label} + whole-column roots",
        )
    stock_reference_depth = np.asarray(
        model["hollow"]["layer_bottom_below_hollow_m"]
    )
    mean_stock = sum(
        model["weights"][label]
        * np.interp(
            stock_reference_depth,
            np.asarray(model[label]["layer_bottom_below_hollow_m"]),
            np.asarray(model[label]["cumulative_carbon_kg_m2"]),
            left=0.0,
        )
        for label in TOPOUNITS
    )
    stock_axis.plot(
        np.r_[0.0, mean_stock],
        np.r_[0.0, stock_reference_depth],
        "--",
        lw=2.2,
        color=COLORS["bog mean"],
        label="ELM bog mean",
    )
    stock_axis.errorbar(
        [176.0],
        [2.25],
        xerr=[40.0],
        yerr=[0.58],
        fmt="s",
        color="#555555",
        ecolor="#888888",
        capsize=3,
        label="McFarlane et al.: 176±40 kg C m$^{-2}$ to 2.25±0.58 m; raised hummock excluded",
    )
    stock_axis.set_xlabel("Cumulative peat C (kg C m$^{-2}$)")
    stock_axis.axhline(0.30, color="#777777", lw=0.9, ls=":")
    stock_axis.grid(alpha=0.2)
    stock_axis.set_title(
        (
            "Diamonds: peat stock + whole-column root C"
            if model["root_carbon_available"]
            else "Whole-column root C was not saved in this history tape"
        ),
        loc="left",
        fontsize=8,
        pad=7,
    )

    for axis in axes.flat:
        axis.set_ylim(3.0, -0.35)
        axis.set_ylabel("Depth below hollow surface (m)")
    axes.flat[0].text(
        0.99,
        0.305,
        "acrotelm boundary",
        transform=axes.flat[0].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94), h_pad=2.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_trajectory_figure(
    model: dict, observations: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    number_of_years = len(model["hollow"]["stock_to_2p25m_timeseries_kg_m2"])
    years = np.arange(1, number_of_years + 1)

    for label in TOPOUNITS:
        axes[0].plot(
            years,
            model[label]["stock_to_2p25m_plus_roots_timeseries_kg_m2"],
            color=COLORS[label],
            lw=1.7,
            label=f"ELM {label}",
        )
    bog_stock = sum(
        model["weights"][label]
        * model[label]["stock_to_2p25m_plus_roots_timeseries_kg_m2"]
        for label in TOPOUNITS
    )
    axes[0].plot(years, bog_stock, "--", color=COLORS["bog mean"], lw=2.2, label="ELM bog mean")
    axes[0].axhspan(176.0 - 40.0, 176.0 + 40.0, color="#777777", alpha=0.18)
    axes[0].axhline(176.0, color="#555555", lw=1.2, label="Observed 176±40 kg C m$^{-2}$")
    axes[0].set(xlabel="AD model year", ylabel="Peat C stock to 2.25 m (kg C m$^{-2}$)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)

    for target_depth, linestyle in ((0.5, "-"), (2.0, "--")):
        bog_delta = np.zeros(number_of_years)
        for label in TOPOUNITS:
            depth = model[label]["depth_below_hollow_m"]
            values = model[label]["delta14c_timeseries_permil"]
            interpolated = np.array(
                [np.interp(target_depth, depth, row) for row in values], dtype=float
            )
            bog_delta += model["weights"][label] * interpolated
        nearby = observations[
            np.abs(observations["depth_m"] - target_depth)
            <= (0.15 if target_depth < 1.0 else 0.30)
        ]["Delta_14C_LLNL"].dropna()
        axes[1].plot(
            years,
            bog_delta,
            linestyle=linestyle,
            lw=2.0,
            label=f"ELM bog mean, {target_depth:g} m",
        )
        if len(nearby):
            mean = float(nearby.mean())
            sd = float(nearby.std())
            axes[1].axhspan(mean - sd, mean + sd, alpha=0.12)
            axes[1].axhline(mean, linestyle=linestyle, lw=1.1, alpha=0.8)
    axes[1].set(
        xlabel="AD model year",
        ylabel=r"Bulk peat $\Delta^{14}$C (‰)",
        title="Diagnostic only: accelerated chronology",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("SPRUCE 100-year AD spinup trajectory relative to McFarlane et al. (2018)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(model: dict, output: Path, history_file: Path) -> None:
    rows = []
    for label in TOPOUNITS:
        depth = np.asarray(model[label]["depth_below_hollow_m"])
        stock = np.asarray(model[label]["cumulative_carbon_kg_m2"])
        rows.append(
            {
                "profile": label,
                "weight_in_bog_mean": model["weights"][label],
                "carbon_stock_to_2p25m_kgC_m2": model[label]["carbon_stock_to_2p25m_kg_m2"],
                "whole_column_root_carbon_kgC_m2": model[label]["root_carbon_kg_m2"],
                "carbon_stock_to_2p25m_plus_roots_kgC_m2": (
                    model[label]["carbon_stock_to_2p25m_kg_m2"]
                    + model[label]["root_carbon_kg_m2"]
                ),
                "carbon_stock_full_profile_kgC_m2": float(stock[-1]),
                "delta14c_at_0p5m_permil": float(
                    np.interp(0.5, depth, model[label]["delta14c_permil"])
                ),
                "delta14c_at_2m_permil": float(
                    np.interp(2.0, depth, model[label]["delta14c_permil"])
                ),
                "history_file": str(history_file),
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", help="ELM h0 history file or glob")
    parser.add_argument(
        "--observations",
        type=Path,
        default=repository / "data/spruce/mcfarlane2018/Peat_Characteristics_T0_20180425.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="SPRUCE peat profiles: ELM versus McFarlane et al. (2018)",
    )
    parser.add_argument(
        "--radiocarbon-note",
        default="Model radiocarbon interpretation depends on atmospheric forcing",
        help="Short note printed in the radiocarbon panel.",
    )
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Do not write the AD spinup-trajectory panel (for continuation-only histories).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    model, history_file = read_model_profiles(arguments.history)
    observations = read_observations(arguments.observations)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    figure = arguments.output_dir / "spruce_mcfarlane_profile_comparison.png"
    trajectory = arguments.output_dir / "spruce_mcfarlane_spinup_trajectory.png"
    summary = arguments.output_dir / "spruce_mcfarlane_profile_summary.csv"
    make_figure(
        model, observations, figure, arguments.title, arguments.radiocarbon_note
    )
    if not arguments.skip_trajectory:
        make_trajectory_figure(model, observations, trajectory)
    write_summary(model, summary, history_file)
    print(figure)
    if not arguments.skip_trajectory:
        print(trajectory)
    print(summary)


if __name__ == "__main__":
    main()

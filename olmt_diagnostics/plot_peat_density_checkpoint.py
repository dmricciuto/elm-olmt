#!/usr/bin/env python3
"""Plot peat carbon density from an ELM restart at a spinup checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from .plot_spruce_mcfarlane_profiles import (
    CARBON_POOLS,
    COLORS,
    TOPOUNITS,
    C14_REFERENCE_RATIO,
    _c14_names,
    _profile,
    binned_observations,
    read_model_profiles,
    read_observations,
)


AD_RATE_VARIABLES = (
    "k_l1",
    "k_l2",
    "k_l3",
    "k_s1",
    "k_s2",
    "k_s3",
    "k_s4",
    "k_frag",
)
POOL_FACTOR_INDEX = {
    "LITR1C_vr": 0,
    "LITR2C_vr": 1,
    "LITR3C_vr": 2,
    "CWDC_vr": 7,
    "SOIL1C_vr": 3,
    "SOIL2C_vr": 4,
    "SOIL3C_vr": 5,
    "SOIL4C_vr": 6,
}


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", help="ELM restart file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--observations",
        type=Path,
        default=repository / "data/spruce/mcfarlane2018/Peat_Characteristics_T0_20180425.csv",
    )
    parser.add_argument("--checkpoint-label", default="50 AD years")
    parser.add_argument("--decomp-depth-efolding", type=float, required=True)
    parser.add_argument("--anoxic-solubilization-fraction", type=float, required=True)
    parser.add_argument("--target-surface-density", type=float, default=25.0)
    parser.add_argument("--target-deep-density", type=float, default=100.0)
    parser.add_argument("--target-efolding-depth", type=float, default=0.25)
    parser.add_argument(
        "--target-history",
        type=Path,
        help=(
            "History file containing PEAT_C_TARGET_DENSITY. When supplied, "
            "plot the model's actual hydrologic target instead of reconstructing "
            "the legacy exponential target."
        ),
    )
    parser.add_argument(
        "--convert-from-ad",
        action="store_true",
        help="Apply the same pool and scalaravg_col multipliers used when ELM exits AD spinup.",
    )
    parser.add_argument(
        "--parameter-file",
        type=Path,
        help="ELM parameter file; defaults to clm_params.nc beside the restart.",
    )
    return parser.parse_args()


def _ad_factors(parameter_file: Path) -> dict[str, float]:
    with netCDF4.Dataset(parameter_file) as dataset:
        rates = np.asarray(
            [float(np.asarray(dataset.variables[name][:]).squeeze()) for name in AD_RATE_VARIABLES]
        )
    factors = np.maximum(1.0, 1.0 / (rates * 365.0))
    factors[7] = max(1.0, 0.5 / (rates[7] * 365.0))
    return {
        pool: float(factors[index]) for pool, index in POOL_FACTOR_INDEX.items()
    }


def _convert_model_from_ad(
    model: dict, restart_file: Path, parameter_file: Path
) -> pd.DataFrame:
    factors = _ad_factors(parameter_file)
    rows = []
    with netCDF4.Dataset(restart_file) as state:
        if "spinup_state" not in state.variables or int(state.variables["spinup_state"][:]) != 1:
            raise RuntimeError(f"{restart_file} is not an AD-spinup restart")
        for label in TOPOUNITS:
            column = int(model[label]["column"])
            count = len(model[label]["depth_m"])
            scalar = _profile(state, "scalaravg_col", column)[:count]
            if np.any(~np.isfinite(scalar) | (scalar <= 0.0)):
                raise RuntimeError(f"Invalid scalaravg_col for {label}: {scalar}")
            converted = np.zeros(count)
            converted_c14 = np.zeros(count)
            for pool in CARBON_POOLS:
                values = _profile(state, pool, column)[:count]
                factor = factors.get(pool, 1.0)
                multiplier = factor / scalar if factor > 1.0 else np.ones(count)
                converted += values * multiplier
                c14_values = None
                for c14_name in _c14_names(pool):
                    try:
                        c14_values = _profile(state, c14_name, column)[:count]
                        break
                    except KeyError:
                        continue
                if c14_values is not None:
                    converted_c14 += c14_values * multiplier
                rows.append(
                    {
                        "topounit": label,
                        "pool": pool,
                        "spinup_factor": factor,
                        "scalaravg_col": float(scalar[0]),
                        "exit_spinup_multiplier": float(multiplier[0]),
                    }
                )
            model[label]["carbon_g_m3"] = converted
            model[label]["delta14c_permil"] = 1000.0 * (
                np.divide(
                    converted_c14,
                    converted * C14_REFERENCE_RATIO,
                    out=np.full(count, np.nan),
                    where=np.isfinite(converted) & (converted > 1.0),
                )
                - 1.0
            )
            thickness = np.asarray(model[label]["thickness_m"])
            depth = np.asarray(model[label]["depth_below_hollow_m"])
            layer_top = depth - 0.5 * thickness
            layer_bottom = depth + 0.5 * thickness
            model[label]["cumulative_carbon_kg_m2"] = np.cumsum(
                converted
                * np.maximum(
                    0.0, layer_bottom - np.maximum(layer_top, 0.0)
                )
            ) / 1000.0
            overlap = np.maximum(
                0.0,
                np.minimum(layer_bottom, 2.25) - np.maximum(layer_top, 0.0),
            )
            model[label]["carbon_stock_to_2p25m_kg_m2"] = float(
                np.sum(converted * overlap) / 1000.0
            )
    return pd.DataFrame(rows)


def _write_c14_diagnostic(
    model: dict,
    observations: pd.DataFrame,
    output_dir: Path,
    title: str,
    converted_from_ad: bool,
) -> None:
    reference_depth = np.asarray(model["hollow"]["depth_below_hollow_m"])
    bog_mean = sum(
        model["weights"][label]
        * np.interp(
            reference_depth,
            np.asarray(model[label]["depth_below_hollow_m"]),
            np.asarray(model[label]["delta14c_permil"]),
        )
        for label in TOPOUNITS
    )
    model_rows = []
    for level, depth in enumerate(reference_depth):
        row = {
            "depth_m": depth,
            "bog_mean_delta14c_permil": bog_mean[level],
        }
        for label in TOPOUNITS:
            row[f"{label}_delta14c_permil"] = np.asarray(
                model[label]["delta14c_permil"]
            )[level]
        model_rows.append(row)
    pd.DataFrame(model_rows).to_csv(output_dir / "c14_profile_model.csv", index=False)
    observations.to_csv(output_dir / "c14_profile_observations.csv", index=False)

    comparison_rows = []
    finite_observed = observations[
        np.isfinite(observations["mean"]) & np.isfinite(observations["depth_m"])
    ].sort_values("depth_m")
    for depth in (0.25, 0.5, 1.0, 1.5, 2.0):
        observed = float(
            np.interp(depth, finite_observed["depth_m"], finite_observed["mean"])
        )
        modeled = float(np.interp(depth, reference_depth, bog_mean))
        comparison_rows.append(
            {
                "depth_m": depth,
                "model_bog_mean_delta14c_permil": modeled,
                "observed_binned_delta14c_permil": observed,
                "model_minus_observed_permil": modeled - observed,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "c14_profile_comparison.csv", index=False
    )

    fig, axis = plt.subplots(figsize=(7.2, 7.0))
    axis.errorbar(
        observations["mean"],
        observations["depth_m"],
        xerr=observations["sd"],
        fmt="o",
        ms=5,
        color="#666666",
        ecolor="#b0b0b0",
        capsize=2,
        label="2012 cores: mean ± SD",
        zorder=2,
    )
    for label in TOPOUNITS:
        axis.plot(
            model[label]["delta14c_permil"],
            model[label]["depth_below_hollow_m"],
            color=COLORS[label],
            lw=1.8,
            label=f"ELM {label}",
            zorder=3,
        )
    axis.plot(
        bog_mean,
        reference_depth,
        color=COLORS["bog mean"],
        lw=2.2,
        ls="--",
        label="ELM bog mean",
        zorder=4,
    )
    axis.axhline(0.30, color="#777777", lw=0.9, ls=":")
    axis.set(
        xlabel=r"Bulk peat $\Delta^{14}$C (‰)",
        ylabel="Depth below hollow surface (m)",
        ylim=(2.25, -0.35),
        title=title,
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="upper left")
    note = (
        "AD diagnostic converted pool-by-pool: each pool preserves its ¹⁴C/C,\n"
        "while pool-specific multipliers can change the bulk ratio."
        if converted_from_ad
        else "Profile read directly from the final-spinup restart;\n"
        "no diagnostic AD pool conversion applied."
    )
    axis.text(
        0.98,
        0.02,
        note,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "c14_profile.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arguments = parse_args()
    model, restart_file = read_model_profiles(arguments.restart)
    conversion = None
    if arguments.convert_from_ad:
        parameter_file = arguments.parameter_file or restart_file.parent / "clm_params.nc"
        conversion = _convert_model_from_ad(model, restart_file, parameter_file)
    observation_samples = read_observations(arguments.observations)
    observations = binned_observations(observation_samples, "carbon_g_m3")
    c14_observations = binned_observations(observation_samples, "Delta_14C_LLNL")
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_depth = np.asarray(model["hollow"]["depth_below_hollow_m"])
    bog_mean = sum(
        model["weights"][label]
        * np.interp(
            reference_depth,
            np.asarray(model[label]["depth_below_hollow_m"]),
            np.asarray(model[label]["carbon_g_m3"]),
        )
        for label in TOPOUNITS
    )
    if arguments.target_history is not None:
        with netCDF4.Dataset(arguments.target_history) as target_state:
            variable = target_state.variables["PEAT_C_TARGET_DENSITY"]
            values = np.ma.filled(variable[-1, ...], np.nan).astype(float)
            level_axis = next(
                index for index, name in enumerate(variable.dimensions[1:])
                if name.startswith("lev")
            )
            column_axis = variable.dimensions[1:].index("column")
            values = np.moveaxis(values, (level_axis, column_axis), (0, 1))
            target = sum(
                model["weights"][label]
                * np.interp(
                    reference_depth,
                    np.asarray(model[label]["depth_below_hollow_m"]),
                    values[: len(model[label]["depth_m"]), int(model[label]["column"])]
                    / 1000.0,
                )
                for label in TOPOUNITS
            )
        target_label = "ELM hydrologic density target"
    else:
        target = arguments.target_deep_density - (
            arguments.target_deep_density - arguments.target_surface_density
        ) * np.exp(-reference_depth / arguments.target_efolding_depth)
        target_label = "Compaction density target"

    rows = []
    for level, depth in enumerate(reference_depth):
        row = {
            "depth_m": depth,
            "target_density_kgC_m3": target[level],
            "bog_mean_density_kgC_m3": bog_mean[level] / 1000.0,
        }
        for label in TOPOUNITS:
            row[f"{label}_density_kgC_m3"] = np.interp(
                depth,
                np.asarray(model[label]["depth_below_hollow_m"]),
                np.asarray(model[label]["carbon_g_m3"]) / 1000.0,
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    csv_file = output_dir / "peat_density_profile.csv"
    frame.to_csv(csv_file, index=False)
    stock_rows = [
        {
            "profile": label,
            "weight_in_bog_mean": model["weights"][label],
            "carbon_stock_to_2p25m_kgC_m2": model[label][
                "carbon_stock_to_2p25m_kg_m2"
            ],
        }
        for label in TOPOUNITS
    ]
    stock_rows.append(
        {
            "profile": "bog mean",
            "weight_in_bog_mean": 1.0,
            "carbon_stock_to_2p25m_kgC_m2": sum(
                model["weights"][label]
                * model[label]["carbon_stock_to_2p25m_kg_m2"]
                for label in TOPOUNITS
            ),
        }
    )
    pd.DataFrame(stock_rows).to_csv(output_dir / "peat_density_summary.csv", index=False)
    if conversion is not None:
        conversion.to_csv(output_dir / "ad_conversion_factors.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.8), sharey=False)
    for axis in axes:
        axis.errorbar(
            observations["mean"] / 1000.0,
            observations["depth_m"],
            xerr=observations["sd"] / 1000.0,
            fmt="o",
            ms=4,
            color="#666666",
            ecolor="#b0b0b0",
            capsize=2,
            label="2012 cores: mean ± SD",
            zorder=2,
        )
        axis.plot(
            target,
            reference_depth,
            color="#2f855a",
            lw=2.2,
            ls=":",
            label=target_label,
            zorder=3,
        )
        for label in TOPOUNITS:
            axis.plot(
                np.asarray(model[label]["carbon_g_m3"]) / 1000.0,
                model[label]["depth_below_hollow_m"],
                color=COLORS[label],
                lw=1.8,
                label=f"ELM {label}",
                zorder=4,
            )
        axis.plot(
            bog_mean / 1000.0,
            reference_depth,
            color=COLORS["bog mean"],
            lw=2.2,
            ls="--",
            label="ELM bog mean",
            zorder=5,
        )
        axis.axhline(0.30, color="#777777", lw=0.9, ls=":")
        axis.set_ylim(2.25, -0.35)
        axis.set_xlabel("Peat C density (kg C m$^{-3}$)")
        axis.grid(alpha=0.2)
    axes[0].set_xlim(left=0.0)
    axes[0].set_title("Full observational and target range", loc="left")
    axes[0].set_ylabel("Depth below hollow surface (m)")
    if arguments.convert_from_ad:
        axes[1].set_xlim(0.0, 120.0)
        axes[1].set_ylim(2.25, 0.80)
        axes[1].set_title("Deep profile enlarged", loc="left")
    else:
        zoom_limit = max(5.0, 1.15 * float(np.nanmax(bog_mean / 1000.0)))
        axes[1].set_xlim(0.0, zoom_limit)
        axes[1].set_title("Model profile enlarged", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "SPRUCE peat density after "
        f"{arguments.checkpoint_label}\n"
        f"decomposition depth e-folding = {arguments.decomp_depth_efolding:g} m; "
        "anoxic DOM-solubilization fraction = "
        f"{arguments.anoxic_solubilization_fraction:g}"
        + ("; converted from AD to physical-equivalent pools" if arguments.convert_from_ad else "")
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.93))
    figure_file = output_dir / "peat_density_profile.png"
    fig.savefig(figure_file, dpi=220, bbox_inches="tight")
    plt.close(fig)

    _write_c14_diagnostic(
        model,
        c14_observations,
        output_dir,
        "SPRUCE bulk-peat Δ¹⁴C after "
        f"{arguments.checkpoint_label}\n"
        + (
            "AD checkpoint converted diagnostically; no historical bomb forcing"
            if arguments.convert_from_ad
            else "final-spinup restart; no historical bomb forcing"
        ),
        arguments.convert_from_ad,
    )

    print(figure_file)
    print(output_dir / "c14_profile.png")
    print(csv_file)
    print(restart_file)


if __name__ == "__main__":
    main()

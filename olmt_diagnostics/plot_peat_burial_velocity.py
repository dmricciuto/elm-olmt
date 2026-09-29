#!/usr/bin/env python3
"""Plot physical and AD-applied peat burial velocities from ELM restarts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from .plot_peat_density_checkpoint import _ad_factors
from .plot_spruce_mcfarlane_profiles import (
    CARBON_POOLS,
    COLORS,
    TOPOUNITS,
    _profile,
    read_model_profiles,
)


SECONDS_PER_YEAR = 365.0 * 86400.0
PLOT_POOLS = ("LITR3C_vr", "SOIL3C_vr", "SOIL4C_vr")
POOL_LABELS = {
    "LITR3C_vr": "Litter 3",
    "SOIL3C_vr": "SOM3",
    "SOIL4C_vr": "SOM4",
}


def _hydrologic_target_profile(
    layer_depth: np.ndarray,
    thickness: np.ndarray,
    acrotelm_depth: float,
    transition_width: float,
    surface_density: float,
    deep_density: float,
) -> np.ndarray:
    transition_top = max(0.0, acrotelm_depth)
    transition_bottom = transition_top + transition_width

    def primitive(depth: np.ndarray) -> np.ndarray:
        return np.where(
            depth <= transition_top,
            0.0,
            np.where(
                depth < transition_bottom,
                (depth - transition_top) ** 2 / (2.0 * transition_width),
                depth - transition_top - 0.5 * transition_width,
            ),
        )

    layer_top = np.maximum(0.0, layer_depth - 0.5 * thickness)
    layer_bottom = layer_top + thickness
    catotelm_fraction = np.clip(
        (primitive(layer_bottom) - primitive(layer_top)) / thickness,
        0.0,
        1.0,
    )
    return 1000.0 * (
        surface_density + (deep_density - surface_density) * catotelm_fraction
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restarts", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parameter-file", type=Path)
    parser.add_argument("--max-depth", type=float, default=2.25)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if len(arguments.restarts) != len(arguments.labels):
        raise ValueError("--labels must contain one label per restart")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    profiles: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    factors = None
    compaction_parameters = None

    for restart, label in zip(arguments.restarts, arguments.labels):
        model, restart_file = read_model_profiles(str(restart))
        parameter_file = arguments.parameter_file or restart_file.parent / "clm_params.nc"
        if factors is None:
            factors = _ad_factors(parameter_file)
            with netCDF4.Dataset(parameter_file) as parameters:
                compaction_parameters = {
                    name: float(np.asarray(parameters.variables[name][:]).squeeze())
                    for name in parameters.variables
                    if name.startswith("peat_compaction_")
                    and np.asarray(parameters.variables[name][:]).size == 1
                }
        profiles[label] = {}
        with netCDF4.Dataset(restart_file) as state:
            spinup_state = int(np.asarray(state.variables["spinup_state"][:]).squeeze())
            for topounit in TOPOUNITS:
                column = int(model[topounit]["column"])
                thickness = np.asarray(model[topounit]["thickness_m"])
                layer_depth = np.asarray(model[topounit]["depth_m"])
                interface_depth = np.r_[0.0, np.cumsum(thickness[:-1])]
                count = min(len(interface_depth), len(_profile(state, "som_adv_coef_vr", column)))
                interface_depth = interface_depth[:count]
                base = _profile(state, "som_adv_coef_vr", column)[:count]
                scalar = _profile(state, "scalaravg_col", column)[:count]
                base_mm_yr = base * SECONDS_PER_YEAR * 1000.0
                pool_concentrations = {
                    pool: _profile(state, pool, column)[:count]
                    for pool in CARBON_POOLS
                    if pool != "DOMC_vr"
                }
                pool_multipliers = {}
                for pool in pool_concentrations:
                    multiplier = np.ones(count)
                    pool_factor = factors.get(pool, 1.0)
                    if spinup_state == 1 and pool_factor > 1.0:
                        multiplier = pool_factor / np.maximum(
                            scalar, np.finfo(float).eps
                        )
                    pool_multipliers[pool] = multiplier
                physical_density = sum(
                    np.maximum(values, 0.0) * pool_multipliers[pool]
                    for pool, values in pool_concentrations.items()
                )
                if "peat_acrotelm_depth" in state.variables:
                    acrotelm_depth = float(
                        np.asarray(state.variables["peat_acrotelm_depth"][:])
                        .squeeze()[column]
                    )
                    target_density = _hydrologic_target_profile(
                        layer_depth[:count],
                        thickness[:count],
                        acrotelm_depth,
                        compaction_parameters["peat_compaction_transition_width"],
                        compaction_parameters["peat_compaction_surface_density"],
                        compaction_parameters["peat_compaction_deep_density"],
                    )
                else:
                    target_density = 1000.0 * (
                        compaction_parameters["peat_compaction_surface_density"]
                        + (
                            compaction_parameters["peat_compaction_deep_density"]
                            - compaction_parameters["peat_compaction_surface_density"]
                        )
                        * (
                            1.0
                            - np.exp(
                                -np.maximum(layer_depth[:count], 0.0)
                                / compaction_parameters["peat_compaction_efolding_depth"]
                            )
                        )
                    )
                physical_flux = np.zeros(count)
                applied_flux = np.zeros(count)
                for interface in range(1, count):
                    donor = interface - 1
                    physical_flux[interface] = (
                        base[interface] * physical_density[donor] * SECONDS_PER_YEAR
                    )
                    applied_flux[interface] = sum(
                        base[interface]
                        * pool_multipliers[pool][interface]
                        * max(values[donor], 0.0)
                        * SECONDS_PER_YEAR
                        for pool, values in pool_concentrations.items()
                    )
                profile = {
                    "depth_m": interface_depth,
                    "physical_mm_yr": base_mm_yr,
                    "physical_flux_g_m2_yr": physical_flux,
                    "applied_flux_g_m2_yr": applied_flux,
                }
                for pool in PLOT_POOLS:
                    multiplier = np.ones(count)
                    if spinup_state == 1 and factors[pool] > 1.0:
                        multiplier = factors[pool] / np.maximum(scalar, np.finfo(float).eps)
                    profile[f"{pool}_mm_yr"] = base_mm_yr * multiplier
                profiles[label][topounit] = profile

                for level, depth in enumerate(interface_depth):
                    row: dict[str, float | str] = {
                        "checkpoint": label,
                        "topounit": topounit,
                        "interface_depth_m": float(depth),
                        "physical_velocity_mm_yr": float(base_mm_yr[level]),
                        "physical_solid_c_density_g_m3": float(
                            physical_density[level]
                        ),
                        "physical_burial_flux_g_m2_yr": float(
                            physical_flux[level]
                        ),
                        "ad_applied_burial_flux_g_m2_yr": float(
                            applied_flux[level]
                        ),
                        "ad_to_physical_flux_ratio": float(
                            applied_flux[level] / physical_flux[level]
                        ) if physical_flux[level] > 0.0 else np.nan,
                        "scalaravg_col": float(scalar[level]),
                    }
                    if level > 0:
                        donor = level - 1
                        donor_target = target_density[donor]
                        row["donor_layer_depth_m"] = float(layer_depth[donor])
                        row["donor_solid_c_density_g_m3"] = float(
                            physical_density[donor]
                        )
                        row["donor_target_density_g_m3"] = float(donor_target)
                        row["donor_excess_density_g_m3"] = float(
                            physical_density[donor] - donor_target
                        )
                    for pool in PLOT_POOLS:
                        row[f"{pool}_applied_velocity_mm_yr"] = float(
                            profile[f"{pool}_mm_yr"][level]
                        )
                        row[f"{pool}_ad_multiplier"] = float(
                            profile[f"{pool}_mm_yr"][level] / base_mm_yr[level]
                        ) if base_mm_yr[level] > 0.0 else float(
                            factors[pool] / max(scalar[level], np.finfo(float).eps)
                        )
                    for pool, values in pool_concentrations.items():
                        row[f"{pool}_physical_density_g_m3"] = float(
                            max(values[level], 0.0) * pool_multipliers[pool][level]
                        )
                    rows.append(row)

    pd.DataFrame(rows).to_csv(arguments.output_dir / "peat_burial_velocity.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 7.0), sharey=True)
    checkpoint_colors = plt.cm.viridis(np.linspace(0.12, 0.78, len(arguments.labels)))
    latest = arguments.labels[-1]
    pool_styles = {"LITR3C_vr": ":", "SOIL3C_vr": "--", "SOIL4C_vr": "-."}
    for axis, topounit in zip(axes, TOPOUNITS):
        for color, label in zip(checkpoint_colors, arguments.labels):
            profile = profiles[label][topounit]
            values = np.where(profile["physical_mm_yr"] > 0.0, profile["physical_mm_yr"], np.nan)
            axis.step(
                values,
                profile["depth_m"],
                where="post",
                color=color,
                lw=2.0,
                label=f"{label}: physical/base",
            )
        for pool in PLOT_POOLS:
            profile = profiles[latest][topounit]
            values = np.where(profile[f"{pool}_mm_yr"] > 0.0, profile[f"{pool}_mm_yr"], np.nan)
            axis.step(
                values,
                profile["depth_m"],
                where="post",
                color=COLORS[topounit],
                ls=pool_styles[pool],
                lw=1.7,
                label=f"{latest}: applied to {POOL_LABELS[pool]}",
            )
        axis.set_xscale("log")
        axis.set_xlim(1.0e-3, 1.0e3)
        axis.set_ylim(arguments.max_depth, 0.0)
        axis.set_title(topounit.capitalize())
        axis.set_xlabel("Downward burial velocity (mm yr$^{-1}$)")
        axis.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("Depth below local peat surface (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.suptitle(
        "Peat storage-and-throughflow burial velocity\n"
        "Base velocity uses the physical one-year compaction timescale; AD applies pool-specific acceleration",
        y=0.98,
    )
    figure.subplots_adjust(bottom=0.25, top=0.87, wspace=0.10)
    figure.savefig(arguments.output_dir / "peat_burial_velocity.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 6.4), sharey=True)
    for axis, topounit in zip(axes, TOPOUNITS):
        for color, label in zip(checkpoint_colors, arguments.labels):
            profile = profiles[label][topounit]
            values = np.where(
                profile["physical_flux_g_m2_yr"] > 0.0,
                profile["physical_flux_g_m2_yr"],
                np.nan,
            )
            axis.step(
                values,
                profile["depth_m"],
                where="post",
                color=color,
                lw=2.0,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_xlim(1.0e-2, 1.0e3)
        axis.set_ylim(arguments.max_depth, 0.0)
        axis.set_title(topounit.capitalize())
        axis.set_xlabel("Physical-equivalent solid-C flux (g C m$^{-2}$ yr$^{-1}$)")
        axis.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("Interface depth below local peat surface (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        "Checkpoint peat burial throughput\n"
        "Flux at each interface; interfaces must not be summed",
        y=0.98,
    )
    figure.subplots_adjust(bottom=0.17, top=0.86, wspace=0.10)
    figure.savefig(arguments.output_dir / "peat_burial_carbon_flux.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(arguments.output_dir / "peat_burial_velocity.png")
    print(arguments.output_dir / "peat_burial_carbon_flux.png")
    print(arguments.output_dir / "peat_burial_velocity.csv")


if __name__ == "__main__":
    main()

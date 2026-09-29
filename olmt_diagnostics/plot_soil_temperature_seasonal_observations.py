#!/usr/bin/env python3
"""Compare seasonal modeled and measured SPRUCE warming profiles."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


BOG_COLUMNS = (1, 2)
OBSERVATION_VARIABLES = (
    ("TSOIL_0", 0.0),
    ("TSOIL_10", 0.10),
    ("TSOIL_20", 0.20),
    ("TSOIL_200", 2.0),
)
SEASONS = (
    ("Winter (DJF)", (12, 1, 2)),
    ("Spring (MAM)", (3, 4, 5)),
    ("Summer (JJA)", (6, 7, 8)),
    ("Autumn (SON)", (9, 10, 11)),
)


def _dates(time) -> np.ndarray:
    decoded = netCDF4.num2date(
        time[:], time.units, getattr(time, "calendar", "noleap")
    )
    return np.asarray([
        dt.datetime(value.year, value.month, value.day, value.hour,
                    value.minute, value.second)
        for value in decoded
    ])


def _read_model(pattern: str, year: int) -> dict[str, np.ndarray]:
    dates = []
    temperatures = []
    depth = None
    for filename in sorted(glob.glob(pattern)):
        with netCDF4.Dataset(filename) as dataset:
            current_dates = _dates(dataset.variables["time"])
            select = np.asarray([value.year == year for value in current_dates])
            if not np.any(select):
                continue
            field = np.asarray(dataset.variables["TSOI"][select, :, :], dtype=float)
            weights = np.asarray(dataset.variables["cols1d_wtgcell"][:], dtype=float)[
                list(BOG_COLUMNS)
            ]
            weights /= weights.sum()
            temperatures.append(
                np.sum(field[:, :, list(BOG_COLUMNS)] * weights[None, None, :], axis=2)
                - 273.15
            )
            dates.extend(current_dates[select])
            depth = np.asarray(dataset.variables["levgrnd"][:], dtype=float)
    if not temperatures:
        raise ValueError(f"No model records for {year}: {pattern}")
    order = np.argsort(np.asarray(dates))
    return {
        "dates": np.asarray(dates)[order],
        "temperature": np.concatenate(temperatures, axis=0)[order, :],
        "depth": depth,
    }


def _read_observations(filename: str, year: int) -> dict[str, np.ndarray]:
    with netCDF4.Dataset(filename) as dataset:
        current_dates = _dates(dataset.variables["DTIME"])
        select = np.asarray([value.year == year for value in current_dates])
        return {
            "dates": current_dates[select],
            "temperature": np.column_stack([
                np.asarray(dataset.variables[name][0, select], dtype=float) - 273.15
                for name, _ in OBSERVATION_VARIABLES
            ]),
            "depth": np.asarray([depth for _, depth in OBSERVATION_VARIABLES]),
        }


def make_plot(control_pattern: str, treatment_pattern: str,
              control_observations: str, treatment_observations: str,
              year: int, output: str) -> None:
    model_control = _read_model(control_pattern, year)
    model_treatment = _read_model(treatment_pattern, year)
    observed_control = _read_observations(control_observations, year)
    observed_treatment = _read_observations(treatment_observations, year)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0), sharex=True, sharey=True)
    rows = []
    for axis, (label, months) in zip(axes.flat, SEASONS):
        model_mask = np.asarray([value.month in months for value in model_control["dates"]])
        observed_mask = np.asarray([value.month in months for value in observed_control["dates"]])
        model_delta = (
            np.mean(model_treatment["temperature"][model_mask, :], axis=0)
            - np.mean(model_control["temperature"][model_mask, :], axis=0)
        )
        observed_delta = (
            np.mean(observed_treatment["temperature"][observed_mask, :], axis=0)
            - np.mean(observed_control["temperature"][observed_mask, :], axis=0)
        )
        modeled_at_observations = np.interp(
            observed_control["depth"], model_control["depth"], model_delta
        )
        rmse = np.sqrt(np.mean((modeled_at_observations - observed_delta) ** 2))
        axis.plot(model_delta, model_control["depth"], color="#e45756", linewidth=2.6,
                  label="ELM T9 − T0")
        axis.scatter(observed_delta, observed_control["depth"], color="black", s=55,
                     zorder=5, label="SPRUCE processed observations")
        axis.axvline(9.0, color="0.25", linestyle="--", linewidth=1.2,
                     label="+9 K target")
        axis.axhspan(2.0, 3.0, color="#f4d35e", alpha=0.20,
                     label="modeled heater interval")
        axis.set_title(f"{label}\nmeasurement-depth RMSE = {rmse:.2f} K")
        axis.grid(True, alpha=0.25)
        axis.set_xlim(-3.0, 15.0)
        axis.set_ylim(5.0, 0.0)
        rows.extend(
            (label, depth, observed, modeled)
            for depth, observed, modeled in zip(
                observed_control["depth"], observed_delta, modeled_at_observations
            )
        )

    for axis in axes[:, 0]:
        axis.set_ylabel("Depth below hollow surface (m)")
    for axis in axes[-1, :]:
        axis.set_xlabel("T9 − T0 soil-temperature difference (K)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"SPRUCE {year} seasonal soil-warming profiles")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w") as handle:
        handle.write("season,depth_m,observed_delta_K,modeled_delta_K\n")
        for row in rows:
            handle.write(f"{row[0]},{row[1]:.3f},{row[2]:.6f},{row[3]:.6f}\n")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0), sharex=True, sharey=True)
    for axis, (label, months) in zip(axes.flat, SEASONS):
        model_mask = np.asarray([value.month in months for value in model_control["dates"]])
        observed_mask = np.asarray([value.month in months for value in observed_control["dates"]])
        model_t0 = np.mean(model_control["temperature"][model_mask, :], axis=0)
        model_t9 = np.mean(model_treatment["temperature"][model_mask, :], axis=0)
        observed_t0 = np.mean(observed_control["temperature"][observed_mask, :], axis=0)
        observed_t9 = np.mean(observed_treatment["temperature"][observed_mask, :], axis=0)
        axis.plot(model_t0, model_control["depth"], color="#4c78a8", linestyle="--",
                  linewidth=2.3, label="ELM T0")
        axis.plot(model_t9, model_treatment["depth"], color="#e45756",
                  linewidth=2.5, label="ELM T9")
        axis.scatter(observed_t0, observed_control["depth"], facecolors="white",
                     edgecolors="#4c78a8", linewidths=1.8, s=58, zorder=5,
                     label="SPRUCE T0 processed observations")
        axis.scatter(observed_t9, observed_treatment["depth"], color="#e45756",
                     edgecolors="black", linewidths=0.6, s=58, zorder=5,
                     label="SPRUCE T9 processed observations")
        axis.axhspan(2.0, 3.0, color="#f4d35e", alpha=0.20,
                     label="modeled heater interval")
        axis.set_title(label)
        axis.grid(True, alpha=0.25)
        axis.set_ylim(5.0, 0.0)
    for axis in axes[:, 0]:
        axis.set_ylabel("Depth below hollow surface (m)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Soil temperature (°C)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"SPRUCE {year} seasonal absolute soil-temperature profiles")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    absolute_path = output_path.with_name(output_path.stem + "_absolute" + output_path.suffix)
    fig.savefig(absolute_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, help="Control history glob")
    parser.add_argument("--treatment", required=True, help="Treatment history glob")
    parser.add_argument("--control-observations", required=True)
    parser.add_argument("--treatment-observations", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    make_plot(args.control, args.treatment, args.control_observations,
              args.treatment_observations, args.year, args.output)


if __name__ == "__main__":
    main()

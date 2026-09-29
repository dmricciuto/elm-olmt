#!/usr/bin/env python3
"""Plot soil-temperature profiles and heater operation for an OLMT pair."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


BOG_COLUMNS = (1, 2)  # hollow and hummock; exclude the auxiliary fen


def _weighted(dataset, variable: str) -> np.ndarray:
    field = np.asarray(dataset.variables[variable][:], dtype=float)
    dimensions = dataset.variables[variable].dimensions
    axis = dimensions.index("column")
    field = np.moveaxis(field, axis, -1)[..., BOG_COLUMNS]
    weights = np.asarray(dataset.variables["cols1d_wtgcell"][:], dtype=float)[
        list(BOG_COLUMNS)
    ]
    weights /= weights.sum()
    return np.sum(field * weights, axis=-1)


def _read(path: str) -> dict[str, np.ndarray]:
    with netCDF4.Dataset(path) as dataset:
        time = dataset.variables["time"]
        decoded = netCDF4.num2date(
                time[:], time.units, getattr(time, "calendar", "noleap"),
                only_use_cftime_datetimes=False,
            )
        dates = np.asarray([
            dt.datetime(date.year, date.month, date.day, date.hour, date.minute, date.second)
            for date in decoded
        ])
        return {
            "dates": dates,
            "depth": np.asarray(dataset.variables["levgrnd"][:], dtype=float),
            "temperature": _weighted(dataset, "TSOI") - 273.15,
            "power": _weighted(dataset, "EFLX_SOIL_HEATING"),
            "control": _weighted(dataset, "TSOI_HEATING_CONTROL") - 273.15,
            "reference": _weighted(dataset, "TSOI_HEATING_REFERENCE") - 273.15,
            "target": _weighted(dataset, "TSOI_HEATING_TARGET") - 273.15,
        }


def _mask(dates: np.ndarray, start: str, end: str) -> np.ndarray:
    strings = np.asarray([date.strftime("%Y-%m-%d") for date in dates])
    return (strings >= start) & (strings <= end)


def make_plots(control_file: str, treatment_file: str, output_dir: str) -> None:
    control = _read(control_file)
    treatment = _read(treatment_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    windows = [
        ("Aug 1–14 (before heating)", "2015-08-01", "2015-08-14", "#4c78a8"),
        ("September", "2015-09-01", "2015-09-30", "#f58518"),
        ("December", "2015-12-01", "2015-12-31", "#b279a2"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.5), sharey=True)
    for label, start, end, color in windows:
        mask = _mask(treatment["dates"], start, end)
        t0 = np.mean(control["temperature"][mask, :], axis=0)
        t9 = np.mean(treatment["temperature"][mask, :], axis=0)
        axes[0].plot(t0, control["depth"], color=color, linestyle="--", linewidth=2)
        axes[0].plot(t9, treatment["depth"], color=color, linewidth=2.4, label=label)
        axes[1].plot(t9 - t0, treatment["depth"], color=color, linewidth=2.4, label=label)

    for axis in axes:
        axis.axhspan(2.0, 3.0, color="#f4d35e", alpha=0.20, label="heater interval")
        axis.axhline(2.0, color="0.35", linestyle=":", linewidth=1.2)
        axis.set_ylim(5.0, 0.0)
        axis.grid(True, alpha=0.25)
        axis.set_ylabel("Soil depth (m)")
    axes[0].set_xlabel("Soil temperature (°C)")
    axes[0].set_title("Absolute profiles\nsolid: T9, dashed: T0")
    axes[1].set_xlabel("T9 − T0 (K)")
    axes[1].set_title("Treatment warming profile")
    axes[1].axvline(9.0, color="black", linestyle="--", linewidth=1.2, label="+9 K target")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="lower right", fontsize=9)
    fig.suptitle("SPRUCE deep-soil heating test: hummock/hollow weighted mean")
    fig.tight_layout()
    fig.savefig(output / "spruce_t0_t9_soil_temperature_profiles.png", dpi=200)
    plt.close(fig)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11.2, 7.2), sharex=True)
    top.plot(control["dates"], control["control"], color="#4c78a8", label="T0 at 2 m")
    top.plot(treatment["dates"], treatment["control"], color="#e45756", label="T9 at 2 m")
    top.plot(treatment["dates"], treatment["target"], color="black", linestyle="--", label="T0 reference + 9 K")
    top.axvline(np.datetime64("2015-08-15"), color="0.25", linestyle=":", linewidth=1.5)
    top.set_ylabel("Temperature (°C)")
    top.legend(ncol=3, fontsize=9)
    top.grid(True, alpha=0.25)
    bottom.plot(treatment["dates"], treatment["power"], color="#f58518")
    bottom.axvline(np.datetime64("2015-08-15"), color="0.25", linestyle=":", linewidth=1.5, label="heating begins")
    bottom.set_ylabel("Heater power (W m⁻²)")
    bottom.set_xlabel("Date")
    bottom.grid(True, alpha=0.25)
    bottom.legend(loc="upper left")
    fig.suptitle("SPRUCE deep-heater controller timing and response")
    fig.tight_layout()
    fig.savefig(output / "spruce_t0_t9_soil_heating_timeseries.png", dpi=200)
    plt.close(fig)

    active = np.flatnonzero(treatment["power"] > 1.0e-12)
    first_active = treatment["dates"][active[0]].strftime("%Y-%m-%d")
    final_error = treatment["target"][-1] - treatment["control"][-1]
    print(f"first_nonzero_heating={first_active}")
    print(f"maximum_power_W_m2={np.max(treatment['power']):.6g}")
    print(f"mean_active_power_W_m2={np.mean(treatment['power'][active]):.6g}")
    print(f"final_control_target_error_K={final_error:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    make_plots(args.control, args.treatment, args.output_dir)


if __name__ == "__main__":
    main()

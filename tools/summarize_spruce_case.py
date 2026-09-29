#!/usr/bin/env python3
"""Summarize carbon, methane, nutrient, and water metrics for one SPRUCE run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86400.0
BOG_TOPOUNITS = (2, 3)
TOPOUNIT_NAMES = {1: "fen", 2: "hollow", 3: "hummock"}

FLUXES = {
    "GPP": ("gC m-2 yr-1", 1.0),
    "NPP": ("gC m-2 yr-1", 1.0),
    "AGNPP": ("gC m-2 yr-1", 1.0),
    "BGNPP": ("gC m-2 yr-1", 1.0),
    "HR": ("gC m-2 yr-1", 1.0),
    "MM_CH4_PROD": ("gC m-2 yr-1", 1.0),
    "MM_CH4_OXID": ("gC m-2 yr-1", 1.0),
    "MM_CH4_PROD_ACET": ("gC m-2 yr-1", 1.0),
    "MM_CH4_PROD_H2": ("gC m-2 yr-1", 1.0),
    "MM_CH4_SURF_DIFF": ("gC m-2 yr-1", 1.0),
    "MM_CH4_SURF_EBUL": ("gC m-2 yr-1", 1.0),
    "MM_CH4_SURF_AERE": ("gC m-2 yr-1", 1.0),
    "MM_SURFACE_CH4_FLUX": ("gC m-2 yr-1", 1000.0),
    "NDEP_TO_SMINN": ("gN m-2 yr-1", 1.0),
    "NDEP_TO_NPOOL": ("gN m-2 yr-1", 1.0),
    "NFIX_TO_ECOSYSN": ("gN m-2 yr-1", 1.0),
    "NFIX_TO_SMINN": ("gN m-2 yr-1", 1.0),
    "SUPPLEMENT_TO_SMINN": ("gN m-2 yr-1", 1.0),
    "GROSS_NMIN": ("gN m-2 yr-1", 1.0),
    "ACTUAL_IMMOB": ("gN m-2 yr-1", 1.0),
    "NET_NMIN": ("gN m-2 yr-1", 1.0),
    "SMINN_TO_PLANT": ("gN m-2 yr-1", 1.0),
    "DENIT": ("gN m-2 yr-1", 1.0),
    "F_NIT": ("gN m-2 yr-1", 1.0),
    "F_DENIT": ("gN m-2 yr-1", 1.0),
    "SOM_N_LEACHED": ("gN m-2 yr-1", 1.0),
    "SMIN_NO3_LEACHED": ("gN m-2 yr-1", 1.0),
    "SMIN_NO3_RUNOFF": ("gN m-2 yr-1", 1.0),
    "MM_LATERAL_NH4_FLUX": ("gN m-2 yr-1", 1.0),
    "MM_LATERAL_NO3_FLUX": ("gN m-2 yr-1", 1.0),
    "QDRAI": ("mm yr-1", 1.0),
    "QOVER": ("mm yr-1", 1.0),
    "QH2OSFC": ("mm yr-1", 1.0),
    "QFLX_LAT_AQU": ("mm yr-1", 1.0),
    "RAIN": ("mm yr-1", 1.0),
    "SNOW": ("mm yr-1", 1.0),
    "QRUNOFF": ("mm yr-1", 1.0),
}

STATES = {
    "TOTVEGC": "gC m-2",
    "TOTSOMC": "gC m-2",
    "TOTLITC": "gC m-2",
    "TOTECOSYSC": "gC m-2",
    "SMINN": "gN m-2",
    "SMIN_NH4": "gN m-2",
    "SMIN_NO3": "gN m-2",
    "TOTECOSYSN": "gN m-2",
    "TOTCOLN": "gN m-2",
    "TOTLITN": "gN m-2",
    "TOTSOMN": "gN m-2",
    "TOTVEGN": "gN m-2",
    "NPOOL": "gN m-2",
    "RETRANSN": "gN m-2",
    "FPG": "1",
    "FPG_P": "1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    return parser.parse_args()


def dates(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    variable = dataset.variables["time"]
    converted = num2date(
        variable[:], variable.units, getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=True,
    )
    return (
        np.asarray([item.year for item in converted], dtype=int),
        np.asarray([item.dayofyr for item in converted], dtype=int),
    )


def stream_files(run_dir: Path) -> dict[str, list[Path]]:
    streams: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.elm.h*.*.nc")):
        stream = path.name.split(".elm.", 1)[1].split(".", 1)[0]
        streams[stream].append(path)
    return streams


def choose_stream(run_dir: Path, variable: str) -> tuple[str, list[Path]]:
    candidates = []
    for stream, files in stream_files(run_dir).items():
        count = 0
        found = False
        dimension = ""
        for path in files:
            with Dataset(path) as dataset:
                item = dataset.variables.get(variable)
                if item is not None and "pft" in item.dimensions:
                    dimension = "pft"
                    found = True
                    count += len(dataset.dimensions["time"])
                elif item is not None and "column" in item.dimensions:
                    dimension = "column"
                    found = True
                    count += len(dataset.dimensions["time"])
                elif item is not None and "lndgrid" in item.dimensions:
                    dimension = "lndgrid"
                    found = True
                    count += len(dataset.dimensions["time"])
        if found:
            candidates.append((count, stream, dimension, files))
    if not candidates:
        return "", []
    selected = max(
        candidates, key=lambda item: (item[0], item[2] != "lndgrid")
    )
    return selected[2], selected[3]


def spatial_metadata(dataset: Dataset, dimension: str) -> dict[str, np.ndarray]:
    if dimension == "lndgrid":
        return {
            "active": np.ones(len(dataset.dimensions[dimension]), dtype=int),
            "topounit": np.zeros(len(dataset.dimensions[dimension]), dtype=int),
            "wtgcell": np.ones(len(dataset.dimensions[dimension]), dtype=float),
            "wttopounit": np.ones(len(dataset.dimensions[dimension]), dtype=float),
        }
    prefix = "pfts1d" if dimension == "pft" else "cols1d"
    return {
        "active": np.asarray(dataset.variables[f"{prefix}_active"][:], dtype=int),
        "topounit": np.asarray(dataset.variables[f"{prefix}_topounit"][:], dtype=int),
        "wtgcell": np.asarray(dataset.variables[f"{prefix}_wtgcell"][:], dtype=float),
        "wttopounit": np.asarray(
            dataset.variables[f"{prefix}_wttopounit"][:], dtype=float
        ),
    }


def entities(metadata: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    active = metadata["active"] == 1
    topounit = metadata["topounit"]
    result = {}
    if np.all(topounit == 0):
        selected = active & (metadata["wtgcell"] > 0.0)
        weights = metadata["wtgcell"][selected]
        return {"gridcell": (selected, weights / weights.sum())}
    for number, name in TOPOUNIT_NAMES.items():
        selected = active & (topounit == number) & (metadata["wttopounit"] > 0.0)
        weights = metadata["wttopounit"][selected]
        if weights.size and weights.sum() > 0.0:
            result[name] = (selected, weights / weights.sum())
    selected = active & (metadata["wtgcell"] > 0.0)
    weights = metadata["wtgcell"][selected]
    if weights.size and weights.sum() > 0.0:
        result["gridcell"] = (selected, weights / weights.sum())
    selected = active & np.isin(topounit, BOG_TOPOUNITS) & (metadata["wtgcell"] > 0.0)
    weights = metadata["wtgcell"][selected]
    if weights.size and weights.sum() > 0.0:
        result["bog"] = (selected, weights / weights.sum())
    return result


def time_space(variable, dimension: str) -> np.ndarray:
    values = np.ma.filled(variable[:], np.nan).astype(float)
    return np.moveaxis(
        values,
        (variable.dimensions.index("time"), variable.dimensions.index(dimension)),
        (0, 1),
    )


def annual_series(
    run_dir: Path, variable_name: str, start_year: int, end_year: int,
    integrate: bool,
) -> tuple[str, dict[str, dict[int, float]]]:
    dimension, files = choose_stream(run_dir, variable_name)
    if not files:
        return "", {}
    totals: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in files:
        with Dataset(path) as dataset:
            variable = dataset.variables.get(variable_name)
            if variable is None:
                continue
            if dimension not in variable.dimensions:
                continue
            years, _ = dates(dataset)
            keep = (years >= start_year) & (years <= end_year)
            if not np.any(keep):
                continue
            values = time_space(variable, dimension)[keep]
            selected_years = years[keep]
            for entity, (selected, weights) in entities(
                spatial_metadata(dataset, dimension)
            ).items():
                spatial_mean = np.nansum(values[:, selected] * weights[None, :], axis=1)
                for year in np.unique(selected_years):
                    sample = spatial_mean[selected_years == year]
                    if integrate:
                        totals[entity][int(year)].extend(sample.tolist())
                    else:
                        totals[entity][int(year)].extend(sample.tolist())
    annual: dict[str, dict[int, float]] = defaultdict(dict)
    for entity, by_year in totals.items():
        for year, samples in by_year.items():
            array = np.asarray(samples, dtype=float)
            annual[entity][year] = float(
                np.nansum(array) * SECONDS_PER_DAY * (365.0 if len(array) == 1 else 1.0)
                if integrate else np.nanmean(array)
            )
    return dimension, annual


def main() -> None:
    arguments = parse_args()
    rows = []
    for variable, (units, unit_scale) in FLUXES.items():
        _, annual = annual_series(
            arguments.run_dir, variable, arguments.start_year,
            arguments.end_year, integrate=True,
        )
        for entity, values in annual.items():
            scaled = np.asarray(list(values.values()), dtype=float) * unit_scale
            rows.append({
                "metric": variable,
                "entity": entity,
                "units": units,
                "start_year": arguments.start_year,
                "end_year": arguments.end_year,
                "n_years": len(scaled),
                "mean": np.nanmean(scaled),
                "interannual_sd": np.nanstd(scaled),
                "minimum": np.nanmin(scaled),
                "maximum": np.nanmax(scaled),
            })
    for variable, units in STATES.items():
        _, annual = annual_series(
            arguments.run_dir, variable, arguments.start_year,
            arguments.end_year, integrate=False,
        )
        for entity, values in annual.items():
            means = np.asarray(list(values.values()), dtype=float)
            rows.append({
                "metric": variable,
                "entity": entity,
                "units": units,
                "start_year": arguments.start_year,
                "end_year": arguments.end_year,
                "n_years": len(means),
                "mean": np.nanmean(means),
                "interannual_sd": np.nanstd(means),
                "minimum": np.nanmin(means),
                "maximum": np.nanmax(means),
            })
    output = pd.DataFrame(rows).sort_values(["metric", "entity"])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(arguments.output, index=False)
    print(output.to_string(index=False))
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()

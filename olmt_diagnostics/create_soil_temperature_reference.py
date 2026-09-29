#!/usr/bin/env python3
"""Create an ELM stream from a T0.00 control-depth temperature history.

The warmed SPRUCE cases use this compact stream as the untreated reference for
their explicit deep-heater controller.  The input is normally an hourly column
history tape containing ``TSOI_HEATING_CONTROL``.  For the three-topounit
SPRUCE configuration, columns 1 and 2 are the hollow and hummock; column 0
(the auxiliary fen) is intentionally excluded by the default selection.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import netCDF4
import numpy as np


def _parse_columns(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _selected_series(dataset, variable: str, columns: list[int]) -> np.ndarray:
    values = np.asarray(dataset.variables[variable][:], dtype=np.float64)
    dimensions = dataset.variables[variable].dimensions
    if "column" not in dimensions:
        return values.reshape(values.shape[0], -1).mean(axis=1)

    column_axis = dimensions.index("column")
    values = np.moveaxis(values, column_axis, 1)
    selected = values[:, columns]
    if "cols1d_wtgcell" in dataset.variables:
        weights = np.asarray(dataset.variables["cols1d_wtgcell"][:], dtype=float)[columns]
        weights = weights / weights.sum()
    else:
        weights = np.full(len(columns), 1.0 / len(columns))
    return np.sum(selected * weights[None, :], axis=1)


def create_reference(files: list[str], output: str, domain: str, variable: str,
                     columns: list[int]) -> None:
    if not files:
        raise ValueError("No T0.00 history files matched")

    time_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    time_units = None
    calendar = "noleap"
    for filename in sorted(files):
        with netCDF4.Dataset(filename) as source:
            if variable not in source.variables:
                raise KeyError(f"{variable} is absent from {filename}")
            time = source.variables["time"]
            if time_units is None:
                time_units = time.units
                calendar = getattr(time, "calendar", calendar)
            elif time.units != time_units:
                raise ValueError("All source files must use the same time units")
            time_values.append(np.asarray(time[:], dtype=np.float64))
            reference_values.append(_selected_series(source, variable, columns))

    time_values_array = np.concatenate(time_values)
    reference_values_array = np.concatenate(reference_values)
    order = np.argsort(time_values_array)
    time_values_array = time_values_array[order]
    reference_values_array = reference_values_array[order]

    with netCDF4.Dataset(domain) as domain_file:
        lon = float(np.asarray(domain_file.variables["xc"][:]).squeeze())
        lat = float(np.asarray(domain_file.variables["yc"][:]).squeeze())
        area = float(np.asarray(domain_file.variables["area"][:]).squeeze())
        mask = int(np.asarray(domain_file.variables["mask"][:]).squeeze())

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(output_path, "w", format="NETCDF4_CLASSIC") as target:
        target.createDimension("time", None)
        target.createDimension("lat", 1)
        target.createDimension("lon", 1)
        time = target.createVariable("time", "f8", ("time",))
        time.units = time_units
        time.calendar = calendar
        time[:] = time_values_array
        lat_var = target.createVariable("lat", "f8", ("lat",))
        lon_var = target.createVariable("lon", "f8", ("lon",))
        area_var = target.createVariable("area", "f8", ("lat", "lon"))
        mask_var = target.createVariable("mask", "i4", ("lat", "lon"))
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        area_var.units = "radians2"
        lat_var[:] = lat
        lon_var[:] = lon
        area_var[:, :] = area
        mask_var[:, :] = mask
        field = target.createVariable(
            "T_SOIL_REFERENCE", "f8", ("time", "lat", "lon"), zlib=True
        )
        field.units = "K"
        field.long_name = "untreated soil temperature at deep-heating control depth"
        field[:, 0, 0] = reference_values_array
        target.source_history_files = ",".join(str(path) for path in sorted(files))
        target.source_columns = ",".join(str(column) for column in columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="History-file glob")
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--variable", default="TSOI_HEATING_CONTROL")
    parser.add_argument(
        "--columns", type=_parse_columns, default=[1, 2],
        help="Comma-separated column indices; default 1,2 excludes SPRUCE fen",
    )
    args = parser.parse_args()
    create_reference(
        glob.glob(args.input), args.output, args.domain, args.variable, args.columns
    )


if __name__ == "__main__":
    main()

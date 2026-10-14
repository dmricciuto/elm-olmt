#!/usr/bin/env python3
"""Extract one or more geographic grid cells from an ELM restart file.

The utility subsets the complete ELM restart hierarchy (grid cells, topounits,
landunits, columns, and PFTs) and rewrites its one-based parent indices so the
result can be used as ``finidat`` for an OLMT point or point-list simulation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from netCDF4 import Dataset


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class HierarchyLevel:
    dimension: str
    prefix: str
    parents: tuple[str, ...]


ELM_HIERARCHY = (
    HierarchyLevel("gridcell", "grid1d", ()),
    HierarchyLevel("topounit", "topo1d", ("gridcell",)),
    HierarchyLevel("landunit", "land1d", ("gridcell", "topounit")),
    HierarchyLevel("column", "cols1d", ("gridcell", "topounit", "landunit")),
    HierarchyLevel("pft", "pfts1d", ("gridcell", "topounit", "landunit", "column")),
)

COORDINATE_PAIRS = (
    ("grid1d_lat", "grid1d_lon"),
    ("LATIXY", "LONGXY"),
    ("lat", "lon"),
    ("latitude", "longitude"),
)


def _validate_points(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    validated = []
    for lat, lon in points:
        lat = float(lat)
        lon = float(lon)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError(f"Coordinates must be finite, got ({lat}, {lon})")
        if lat < -90.0 or lat > 90.0:
            raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
        validated.append((lat, lon))
    if not validated:
        raise ValueError("At least one latitude/longitude pair is required")
    return validated


def _coordinate_variables(dataset: Dataset) -> tuple[str, str, str]:
    for lat_name, lon_name in COORDINATE_PAIRS:
        if lat_name not in dataset.variables or lon_name not in dataset.variables:
            continue
        lat_var = dataset.variables[lat_name]
        lon_var = dataset.variables[lon_name]
        if lat_var.ndim != 1 or lon_var.ndim != 1 or lat_var.dimensions != lon_var.dimensions:
            continue
        dimension = lat_var.dimensions[0]
        if dimension in dataset.dimensions:
            return lat_name, lon_name, dimension
    pairs = ", ".join(f"{lat}/{lon}" for lat, lon in COORDINATE_PAIRS)
    raise ValueError(f"Could not find one-dimensional restart coordinates; tried {pairs}")


def domain_points(path: os.PathLike[str] | str) -> list[tuple[float, float]]:
    """Return active domain-cell centers in the domain file's native order."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Domain file does not exist: {path}")
    with Dataset(path, "r") as domain:
        coordinate_pairs = (
            ("yc", "xc"),
            ("LATIXY", "LONGXY"),
            ("lat", "lon"),
            ("latitude", "longitude"),
        )
        for lat_name, lon_name in coordinate_pairs:
            if lat_name in domain.variables and lon_name in domain.variables:
                break
        else:
            pairs = ", ".join(f"{lat}/{lon}" for lat, lon in coordinate_pairs)
            raise ValueError(f"Could not find domain coordinates in {path}; tried {pairs}")

        lat_variable = domain.variables[lat_name]
        lon_variable = domain.variables[lon_name]
        lat = np.asarray(lat_variable[:], dtype=np.float64)
        lon = np.asarray(lon_variable[:], dtype=np.float64)
        if lat.ndim == 1 and lon.ndim == 1 and lat_variable.dimensions != lon_variable.dimensions:
            lon, lat = np.meshgrid(lon, lat)
        elif lat.shape != lon.shape:
            raise ValueError(
                f"Domain coordinate shapes do not match in {path}: {lat.shape} versus {lon.shape}"
            )

        active = np.isfinite(lat) & np.isfinite(lon)
        if "mask" in domain.variables:
            mask = np.asarray(domain.variables["mask"][:])
            if mask.shape != lat.shape:
                squeezed_mask = np.squeeze(mask)
                if squeezed_mask.shape == lat.shape:
                    mask = squeezed_mask
                else:
                    try:
                        mask = np.broadcast_to(mask, lat.shape)
                    except ValueError as error:
                        raise ValueError(
                            f"Domain mask shape {mask.shape} does not match coordinates {lat.shape}"
                        ) from error
            active &= mask > 0
        points = list(zip(lat[active].tolist(), lon[active].tolist()))
    return _validate_points(points)


def nearest_gridcells(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    points: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return source indices and great-circle distances for requested points."""
    lat = np.asarray(grid_lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(grid_lon, dtype=np.float64).reshape(-1)
    valid = np.isfinite(lat) & np.isfinite(lon) & (lat >= -90.0) & (lat <= 90.0)
    if not np.any(valid):
        raise ValueError("Restart file has no valid grid-cell coordinates")

    valid_indices = np.flatnonzero(valid)
    lat_radians = np.radians(lat[valid])
    lon_radians = np.radians(lon[valid])
    selected = []
    distances = []
    for point_lat, point_lon in _validate_points(points):
        point_lat_radians = math.radians(point_lat)
        point_lon_radians = math.radians(point_lon)
        delta_lat = lat_radians - point_lat_radians
        delta_lon = (lon_radians - point_lon_radians + math.pi) % (2.0 * math.pi) - math.pi
        haversine = (
            np.sin(delta_lat / 2.0) ** 2
            + math.cos(point_lat_radians) * np.cos(lat_radians) * np.sin(delta_lon / 2.0) ** 2
        )
        angular_distance = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
        nearest_valid = int(np.argmin(angular_distance))
        selected.append(int(valid_indices[nearest_valid]))
        distances.append(float(EARTH_RADIUS_KM * angular_distance[nearest_valid]))
    return np.asarray(selected, dtype=np.int64), np.asarray(distances, dtype=np.float64)


def _reference_name(dataset: Dataset, level: HierarchyLevel, parent: str) -> str | None:
    candidate = f"{level.prefix}_{parent}_index"
    if candidate in dataset.variables:
        return candidate
    return None


def _hierarchy_levels(dataset: Dataset, grid_dimension: str) -> list[HierarchyLevel]:
    if grid_dimension == "gridcell":
        return [level for level in ELM_HIERARCHY if level.dimension in dataset.dimensions]
    return [HierarchyLevel(grid_dimension, "", ())]


def _build_hierarchy_selection(
    dataset: Dataset,
    grid_dimension: str,
    grid_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    levels = _hierarchy_levels(dataset, grid_dimension)
    selectors: dict[str, np.ndarray] = {grid_dimension: np.asarray(grid_indices, dtype=np.int64)}
    occurrences: dict[str, np.ndarray] = {
        grid_dimension: np.arange(len(grid_indices), dtype=np.int64)
    }
    lookup: dict[str, dict[tuple[int, int], int]] = {grid_dimension: {}}
    for output_index, source_index in enumerate(grid_indices):
        lookup[grid_dimension][(output_index, int(source_index) + 1)] = output_index + 1

    replacements: dict[str, np.ndarray] = {}
    for level in levels[1:]:
        grid_reference = _reference_name(dataset, level, grid_dimension)
        if grid_reference is not None:
            reference_values = np.asarray(dataset.variables[grid_reference][:], dtype=np.int64)
            entries = [
                (int(source_index), occurrence)
                for occurrence, grid_index in enumerate(grid_indices)
                for source_index in np.flatnonzero(reference_values == int(grid_index) + 1)
            ]
        else:
            available_parent = next(
                (parent for parent in reversed(level.parents) if parent in selectors), None
            )
            if available_parent is None:
                raise ValueError(
                    f"Cannot determine which {level.dimension} records belong to selected grid cells"
                )
            parent_reference = _reference_name(dataset, level, available_parent)
            if parent_reference is None:
                raise ValueError(
                    f"Missing {level.dimension}-to-{available_parent} mapping in restart file"
                )
            reference_values = np.asarray(dataset.variables[parent_reference][:], dtype=np.int64)
            entries = []
            parent_selector = selectors[available_parent]
            parent_occurrences = occurrences[available_parent]
            for occurrence in range(len(grid_indices)):
                source_parent_ids = parent_selector[parent_occurrences == occurrence] + 1
                for source_index in np.flatnonzero(np.isin(reference_values, source_parent_ids)):
                    entries.append((int(source_index), occurrence))

        selectors[level.dimension] = np.asarray([entry[0] for entry in entries], dtype=np.int64)
        occurrences[level.dimension] = np.asarray([entry[1] for entry in entries], dtype=np.int64)
        lookup[level.dimension] = {
            (occurrence, source_index + 1): output_index + 1
            for output_index, (source_index, occurrence) in enumerate(entries)
        }

        for parent in level.parents:
            reference_name = _reference_name(dataset, level, parent)
            if reference_name is None or parent not in lookup:
                continue
            source_references = np.asarray(dataset.variables[reference_name][:], dtype=np.int64)
            remapped = []
            for source_index, occurrence in entries:
                source_parent = int(source_references[source_index])
                try:
                    remapped.append(lookup[parent][(occurrence, source_parent)])
                except KeyError as error:
                    raise ValueError(
                        f"{reference_name} points outside the selected {parent} hierarchy: "
                        f"source {level.dimension} {source_index + 1} -> {source_parent}"
                    ) from error
            replacements[reference_name] = np.asarray(remapped, dtype=source_references.dtype)
    return selectors, replacements


def _copy_attributes(source, target, excluded: set[str] | None = None) -> None:
    excluded = excluded or set()
    for name in source.ncattrs():
        if name not in excluded and name != "_NCProperties":
            target.setncattr(name, source.getncattr(name))


def _creation_options(variable) -> dict:
    options = {}
    if "_FillValue" in variable.ncattrs():
        options["fill_value"] = variable.getncattr("_FillValue")
    try:
        filters = variable.filters()
    except (AttributeError, RuntimeError):
        filters = None
    if filters and filters.get("zlib"):
        options.update(
            zlib=True,
            complevel=int(filters.get("complevel", 4)),
            shuffle=bool(filters.get("shuffle", True)),
            fletcher32=bool(filters.get("fletcher32", False)),
        )
    try:
        endian = variable.endian()
    except (AttributeError, RuntimeError):
        endian = "native"
    if endian in {"little", "big"}:
        options["endian"] = endian
    return options


def _read_subset(variable, selectors: dict[str, np.ndarray]):
    selected_axes = [
        (axis, selectors[dimension])
        for axis, dimension in enumerate(variable.dimensions)
        if dimension in selectors
    ]
    variable.set_auto_maskandscale(False)
    if not selected_axes:
        return variable[...]
    if len(selected_axes) == 1:
        axis, indices = selected_axes[0]
        key = [slice(None)] * variable.ndim
        key[axis] = indices
        return variable[tuple(key)]
    data = variable[...]
    for axis, indices in selected_axes:
        data = np.take(data, indices, axis=axis)
    return data


def subset_restart(
    input_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    points: Sequence[tuple[float, float]],
    max_distance_km: float | None = 250.0,
    overwrite: bool = False,
) -> list[dict[str, float | int]]:
    """Write an ELM restart containing nearest grid cells for ``points``."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output restart paths must differ")
    if not input_path.is_file():
        raise FileNotFoundError(f"Restart file does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = _validate_points(points)

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    matches = []
    try:
        with Dataset(input_path, "r") as source:
            lat_name, lon_name, grid_dimension = _coordinate_variables(source)
            grid_lat = np.asarray(source.variables[lat_name][:], dtype=np.float64)
            grid_lon = np.asarray(source.variables[lon_name][:], dtype=np.float64)
            grid_indices, distances = nearest_gridcells(grid_lat, grid_lon, points)
            if max_distance_km is not None and max_distance_km > 0:
                too_far = np.flatnonzero(distances > max_distance_km)
                if len(too_far):
                    i = int(too_far[0])
                    raise ValueError(
                        f"Nearest restart grid cell to ({points[i][0]}, {points[i][1]}) is "
                        f"{distances[i]:.1f} km away, exceeding {max_distance_km:g} km"
                    )
            selectors, replacements = _build_hierarchy_selection(
                source, grid_dimension, grid_indices
            )
            for requested, source_index, distance in zip(points, grid_indices, distances):
                matches.append(
                    {
                        "requested_lat": requested[0],
                        "requested_lon": requested[1],
                        "source_index": int(source_index),
                        "grid_lat": float(grid_lat[source_index]),
                        "grid_lon": float(grid_lon[source_index]),
                        "distance_km": float(distance),
                    }
                )

            with Dataset(temporary_path, "w", format=source.data_model) as target:
                _copy_attributes(source, target)
                for name, dimension in source.dimensions.items():
                    if name in selectors:
                        size = len(selectors[name])
                    else:
                        size = None if dimension.isunlimited() else len(dimension)
                    target.createDimension(name, size)

                for name, variable in source.variables.items():
                    target_variable = target.createVariable(
                        name, variable.datatype, variable.dimensions, **_creation_options(variable)
                    )
                    _copy_attributes(variable, target_variable, {"_FillValue"})
                    target_variable.set_auto_maskandscale(False)
                    if name in replacements:
                        target_variable[...] = replacements[name]
                    else:
                        target_variable[...] = _read_subset(variable, selectors)

                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                command = " ".join(sys.argv)
                previous_history = getattr(source, "history", "")
                history_line = f"{now}: OLMT subset_restart.py {command}".rstrip()
                target.setncattr("history", history_line + ("\n" + previous_history if previous_history else ""))
                target.setncattr("olmt_restart_subset_source", str(input_path))
                target.setncattr(
                    "olmt_restart_subset_source_grid_indices",
                    ",".join(str(int(index)) for index in grid_indices),
                )
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return matches


def restart_matches_domain(
    restart_path: os.PathLike[str] | str,
    domain_path: os.PathLike[str] | str,
    tolerance_km: float = 0.01,
) -> bool:
    """Return whether active domain cells already match the restart in order."""
    points = domain_points(domain_path)
    with Dataset(restart_path, "r") as restart:
        lat_name, lon_name, grid_dimension = _coordinate_variables(restart)
        if len(restart.dimensions[grid_dimension]) != len(points):
            return False
        grid_lat = np.asarray(restart.variables[lat_name][:], dtype=np.float64)
        grid_lon = np.asarray(restart.variables[lon_name][:], dtype=np.float64)
    indices, distances = nearest_gridcells(grid_lat, grid_lon, points)
    return bool(
        np.array_equal(indices, np.arange(len(points), dtype=np.int64))
        and np.all(distances <= tolerance_km)
    )


def subset_restart_from_domain(
    input_path: os.PathLike[str] | str,
    domain_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    max_distance_km: float | None = 250.0,
    overwrite: bool = False,
) -> list[dict[str, float | int]]:
    """Subset ``input_path`` to active cells and ordering from ``domain_path``."""
    return subset_restart(
        input_path,
        output_path,
        domain_points(domain_path),
        max_distance_km=max_distance_km,
        overwrite=overwrite,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract nearest grid cells from an ELM restart while preserving and "
            "remapping its topounit/landunit/column/PFT hierarchy."
        )
    )
    parser.add_argument("input", help="Input ELM restart NetCDF file")
    parser.add_argument("output", help="Output subset restart NetCDF file")
    parser.add_argument(
        "--domain",
        required=True,
        help="Generated ELM domain file defining active cells and output ordering",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=250.0,
        help="Reject a nearest cell farther away than this (default: 250; <=0 disables)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        matches = subset_restart_from_domain(
            args.input,
            args.domain,
            args.output,
            max_distance_km=args.max_distance_km,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"ERROR: {error}\n")

    print(f"Wrote {len(matches)} grid cell(s) to {Path(args.output).resolve()}")
    for number, match in enumerate(matches, start=1):
        print(
            f"  {number}: requested ({match['requested_lat']:.6f}, "
            f"{match['requested_lon']:.6f}) -> source index {match['source_index']} "
            f"at ({match['grid_lat']:.6f}, {match['grid_lon']:.6f}), "
            f"{match['distance_km']:.2f} km"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

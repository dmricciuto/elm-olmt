"""SPRUCE DOM, acetate, and methane porewater-profile diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd

from .spruce import CASE_LABELS, CASE_ORDER, treatment_run_dirs
from .spruce_carbon import CASE_COLORS


CARBON_ATOMIC_MASS_G_MOL = 12.011
WATER_DENSITY_KG_M3 = 1000.0
BOG_TOPOUNITS = (2, 3)
PROFILE_DEFINITIONS = {
    "doc": {
        "variable": "MM_DOM_POREWATER_C",
        "label": "DOC (g C m$^{-3}$ water)",
        "observation_file": "CDOCS.txt",
    },
    "acetate": {
        "variable": "MM_ACETATE_C_SAT",
        "label": "Acetate (mmol L$^{-1}$)",
        "observation_file": "CACES.txt",
    },
    "ch4": {
        "variable": "MM_CH4_POREWATER",
        "label": "CH$_4$ (mmol L$^{-1}$)",
        "observation_file": "CCON_CH4S.txt",
    },
}


def _dates(dataset: Dataset) -> pd.DatetimeIndex:
    variable = dataset.variables["time"]
    values = num2date(
        variable[:], variable.units, getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False, only_use_python_datetimes=False,
    )
    return pd.DatetimeIndex(pd.Timestamp(str(value)[:10]) for value in values)


def _history_files(run_dir: Path, variable: str) -> list[Path]:
    stream = None
    for path in sorted(run_dir.glob("*.elm.h*.*.nc")):
        with Dataset(path) as dataset:
            item = dataset.variables.get(variable)
            if item is not None and {"time", "column"}.issubset(item.dimensions):
                stream = path.name.split(".elm.", 1)[1].split(".", 1)[0]
                break
    if stream is None:
        raise RuntimeError(
            f"No column-resolved {variable} in {run_dir}; add "
            f"{variable}_col to the OLMT postprocessing variables."
        )
    return sorted(run_dir.glob(f"*.elm.{stream}.*.nc"))


def _time_layer_column(variable) -> np.ndarray:
    dimensions = variable.dimensions
    level = next((name for name in dimensions if name.startswith("lev")), None)
    if level is None or "time" not in dimensions or "column" not in dimensions:
        raise RuntimeError(
            f"Expected time, layer, and column dimensions for {variable.name}; "
            f"found {dimensions}"
        )
    values = np.ma.filled(variable[:], np.nan).astype(float)
    return np.moveaxis(
        values,
        (dimensions.index("time"), dimensions.index(level), dimensions.index("column")),
        (0, 1, 2),
    )


def _bog_columns_and_weights(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(dataset.variables["cols1d_active"][:], dtype=int)
    topounit = np.asarray(dataset.variables["cols1d_topounit"][:], dtype=int)
    weights = np.asarray(dataset.variables["cols1d_wtgcell"][:], dtype=float)
    selected = (active == 1) & np.isin(topounit, BOG_TOPOUNITS) & (weights > 0.0)
    for name in ("cols1d_itype_lunit", "cols1d_ityplunit"):
        if name in dataset.variables:
            selected &= np.asarray(dataset.variables[name][:], dtype=int) == 1
            break
    columns = np.flatnonzero(selected)
    if not columns.size:
        raise RuntimeError("No active natural-vegetation hollow/hummock columns")
    normalized = weights[columns] / weights[columns].sum()
    return columns, normalized


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average a time x layer x selected-column array over valid columns."""
    valid = np.isfinite(values)
    effective = valid * weights[None, None, :]
    denominator = effective.sum(axis=2)
    numerator = np.nansum(values * weights[None, None, :], axis=2)
    return np.divide(
        numerator, denominator, out=np.full(denominator.shape, np.nan),
        where=denominator > 0.0,
    )


def _layer_thickness(depth: np.ndarray) -> np.ndarray:
    interface = np.zeros(depth.size + 1)
    for level, center in enumerate(depth):
        interface[level + 1] = 2.0 * center - interface[level]
    return np.diff(interface)


def _profiles_from_file(
    path: Path, start_year: int, end_year: int, doy_start: int, doy_end: int
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    with Dataset(path) as dataset:
        dates = _dates(dataset)
        keep = (
            (dates.year >= start_year)
            & (dates.year <= end_year)
            & (dates.dayofyear >= doy_start)
            & (dates.dayofyear <= doy_end)
        )
        if not np.any(keep):
            return None
        columns, weights = _bog_columns_and_weights(dataset)
        depth = np.asarray(dataset.variables["levdcmp"][:], dtype=float)

        doc = _time_layer_column(dataset.variables["MM_DOM_POREWATER_C"])[
            keep, :, :
        ][:, :, columns]
        ch4 = _time_layer_column(dataset.variables["MM_CH4_POREWATER"])[
            keep, :, :
        ][:, :, columns]
        acetate_c = _time_layer_column(dataset.variables["MM_ACETATE_C_SAT"])[
            keep, :, :
        ][:, :, columns]
        liquid = _time_layer_column(dataset.variables["SOILLIQ"])[
            keep, : depth.size, :
        ][:, :, columns]
        liquid_fraction = liquid / (
            WATER_DENSITY_KG_M3 * _layer_thickness(depth)[None, :, None]
        )
        # MM_ACETATE_C_SAT is a carbon-mass inventory per bulk layer volume.
        # Divide by liquid fraction and by two carbon atoms per acetate molecule.
        # Numerically mol m-3 equals mmol L-1.
        acetate = np.divide(
            acetate_c,
            liquid_fraction * (2.0 * CARBON_ATOMIC_MASS_G_MOL),
            out=np.full_like(acetate_c, np.nan),
            where=liquid_fraction > 1.0e-12,
        )
        return depth, {
            "doc": _weighted_mean(doc, weights),
            "acetate": _weighted_mean(acetate, weights),
            "ch4": _weighted_mean(ch4, weights),
        }


def read_treatment_profiles(
    run_root: Path,
    case_prefix: str,
    start_year: int = 2019,
    end_year: int = 2023,
    doy_start: int = 121,
    doy_end: int = 260,
) -> pd.DataFrame:
    """Read hummock+hollow weighted porewater profiles for all treatments."""
    runs = treatment_run_dirs(run_root, case_prefix)
    rows = []
    for case in CASE_ORDER:
        chunks = {name: [] for name in PROFILE_DEFINITIONS}
        depth = None
        for path in _history_files(runs[case], "MM_DOM_POREWATER_C"):
            result = _profiles_from_file(
                path, start_year, end_year, doy_start, doy_end
            )
            if result is None:
                continue
            file_depth, profiles = result
            if depth is None:
                depth = file_depth
            elif not np.allclose(depth, file_depth):
                raise RuntimeError(f"Layer depths change between history files in {runs[case]}")
            for name, values in profiles.items():
                chunks[name].append(values)
        if depth is None or any(not values for values in chunks.values()):
            raise RuntimeError(
                f"No profile samples selected for {case} during "
                f"{start_year}-{end_year}, DOY {doy_start}-{doy_end}"
            )
        for name, values in chunks.items():
            samples = np.concatenate(values, axis=0)
            for level, z in enumerate(depth):
                layer = samples[:, level]
                rows.append(
                    {
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "variable": name,
                        "depth_m": z,
                        "mean": np.nanmean(layer),
                        "p05": np.nanpercentile(layer, 5),
                        "p95": np.nanpercentile(layer, 95),
                        "n": int(np.isfinite(layer).sum()),
                    }
                )
    return pd.DataFrame(rows)


def read_profile_observations(directory: Path, year: int = 2013) -> pd.DataFrame:
    rows = []
    for name, definition in PROFILE_DEFINITIONS.items():
        path = directory / definition["observation_file"]
        frame = pd.read_csv(path, sep=r"\s+")
        frame.columns = [column.lower() for column in frame.columns]
        frame = frame[pd.to_numeric(frame["year"], errors="coerce") == year].copy()
        value = pd.to_numeric(frame["value"], errors="coerce").to_numpy(float)
        uncertainty = pd.to_numeric(
            frame["uncertainty"], errors="coerce"
        ).to_numpy(float)
        if name == "doc":
            value *= CARBON_ATOMIC_MASS_G_MOL
            uncertainty *= CARBON_ATOMIC_MASS_G_MOL
        uncertainty[uncertainty < 0.0] = np.nan
        for depth, observed, error in zip(
            pd.to_numeric(frame["depth"], errors="coerce") / 100.0,
            value,
            uncertainty,
        ):
            rows.append(
                {
                    "variable": name,
                    "year": year,
                    "depth_m": depth,
                    "observed": observed,
                    "uncertainty": error,
                }
            )
    return pd.DataFrame(rows)


def plot_profiles(
    profiles: pd.DataFrame,
    observations: pd.DataFrame,
    output: Path,
    start_year: int,
    end_year: int,
    doy_start: int,
    doy_end: int,
    observation_year: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.2), sharey=True)
    for axis, (name, definition) in zip(axes, PROFILE_DEFINITIONS.items()):
        observed = observations[observations["variable"] == name]
        axis.errorbar(
            observed["observed"], observed["depth_m"],
            xerr=observed["uncertainty"], fmt="o", ms=4.5,
            markerfacecolor="white", markeredgecolor="0.2", color="0.45",
            ecolor="0.65", alpha=0.8, capsize=2, label=f"Observed ({observation_year})",
            zorder=5,
        )
        for case in CASE_ORDER:
            selected = profiles[
                (profiles["variable"] == name) & (profiles["case"] == case)
            ].sort_values("depth_m")
            axis.plot(
                selected["mean"], selected["depth_m"],
                color=CASE_COLORS[case], lw=1.9, marker=".", ms=4,
                label=CASE_LABELS[case],
            )
        axis.set_xlabel(definition["label"])
        axis.set_ylim(2.5, 0.0)
        axis.set_xlim(left=0.0)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Depth below local peat surface (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=8.5)
    fig.suptitle("SPRUCE porewater profiles: ELM treatments versus observations")
    fig.text(
        0.5, 0.055,
        f"ELM means: {start_year}-{end_year}, DOY {doy_start}-{doy_end}; "
        "hummock+hollow weighted mean, fen excluded.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_profile_products(
    profiles: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: Path,
    start_year: int,
    end_year: int,
    doy_start: int,
    doy_end: int,
    observation_year: int,
    source: dict[str, object] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plot": output_dir / "porewater_dom_acetate_ch4_profiles.png",
        "model": output_dir / "porewater_profiles_model.csv",
        "observations": output_dir / "porewater_profiles_observations.csv",
        "manifest": output_dir / "porewater_profiles_manifest.json",
    }
    profiles.to_csv(paths["model"], index=False)
    observations.to_csv(paths["observations"], index=False)
    plot_profiles(
        profiles, observations, paths["plot"], start_year, end_year,
        doy_start, doy_end, observation_year,
    )
    manifest = {
        "diagnostic": "SPRUCE porewater DOM, acetate, and methane profiles",
        "model_area_basis": "hummock+hollow area; fen excluded",
        "model_period": {"start_year": start_year, "end_year": end_year},
        "model_day_of_year_window": [doy_start, doy_end],
        "observation_year": observation_year,
        "acetate_conversion": (
            "MM_ACETATE_C_SAT / liquid_water_fraction / "
            "(2 * 12.011 g C mol-1); mol m-3 = mmol L-1"
        ),
        **(source or {}),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return paths

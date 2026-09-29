"""SPRUCE treatment diagnostics and observation comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd


CASE_ORDER = ["TAMB", "T0.00", "T2.25", "T4.50", "T6.75", "T9.00"]
CASE_LABELS = {
    "TAMB": "Ambient",
    "T0.00": "T0",
    "T2.25": "T+2.25",
    "T4.50": "T+4.50",
    "T6.75": "T+6.75",
    "T9.00": "T+9.00",
}
OBS_GROUP_BY_CASE = {
    "TAMB": "ambient_reference",
    "T0.00": "T0_ambient_CO2",
    "T2.25": "T2.25",
    "T4.50": "T4.50",
    "T6.75": "T6.75",
    "T9.00": "T9.00",
}
HOLLOW_TOPOUNIT = 2


def _dates(dataset: Dataset) -> pd.DatetimeIndex:
    time = dataset.variables["time"]
    values = num2date(
        time[:],
        time.units,
        getattr(time, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    return pd.DatetimeIndex(pd.Timestamp(str(value)[:10]) for value in values)


def _decode_string(value: object) -> str:
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8", errors="ignore").strip("\x00 ")
    return str(value).strip()


def treatment_run_dirs(run_root: Path, case_prefix: str) -> dict[str, Path]:
    """Return the six standard treatment run directories in canonical order."""
    result = {}
    for case in CASE_ORDER:
        run_dir = run_root / f"{case_prefix}_{case}" / "run"
        if run_dir.is_dir():
            result[case] = run_dir
    missing = [case for case in CASE_ORDER if case not in result]
    if missing:
        raise FileNotFoundError(
            f"Missing SPRUCE treatment run directories for {', '.join(missing)} "
            f"under {run_root} (prefix {case_prefix})"
        )
    return result


def _column_history_files(run_dir: Path, variable: str = "ZWT") -> list[Path]:
    """Discover the history tape holding a column-resolved variable."""
    candidates = sorted(run_dir.glob("*.elm.h*.*.nc"))
    stream = None
    for path in candidates:
        with Dataset(path) as dataset:
            var = dataset.variables.get(variable)
            if var is not None and "column" in var.dimensions and "time" in var.dimensions:
                stream = path.name.split(".elm.", 1)[1].split(".", 1)[0]
                break
    if stream is None:
        raise RuntimeError(f"No column-resolved {variable} history stream in {run_dir}")
    files = sorted(run_dir.glob(f"*.elm.{stream}.*.nc"))
    if not files:
        raise RuntimeError(f"No files found for discovered stream {stream} in {run_dir}")
    return files


def _hollow_columns(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    topounit = np.asarray(dataset.variables["cols1d_topounit"][:], dtype=int)
    active = np.asarray(dataset.variables["cols1d_active"][:], dtype=int)
    grid_weight = np.asarray(dataset.variables["cols1d_wtgcell"][:], dtype=float)
    columns = np.where(
        (topounit == HOLLOW_TOPOUNIT) & (active == 1) & (grid_weight > 0)
    )[0]
    if not len(columns):
        raise RuntimeError("No active hollow columns (topounit 2) in history file")
    if "cols1d_wttopounit" in dataset.variables:
        weights = np.asarray(dataset.variables["cols1d_wttopounit"][columns], dtype=float)
    else:
        weights = np.ones(len(columns), dtype=float)
    if not np.isfinite(weights).any() or np.nansum(weights) <= 0:
        weights = np.ones(len(columns), dtype=float)
    weights = weights / np.nansum(weights)
    return columns, weights


def _weighted_column_mean(
    dataset: Dataset, variable: str, columns: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    values = np.ma.filled(dataset.variables[variable][:, columns], np.nan).astype(float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    valid = np.isfinite(values)
    effective_weights = valid * weights[np.newaxis, :]
    denominator = effective_weights.sum(axis=1)
    numerator = np.nansum(values * weights[np.newaxis, :], axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[0], np.nan),
        where=denominator > 0,
    )


def _weighted_column_inventory(
    dataset: Dataset, variable: str, columns: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Sum a layer-resolved variable and average its inventory across columns."""
    netcdf_variable = dataset.variables[variable]
    dimensions = netcdf_variable.dimensions
    if "time" not in dimensions or "column" not in dimensions:
        raise RuntimeError(
            f"{variable} must have time and column dimensions; found {dimensions}"
        )
    values = np.ma.filled(netcdf_variable[:], np.nan).astype(float)
    values = np.moveaxis(
        values,
        (dimensions.index("time"), dimensions.index("column")),
        (0, 1),
    )
    values = values[:, columns, ...]
    if values.ndim > 2:
        values = np.nansum(values, axis=tuple(range(2, values.ndim)))
    return np.nansum(values * weights[np.newaxis, :], axis=1)


def read_hollow_water_table(
    run_dir: Path,
    case: str,
    start_year: int | None = None,
    end_year: int | None = None,
    missing_h2osfc: str = "error",
    soilice_threshold_kg_m2: float = 0.1,
) -> pd.DataFrame:
    """Read daily hollow water-table height from a treatment case.

    Height is positive above the hollow surface and is defined as
    ``-ZWT + H2OSFC / 1000``.  ``missing_h2osfc='zero'`` is an explicit
    compatibility mode for older runs that did not archive column H2OSFC.
    """
    if missing_h2osfc not in {"error", "zero"}:
        raise ValueError("missing_h2osfc must be 'error' or 'zero'")
    frames = []
    for path in _column_history_files(run_dir):
        with Dataset(path) as dataset:
            dates = _dates(dataset)
            selected = np.ones(len(dates), dtype=bool)
            if start_year is not None:
                selected &= dates.year >= start_year
            if end_year is not None:
                selected &= dates.year <= end_year
            if not selected.any():
                continue
            columns, weights = _hollow_columns(dataset)
            zwt = _weighted_column_mean(dataset, "ZWT", columns, weights)
            has_h2osfc = "H2OSFC" in dataset.variables and "column" in dataset.variables["H2OSFC"].dimensions
            if has_h2osfc:
                h2osfc = _weighted_column_mean(dataset, "H2OSFC", columns, weights)
            elif missing_h2osfc == "zero":
                h2osfc = np.zeros_like(zwt)
            else:
                raise RuntimeError(
                    f"{path} has no column-resolved H2OSFC. Rerun with "
                    "H2OSFC_col in the OLMT postprocessing variables, or use "
                    "missing_h2osfc='zero' for a clearly labeled approximation."
                )
            if "SOILICE" not in dataset.variables:
                raise RuntimeError(
                    f"{path} has no SOILICE; include SOILICE_col in the OLMT "
                    "postprocessing variables to shade frozen periods."
                )
            soilice = _weighted_column_inventory(
                dataset, "SOILICE", columns, weights
            )
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates[selected],
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "model_zwt_m": zwt[selected],
                        "model_h2osfc_mm": h2osfc[selected],
                        "model_wt_height_m": -zwt[selected] + h2osfc[selected] / 1000.0,
                        "h2osfc_available": has_h2osfc,
                        "soilice_total_kg_m2": soilice[selected],
                        "soilice_present": soilice[selected] > soilice_threshold_kg_m2,
                    }
                )
            )
    if not frames:
        raise RuntimeError(f"No model dates selected from {run_dir}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "case"])


def read_water_table_observations(
    obs_file: Path, start_year: int | None = None, end_year: int | None = None
) -> pd.DataFrame:
    """Read and average observed hollow-relative water-table heights by treatment."""
    with Dataset(obs_file) as dataset:
        dates = _dates(dataset)
        groups = [_decode_string(value) for value in dataset.variables["plot_group"][:]]
        height = np.ma.filled(
            dataset.variables["wt_height_above_hollow_m"][:], np.nan
        ).astype(float)
    result = pd.DataFrame({"date": dates})
    for case, group in OBS_GROUP_BY_CASE.items():
        indices = np.asarray([i for i, value in enumerate(groups) if value == group])
        if not len(indices):
            raise RuntimeError(f"Observation group {group!r} required for {case} is absent")
        values = height[:, indices]
        counts = np.sum(np.isfinite(values), axis=1)
        result[case] = np.divide(
            np.nansum(values, axis=1),
            counts,
            out=np.full(values.shape[0], np.nan),
            where=counts > 0,
        )
    if start_year is not None:
        result = result[result["date"].dt.year >= start_year]
    if end_year is not None:
        result = result[result["date"].dt.year <= end_year]
    return result.reset_index(drop=True)


def compare_hollow_water_tables(
    run_root: Path,
    case_prefix: str,
    obs_file: Path,
    start_year: int | None = None,
    end_year: int | None = None,
    missing_h2osfc: str = "error",
    soilice_threshold_kg_m2: float = 0.1,
) -> pd.DataFrame:
    """Return aligned model and observed daily hollow water-table heights."""
    runs = treatment_run_dirs(run_root, case_prefix)
    observations = read_water_table_observations(obs_file, start_year, end_year)
    frames = []
    for case in CASE_ORDER:
        model = read_hollow_water_table(
            runs[case], case, start_year, end_year, missing_h2osfc,
            soilice_threshold_kg_m2,
        )
        obs = observations[["date", case]].rename(columns={case: "observed_wt_height_m"})
        frames.append(model.merge(obs, on="date", how="left"))
    return pd.concat(frames, ignore_index=True)


def water_table_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched model/observation water tables on thawed days only.

    ``r_squared`` is the squared Pearson correlation.  Bias is ELM minus
    observed water-table height.
    """
    rows = []
    for case in CASE_ORDER:
        subset = comparison[comparison["case"] == case]
        valid = subset.loc[
            ~subset["soilice_present"].astype(bool),
            ["model_wt_height_m", "observed_wt_height_m"],
        ].dropna()
        difference = valid["model_wt_height_m"] - valid["observed_wt_height_m"]
        if (
            len(valid) >= 2
            and valid["model_wt_height_m"].nunique() > 1
            and valid["observed_wt_height_m"].nunique() > 1
        ):
            correlation = valid["model_wt_height_m"].corr(
                valid["observed_wt_height_m"]
            )
            r_squared = correlation**2
        else:
            r_squared = np.nan
        rows.append(
            {
                "case": case,
                "case_label": CASE_LABELS[case],
                "n": len(valid),
                "period": "non-ice matched days",
                "model_mean_m": valid["model_wt_height_m"].mean(),
                "observed_mean_m": valid["observed_wt_height_m"].mean(),
                "bias_m": difference.mean(),
                "rmse_m": np.sqrt(np.mean(difference**2)) if len(valid) else np.nan,
                "r_squared": r_squared,
                "h2osfc_available": bool(subset["h2osfc_available"].all()),
                "frozen_days": int(subset["soilice_present"].sum()),
                "frozen_fraction": float(subset["soilice_present"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_hollow_water_tables(comparison: pd.DataFrame, output: Path) -> None:
    """Create a standard six-panel model/observation water-table figure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = water_table_metrics(comparison).set_index("case")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, case in zip(axes.ravel(), CASE_ORDER):
        subset = comparison[comparison["case"] == case].sort_values("date")
        ax.fill_between(
            subset["date"], 0.0, 1.0,
            where=subset["soilice_present"].to_numpy(dtype=bool),
            transform=ax.get_xaxis_transform(),
            color="0.75", alpha=0.45, linewidth=0, step="mid",
            label="Ice in hollow column",
        )
        ax.plot(
            subset["date"], subset["observed_wt_height_m"],
            color="black", linewidth=1.0, label="Observed",
        )
        ax.plot(
            subset["date"], subset["model_wt_height_m"],
            color="#2166ac", linewidth=0.9, label="ELM",
        )
        ax.axhline(0.0, color="0.45", linewidth=0.7)
        ax.set_title(CASE_LABELS[case], loc="left", fontsize=11)
        metric = metrics.loc[case]
        annotation = (
            f"Thawed n={int(metric['n'])}\n"
            f"R²={metric['r_squared']:.2f}   RMSE={100.0 * metric['rmse_m']:.1f} cm\n"
            f"Mean ELM/obs={100.0 * metric['model_mean_m']:.1f}/"
            f"{100.0 * metric['observed_mean_m']:.1f} cm\n"
            f"Bias={100.0 * metric['bias_m']:+.1f} cm"
        )
        ax.text(
            0.015, 0.035, annotation,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8,
                  "edgecolor": "0.75"},
        )
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].legend(frameon=False, loc="lower right")
    for ax in axes[:, 0]:
        ax.set_ylabel("water-table height above hollow (m)")
    exact = bool(comparison["h2osfc_available"].all())
    title = "SPRUCE hollow water table: ELM versus observations"
    subtitle = "ELM = -ZWT + H2OSFC/1000"
    if not exact:
        subtitle += "; H2OSFC was not archived, so H2OSFC=0 in this legacy-output plot"
    fig.suptitle(f"{title}\n{subtitle}", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_water_table_products(
    comparison: pd.DataFrame,
    output_dir: Path,
    source: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Write the standard plot, aligned daily data, metrics, and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plot": output_dir / "hollow_water_table_vs_observations.png",
        "daily": output_dir / "hollow_water_table_vs_observations_daily.csv",
        "metrics": output_dir / "hollow_water_table_vs_observations_metrics.csv",
        "manifest": output_dir / "hollow_water_table_vs_observations_manifest.json",
    }
    plot_hollow_water_tables(comparison, paths["plot"])
    comparison.to_csv(paths["daily"], index=False)
    water_table_metrics(comparison).to_csv(paths["metrics"], index=False)
    manifest = {
        "diagnostic": "SPRUCE treatment hollow water table",
        "formula": "model_wt_height_m = -ZWT_m + H2OSFC_mm / 1000",
        "metrics_period": "matched non-ice days only",
        "bias_definition": "ELM minus observed water-table height",
        "r_squared_definition": "squared Pearson correlation",
        "hollow_topounit": HOLLOW_TOPOUNIT,
        "treatments": CASE_ORDER,
        "h2osfc_available": bool(comparison["h2osfc_available"].all()),
        **(source or {}),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return paths

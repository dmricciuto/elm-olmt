"""SPRUCE treatment carbon-budget diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd

from .spruce import CASE_LABELS, CASE_ORDER, treatment_run_dirs


SECONDS_PER_DAY = 86400.0
BOG_TOPOUNITS = (2, 3)
FIT_CASES = ("T0.00", "T2.25", "T4.50", "T6.75", "T9.00")
TREATMENT_WARMING_C = {
    "TAMB": np.nan,
    "T0.00": 0.0,
    "T2.25": 2.25,
    "T4.50": 4.5,
    "T6.75": 6.75,
    "T9.00": 9.0,
}
CASE_COLORS = {
    "TAMB": "#4d4d4d",
    "T0.00": "#1b9e77",
    "T2.25": "#d95f02",
    "T4.50": "#7570b3",
    "T6.75": "#e7298a",
    "T9.00": "#66a61e",
}

METRICS = {
    "agnpp_tree": {
        "label": "Tree aboveground NPP",
        "model_variable": "AGNPP",
        "model_dimension": "pft",
        "pft_indices": (3, 5),
        "observation_variable": "AGNPP_pftgroup_tree",
    },
    "agnpp_shrub": {
        "label": "Shrub aboveground NPP",
        "model_variable": "AGNPP",
        "model_dimension": "pft",
        "pft_indices": (14,),
        "observation_variable": "AGNPP_pftgroup_shrub",
    },
    "bgnpp_woody": {
        "label": "Tree + shrub belowground NPP",
        "model_variable": "BGNPP",
        "model_dimension": "pft",
        "pft_indices": (3, 5, 14),
        "observation_variable": "BGNPP_pftgroup_woody",
    },
    "npp_moss": {
        "label": "Moss NPP",
        "model_variable": "NPP",
        "model_dimension": "pft",
        "pft_indices": (18,),
        "observation_variable": "NPP_pftgroup_moss",
    },
    "hr": {
        "label": "Heterotrophic respiration",
        "model_variable": "HR",
        "model_dimension": "column",
        "pft_indices": (),
        "observation_variable": "HR",
    },
}


def _dates(dataset: Dataset) -> pd.DatetimeIndex:
    time = dataset.variables["time"]
    values = num2date(
        time[:], time.units, getattr(time, "calendar", "standard"),
        only_use_cftime_datetimes=False, only_use_python_datetimes=False,
    )
    return pd.DatetimeIndex(pd.Timestamp(str(value)[:10]) for value in values)


def _history_files(run_dir: Path, variable: str, dimension: str) -> list[Path]:
    stream = None
    for path in sorted(run_dir.glob("*.elm.h*.*.nc")):
        with Dataset(path) as dataset:
            item = dataset.variables.get(variable)
            if item is not None and dimension in item.dimensions and "time" in item.dimensions:
                stream = path.name.split(".elm.", 1)[1].split(".", 1)[0]
                break
    if stream is None:
        raise RuntimeError(
            f"No {dimension}-resolved {variable} in {run_dir}. Add "
            f"{variable}_{'pft' if dimension == 'pft' else 'col'} to the OLMT "
            "postprocessing variables and rerun the treatment stage."
        )
    return sorted(run_dir.glob(f"*.elm.{stream}.*.nc"))


def _axis_values(variable, dimension: str) -> np.ndarray:
    values = np.ma.filled(variable[:], np.nan).astype(float)
    return np.moveaxis(
        values,
        (variable.dimensions.index("time"), variable.dimensions.index(dimension)),
        (0, 1),
    )


def _annual_pft_flux(
    run_dir: Path,
    variable: str,
    pft_indices: tuple[int, ...],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    frames = []
    for path in _history_files(run_dir, variable, "pft"):
        with Dataset(path) as dataset:
            dates = _dates(dataset)
            values = _axis_values(dataset.variables[variable], "pft")
            active = np.asarray(dataset.variables["pfts1d_active"][:], dtype=int)
            topounit = np.asarray(dataset.variables["pfts1d_topounit"][:], dtype=int)
            pft_type = np.asarray(dataset.variables["pfts1d_itype_veg"][:], dtype=int)
            weights = np.asarray(dataset.variables["pfts1d_wtgcell"][:], dtype=float)
            bog = (active == 1) & np.isin(topounit, BOG_TOPOUNITS) & (weights > 0)
            selected = bog & np.isin(pft_type, pft_indices)
            bog_area = weights[bog].sum()
            if bog_area <= 0 or not selected.any():
                raise RuntimeError(f"Missing active bog PFTs {pft_indices} in {path}")
            flux = np.nansum(values[:, selected] * weights[selected], axis=1) / bog_area
            frames.append(pd.DataFrame({"date": dates, "model_flux_gC_m2_s": flux}))
    daily = pd.concat(frames, ignore_index=True).drop_duplicates("date")
    daily = daily[(daily["date"].dt.year >= start_year) & (daily["date"].dt.year <= end_year)]
    daily["year"] = daily["date"].dt.year
    daily["model_daily_gC_m2"] = daily["model_flux_gC_m2_s"] * SECONDS_PER_DAY
    return daily.groupby("year", as_index=False).agg(
        model_gC_m2_yr=("model_daily_gC_m2", "sum"),
        model_days=("date", "nunique"),
    )


def _annual_column_flux(
    run_dir: Path, variable: str, start_year: int, end_year: int
) -> pd.DataFrame:
    frames = []
    for path in _history_files(run_dir, variable, "column"):
        with Dataset(path) as dataset:
            dates = _dates(dataset)
            values = _axis_values(dataset.variables[variable], "column")
            active = np.asarray(dataset.variables["cols1d_active"][:], dtype=int)
            topounit = np.asarray(dataset.variables["cols1d_topounit"][:], dtype=int)
            weights = np.asarray(dataset.variables["cols1d_wtgcell"][:], dtype=float)
            bog = (active == 1) & np.isin(topounit, BOG_TOPOUNITS) & (weights > 0)
            bog_area = weights[bog].sum()
            if bog_area <= 0:
                raise RuntimeError(f"No active bog columns in {path}")
            flux = np.nansum(values[:, bog] * weights[bog], axis=1) / bog_area
            frames.append(pd.DataFrame({"date": dates, "model_flux_gC_m2_s": flux}))
    daily = pd.concat(frames, ignore_index=True).drop_duplicates("date")
    daily = daily[(daily["date"].dt.year >= start_year) & (daily["date"].dt.year <= end_year)]
    daily["year"] = daily["date"].dt.year
    daily["model_daily_gC_m2"] = daily["model_flux_gC_m2_s"] * SECONDS_PER_DAY
    return daily.groupby("year", as_index=False).agg(
        model_gC_m2_yr=("model_daily_gC_m2", "sum"),
        model_days=("date", "nunique"),
    )


def read_budget_observations(path: Path) -> pd.DataFrame:
    observations = pd.read_csv(path)
    observations["obs"] = pd.to_numeric(observations["obs"], errors="coerce")
    observations["obs_err"] = pd.to_numeric(observations["obs_err"], errors="coerce")
    observations["year"] = pd.to_numeric(observations["year"], errors="coerce")
    observations["temperature_offset_C"] = pd.to_numeric(
        observations["temperature_offset_C"], errors="coerce"
    )
    return observations


def compare_carbon_budget(
    run_root: Path,
    case_prefix: str,
    observation_file: Path,
    start_year: int = 2016,
    end_year: int = 2023,
) -> pd.DataFrame:
    """Build aligned annual model/observation records for all budget metrics."""
    runs = treatment_run_dirs(run_root, case_prefix)
    observations = read_budget_observations(observation_file)
    frames = []
    for metric, definition in METRICS.items():
        obs_metric = observations[
            observations["model_var"] == definition["observation_variable"]
        ].copy()
        for case in CASE_ORDER:
            if definition["model_dimension"] == "pft":
                model = _annual_pft_flux(
                    runs[case], definition["model_variable"],
                    definition["pft_indices"], start_year, end_year,
                )
            else:
                model = _annual_column_flux(
                    runs[case], definition["model_variable"], start_year, end_year
                )
            obs_case = obs_metric[obs_metric["treatment"] == case][
                ["year", "obs", "obs_err", "temperature_offset_C"]
            ]
            joined = model.merge(obs_case, on="year", how="outer")
            joined["metric"] = metric
            joined["metric_label"] = definition["label"]
            joined["case"] = case
            joined["case_label"] = CASE_LABELS[case]
            joined["warming_C"] = TREATMENT_WARMING_C[case]
            frames.append(joined)
    columns = [
        "metric", "metric_label", "case", "case_label", "warming_C", "year",
        "model_gC_m2_yr", "model_days", "obs", "obs_err",
    ]
    return pd.concat(frames, ignore_index=True)[columns].sort_values(
        ["metric", "case", "year"]
    )


def temperature_response(comparison: pd.DataFrame) -> pd.DataFrame:
    """Fit treatment-mean intercepts and slopes across T0 through T9."""
    rows = []
    for metric, definition in METRICS.items():
        metric_data = comparison[
            (comparison["metric"] == metric) & comparison["case"].isin(FIT_CASES)
        ]
        for source, column in [("ELM", "model_gC_m2_yr"), ("Observed", "obs")]:
            matched = metric_data.dropna(subset=[column, "obs"]).copy()
            treatment_means = matched.groupby(
                ["case", "warming_C"], as_index=False
            )[column].mean()
            if len(treatment_means) < 2:
                slope = intercept = r_squared = np.nan
            else:
                x = treatment_means["warming_C"].to_numpy(dtype=float)
                y = treatment_means[column].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, 1)
                fitted = intercept + slope * x
                residual = np.sum((y - fitted) ** 2)
                total = np.sum((y - y.mean()) ** 2)
                r_squared = 1.0 - residual / total if total > 0 else np.nan
            rows.append(
                {
                    "metric": metric,
                    "metric_label": definition["label"],
                    "source": source,
                    "intercept_gC_m2_yr": intercept,
                    "slope_gC_m2_yr_C": slope,
                    "r_squared": r_squared,
                    "n_treatments": len(treatment_means),
                    "fit_cases": ",".join(FIT_CASES),
                }
            )
    return pd.DataFrame(rows)


def carbon_timeseries_figure(comparison: pd.DataFrame, metric: str):
    definition = METRICS[metric]
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 9.5), sharex=True, sharey=True)
    for ax, case in zip(axes.ravel(), CASE_ORDER):
        data = comparison[
            (comparison["metric"] == metric) & (comparison["case"] == case)
        ].sort_values("year")
        ax.plot(
            data["year"], data["model_gC_m2_yr"], color="#2166ac",
            marker="o", markersize=3.5, linewidth=1.5, label="ELM",
        )
        ax.plot(
            data["year"], data["obs"], color="black",
            marker="s", markersize=3.5, linewidth=1.1, label="Observed",
        )
        ax.set_title(CASE_LABELS[case], loc="left", fontsize=11)
        ax.grid(True, color="0.88", linewidth=0.7)
    axes[0, 0].legend(frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("gC m$^{-2}$ yr$^{-1}$")
    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    fig.suptitle(f"SPRUCE {definition['label']}: ELM versus observations", y=0.985)
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.075, top=0.915,
        hspace=0.26, wspace=0.03,
    )
    return fig


def temperature_response_figure(response: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    labels = [METRICS[metric]["label"] for metric in METRICS]
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    for ax, column, ylabel, title in [
        (axes[0], "intercept_gC_m2_yr", "gC m$^{-2}$ yr$^{-1}$", "Intercept at 0 °C warming"),
        (axes[1], "slope_gC_m2_yr_C", "gC m$^{-2}$ yr$^{-1}$ °C$^{-1}$", "Slope across T0-T9"),
    ]:
        for offset, source, color in [(-width / 2, "Observed", "#4d4d4d"), (width / 2, "ELM", "#2166ac")]:
            values = []
            for metric in METRICS:
                row = response[(response["metric"] == metric) & (response["source"] == source)]
                values.append(float(row[column].iloc[0]))
            bars = ax.bar(x + offset, values, width, label=source, color=color)
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="0.88", linewidth=0.7)
    axes[0].legend(frameon=False)
    fig.suptitle("SPRUCE carbon-budget temperature responses (T0-T9)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def write_carbon_budget_products(
    comparison: pd.DataFrame,
    output_dir: Path,
    source: dict[str, object] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "annual": output_dir / "carbon_budget_model_vs_observations_annual.csv",
        "response": output_dir / "carbon_budget_temperature_response.csv",
        "pdf": output_dir / "carbon_budget_diagnostics.pdf",
        "response_plot": output_dir / "carbon_budget_temperature_response.png",
        "manifest": output_dir / "carbon_budget_manifest.json",
    }
    comparison.to_csv(paths["annual"], index=False)
    response = temperature_response(comparison)
    response.to_csv(paths["response"], index=False)
    with PdfPages(paths["pdf"]) as pdf:
        for metric in METRICS:
            fig = carbon_timeseries_figure(comparison, metric)
            path = output_dir / f"carbon_budget_{metric}_timeseries.png"
            fig.savefig(path, dpi=200)
            pdf.savefig(fig)
            plt.close(fig)
            paths[f"{metric}_plot"] = path
        fig = temperature_response_figure(response)
        fig.savefig(paths["response_plot"], dpi=200)
        pdf.savefig(fig)
        plt.close(fig)
    manifest = {
        "diagnostic": "SPRUCE treatment carbon budget",
        "bog_topounits": list(BOG_TOPOUNITS),
        "fit_cases": list(FIT_CASES),
        "model_area_basis": "hummock+hollow area; fen excluded",
        "metrics": METRICS,
        **(source or {}),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return paths

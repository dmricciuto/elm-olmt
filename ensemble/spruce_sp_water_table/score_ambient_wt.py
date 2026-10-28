"""Rank completed SPRUCE SP ambient members against hollow water tables.

The ranking separates the thawed-period level error from event-scale response.
Frozen-period error and 2022-2023 holdout error are reported, not optimized.
Incomplete members are skipped so this can be run while the ensemble progresses.
Run from the OLMT root with ``python -m ensemble.spruce_sp_water_table.score_ambient_wt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from olmt_diagnostics.spruce import (
    read_hollow_water_table,
    read_water_table_observations,
)


def rmse_cm(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    return float(100 * np.sqrt(np.mean(clean**2))) if len(clean) else float("nan")


def score_member(run_dir: Path, observations: pd.DataFrame) -> dict[str, float | int]:
    model = read_hollow_water_table(run_dir, "TAMB", 2015, 2023)
    if len(model) < 3200 or model["date"].min().year > 2015 or model["date"].max().year < 2023:
        raise ValueError("treatment history is incomplete")
    frame = model.merge(observations, on="date", how="inner").sort_values("date")
    frame["error_m"] = frame["model_wt_height_m"] - frame["observed_m"]
    thawed = ~frame["soilice_present"]
    train = frame["date"].dt.year.between(2015, 2020)
    holdout = frame["date"].dt.year.between(2022, 2023)
    summer_2021 = (frame["date"].dt.year == 2021) & frame["date"].dt.month.between(5, 10)

    model_delta = frame["model_wt_height_m"].diff(7)
    observed_delta = frame["observed_m"].diff(7)
    seven_days_apart = frame["date"].diff(7).dt.days == 7
    no_ice_window = ~frame["soilice_present"].rolling(8, min_periods=8).max().fillna(1).astype(bool)
    event_mask = train & seven_days_apart & no_ice_window

    training_rmse = rmse_cm(frame.loc[train & thawed, "error_m"])
    event_rmse = rmse_cm((model_delta - observed_delta).where(event_mask))
    model_min = frame.loc[summer_2021, "model_wt_height_m"].min()
    observed_min = frame.loc[summer_2021, "observed_m"].min()
    min_error_cm = float(100 * (model_min - observed_min))
    return {
        "training_thawed_days": int((train & thawed & frame["error_m"].notna()).sum()),
        "training_thawed_rmse_cm": training_rmse,
        "training_thawed_bias_cm": float(100 * frame.loc[train & thawed, "error_m"].mean()),
        "training_7day_change_pairs": int((event_mask & observed_delta.notna()).sum()),
        "training_7day_change_rmse_cm": event_rmse,
        "summer_2021_model_min_cm": float(100 * model_min),
        "summer_2021_observed_min_cm": float(100 * observed_min),
        "summer_2021_min_error_cm": min_error_cm,
        "holdout_2022_2023_thawed_rmse_cm": rmse_cm(frame.loc[holdout & thawed, "error_m"]),
        "all_years_frozen_rmse_cm": rmse_cm(frame.loc[~thawed, "error_m"]),
        "score_cm": training_rmse + 0.75 * event_rmse + 0.25 * abs(min_error_cm),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-case", required=True, help="Full OLMT base case name, including site/compset")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--parm-list", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = [line.split()[0] for line in args.parm_list.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    first_sample = next(line for line in args.samples.read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#"))
    samples = np.loadtxt(args.samples, delimiter="," if "," in first_sample else None)
    samples = np.atleast_2d(samples)
    if samples.shape[1] != len(names):
        raise ValueError("sample columns and parameter names do not match")
    observations = read_water_table_observations(args.observations, 2015, 2023)
    observations = observations[["date", "TAMB"]].rename(columns={"TAMB": "observed_m"})

    rows = []
    for member, values in enumerate(samples, start=1):
        run_dir = args.run_root / f"{args.base_case}_TAMB" / f"g{member:05d}"
        if not run_dir.exists():
            continue
        try:
            metrics = score_member(run_dir, observations)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"Skipping member {member}: {exc}")
            continue
        rows.append({"member": member, **metrics, **dict(zip(names, values))})
    if not rows:
        raise RuntimeError("No completed members with usable daily water tables")
    result = pd.DataFrame(rows).sort_values("score_cm")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} of {len(samples)} members; ranked results: {args.output}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

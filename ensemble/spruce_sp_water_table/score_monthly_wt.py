"""Score the SPRUCE SP ambient ensemble on matched calendar-month water tables.

Monthly model and observed means use the same observed days, include days with
soil ice, and require at least 15 paired days per month by default. The
2015--2020 score is a training diagnostic; 2021--2023 is held out. A full-period
score is also written for users who want to fit all available observations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ensemble.spruce_sp_water_table.score_ambient_wt import rmse_cm
from olmt_diagnostics.spruce import (
    read_hollow_water_table,
    read_water_table_observations,
)


def monthly_pairs(frame: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """Aggregate only paired daily samples; do not fill observation gaps."""
    paired = frame.dropna(subset=["model_wt_height_m", "observed_m"]).copy()
    paired["month"] = paired["date"].dt.to_period("M")
    monthly = paired.groupby("month", observed=True).agg(
        model_m=("model_wt_height_m", "mean"),
        observed_m=("observed_m", "mean"),
        paired_days=("date", "size"),
        ice_days=("soilice_present", "sum"),
    )
    monthly = monthly.loc[monthly["paired_days"] >= min_days].reset_index()
    monthly["year"] = monthly["month"].dt.year
    monthly["error_m"] = monthly["model_m"] - monthly["observed_m"]
    return monthly


def member_metrics(frame: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    train = monthly["year"].between(2015, 2020)
    holdout = monthly["year"].between(2021, 2023)
    daily_train = frame["date"].dt.year.between(2015, 2020)
    thawed = ~frame["soilice_present"]
    daily_error = frame["model_wt_height_m"] - frame["observed_m"]

    def bias_cm(values: pd.Series) -> float:
        return float(100 * values.mean()) if values.notna().any() else np.nan

    return {
        "train_months": int(train.sum()),
        "holdout_months": int(holdout.sum()),
        "all_months": len(monthly),
        "ice_affected_months": int((monthly["ice_days"] > 0).sum()),
        "train_monthly_rmse_cm": rmse_cm(monthly.loc[train, "error_m"]),
        "train_monthly_bias_cm": bias_cm(monthly.loc[train, "error_m"]),
        "holdout_monthly_rmse_cm": rmse_cm(monthly.loc[holdout, "error_m"]),
        "holdout_monthly_bias_cm": bias_cm(monthly.loc[holdout, "error_m"]),
        "all_monthly_rmse_cm": rmse_cm(monthly["error_m"]),
        "all_monthly_bias_cm": bias_cm(monthly["error_m"]),
        "ice_affected_monthly_rmse_cm": rmse_cm(
            monthly.loc[monthly["ice_days"] > 0, "error_m"]
        ),
        "ice_free_monthly_rmse_cm": rmse_cm(
            monthly.loc[monthly["ice_days"] == 0, "error_m"]
        ),
        "train_thawed_daily_rmse_cm": rmse_cm(
            daily_error.where(daily_train & thawed)
        ),
    }


def daily_response_metrics(frame: pd.DataFrame) -> dict:
    """Daily level and seven-day response errors for 2019--2023, ice included."""
    paired = frame.dropna(subset=["model_wt_height_m", "observed_m"]).copy()
    paired = paired.loc[paired["date"].dt.year.between(2019, 2023)]
    paired = paired.sort_values("date")
    error = paired["model_wt_height_m"] - paired["observed_m"]
    train = paired["date"].dt.year.between(2019, 2021)
    holdout = paired["date"].dt.year.between(2022, 2023)
    seven_days = paired["date"].diff(7).dt.days == 7
    observed_change = paired["observed_m"].diff(7).where(seven_days)
    modeled_change = paired["model_wt_height_m"].diff(7).where(seven_days)
    change_error = modeled_change - observed_change
    # A 1 cm/week observed change avoids labeling tiny fluctuations as events.
    recharge = observed_change >= 0.01
    drawdown = observed_change <= -0.01
    growing_season = paired["date"].dt.month.between(5, 10)
    summer_recharge = recharge & growing_season
    summer_drawdown = drawdown & growing_season

    def gain(mask: pd.Series) -> float:
        observed = observed_change[mask].to_numpy(dtype=float)
        modeled = modeled_change[mask].to_numpy(dtype=float)
        denominator = np.sum(observed**2)
        return float(np.sum(observed * modeled) / denominator) if denominator else np.nan

    return {
        "daily_pairs_2019_2023": len(paired),
        "daily_rmse_cm_2019_2023": rmse_cm(error),
        "daily_bias_cm_2019_2023": float(100 * error.mean()),
        "daily_rmse_cm_2019_2021": rmse_cm(error.where(train)),
        "daily_rmse_cm_2022_2023": rmse_cm(error.where(holdout)),
        "daily_ice_rmse_cm_2019_2023": rmse_cm(error.where(paired["soilice_present"])),
        "daily_thawed_rmse_cm_2019_2023": rmse_cm(error.where(~paired["soilice_present"])),
        "change_pairs_7day": int(seven_days.sum()),
        "change_rmse_cm_7day": rmse_cm(change_error),
        "recharge_pairs_7day": int(recharge.sum()),
        "recharge_rmse_cm_7day": rmse_cm(change_error.where(recharge)),
        "recharge_gain_7day": gain(recharge),
        "drawdown_pairs_7day": int(drawdown.sum()),
        "drawdown_rmse_cm_7day": rmse_cm(change_error.where(drawdown)),
        "drawdown_gain_7day": gain(drawdown),
        "growing_recharge_pairs_7day": int(summer_recharge.sum()),
        "growing_recharge_rmse_cm_7day": rmse_cm(change_error.where(summer_recharge)),
        "growing_recharge_gain_7day": gain(summer_recharge),
        "growing_drawdown_pairs_7day": int(summer_drawdown.sum()),
        "growing_drawdown_rmse_cm_7day": rmse_cm(change_error.where(summer_drawdown)),
        "growing_drawdown_gain_7day": gain(summer_drawdown),
    }


def score_one(task: tuple) -> tuple[int, dict | None, pd.DataFrame | None, str | None]:
    member, values, names, run_root, base_case, observations, min_days = task
    run_dir = run_root / f"{base_case}_TAMB" / f"g{member:05d}"
    try:
        model = read_hollow_water_table(run_dir, "TAMB", 2015, 2023)
        if (len(model) < 3200 or model["date"].min().year > 2015 or
                model["date"].max().year < 2023):
            raise ValueError("treatment history is incomplete")
        frame = model.merge(observations, on="date", how="inner")
        monthly = monthly_pairs(frame, min_days)
        if not monthly["year"].between(2015, 2020).any() or not monthly[
                "year"].between(2021, 2023).any():
            raise ValueError("missing training or holdout months")
        row = {"member": member, **member_metrics(frame, monthly),
               **daily_response_metrics(frame),
               **dict(zip(names, values))}
        return member, row, monthly, None
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        return member, None, None, str(exc)


def plot_comparison(
    scores: pd.DataFrame, monthly_by_member: dict[int, pd.DataFrame], output: Path
) -> None:
    baseline = 1
    best_train = int(scores.sort_values("train_monthly_rmse_cm").iloc[0]["member"])
    best_all = int(scores.sort_values("all_monthly_rmse_cm").iloc[0]["member"])
    selected = [(baseline, "Baseline", "0.55"),
                (best_train, "Best 2015–20", "#2166ac"),
                (best_all, "Best 2015–23", "#b2182b")]
    fig, (ax, scatter) = plt.subplots(2, 1, figsize=(12, 8),
                                     gridspec_kw={"height_ratios": [2, 1]})
    # Reindex the displayed series so missing observed years remain gaps;
    # plotting only the available months would imply a false 2015--2018 trend.
    full_months = pd.period_range("2015-01", "2023-12", freq="M")
    dates = full_months.to_timestamp()
    obs = monthly_by_member[baseline].set_index("month").reindex(full_months)
    ax.plot(dates, obs["observed_m"] * 100,
            color="black", lw=1.8, label="Observed")
    shown = set()
    for member, label, color in selected:
        if member in shown:
            continue
        shown.add(member)
        series = monthly_by_member[member].set_index("month").reindex(full_months)
        ax.plot(dates, series["model_m"] * 100,
                color=color, lw=1.3, label=f"{label} (#{member})")
    ax.set_ylabel("Water-table height above hollow (cm)")
    ax.set_title("SPRUCE ambient: matched monthly means, ice periods included")
    ax.axhline(0, color="0.7", lw=0.7)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)

    scatter.scatter(scores["train_monthly_rmse_cm"],
                    scores["holdout_monthly_rmse_cm"], s=9, c="0.55", alpha=0.5)
    for member, label, color in selected:
        row = scores.loc[scores["member"] == member].iloc[0]
        scatter.scatter(row["train_monthly_rmse_cm"],
                        row["holdout_monthly_rmse_cm"], s=75, color=color,
                        edgecolor="black", linewidth=0.5, zorder=4)
        scatter.annotate(f"#{member}",
                         (row["train_monthly_rmse_cm"],
                          row["holdout_monthly_rmse_cm"]),
                         xytext=(4, 3), textcoords="offset points", fontsize=8)
    scatter.set_xlabel("2015–2020 monthly RMSE (cm)")
    scatter.set_ylabel("2021–2023 monthly RMSE (cm)")
    scatter.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-case", required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--parm-list", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-days", type=int, default=15)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.min_days < 1 or args.workers < 1:
        parser.error("--min-days and --workers must be positive")

    names = [line.split()[0] for line in args.parm_list.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    first = next(line for line in args.samples.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#"))
    samples = np.atleast_2d(np.loadtxt(
        args.samples, delimiter="," if "," in first else None
    ))
    if samples.shape[1] != len(names):
        raise ValueError("sample columns and parameter names do not match")

    observations = read_water_table_observations(args.observations, 2015, 2023)
    observations = observations[["date", "TAMB"]].rename(
        columns={"TAMB": "observed_m"}
    )
    rows = []
    monthly_by_member = {}
    tasks = ((member, values, names, args.run_root, args.base_case,
              observations, args.min_days)
             for member, values in enumerate(samples, start=1))
    if args.workers == 1:
        results = map(score_one, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(score_one, tasks, chunksize=1)
    for member, row, monthly, error in results:
        if error is not None:
            print(f"Skipping member {member}: {error}", flush=True)
        else:
            rows.append(row)
            monthly_by_member[member] = monthly
        if member % 50 == 0:
            print(f"Processed {member}/{len(samples)}", flush=True)
    if args.workers > 1:
        pool.shutdown()
    if not rows:
        raise RuntimeError("No completed members with usable daily water tables")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(rows).sort_values("train_monthly_rmse_cm")
    scores.to_csv(args.output_dir / "monthly_member_scores.csv", index=False)
    all_monthly = pd.concat(
        [monthly_by_member[m].assign(member=m) for m in sorted(monthly_by_member)],
        ignore_index=True,
    )
    all_monthly.to_csv(args.output_dir / "monthly_all_member_series.csv", index=False)
    selected = set(scores.head(10)["member"].astype(int))
    selected.update(scores.nsmallest(10, "all_monthly_rmse_cm")["member"].astype(int))
    selected.add(1)
    pd.concat([monthly_by_member[m].assign(member=m) for m in sorted(selected)],
              ignore_index=True).to_csv(
                  args.output_dir / "monthly_selected_series.csv", index=False)
    # The first two design rows are deliberate anchors, not Sobol draws.
    sampled = scores[scores["member"] > 2]
    correlation = sampled[names].corrwith(
        sampled["train_monthly_rmse_cm"], method="spearman"
    ).sort_values(key=abs, ascending=False)
    correlation.rename("spearman_with_train_rmse").to_csv(
        args.output_dir / "parameter_rank_correlations.csv")
    plot_comparison(scores, monthly_by_member,
                    args.output_dir / "monthly_wt_comparison.png")
    print(f"Scored {len(scores)} of {len(samples)} members", flush=True)
    print(scores.head(12).to_string(index=False), flush=True)
    print("Parameter rank correlations with training RMSE:", flush=True)
    print(correlation.to_string(), flush=True)


if __name__ == "__main__":
    main()

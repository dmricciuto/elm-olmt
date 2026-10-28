"""Calibrate SPRUCE hollow water table from the completed SP ensemble.

The existing OLMT time-series surrogate is trained on nonoverlapping seven-day
means. An outer member holdout keeps its SVD basis and neural networks entirely
separate from surrogate validation. MCMC uses an AR(1) residual likelihood, so
adjacent weekly errors are not treated as independent observations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olmt_diagnostics.spruce import (  # noqa: E402
    read_hollow_water_table,
    read_water_table_observations,
)

HERE = Path(__file__).resolve().parent
PARAMETER_NAMES = (
    "peat_jules_sucsat_scale",
    "peat_jules_b_scale",
    "peat_hydraulic_hksat_scale",
    "rsub_top_globalmax_peat",
    "peat_drainage_decay_depth",
    "soil_ice_impedance_exponent",
    "moss_water_hydraulic_conductivity_sat",
    "moss_water_layer_thickness",
    "moss_water_substrate_rewetting_timescale_days",
)
TARGET = "ZWT_WEEKLY_HEIGHT"


def noleap_index(date: pd.Series) -> np.ndarray:
    year = date.dt.year.to_numpy()
    day = date.dt.dayofyear.to_numpy() - 1
    after_february = (date.dt.month.to_numpy() > 2) & date.dt.is_leap_year.to_numpy()
    return (year - 2019) * 365 + day - after_february.astype(int)


def observation_weeks(path: Path) -> pd.DataFrame:
    obs = read_water_table_observations(path, 2019, 2023)
    obs = obs[["date", "TAMB"]].rename(columns={"TAMB": "observed_m"})
    obs["week"] = noleap_index(obs["date"]) // 7
    valid = obs.dropna(subset=["observed_m"])
    weeks = valid.groupby("week").agg(
        observed_m=("observed_m", "mean"),
        n_observed=("observed_m", "size"),
        first_date=("date", "min"),
    )
    weeks = weeks.loc[weeks["n_observed"] >= 5].reset_index()
    return weeks


def ice_free_week_mask(weeks: pd.DataFrame, daily: pd.DataFrame) -> np.ndarray:
    """Retain only complete seven-day weeks without hollow-column soil ice.

    The surrogate has one output per fixed calendar week, so the ice mask must
    also be fixed across candidate parameter sets. ``daily`` is a reference
    member's hollow history, not a candidate-dependent MCMC prediction.
    """
    daily = daily[["date", "soilice_present"]].copy()
    daily["week"] = noleap_index(daily["date"]) // 7
    status = daily.groupby("week").agg(
        n_days=("soilice_present", "size"),
        n_ice_days=("soilice_present", "sum"),
    )
    aligned = status.reindex(weeks["week"].to_numpy())
    return ((aligned["n_days"] == 7) & (aligned["n_ice_days"] == 0)).to_numpy()


def extract_member(task: tuple[int, Path, Path, np.ndarray]) -> tuple[int, np.ndarray]:
    member, root, obs_path, retained_weeks = task
    observed = read_water_table_observations(obs_path, 2019, 2023)
    observed = observed[["date", "TAMB"]].rename(columns={"TAMB": "observed_m"})
    observed["week"] = noleap_index(observed["date"]) // 7
    model = read_hollow_water_table(root / f"g{member:05d}", "TAMB", 2019, 2023)
    matched = observed.merge(model[["date", "model_wt_height_m"]], on="date", how="left")
    matched = matched.dropna(subset=["observed_m", "model_wt_height_m"])
    weekly = matched.groupby("week")["model_wt_height_m"].mean()
    values = weekly.reindex(retained_weeks).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"member {member} has missing matched weekly model values")
    return member, values


def extract(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    weeks = observation_weeks(args.observations)
    samples = np.atleast_2d(np.loadtxt(args.samples, delimiter=","))
    if samples.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(f"expected {len(PARAMETER_NAMES)} parameters, got {samples.shape}")
    tasks = (
        (member, args.ensemble_root, args.observations, weeks["week"].to_numpy())
        for member in range(1, len(samples) + 1)
    )
    matrix = np.empty((len(samples), len(weeks)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, (member, values) in enumerate(pool.map(extract_member, tasks), 1):
            matrix[member - 1] = values
            if count % 64 == 0:
                print(f"Extracted {count}/{len(samples)} members", flush=True)
    weeks.to_csv(args.output / "weekly_observations.csv", index=False)
    np.save(args.output / "weekly_model_m.npy", matrix)
    np.save(args.output / "samples_physical.npy", samples)
    print(f"Wrote {matrix.shape} weekly model matrix", flush=True)


def transformed_samples(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical = np.load(args.output / "samples_physical.npy")
    metadata = json.loads(args.metadata.read_text())
    if tuple(metadata["parameter_order"]) != PARAMETER_NAMES:
        raise ValueError("sample metadata parameter order differs from expected order")
    bounds = [line.split() for line in args.parm_list.read_text().splitlines()
              if line.strip() and not line.startswith("#")]
    if tuple(row[0] for row in bounds) != PARAMETER_NAMES:
        raise ValueError("parameter bounds order differs from samples")
    x = physical.copy()
    pmin = np.array([float(row[2]) for row in bounds])
    pmax = np.array([float(row[3]) for row in bounds])
    for col, name in enumerate(PARAMETER_NAMES):
        if metadata["transforms"][name] == "log":
            x[:, col] = np.log(x[:, col])
            pmin[col] = np.log(pmin[col])
            pmax[col] = np.log(pmax[col])
    return x, pmin, pmax


def load_case(args: argparse.Namespace):
    # This pickle is generated by this same OLMT checkout during case setup.
    import model_ELM  # noqa: F401

    with args.case_pickle.open("rb") as handle:
        case = pickle.load(handle)
    case.OLMTdir = str(REPO)
    case.UQ_output = str(args.output)
    return case


def fit(args: argparse.Namespace) -> None:
    y = np.load(args.output / "weekly_model_m.npy")
    x, pmin, pmax = transformed_samples(args)
    indices = np.arange(len(x))
    train, heldout = train_test_split(indices, test_size=0.2, random_state=42)
    case = load_case(args)
    case.samples = x[train].T.copy()
    case.nsamples = len(train)
    case.nparms_ensemble = x.shape[1]
    case.ensemble_parms = list(PARAMETER_NAMES)
    case.ensemble_pmin = pmin
    case.ensemble_pmax = pmax
    case.output = {TARGET: y[train].T.copy()}
    case.surrogate = {}
    case.pscaler = {}
    case.yscaler = {}
    case.svd_components = {}
    case.use_svd = {}
    case.qoi_bad = {}
    case.qoi_bad_meanval = {}
    case.surrogate_exclude_zeros = False
    trained = case.train_surrogate([TARGET])
    if trained != [TARGET]:
        raise RuntimeError(f"OLMT could not train weekly water-table surrogate: {trained}")
    predicted = case.run_surrogate(x[heldout], [TARGET])[TARGET]
    if predicted.shape != y[heldout].shape:
        raise ValueError(f"surrogate output shape {predicted.shape} != {y[heldout].shape}")
    error_cm = 100 * (predicted - y[heldout])
    member_rmse = np.sqrt(np.mean(error_cm**2, axis=1))
    weekly_rmse = np.sqrt(np.mean(error_cm**2, axis=0))
    validation = {
        "n_train_members": len(train),
        "n_heldout_members": len(heldout),
        "n_weeks": y.shape[1],
        "overall_rmse_cm": float(np.sqrt(np.mean(error_cm**2))),
        "median_member_rmse_cm": float(np.median(member_rmse)),
        "p90_member_rmse_cm": float(np.percentile(member_rmse, 90)),
        "median_week_rmse_cm": float(np.median(weekly_rmse)),
        "max_week_rmse_cm": float(np.max(weekly_rmse)),
        "members": [{"member": int(i + 1), "rmse_cm": float(r)}
                    for i, r in zip(heldout, member_rmse)],
    }
    (args.output / "surrogate_validation.json").write_text(json.dumps(validation, indent=2))
    with (args.output / "weekly_surrogate.pkl").open("wb") as handle:
        pickle.dump(case, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(args.output / "member_split.npz", train=train, heldout=heldout)
    print(json.dumps({k: v for k, v in validation.items() if k != "members"}, indent=2),
          flush=True)


def ar1_log_likelihood(residual: np.ndarray, gaps: np.ndarray,
                       phi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """AR(1) Gaussian likelihood with gaps in observed weekly windows."""
    phi = phi[:, None]
    sigma = sigma[:, None]
    marginal_var = sigma**2 / (1 - phi**2)
    result = -0.5 * (np.log(2 * np.pi * marginal_var[:, 0])
                     + residual[:, 0] ** 2 / marginal_var[:, 0])
    powers = phi ** gaps[None, :]
    conditional_var = marginal_var * (1 - powers**2)
    innovation = residual[:, 1:] - powers * residual[:, :-1]
    result += -0.5 * np.sum(np.log(2 * np.pi * conditional_var)
                             + innovation**2 / conditional_var, axis=1)
    return result


def mcmc(args: argparse.Namespace) -> None:
    import emcee

    validation = json.loads((args.output / "surrogate_validation.json").read_text())
    if validation["median_member_rmse_cm"] > args.max_validation_rmse_cm:
        raise ValueError("Held-out surrogate RMSE exceeds threshold; do not calibrate")
    with (args.output / "weekly_surrogate.pkl").open("rb") as handle:
        case = pickle.load(handle)
    weeks = pd.read_csv(args.output / "weekly_observations.csv", parse_dates=["first_date"])
    x, pmin, pmax = transformed_samples(args)
    model = np.load(args.output / "weekly_model_m.npy")
    if args.ice_free_reference_run is None:
        ice_free = np.ones(len(weeks), dtype=bool)
    else:
        reference = read_hollow_water_table(
            args.ice_free_reference_run, "TAMB", 2019, 2023,
            soilice_threshold_kg_m2=args.ice_threshold_kg_m2,
        )
        ice_free = ice_free_week_mask(weeks, reference)
    train = weeks["first_date"].dt.year.between(2019, 2021).to_numpy() & ice_free
    test = weeks["first_date"].dt.year.between(2022, 2023).to_numpy() & ice_free
    min_train, min_test = (40, 30) if args.ice_free_reference_run else (100, 80)
    if train.sum() < min_train or test.sum() < min_test:
        raise ValueError("insufficient calibration or temporal holdout weeks")
    yobs = weeks.loc[train, "observed_m"].to_numpy()
    gaps = np.diff(weeks.loc[train, "week"].to_numpy())
    prior_lo = np.r_[pmin, 0.0, np.log(0.002)]
    prior_hi = np.r_[pmax, 0.99, np.log(0.20)]

    def log_probability(theta: np.ndarray) -> np.ndarray:
        valid = np.all((theta >= prior_lo) & (theta <= prior_hi), axis=1)
        result = np.full(len(theta), -np.inf)
        if not valid.any():
            return result
        selected = theta[valid]
        prediction = case.run_surrogate(selected[:, :len(PARAMETER_NAMES)], [TARGET])[TARGET]
        residual = prediction[:, train] - yobs
        result[valid] = ar1_log_likelihood(
            residual, gaps, selected[:, -2], np.exp(selected[:, -1]))
        return result

    # Initialize around well-scoring *real* ensemble members, not an
    # unsupported surrogate optimum. The walkers still explore the full prior.
    level_rmse = np.sqrt(np.mean((model[:, train]
                                  - weeks.loc[train, "observed_m"].to_numpy())**2, axis=1))
    candidates = np.argsort(level_rmse)[:max(80, args.walkers)]
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(candidates, size=args.walkers, replace=False)
    starts = x[picked].copy()
    starts += rng.normal(0, 0.01, starts.shape) * (pmax - pmin)
    starts = np.clip(starts, pmin + 1e-9, pmax - 1e-9)
    initial = np.column_stack((starts,
                               rng.uniform(0.55, 0.9, args.walkers),
                               rng.uniform(np.log(0.025), np.log(0.09), args.walkers)))
    # emcee uses NumPy's legacy global RNG unless an explicit sampler state is
    # supplied; seed it as well as the Generator used for walker initialization.
    np.random.seed(args.seed)
    sampler = emcee.EnsembleSampler(args.walkers, len(prior_lo), log_probability,
                                    vectorize=True)
    sampler.run_mcmc(initial, args.steps, progress=True)
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    np.savez_compressed(args.output / "mcmc_weekly_ar1.npz", chain=chain,
                        log_prob=log_prob, acceptance_fraction=sampler.acceptance_fraction)
    burn = args.steps // 2
    flat = chain[burn::max(1, args.thin)].reshape(-1, chain.shape[-1])
    flat_prob = log_prob[burn::max(1, args.thin)].reshape(-1)
    best = flat[np.argmax(flat_prob)]
    predicted = case.run_surrogate(best[:len(PARAMETER_NAMES)][None, :], [TARGET])[TARGET][0]
    if args.ice_free_reference_run:
        best_real_member = int(np.argmin(np.mean(
            (model[:, train] - weeks.loc[train, "observed_m"].to_numpy())**2, axis=1)))
    else:
        # Preserve the original all-week diagnostic for existing workflows.
        best_real_member = int(np.argmin(np.mean(
            (model - weeks["observed_m"].to_numpy())**2, axis=1)))
    baseline_holdout_rmse = float(100 * np.sqrt(np.mean(
        (model[0, test] - weeks.loc[test, "observed_m"].to_numpy())**2)))
    best_real_holdout_rmse = float(100 * np.sqrt(np.mean(
        (model[best_real_member, test]
         - weeks.loc[test, "observed_m"].to_numpy())**2)))
    try:
        autocorrelation_time = sampler.get_autocorr_time(quiet=True)
        max_autocorrelation_time = float(np.max(autocorrelation_time))
    except (ValueError, FloatingPointError):
        max_autocorrelation_time = float("nan")
    train_rmse = float(100 * np.sqrt(np.mean(
        (predicted[train] - weeks.loc[train, "observed_m"].to_numpy())**2)))
    holdout_rmse = float(100 * np.sqrt(np.mean(
        (predicted[test] - weeks.loc[test, "observed_m"].to_numpy())**2)))
    summary = {
        "provisional": True,
        "steps": args.steps,
        "walkers": args.walkers,
        "acceptance_mean": float(np.mean(sampler.acceptance_fraction)),
        "max_autocorrelation_steps": max_autocorrelation_time,
        "steps_per_max_autocorrelation": float(args.steps / max_autocorrelation_time),
        "convergence_screen_pass": bool(args.steps >= 50 * max_autocorrelation_time),
        "train_weeks": int(train.sum()),
        "holdout_weeks": int(test.sum()),
        "ice_free_only": args.ice_free_reference_run is not None,
        "ice_free_reference_run": (str(args.ice_free_reference_run)
                                   if args.ice_free_reference_run else None),
        "ice_threshold_kg_m2": (args.ice_threshold_kg_m2
                                 if args.ice_free_reference_run else None),
        "ice_free_week_rule": ("all seven days ice-free"
                               if args.ice_free_reference_run else None),
        "train_rmse_cm": train_rmse,
        "holdout_rmse_cm": holdout_rmse,
        "baseline_member_holdout_rmse_cm": baseline_holdout_rmse,
        "temporal_holdout_screen_pass": bool(holdout_rmse <= baseline_holdout_rmse),
        "best_transformed": dict(zip(PARAMETER_NAMES, map(float, best[:len(PARAMETER_NAMES)]))),
        "phi": float(best[-2]),
        "innovation_sigma_cm": float(100 * np.exp(best[-1])),
        "best_train_level_members": [int(i + 1) for i in np.argsort(level_rmse)[:10]],
    }
    if args.ice_free_reference_run:
        summary["best_calibration_real_member"] = best_real_member + 1
        summary["best_calibration_real_member_holdout_rmse_cm"] = best_real_holdout_rmse
    else:
        summary["best_full_period_real_member"] = best_real_member + 1
        summary["best_full_period_real_member_holdout_rmse_cm"] = best_real_holdout_rmse
    metadata = json.loads(args.metadata.read_text())
    physical = best[:len(PARAMETER_NAMES)].copy()
    for col, name in enumerate(PARAMETER_NAMES):
        if metadata["transforms"][name] == "log":
            physical[col] = np.exp(physical[col])
    summary["best_physical"] = dict(zip(PARAMETER_NAMES, map(float, physical)))
    (args.output / "mcmc_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"week": weeks["week"], "first_date": weeks["first_date"],
                  "observed_m": weeks["observed_m"],
                  "posterior_best_m": predicted,
                  "ice_free_week": ice_free,
                  "used_for_calibration": train,
                  "used_for_holdout": test}).to_csv(
                      args.output / "mcmc_best_weekly_series.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "fit", "mcmc"))
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=Path, default=REPO / "data/spruce/spruce079/spruce_daily_water_table_processed_20250408.nc")
    parser.add_argument("--samples", type=Path, default=HERE / "samples_SPRUCE_SP_ambient_WT_sobol512_20261028.txt")
    parser.add_argument("--metadata", type=Path, default=HERE / "metadata_SPRUCE_SP_ambient_WT_sobol512_20261028.json")
    parser.add_argument("--parm-list", type=Path, default=HERE / "parm_list_SPRUCE_SP_ambient_WT_20261028.txt")
    parser.add_argument("--case-pickle", type=Path, default=REPO / "pklfiles/SPRSPWT512_20261028_US-SPR_ICBELMBC_TAMB.pkl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-validation-rmse-cm", type=float, default=3.0)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ice-free-reference-run", type=Path,
                        help="Reference TAMB member directory defining fixed ice-free weeks")
    parser.add_argument("--ice-threshold-kg-m2", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    {"extract": extract, "fit": fit, "mcmc": mcmc}[options.stage](options)

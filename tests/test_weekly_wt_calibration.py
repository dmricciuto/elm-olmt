"""Focused tests for the SPRUCE weekly water-table calibration adapter."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ensemble.spruce_sp_water_table.calibrate_weekly_wt import (
    ar1_log_likelihood,
    ice_free_week_mask,
    noleap_index,
    observation_weeks,
    transformed_samples,
)


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "ensemble/spruce_sp_water_table"


def test_noleap_index_skips_february_29() -> None:
    dates = pd.Series(pd.to_datetime(["2020-02-28", "2020-03-01", "2021-01-01"]))
    np.testing.assert_array_equal(np.diff(noleap_index(dates)), [1, 306])


def test_observation_weeks_are_dense_enough() -> None:
    path = ROOT / "data/spruce/spruce079/spruce_daily_water_table_processed_20250408.nc"
    if not path.exists():
        pytest.skip("SPRUCE observed water-table data are not installed")
    weeks = observation_weeks(path)
    assert len(weeks) > 240
    assert weeks["n_observed"].min() >= 5
    assert weeks["week"].is_unique


def test_sample_transform_matches_metadata(tmp_path: Path) -> None:
    physical = np.atleast_2d(np.loadtxt(
        HERE / "samples_SPRUCE_SP_ambient_WT_sobol512_20261028.txt",
        delimiter=",",
    ))
    np.save(tmp_path / "samples_physical.npy", physical)
    args = SimpleNamespace(
        output=tmp_path,
        metadata=HERE / "metadata_SPRUCE_SP_ambient_WT_sobol512_20261028.json",
        parm_list=HERE / "parm_list_SPRUCE_SP_ambient_WT_20261028.txt",
    )
    transformed, low, high = transformed_samples(args)
    np.testing.assert_allclose(transformed[:, 3], np.log(physical[:, 3]))
    np.testing.assert_allclose(transformed[:, 1], physical[:, 1])
    assert np.all((transformed >= low) & (transformed <= high))


def test_ar1_likelihood_rewards_smaller_residuals() -> None:
    residual = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    scores = ar1_log_likelihood(
        residual, np.array([1, 2]), np.array([0.5, 0.5]), np.array([0.03, 0.03])
    )
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]


def test_ice_free_week_mask_requires_seven_unfrozen_days() -> None:
    dates = pd.date_range("2019-01-01", periods=21, freq="D")
    daily = pd.DataFrame({"date": dates, "soilice_present": False})
    daily.loc[9, "soilice_present"] = True
    daily = daily.drop(index=20)  # An incomplete but unfrozen week is excluded.
    weeks = pd.DataFrame({"week": [0, 1, 2]})
    np.testing.assert_array_equal(
        ice_free_week_mask(weeks, daily), [True, False, False]
    )

import tempfile
import unittest
from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from olmt_diagnostics.spruce import read_hollow_water_table, water_table_metrics
from tools.compare_spruce_spinup_water_table import forcing_alignment


class SpruceDiagnosticsTest(unittest.TestCase):
    def make_history(self, root, include_h2osfc=True):
        path = Path(root) / "case.elm.h7.2016-01-01-00000.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", 2)
            dataset.createDimension("column", 3)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = "days since 2016-01-01 00:00:00"
            time[:] = [0.5, 1.5]
            for name, values in {
                "cols1d_topounit": [1, 2, 2],
                "cols1d_active": [1, 1, 1],
            }.items():
                variable = dataset.createVariable(name, "i4", ("column",))
                variable[:] = values
            for name, values in {
                "cols1d_wtgcell": [0.2, 0.4, 0.4],
                "cols1d_wttopounit": [1.0, 0.25, 0.75],
            }.items():
                variable = dataset.createVariable(name, "f8", ("column",))
                variable[:] = values
            zwt = dataset.createVariable("ZWT", "f8", ("time", "column"))
            zwt[:] = [[9.0, 0.2, 0.4], [9.0, 0.1, 0.3]]
            dataset.createDimension("levgrnd", 2)
            soilice = dataset.createVariable(
                "SOILICE", "f8", ("time", "levgrnd", "column")
            )
            soilice[:] = [
                [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
            if include_h2osfc:
                h2osfc = dataset.createVariable("H2OSFC", "f8", ("time", "column"))
                h2osfc[:] = [[0.0, 10.0, 30.0], [0.0, 20.0, 40.0]]
        return path

    def test_height_uses_weighted_hollow_zwt_and_surface_water(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_history(root)
            result = read_hollow_water_table(Path(root), "TAMB")
        np.testing.assert_allclose(result["model_zwt_m"], [0.35, 0.25])
        np.testing.assert_allclose(result["model_h2osfc_mm"], [25.0, 35.0])
        np.testing.assert_allclose(result["model_wt_height_m"], [-0.325, -0.215])
        np.testing.assert_allclose(result["soilice_total_kg_m2"], [0.225, 0.0])
        self.assertEqual(result["soilice_present"].tolist(), [True, False])
        self.assertTrue(result["h2osfc_available"].all())

    def test_missing_h2osfc_requires_explicit_legacy_policy(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_history(root, include_h2osfc=False)
            with self.assertRaisesRegex(RuntimeError, "no column-resolved H2OSFC"):
                read_hollow_water_table(Path(root), "TAMB")
            result = read_hollow_water_table(
                Path(root), "TAMB", missing_h2osfc="zero"
            )
        np.testing.assert_allclose(result["model_wt_height_m"], [-0.35, -0.25])
        self.assertFalse(result["h2osfc_available"].any())

    def test_metrics_use_only_non_ice_days(self):
        import pandas as pd

        frames = []
        for case in ("TAMB", "T0.00", "T2.25", "T4.50", "T6.75", "T9.00"):
            frames.append(
                pd.DataFrame(
                    {
                        "case": [case] * 4,
                        "model_wt_height_m": [99.0, 0.0, 1.0, 2.0],
                        "observed_wt_height_m": [-99.0, 0.0, 1.0, 2.0],
                        "soilice_present": [True, False, False, False],
                        "h2osfc_available": [True] * 4,
                    }
                )
            )
        metrics = water_table_metrics(pd.concat(frames, ignore_index=True))
        np.testing.assert_allclose(metrics["model_mean_m"], 1.0)
        np.testing.assert_allclose(metrics["observed_mean_m"], 1.0)
        np.testing.assert_allclose(metrics["bias_m"], 0.0)
        np.testing.assert_allclose(metrics["rmse_m"], 0.0)
        np.testing.assert_allclose(metrics["r_squared"], 1.0)
        self.assertTrue((metrics["n"] == 3).all())

    def test_cpl_bypass_spinup_forcing_phase_matches_elm(self):
        model = {
            "year": np.arange(1, 10),
            "doy": np.full(9, 100),
            "wt": np.arange(9, dtype=float),
            "ice": np.zeros(9, dtype=bool),
        }
        observed = {
            "year": np.arange(2015, 2024),
            "doy": np.full(9, 100),
            "wt": np.arange(9, dtype=float),
        }
        aligned, _ = forcing_alignment(model, observed, 2015, 2023)
        np.testing.assert_array_equal(
            aligned["forcing_year"],
            [2021, 2022, 2023, 2015, 2016, 2017, 2018, 2019, 2020],
        )


if __name__ == "__main__":
    unittest.main()

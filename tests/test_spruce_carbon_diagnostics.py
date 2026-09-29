import tempfile
import unittest
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pandas as pd

from olmt_diagnostics.spruce_carbon import (
    FIT_CASES,
    METRICS,
    _annual_column_flux,
    _annual_pft_flux,
    temperature_response,
)


class SpruceCarbonDiagnosticsTest(unittest.TestCase):
    def make_history(self, root):
        path = Path(root) / "case.elm.h4.2016-01-01-00000.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", 2)
            dataset.createDimension("pft", 5)
            dataset.createDimension("column", 3)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = "days since 2016-01-01 00:00:00"
            time.calendar = "noleap"
            time[:] = [0.5, 1.5]
            for name, values in {
                "pfts1d_active": [1, 1, 1, 1, 1],
                "pfts1d_topounit": [1, 2, 2, 3, 3],
                "pfts1d_itype_veg": [0, 3, 5, 14, 18],
            }.items():
                variable = dataset.createVariable(name, "i4", ("pft",))
                variable[:] = values
            weights = dataset.createVariable("pfts1d_wtgcell", "f8", ("pft",))
            weights[:] = [0.5, 0.1, 0.1, 0.15, 0.15]
            for name in ["AGNPP", "BGNPP", "NPP"]:
                variable = dataset.createVariable(name, "f8", ("time", "pft"))
                variable[:] = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]) * 1e-6
            for name, values in {
                "cols1d_active": [1, 1, 1],
                "cols1d_topounit": [1, 2, 3],
            }.items():
                variable = dataset.createVariable(name, "i4", ("column",))
                variable[:] = values
            col_weights = dataset.createVariable("cols1d_wtgcell", "f8", ("column",))
            col_weights[:] = [0.5, 0.2, 0.3]
            hr = dataset.createVariable("HR", "f8", ("time", "column"))
            hr[:] = np.array([[0, 1, 2], [0, 1, 2]]) * 1e-6

    def test_area_weighted_annual_pft_and_column_fluxes(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_history(root)
            tree = _annual_pft_flux(Path(root), "AGNPP", (3, 5), 2016, 2016)
            hr = _annual_column_flux(Path(root), "HR", 2016, 2016)
        self.assertEqual(tree["model_days"].iloc[0], 2)
        self.assertAlmostEqual(tree["model_gC_m2_yr"].iloc[0], 0.10368)
        self.assertAlmostEqual(hr["model_gC_m2_yr"].iloc[0], 0.27648)

    def test_temperature_response_uses_t0_through_t9(self):
        rows = []
        warming = dict(zip(FIT_CASES, [0.0, 2.25, 4.5, 6.75, 9.0]))
        for metric in METRICS:
            for case in FIT_CASES:
                x = warming[case]
                rows.append(
                    {
                        "metric": metric,
                        "case": case,
                        "warming_C": x,
                        "model_gC_m2_yr": 10.0 + 3.0 * x,
                        "obs": 20.0 + 2.0 * x,
                    }
                )
        result = temperature_response(pd.DataFrame(rows))
        observed = result[result["source"] == "Observed"]
        model = result[result["source"] == "ELM"]
        np.testing.assert_allclose(observed["intercept_gC_m2_yr"], 20.0)
        np.testing.assert_allclose(observed["slope_gC_m2_yr_C"], 2.0)
        np.testing.assert_allclose(model["intercept_gC_m2_yr"], 10.0)
        np.testing.assert_allclose(model["slope_gC_m2_yr_C"], 3.0)


if __name__ == "__main__":
    unittest.main()

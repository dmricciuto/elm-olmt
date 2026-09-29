import tempfile
import unittest
from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from olmt_diagnostics.spruce_profiles import (
    CARBON_ATOMIC_MASS_G_MOL,
    _profiles_from_file,
    read_profile_observations,
)


class SpruceProfileDiagnosticsTest(unittest.TestCase):
    def make_history(self, root: str) -> Path:
        path = Path(root) / "case.elm.h3.2019-01-01-00000.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", 2)
            dataset.createDimension("levdcmp", 2)
            dataset.createDimension("levgrnd", 2)
            dataset.createDimension("column", 3)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = "days since 2019-01-01 00:00:00"
            time.calendar = "noleap"
            time[:] = [149.5, 150.5]
            depth = dataset.createVariable("levdcmp", "f8", ("levdcmp",))
            depth[:] = [0.1, 0.3]
            for name, values in {
                "cols1d_active": [1, 1, 1],
                "cols1d_topounit": [1, 2, 3],
                "cols1d_itype_lunit": [1, 1, 1],
            }.items():
                variable = dataset.createVariable(name, "i4", ("column",))
                variable[:] = values
            weight = dataset.createVariable("cols1d_wtgcell", "f8", ("column",))
            weight[:] = [0.2, 0.3, 0.5]

            shape = (2, 2, 3)
            doc = dataset.createVariable(
                "MM_DOM_POREWATER_C", "f8", ("time", "levdcmp", "column")
            )
            ch4 = dataset.createVariable(
                "MM_CH4_POREWATER", "f8", ("time", "levdcmp", "column")
            )
            acetate = dataset.createVariable(
                "MM_ACETATE_C_SAT", "f8", ("time", "levdcmp", "column")
            )
            liquid = dataset.createVariable(
                "SOILLIQ", "f8", ("time", "levgrnd", "column")
            )
            doc_values = np.zeros(shape)
            doc_values[:, :, 1] = 10.0
            doc_values[:, :, 2] = 20.0
            doc[:] = doc_values
            ch4_values = np.zeros(shape)
            ch4_values[:, :, 1] = 1.0
            ch4_values[:, :, 2] = 3.0
            ch4[:] = ch4_values
            acetate_values = np.zeros(shape)
            acetate_values[:, :, 1:] = CARBON_ATOMIC_MASS_G_MOL
            acetate[:] = acetate_values
            liquid[:] = np.full(shape, 100.0)
        return path

    def test_bog_weighting_and_acetate_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_history(root)
            result = _profiles_from_file(path, 2019, 2019, 121, 260)
        self.assertIsNotNone(result)
        depth, profiles = result
        np.testing.assert_allclose(depth, [0.1, 0.3])
        # Hollow/hummock normalized weights are 0.375 and 0.625.
        np.testing.assert_allclose(profiles["doc"], 16.25)
        np.testing.assert_allclose(profiles["ch4"], 2.25)
        # 100 kg m-2 water in a 0.2-m layer gives theta=0.5.
        np.testing.assert_allclose(profiles["acetate"], 1.0)

    def test_observation_reader_converts_doc_carbon_units(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            for filename, value in (
                ("CDOCS.txt", 2.0),
                ("CACES.txt", 3.0),
                ("CCON_CH4S.txt", 4.0),
            ):
                (directory / filename).write_text(
                    f"Year DOY Depth Value uncertainty\n2013 180 50 {value} 0.5\n"
                )
            observations = read_profile_observations(directory, 2013)
        doc = observations[observations["variable"] == "doc"].iloc[0]
        acetate = observations[observations["variable"] == "acetate"].iloc[0]
        self.assertAlmostEqual(doc["observed"], 2.0 * CARBON_ATOMIC_MASS_G_MOL)
        self.assertAlmostEqual(acetate["observed"], 3.0)


if __name__ == "__main__":
    unittest.main()

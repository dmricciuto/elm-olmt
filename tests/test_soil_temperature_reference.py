import importlib.util
import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np


SCRIPT = Path(__file__).parents[1] / "olmt_diagnostics" / "create_soil_temperature_reference.py"
SPEC = importlib.util.spec_from_file_location("soil_temperature_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SoilTemperatureReferenceTest(unittest.TestCase):
    def test_hummock_hollow_weighted_reference_excludes_fen(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "t0.elm.h1.2015.nc"
            domain_path = directory / "domain.nc"
            output_path = directory / "reference.nc"

            with netCDF4.Dataset(source_path, "w") as source:
                source.createDimension("time", 2)
                source.createDimension("column", 3)
                time = source.createVariable("time", "f8", ("time",))
                time.units = "days since 2015-01-01 00:00:00"
                time.calendar = "noleap"
                time[:] = [0.0, 1.0 / 24.0]
                weights = source.createVariable("cols1d_wtgcell", "f8", ("column",))
                weights[:] = [0.2, 0.3, 0.5]
                temperature = source.createVariable(
                    "TSOI_HEATING_CONTROL", "f8", ("time", "column")
                )
                temperature[:] = [[250.0, 270.0, 280.0], [251.0, 272.0, 282.0]]

            with netCDF4.Dataset(domain_path, "w") as domain:
                domain.createDimension("nj", 1)
                domain.createDimension("ni", 1)
                for name, value in (("xc", -93.45), ("yc", 47.50), ("area", 1e-8)):
                    variable = domain.createVariable(name, "f8", ("nj", "ni"))
                    variable[:] = value
                mask = domain.createVariable("mask", "i4", ("nj", "ni"))
                mask[:] = 1

            MODULE.create_reference(
                [str(source_path)], str(output_path), str(domain_path),
                "TSOI_HEATING_CONTROL", [1, 2]
            )

            with netCDF4.Dataset(output_path) as result:
                actual = result.variables["T_SOIL_REFERENCE"][:, 0, 0]
                expected = np.array([
                    (0.3 * 270.0 + 0.5 * 280.0) / 0.8,
                    (0.3 * 272.0 + 0.5 * 282.0) / 0.8,
                ])
                np.testing.assert_allclose(actual, expected)
                self.assertEqual(result.source_columns, "1,2")


if __name__ == "__main__":
    unittest.main()

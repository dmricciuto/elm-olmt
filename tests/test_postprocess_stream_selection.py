import tempfile
import unittest
from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from model_ELM.postprocess import (
    _postprocess_file_list_for_variable,
    do_timeaverage,
    read_postprocess_values,
)


class PostprocessStreamSelectionTest(unittest.TestCase):
    def make_history(self, root, stream, dimension, size, records=2):
        path = Path(root) / f"case.elm.h{stream}.0001-01-01-00000.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", records)
            dataset.createDimension(dimension, size)
            variable = dataset.createVariable("ZWT", "f8", ("time", dimension))
            variable[:] = np.arange(records * size).reshape(records, size)
        return path

    def test_indexed_column_uses_compatible_high_resolution_stream(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_history(root, 2, "lndgrid", 1, records=2)
            indexed = self.make_history(root, 3, "column", 3, records=5)
            Path(root, "lnd_in").write_text(
                " hist_mfilt = 1,1,365,365\n"
                " hist_nhtfrq = -8760,-8760,-24,-24\n"
            )

            class Case:
                casename = "case"

            files, _, _, _, _, stream = _postprocess_file_list_for_variable(
                Case(), root, "ZWT", "ZWT_col", 2, startyear=1, endyear=1
            )

            self.assertEqual(stream, 3)
            self.assertEqual(files, [str(indexed)])
            values, _ = read_postprocess_values(files, "ZWT", "ZWT_col", index=2)
            np.testing.assert_array_equal(values, [2, 5, 8, 11, 14])

    def test_vertical_column_slice_preserves_layers(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "case.elm.h3.0001-01-01-00000.nc"
            with Dataset(path, "w") as dataset:
                dataset.createDimension("time", 4)
                dataset.createDimension("levdcmp", 2)
                dataset.createDimension("column", 3)
                variable = dataset.createVariable(
                    "PROFILE", "f8", ("time", "levdcmp", "column")
                )
                variable[:] = np.arange(24).reshape(4, 2, 3)

            values, _ = read_postprocess_values(
                [str(path)], "PROFILE", "PROFILE_col", index=1
            )
            self.assertEqual(values.shape, (4, 2))
            np.testing.assert_array_equal(
                values, np.arange(24).reshape(4, 2, 3)[:, :, 1]
            )
            averaged = do_timeaverage(values, 2)
            self.assertEqual(averaged.shape, (2, 2))
            np.testing.assert_allclose(
                averaged,
                [values[:2].mean(axis=0), values[2:].mean(axis=0)],
            )

    def test_unindexed_variable_finds_daily_grid_stream(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_history(root, 0, "lndgrid", 1, records=12)
            daily = self.make_history(root, 2, "lndgrid", 1, records=365)
            Path(root, "lnd_in").write_text(
                " hist_mfilt = 1,1,365,365\n"
                " hist_nhtfrq = 0,-24,-24,-24\n"
            )

            class Case:
                casename = "case"

            files, _, _, _, _, stream = _postprocess_file_list_for_variable(
                Case(), root, "ZWT", "ZWT", 1, startyear=1, endyear=1
            )

            self.assertEqual(stream, 2)
            self.assertEqual(files, [str(daily)])


if __name__ == "__main__":
    unittest.main()

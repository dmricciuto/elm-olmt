import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from subset_restart import (
    domain_points,
    restart_matches_domain,
    subset_restart_from_domain,
)


def create_restart(path):
    with Dataset(path, "w") as ds:
        dimensions = {
            "gridcell": 3,
            "topounit": 3,
            "landunit": 5,
            "column": 6,
            "pft": 8,
            "levgrnd": 2,
        }
        for name, size in dimensions.items():
            ds.createDimension(name, size)

        def variable(name, dimension, values, dtype="i4"):
            var = ds.createVariable(name, dtype, (dimension,))
            var[:] = values
            return var

        variable("grid1d_lat", "gridcell", [0.0, 10.0, 20.0], "f8")
        variable("grid1d_lon", "gridcell", [179.5, 20.0, 30.0], "f8")
        variable("grid_value", "gridcell", [100, 200, 300])

        variable("topo1d_gridcell_index", "topounit", [1, 2, 3])
        variable("topo_value", "topounit", [11, 22, 33])

        variable("land1d_gridcell_index", "landunit", [1, 1, 2, 3, 3])
        variable("land1d_topounit_index", "landunit", [1, 1, 2, 3, 3])
        variable("land_value", "landunit", [10, 11, 20, 30, 31])

        variable("cols1d_gridcell_index", "column", [1, 1, 2, 3, 3, 3])
        variable("cols1d_topounit_index", "column", [1, 1, 2, 3, 3, 3])
        variable("cols1d_landunit_index", "column", [1, 2, 3, 4, 4, 5])
        column_state = ds.createVariable("column_state", "f8", ("column", "levgrnd"), zlib=True)
        column_state[:] = np.arange(12, dtype=float).reshape(6, 2)

        variable("pfts1d_gridcell_index", "pft", [1, 1, 1, 2, 3, 3, 3, 3])
        variable("pfts1d_topounit_index", "pft", [1, 1, 1, 2, 3, 3, 3, 3])
        variable("pfts1d_landunit_index", "pft", [1, 1, 2, 3, 4, 4, 4, 5])
        variable("pfts1d_column_index", "pft", [1, 1, 2, 3, 4, 4, 5, 6])
        variable("pft_value", "pft", np.arange(8))


def create_domain(path, points, mask=None):
    with Dataset(path, "w") as ds:
        ds.createDimension("nj", 1)
        ds.createDimension("ni", len(points))
        lat = ds.createVariable("yc", "f8", ("nj", "ni"))
        lon = ds.createVariable("xc", "f8", ("nj", "ni"))
        active = ds.createVariable("mask", "i4", ("nj", "ni"))
        lat[:] = [[point[0] for point in points]]
        lon[:] = [[point[1] for point in points]]
        active[:] = [mask if mask is not None else [1] * len(points)]


class SubsetRestartTests(unittest.TestCase):
    def test_subsets_hierarchy_in_requested_order_and_remaps_parents(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.nc"
            domain = Path(temp) / "domain.nc"
            output = Path(temp) / "subset.nc"
            create_restart(source)
            create_domain(domain, [(20.1, 30.1), (0.1, -180.4)])

            matches = subset_restart_from_domain(source, domain, output)

            self.assertEqual([match["source_index"] for match in matches], [2, 0])
            with Dataset(output) as ds:
                self.assertEqual(len(ds.dimensions["gridcell"]), 2)
                self.assertEqual(len(ds.dimensions["landunit"]), 4)
                self.assertEqual(len(ds.dimensions["column"]), 5)
                self.assertEqual(len(ds.dimensions["pft"]), 7)
                np.testing.assert_array_equal(ds.variables["grid_value"][:], [300, 100])
                np.testing.assert_array_equal(ds.variables["land_value"][:], [30, 31, 10, 11])
                np.testing.assert_array_equal(ds.variables["land1d_gridcell_index"][:], [1, 1, 2, 2])
                np.testing.assert_array_equal(ds.variables["land1d_topounit_index"][:], [1, 1, 2, 2])
                np.testing.assert_array_equal(ds.variables["cols1d_landunit_index"][:], [1, 1, 2, 3, 4])
                np.testing.assert_array_equal(ds.variables["pfts1d_column_index"][:], [1, 1, 2, 3, 4, 4, 5])
                np.testing.assert_array_equal(
                    ds.variables["column_state"][:],
                    np.asarray([[6, 7], [8, 9], [10, 11], [0, 1], [2, 3]], dtype=float),
                )

    def test_duplicate_nearest_cells_are_duplicated_with_independent_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.nc"
            domain = Path(temp) / "domain.nc"
            output = Path(temp) / "subset.nc"
            create_restart(source)
            create_domain(domain, [(10.0, 20.0), (10.1, 20.1)])
            subset_restart_from_domain(source, domain, output)

            with Dataset(output) as ds:
                np.testing.assert_array_equal(ds.variables["grid_value"][:], [200, 200])
                np.testing.assert_array_equal(ds.variables["land_value"][:], [20, 20])
                np.testing.assert_array_equal(ds.variables["land1d_gridcell_index"][:], [1, 2])
                np.testing.assert_array_equal(ds.variables["pfts1d_column_index"][:], [1, 2])

    def test_domain_controls_order_and_ignores_masked_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            domain = Path(temp) / "domain.nc"
            create_domain(domain, [(20.0, 30.0), (10.0, 20.0), (0.0, 179.5)], [1, 0, 1])
            self.assertEqual(domain_points(domain), [(20.0, 30.0), (0.0, 179.5)])

    def test_domain_accepts_1d_coordinates_with_singleton_2d_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            domain = Path(temp) / "domain.nc"
            with Dataset(domain, "w") as ds:
                ds.createDimension("nj", 1)
                ds.createDimension("ni", 3)
                ds.createVariable("yc", "f8", ("ni",))[:] = [1.0, 2.0, 3.0]
                ds.createVariable("xc", "f8", ("ni",))[:] = [10.0, 20.0, 30.0]
                ds.createVariable("mask", "i4", ("nj", "ni"))[:] = [[1, 0, 1]]
            self.assertEqual(domain_points(domain), [(1.0, 10.0), (3.0, 30.0)])

    def test_rejects_distant_gridcell(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.nc"
            domain = Path(temp) / "domain.nc"
            output = Path(temp) / "subset.nc"
            create_restart(source)
            create_domain(domain, [(-80.0, 0.0)])
            with self.assertRaisesRegex(ValueError, "exceeding 10 km"):
                subset_restart_from_domain(
                    source, domain, output, max_distance_km=10.0
                )

    def test_detects_when_restart_already_matches_domain(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.nc"
            domain = Path(temp) / "domain.nc"
            create_restart(source)
            create_domain(domain, [(0.0, 179.5), (10.0, 20.0), (20.0, 30.0)])
            self.assertTrue(restart_matches_domain(source, domain))


if __name__ == "__main__":
    unittest.main()

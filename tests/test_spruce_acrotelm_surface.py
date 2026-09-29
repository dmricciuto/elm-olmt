import unittest

import numpy as np
import xarray as xr

from model_ELM.makepointdata import add_topounit_dimension, spruce_topounit_fractions


class SpruceAcrotelmSurfaceTests(unittest.TestCase):

    SPRUCE_FRACAREA = spruce_topounit_fractions()

    @staticmethod
    def base_surface():
        return xr.Dataset(
            data_vars={
                "LATIXY": (("gridcell",), [47.563]),
                "LONGXY": (("gridcell",), [266.562]),
                "PCT_NATVEG": (("gridcell",), [100.0]),
            },
            coords={"gridcell": [0]},
        )

    def test_spruce_local_depths_share_hollow_referenced_boundary(self):
        elevations = np.array([464.95, 465.00, 465.15])
        boundary_elevation = elevations[1] - 0.30
        local_depths = elevations - boundary_elevation

        surface = add_topounit_dimension(
            None,
            self.base_surface(),
            "LATIXY",
            "LONGXY",
            num_topounits=3,
            fracarea=self.SPRUCE_FRACAREA,
            elevations=elevations,
            distances=[0, 3, 1],
            is_bog=[0, 1, 1],
            peat_depth=[2.95, 3.00, 3.15],
            till_ksat=[0.0, 0.1 / 86400.0, 0.1 / 86400.0],
            drainage_outlet_depth=[0.4, 0.4, 0.4],
            acrotelm_depth=local_depths,
        )

        actual = surface["TopounitAcrotelmDepth"].values[:, 0]
        np.testing.assert_allclose(actual, [0.25, 0.30, 0.45])
        np.testing.assert_allclose(elevations - actual, boundary_elevation)
        np.testing.assert_allclose(
            surface["TopounitFracArea"].values[:, 0], self.SPRUCE_FRACAREA
        )
        np.testing.assert_allclose(sum(self.SPRUCE_FRACAREA), 1.0)
        np.testing.assert_array_equal(
            surface["TopounitStructureSnowRetention"].values[:, 0], [1.0, 1.0, 1.0]
        )

    def test_zero_default_retains_prognostic_mode(self):
        surface = add_topounit_dimension(
            None,
            self.base_surface(),
            "LATIXY",
            "LONGXY",
            num_topounits=2,
            fracarea=[0.5, 0.5],
            elevations=[100.0, 100.15],
            peat_depth=[3.0, 3.15],
        )

        np.testing.assert_array_equal(
            surface["TopounitAcrotelmDepth"].values[:, 0], [0.0, 0.0]
        )


if __name__ == "__main__":
    unittest.main()

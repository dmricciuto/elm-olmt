import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np

from tools.migrate_myci_parameters import (
    GLOBAL_PARAMETERS, PFT_PARAMETERS, migrate,
)


class MYCIParameterMigrationTest(unittest.TestCase):
    def write_file(self, path, names, global_dim, include_parameters):
        with netCDF4.Dataset(path, "w") as nc:
            nc.createDimension("pft", len(names))
            nc.createDimension("string_length", 40)
            nc.createDimension(global_dim, 1)
            pftname = nc.createVariable("pftname", "S1", ("pft", "string_length"))
            pftname[:] = np.asarray([list(name.ljust(40)) for name in names],
                                    dtype="S1")
            if include_parameters:
                for name in PFT_PARAMETERS:
                    var = nc.createVariable(name, "f8", ("pft",), fill_value=-1.0e20)
                    var[:] = np.arange(len(names), dtype=float) + 1.0
                for name in GLOBAL_PARAMETERS:
                    var = nc.createVariable(name, "f8", (global_dim,))
                    var[:] = [2.0]

    def test_split_pfts_are_mapped_by_name(self):
        source_names = [
            "not_vegetated", "needleleaf_evergreen_boreal_tree",
            "needleleaf_deciduous_boreal_tree", "broadleaf_deciduous_boreal_shrub",
        ]
        target_names = [
            "not_vegetated", "upland_needleleaf_evergreen_boreal_tree",
            "peatlnd_needleleaf_evergreen_boreal_tree",
            "needleleaf_deciduous_boreal_tree",
            "peatlnd_needleleaf_deciduous_boreal_tree",
            "upland_broadleaf_deciduous_boreal_shrub",
            "peatlnd_broadleaf_deciduous_boreal_shrub", "moss",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, base, output = (root / name for name in
                                    ("source.nc", "base.nc", "output.nc"))
            self.write_file(source, source_names, "allpft", True)
            self.write_file(base, target_names, "allpfts", False)
            migrate(source, base, output)
            with netCDF4.Dataset(output) as nc:
                values = nc["vmax_fungi_son"][:]
                self.assertEqual(values[1], values[2])
                self.assertEqual(values[3], values[4])
                self.assertEqual(values[5], values[6])
                self.assertNotEqual(values[1], values[3])
                self.assertTrue(np.ma.is_masked(values[7]))
                self.assertEqual(nc["q10_upt"].dimensions, ("allpfts",))
                self.assertEqual(len(nc.dimensions["pft"]), len(target_names))


if __name__ == "__main__":
    unittest.main()

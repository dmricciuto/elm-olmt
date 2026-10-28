"""Regression checks for FLUXNET local-time forcing in ELM's UTC clock."""

import io
import sys
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from fluxnet_shuttle_to_olmt import _hourly_forcing, _utc_offset_hours


class ShuttleTimeAlignmentTest(unittest.TestCase):
    def test_bif_utc_offset(self):
        out = io.BytesIO()
        with ZipFile(out, "w") as archive:
            archive.writestr(
                "AMF_US-MMS_FLUXNET_BIF_2000.csv",
                "SITE_ID,GROUP_ID,VARIABLE_GROUP,VARIABLE,DATAVALUE\n"
                "US-MMS,1,UTC_OFFSET,UTC_OFFSET,-5\n",
            )
        out.seek(0)
        with ZipFile(out) as archive:
            self.assertEqual(_utc_offset_hours(archive), -5)

    def test_hourly_local_standard_time_to_utc_across_leap_day(self):
        local = pd.date_range("2000-01-01", "2000-12-31 23:00", freq="h")
        frame = pd.DataFrame({
            "TIMESTAMP_START": local.strftime("%Y%m%d%H%M"),
            "TA_F": np.full(len(local), 20.0),
            "VPD_F": np.full(len(local), 10.0),
            "WS_F": np.full(len(local), 2.0),
            "PA_F": np.full(len(local), 100.0),
            "SW_IN_F": local.hour.astype(float),
            "LW_IN_F": np.full(len(local), 300.0),
            "P_F": np.zeros(len(local)),
        })
        forcing, first, last, _, missing, edge_hours = _hourly_forcing(frame, -5)
        self.assertEqual((first, last, edge_hours), (2000, 2000, 5))
        self.assertEqual(len(forcing), 8760)
        self.assertEqual(sum(missing.values()), 0)
        self.assertEqual(forcing.loc["2000-07-01 17:00", "FSDS"], 12.0)
        self.assertEqual(forcing.loc["2000-03-01 00:00", "FSDS"], 19.0)
        self.assertNotIn(pd.Timestamp("2000-02-29 12:00"), forcing.index)


if __name__ == "__main__":
    unittest.main()

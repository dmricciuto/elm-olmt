"""Regression tests for SPRUCE.032 seasonal temperature summaries."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from olmt_diagnostics.spruce_soil_temperature_benchmark import (
    _expected_halfhours, _probe_columns, summarize_plot,
)


class SoilTemperatureBenchmarkTests(unittest.TestCase):
    def test_leap_year_djf_expected_slots(self):
        self.assertEqual(_expected_halfhours(2016, "DJF"), 91 * 48)
        self.assertEqual(_expected_halfhours(2017, "DJF"), 90 * 48)

    def test_equal_year_and_probe_weighting_with_missing_sensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            rows = []
            for year, a_value, b_value in ((2016, 1.0, 3.0),
                                           (2017, 5.0, 7.0)):
                for timestamp in pd.date_range(f"{year}-03-01", periods=12,
                                               freq="30min"):
                    row = {"TIMESTAMP": timestamp.strftime("%Y-%m-%d %H:%M")}
                    for probe in _probe_columns():
                        row[probe] = (-9999 if "_C" in probe else
                                      a_value if "_A" in probe else b_value)
                    rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            result = summarize_plot(path, "T0.00", 6, 2016, 2017,
                                    min_coverage=0.002, chunk_size=7)
            spring_surface = result[(result.season == "MAM") &
                                    (result.depth_m == 0.0)].iloc[0]
            self.assertAlmostEqual(spring_surface.temperature_C, 4.0)
            self.assertEqual(spring_surface.n_years, 2)
            self.assertEqual(spring_surface.qualified_probe_years, 4)
            self.assertEqual(spring_surface.possible_probe_years, 6)
            self.assertEqual(spring_surface.valid_halfhours, 48)
            self.assertTrue(np.isnan(result[(result.season == "DJF") &
                                             (result.depth_m == 0.0)]
                                      .iloc[0].temperature_C))


if __name__ == "__main__":
    unittest.main()

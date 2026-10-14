"""Check hollow-column seasonal TSOI extraction and PDF-job wiring."""

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from netCDF4 import Dataset, date2num
import numpy as np
import pandas as pd

from elm_olmt import _create_spruce_diagnostics_script
from olmt_diagnostics.plot_spruce_soil_temperature_model_vs_obs import (
    plot_comparison, summarize_model,
)
from olmt_diagnostics.spruce_soil_temperature_benchmark import TREATMENT_PLOTS


class SoilTemperatureModelTests(unittest.TestCase):
    def test_seasons_use_active_hollow_not_fen_or_hummock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = [datetime(1861, month, 15) for month in (1, 4, 7, 10)]
            for treatment_index, treatment in enumerate(TREATMENT_PLOTS):
                case = f"SYNTHETIC_{treatment}"
                directory = root / case / "run"
                directory.mkdir(parents=True)
                path = directory / f"{case}.elm.h1.1861-01-01-00000.nc"
                with Dataset(path, "w") as dataset:
                    dataset.createDimension("time", 4)
                    dataset.createDimension("levgrnd", 2)
                    dataset.createDimension("column", 3)
                    time = dataset.createVariable("time", "f8", ("time",))
                    time.units = "days since 1861-01-01 00:00:00"
                    time.calendar = "noleap"
                    time[:] = date2num(dates, time.units, calendar=time.calendar)
                    dataset.createVariable("levgrnd", "f8", ("levgrnd",))[:] = [0.1, 1.0]
                    dataset.createVariable("cols1d_topounit", "i4", ("column",))[:] = [1, 2, 3]
                    dataset.createVariable("cols1d_wtgcell", "f8", ("column",))[:] = [0.5, 0.2, 0.3]
                    temperature = dataset.createVariable(
                        "TSOI", "f8", ("time", "levgrnd", "column"))
                    temperature.units = "K"
                    values = np.full((4, 2, 3), 273.15)
                    values[:, :, 0] += 50  # fen must not contaminate comparison
                    values[:, :, 2] += 30  # hummock must not contaminate comparison
                    for season_index in range(4):
                        values[season_index, :, 1] += 5 + treatment_index + season_index
                    temperature[:] = values
            summary = summarize_model(root, "SYNTHETIC", 1861, 1861)
            selected = summary[(summary.treatment == "T9.00") &
                               (summary.season == "JJA") &
                               (summary.depth_m == 1.0)].iloc[0]
            self.assertAlmostEqual(selected.temperature_C, 12.0)
            self.assertEqual(selected.n_years, 1)
            observations = pd.read_csv(
                Path(__file__).resolve().parents[1] / "data" / "spruce" /
                "soil_temperature_2026" / "seasonal_mean_2023.csv"
            )
            figure = root / "comparison.png"
            plot_comparison(observations, summary, figure, "1861", "2023", True)
            self.assertGreater(figure.stat().st_size, 10000)

    def test_aggregate_job_receives_observation_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = _create_spruce_diagnostics_script(
                temporary, "/run", "CASE", str(Path(temporary) / "diags"),
                1860, 1861, 1860, 1861, True,
                "/obs/seasonal_mean_2023.csv", 1861, 1861, "2023")
            command = Path(script).read_text()
            self.assertIn("--soil-temperature-observations", command)
            self.assertIn("/obs/seasonal_mean_2023.csv", command)
            self.assertIn("--soil-temperature-model-start-year 1861", command)
            self.assertIn("--soil-temperature-observation-years 2023", command)


if __name__ == "__main__":
    unittest.main()

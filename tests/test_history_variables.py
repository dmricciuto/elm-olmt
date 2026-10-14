import configparser
import importlib.util
import re
import unittest
from pathlib import Path

from elm_olmt import next_soil_heating_history_tape

# Import the history helper without importing model_ELM's optional plotting and
# calibration dependencies through its package initializer.
_histvars_path = Path(__file__).resolve().parents[1] / "model_ELM" / "set_histvars.py"
_histvars_spec = importlib.util.spec_from_file_location("olmt_set_histvars", _histvars_path)
_histvars_module = importlib.util.module_from_spec(_histvars_spec)
_histvars_spec.loader.exec_module(_histvars_module)
set_histvars = _histvars_module.set_histvars


class HistoryVariableTest(unittest.TestCase):
    def make_case(self, variables):
        class Case:
            pass

        case = Case()
        case.postproc_vars = variables
        case.postproc_freq = "daily"
        case.sitegroup = "AmeriFlux"
        case.calls = {}
        case.customize_namelist = (
            lambda variable, value: case.calls.__setitem__(variable, value))
        return case

    def test_transient_indexed_only_uses_one_daily_tape(self):
        case = self.make_case(["ZWT_col", "NPP_pft"])

        set_histvars(case)

        self.assertEqual(case.calls["hist_mfilt"], "1,365")
        self.assertEqual(case.calls["hist_nhtfrq"], "-8760,-24")
        self.assertEqual(case.calls["hist_dov2xy"], ".true.,.false.")
        self.assertEqual(case.calls["hist_fincl2"], "'ZWT','NPP'")
        self.assertNotIn("hist_fincl3", case.calls)
        self.assertNotEqual(case.calls["hist_fincl2"], "")

    def test_transient_mixed_variables_use_separate_daily_tapes(self):
        case = self.make_case(["GPP", "ZWT_col"])

        set_histvars(case)

        self.assertEqual(case.calls["hist_mfilt"], "1,365,365")
        self.assertEqual(case.calls["hist_dov2xy"], ".true.,.true.,.false.")
        self.assertEqual(case.calls["hist_fincl2"], "'GPP'")
        self.assertEqual(case.calls["hist_fincl3"], "'ZWT'")

    def test_hourly_heating_tape_follows_indexed_daily_output(self):
        variables = ["ZWT_col", "NPP_pft"]
        case = self.make_case(variables)
        set_histvars(case)

        reference_tape = next_soil_heating_history_tape(
            variables, case.sitegroup)
        self.assertEqual(reference_tape, 2)  # h2; Fortran tape index 3
        self.assertIn("hist_fincl2", case.calls)
        self.assertNotIn("hist_fincl3", case.calls)

    def test_hourly_heating_tape_follows_mixed_daily_output(self):
        self.assertEqual(next_soil_heating_history_tape(
            ["GPP", "ZWT_col"], "AmeriFlux"), 3)

    def test_hourly_heating_tape_follows_default_output(self):
        self.assertEqual(next_soil_heating_history_tape([], "Peatlands"), 1)

    def test_spruce_configuration_uses_next_available_tape(self):
        config_path = Path(__file__).resolve().parents[1] / "config_files" / "SPRUCE_microbe_20260924.cfg"
        config = configparser.ConfigParser(interpolation=None)
        config.read(config_path)
        variables = re.findall(r"'([^']+)'", config["postprocessing"]["variables"])
        configured_tape = int(config["treatment_options"]["soil_heating_reference_history_tape"])
        self.assertEqual(configured_tape, next_soil_heating_history_tape(
            variables, config["simulation"]["sitegroup"]))


if __name__ == "__main__":
    unittest.main()

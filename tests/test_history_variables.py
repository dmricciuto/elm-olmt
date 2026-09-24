import unittest

from model_ELM.set_histvars import set_histvars


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


if __name__ == "__main__":
    unittest.main()

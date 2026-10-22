import unittest

from model_ELM.main import ELMcase


class LegacyHumholTest(unittest.TestCase):
    def make_case(self, options, cppdefs=''):
        case = ELMcase.__new__(ELMcase)
        case.case_options = options
        case.humhol = False
        case.cppdefs = cppdefs
        return case

    def test_modern_runtime_option_does_not_add_cppdef(self):
        case = self.make_case({'use_humhol': '.true.'})
        self.assertFalse(case.configure_humhol_options())
        self.assertTrue(case.humhol)
        self.assertEqual(case.cppdefs, '')

    def test_disabled_legacy_switch_leaves_modern_behavior_unchanged(self):
        case = self.make_case({'use_humhol': '.true.',
                               'legacy_humhol_cppdef': 'False'})
        self.assertFalse(case.configure_humhol_options())
        self.assertTrue(case.humhol)
        self.assertEqual(case.cppdefs, '')

    def test_legacy_switch_adds_cppdef_once(self):
        case = self.make_case({'use_humhol': '.true.',
                               'legacy_humhol_cppdef': 'True'}, 'OTHER')
        self.assertTrue(case.configure_humhol_options())
        self.assertTrue(case.humhol)
        self.assertEqual(case.cppdefs, 'OTHER,HUM_HOL')
        self.assertTrue(case.configure_humhol_options())
        self.assertEqual(case.cppdefs, 'OTHER,HUM_HOL')

    def test_legacy_switch_alone_enables_multitopounit(self):
        case = self.make_case({'legacy_humhol_cppdef': '.true.'})
        self.assertTrue(case.configure_humhol_options())
        self.assertTrue(case.humhol)
        self.assertEqual(case.cppdefs, 'HUM_HOL')

    def test_legacy_switch_suppresses_only_runtime_namelist_option(self):
        excluded = {'legacy_humhol_cppdef', 'humhol'}
        self.assertFalse(ELMcase.case_option_is_namelist(
            'use_humhol', excluded, True))
        self.assertTrue(ELMcase.case_option_is_namelist(
            'use_humhol', excluded, False))
        self.assertFalse(ELMcase.case_option_is_namelist(
            'legacy_humhol_cppdef', excluded, True))
        self.assertTrue(ELMcase.case_option_is_namelist(
            'use_nofire', excluded, True))

    def test_conflicting_runtime_false_is_rejected(self):
        case = self.make_case({'use_humhol': '.false.',
                               'legacy_humhol_cppdef': 'True'})
        with self.assertRaisesRegex(ValueError, 'requires use_humhol'):
            case.configure_humhol_options()


if __name__ == '__main__':
    unittest.main()

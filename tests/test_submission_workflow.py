import os
import tempfile
import unittest

from model_ELM.main import ELMcase


class SubmissionWorkflowTest(unittest.TestCase):
    def make_case(self, root):
        case = ELMcase.__new__(ELMcase)
        case.casedir = os.path.join(root, "case")
        case.OLMTdir = root
        case.noslurm = False
        case.postproc_vars = []
        case.create_pkl = lambda outdir: None
        case.slurm_submit_args = lambda *args, **kwargs: []
        os.makedirs(os.path.join(case.casedir, "Buildconf"))
        return case

    def test_dependent_case_uses_case_submit_prerequisite(self):
        with tempfile.TemporaryDirectory() as root:
            case = self.make_case(root)

            case_submit = os.path.join(case.casedir, "case.submit")
            with open(case_submit, "w") as stream:
                stream.write(
                    "#!/bin/sh\n"
                    "test \"$1\" = '--prereq' || exit 21\n"
                    "test \"$2\" = '42' || exit 22\n"
                    "echo 'Submitted batch job 12345'\n"
                )
            os.chmod(case_submit, 0o755)

            old_cwd = os.getcwd()
            try:
                job = case.submit_case(depend=42)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(job, 12345)

    def test_dependent_finidat_is_relative_link(self):
        with tempfile.TemporaryDirectory() as root:
            case = self.make_case(root)
            case.runroot = os.path.join(root, "runs")
            case.rundir = os.path.join(case.runroot, "dependent", "run")
            case.finidat_source = ""
            case.link_finidat = False

            case.set_finidat_file(finidat_case="ad_spinup", finidat_year=208)

            expected_name = "ad_spinup.elm.r.0208-01-01-00000.nc"
            expected_source = os.path.join(
                case.runroot, "ad_spinup", "run", expected_name)
            self.assertEqual(case.finidat, expected_name)
            self.assertEqual(case.finidat_source, expected_source)
            self.assertTrue(case.link_finidat)

            link_path = case.create_finidat_link()
            self.assertTrue(os.path.islink(link_path))
            self.assertEqual(
                os.readlink(link_path), os.path.relpath(expected_source, case.rundir))
            self.assertEqual(case.create_finidat_link(), link_path)

            os.makedirs(os.path.dirname(expected_source))
            with open(expected_source, "w") as stream:
                stream.write("restart-data")
            with open(link_path) as stream:
                self.assertEqual(stream.read(), "restart-data")

    def test_explicit_finidat_remains_absolute(self):
        with tempfile.TemporaryDirectory() as root:
            case = self.make_case(root)
            case.finidat_source = ""
            case.link_finidat = False
            explicit = os.path.join(root, "initial.elm.r.1850-01-01-00000.nc")

            case.set_finidat_file(finidat=explicit)

            self.assertEqual(case.finidat, explicit)
            self.assertFalse(case.link_finidat)
            self.assertEqual(case.create_finidat_link(), "")

    def test_failure_reports_command_and_case_status(self):
        with tempfile.TemporaryDirectory() as root:
            case = self.make_case(root)
            case.finidat = ""
            with open(os.path.join(case.casedir, "CaseStatus"), "w") as stream:
                stream.write("case.submit error missing input\n")
            submit = os.path.join(case.casedir, "case.submit")
            with open(submit, "w") as stream:
                stream.write("#!/bin/sh\necho scheduler-rejected >&2\nexit 7\n")
            os.chmod(submit, 0o755)

            old_cwd = os.getcwd()
            try:
                with self.assertRaises(RuntimeError) as caught:
                    case.submit_case()
            finally:
                os.chdir(old_cwd)

            message = str(caught.exception)
            self.assertIn("Command: ./case.submit", message)
            self.assertIn("Return code: 7", message)
            self.assertIn("scheduler-rejected", message)
            self.assertIn("case.submit error missing input", message)


if __name__ == "__main__":
    unittest.main()

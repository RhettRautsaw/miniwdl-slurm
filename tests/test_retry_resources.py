import json
import logging
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from WDL.runtime import config
from WDL.runtime.backend.singularity import SingularityContainer

from miniwdl_slurm import SlurmSingularity


class RetryResourcesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.logger = logging.getLogger("test_retry_resources")
        self.cfg = config.Loader(self.logger, filenames=[])
        self.container = SlurmSingularity(self.cfg, "task", directory.name)
        self.container.runtime_values.update(
            {"time_minutes": 60, "memory_reservation": 1024 ** 3}
        )
        self.submit_log = Path(directory.name) / "slurm_singularity.log.txt"
        self.submit_log.write_text("12345\n")

    def run_attempt(self, exit_code=1, state="TIMEOUT", error=None):
        result = subprocess.CompletedProcess(
            [], 0, stdout=f"JobId=12345 JobState={state} ExitCode=0:15\n",
            stderr=""
        )
        with patch.object(SingularityContainer, "_run", return_value=exit_code):
            with patch("miniwdl_slurm.subprocess.run", return_value=result,
                       side_effect=error) as query:
                actual = self.container._run(self.logger, lambda: False, "command")
        self.assertEqual(actual, exit_code)
        return query

    def retry_args(self):
        self.container.reset(self.logger)
        return self.container._slurm_invocation(self.logger)

    def test_timeout_increases_time_and_logs_reason(self):
        query = self.run_attempt()
        query.assert_called_once_with(
            ["scontrol", "--oneliner", "show", "job", "12345"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=True, timeout=10
        )
        with self.assertLogs(self.logger, level="INFO") as logs:
            args = self.retry_args()
        self.assertEqual(args[args.index("--time") + 1], "90")
        self.assertEqual(args[args.index("--mem") + 1], "1024M")
        message = "\n".join(logs.output)
        for text in ["increased Slurm time limit", "try 2", "TIMEOUT",
                     "previous exit code 1", "60 -> 90 minutes"]:
            self.assertIn(text, message)

    def test_consecutive_timeouts_compound_and_round_up(self):
        self.container.runtime_values["time_minutes"] = 3
        for expected in ["5", "7", "11"]:
            self.run_attempt()
            args = self.retry_args()
            self.assertEqual(args[args.index("--time") + 1], expected)
        self.assertEqual(self.container.runtime_values["time_minutes"], 3)

    def test_other_failures_do_not_increase_time(self):
        for exit_code, state in [(1, "FAILED"), (124, "FAILED"),
                                 (137, "OUT_OF_MEMORY"), (143, "CANCELLED"),
                                 (253, "OUT_OF_MEMORY")]:
            with self.subTest(exit_code=exit_code, state=state):
                self.run_attempt(exit_code, state)
                args = self.retry_args()
                self.assertEqual(args[args.index("--time") + 1], "60")
                if exit_code in SlurmSingularity.OOM_EXIT_CODES:
                    self.assertGreater(self.container._oom_retry_count, 0)

    def test_other_failure_resets_timeout_count(self):
        for state, expected in [("TIMEOUT", "90"), ("FAILED", "60"),
                                ("TIMEOUT", "90")]:
            self.run_attempt(state=state)
            args = self.retry_args()
            self.assertEqual(args[args.index("--time") + 1], expected)

    def test_dynamic_rules_use_adjusted_time(self):
        self.cfg.override({"slurm": {"dynamic_partition": json.dumps([
            {"time_minutes__ge": 90, "args": "--partition long"},
            {"args": "--partition short"}
        ])}})
        args = self.container._slurm_invocation(self.logger)
        self.assertEqual(args[args.index("--partition") + 1], "short")
        self.run_attempt()
        args = self.retry_args()
        self.assertEqual(args[args.index("--partition") + 1], "long")

    def test_missing_or_zero_time_skips_state_query(self):
        for value in [None, 0]:
            with self.subTest(time_minutes=value):
                self.container.runtime_values.pop("time_minutes", None)
                if value is not None:
                    self.container.runtime_values["time_minutes"] = value
                query = self.run_attempt()
                query.assert_not_called()
                self.retry_args()
                self.assertEqual(self.container._retry_adjusted_time_minutes(), value)

    def test_success_and_accepted_exit_codes_skip_state_query(self):
        self.run_attempt(exit_code=0).assert_not_called()
        self.container.runtime_values["returnCodes"] = [1]
        self.run_attempt(exit_code=1).assert_not_called()

    def test_query_errors_warn_without_changing_exit_code(self):
        for error in [FileNotFoundError("scontrol missing"),
                      subprocess.TimeoutExpired("scontrol", 10),
                      subprocess.CalledProcessError(1, "scontrol")]:
            with self.subTest(error=error):
                self.container._timeout_retry_count = 1
                with self.assertLogs(self.logger, level="WARNING") as logs:
                    self.run_attempt(error=error)
                self.assertIn("time limit will not be increased", logs.output[0])
                args = self.retry_args()
                self.assertEqual(args[args.index("--time") + 1], "60")

    def test_missing_job_id_and_state_do_not_guess_timeout(self):
        self.submit_log.write_text("sbatch: error: submission failed\n")
        with self.assertLogs(self.logger, level="WARNING"):
            self.run_attempt().assert_not_called()
        self.assertEqual(self.container._timeout_retry_count, 0)
        self.submit_log.write_text("12345\n")
        result = subprocess.CompletedProcess([], 0, stdout="Reason=TimeLimit\n")
        with patch("miniwdl_slurm.subprocess.run", return_value=result):
            with self.assertLogs(self.logger, level="WARNING"):
                state = self.container._slurm_job_state(
                    self.logger, str(self.submit_log)
                )
        self.assertIsNone(state)

    def test_cluster_job_id_with_submission_warnings(self):
        self.submit_log.write_text("sbatch: warning: message\n12345;remote\n")
        query = self.run_attempt()
        self.assertEqual(query.call_args.args[0], [
            "scontrol", "--oneliner", "--clusters=remote", "show", "job", "12345"
        ])

    def test_completing_requires_explicit_time_limit_reason(self):
        for reason, expected in [("TimeLimit", "90"), ("NonZeroExitCode", "60"),
                                 ("OutOfMemory", "60"), ("PartitionTimeLimit", "60")]:
            with self.subTest(reason=reason):
                output = f"JobId=12345 JobState=COMPLETING Reason={reason}\n"
                result = subprocess.CompletedProcess([], 0, stdout=output)
                with patch.object(SingularityContainer, "_run", return_value=1):
                    with patch("miniwdl_slurm.subprocess.run", return_value=result):
                        self.container._run(self.logger, lambda: False, "command")
                args = self.retry_args()
                self.assertEqual(args[args.index("--time") + 1], expected)

    def test_termination_cancels_job_without_state_query(self):
        self.submit_log.write_text("12345;remote\n")
        with patch.object(SingularityContainer, "_run", return_value=1):
            with patch("miniwdl_slurm.subprocess.run") as command:
                self.container._run(self.logger, lambda: True, "command")
        command.assert_called_once_with(["scancel", "--clusters=remote", "12345"])
        self.assertEqual(self.container._timeout_retry_count, 0)

    def test_backend_exception_clears_previous_timeout(self):
        self.container._timeout_retry_count = 1
        with patch.object(SingularityContainer, "_run", side_effect=RuntimeError):
            with patch("miniwdl_slurm.subprocess.run") as query:
                with self.assertRaises(RuntimeError):
                    self.container._run(self.logger, lambda: False, "command")
        query.assert_not_called()
        self.assertEqual(self.container._timeout_retry_count, 0)


if __name__ == "__main__":
    unittest.main()

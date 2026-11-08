from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "mlops_pipeline" / "09_failure_recovery.py"
SPEC = importlib.util.spec_from_file_location("lab9_failure_recovery", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load failure-recovery module from {MODULE_PATH}")
FAILURE_RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FAILURE_RECOVERY)


class FailureRecoveryTests(unittest.TestCase):
    def test_scenario_pass_requires_expected_halt(self):
        result = FAILURE_RECOVERY.make_scenario_result(
            name="missing_data",
            expected="HALT",
            actual="HALT",
            passed=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "PASSED")

    def test_unexpected_continuation_fails_scenario(self):
        result = FAILURE_RECOVERY.make_scenario_result(
            name="missing_data",
            expected="HALT",
            actual="CONTINUED",
            passed=False,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "FAILED")

    def test_report_fails_if_any_repetition_has_a_failed_scenario(self):
        report = FAILURE_RECOVERY.build_report(
            [
                {
                    "repetition": 1,
                    "results": [
                        {"scenario": "missing_data", "passed": True},
                    ],
                },
                {
                    "repetition": 2,
                    "results": [
                        {"scenario": "missing_data", "passed": False},
                    ],
                },
            ]
        )

        self.assertEqual(report["overall_status"], "FAILED")

    def test_report_passes_only_when_all_scenarios_pass(self):
        report = FAILURE_RECOVERY.build_report(
            [
                {
                    "repetition": 1,
                    "results": [
                        {"scenario": "missing_data", "passed": True},
                        {"scenario": "model_recovery", "passed": True},
                    ],
                },
            ]
        )

        self.assertEqual(report["overall_status"], "PASSED")

    def test_sandbox_guard_rejects_paths_outside_temporary_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)

            with self.assertRaises(ValueError):
                FAILURE_RECOVERY.ensure_sandbox_path(
                    temporary_root,
                    PROJECT_ROOT,
                )


if __name__ == "__main__":
    unittest.main()

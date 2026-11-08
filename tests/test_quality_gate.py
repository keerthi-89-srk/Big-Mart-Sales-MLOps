from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "mlops_pipeline" / "07_quality_gate.py"
SPEC = importlib.util.spec_from_file_location("lab7_quality_gate", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load quality gate module from {MODULE_PATH}")
QUALITY_GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY_GATE)


def valid_config() -> dict:
    return {
        "target_column": "Item_Outlet_Sales",
        "baseline": {
            "registered_model_name": "BigMart_Sales_Model",
            "model_version": 1,
            "run_id": "baseline-run",
        },
        "quality_gate": {
            "metrics": {
                "MAE": {"operator": "<=", "threshold": 740.0},
                "RMSE": {"operator": "<=", "threshold": 1060.0},
                "R2": {"operator": ">=", "threshold": 0.59},
            }
        },
    }


def valid_evaluation() -> dict:
    return {
        "target_column": "Item_Outlet_Sales",
        "MAE": 720.0,
        "RMSE": 1030.0,
        "R2_Score": 0.61,
    }


class QualityGateTests(unittest.TestCase):
    def test_passes_when_all_regression_metrics_meet_thresholds(self):
        result = QUALITY_GATE.evaluate_quality_gate(
            valid_config(),
            valid_evaluation(),
        )

        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(
            set(result["metric_results"]),
            {"MAE", "RMSE", "R2"},
        )

    def test_fails_when_any_metric_misses_threshold(self):
        evaluation = valid_evaluation()
        evaluation["MAE"] = 800.0

        result = QUALITY_GATE.evaluate_quality_gate(valid_config(), evaluation)

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["metric_results"]["MAE"]["status"], "FAILED")
        self.assertEqual(result["metric_results"]["RMSE"]["status"], "PASSED")

    def test_rejects_wrong_target(self):
        evaluation = valid_evaluation()
        evaluation["target_column"] = "Churn"

        with self.assertRaisesRegex(ValueError, "Evaluation target"):
            QUALITY_GATE.evaluate_quality_gate(valid_config(), evaluation)

    def test_rejects_missing_metric(self):
        evaluation = valid_evaluation()
        del evaluation["RMSE"]

        with self.assertRaisesRegex(ValueError, "missing metric 'RMSE'"):
            QUALITY_GATE.evaluate_quality_gate(valid_config(), evaluation)

    def test_rejects_non_finite_metric(self):
        evaluation = valid_evaluation()
        evaluation["MAE"] = float("nan")

        with self.assertRaisesRegex(ValueError, "finite number"):
            QUALITY_GATE.evaluate_quality_gate(valid_config(), evaluation)

    def test_rejects_negative_error_metric(self):
        evaluation = valid_evaluation()
        evaluation["RMSE"] = -1.0

        with self.assertRaisesRegex(ValueError, "RMSE cannot be negative"):
            QUALITY_GATE.evaluate_quality_gate(valid_config(), evaluation)

    def test_writes_unique_reports_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_dir = Path(temporary_directory)
            same_path = report_dir / "lab7_quality_gate_fixed.json"
            with patch.object(
                QUALITY_GATE,
                "_new_report_path",
                return_value=same_path,
            ):
                first = QUALITY_GATE.write_report(
                    report_dir,
                    {"status": "PASSED"},
                )
                first_content = first.read_text(encoding="utf-8")
                second = QUALITY_GATE.write_report(
                    report_dir,
                    {"status": "FAILED"},
                )

            self.assertNotEqual(first, second)
            self.assertEqual(first.read_text(encoding="utf-8"), first_content)
            self.assertEqual(
                json.loads(second.read_text(encoding="utf-8"))["status"],
                "FAILED",
            )


if __name__ == "__main__":
    unittest.main()

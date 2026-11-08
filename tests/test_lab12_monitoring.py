from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "mlops_pipeline" / "12_advanced_monitoring.py"
SPEC = importlib.util.spec_from_file_location("lab12_advanced_monitoring", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load monitoring module from {MODULE_PATH}")
MONITORING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITORING)


class AdvancedMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_path = PROJECT_ROOT / "data" / "raw" / "Train.csv"
        cls.test_path = PROJECT_ROOT / "data" / "raw" / "Test.csv"
        cls.train = pd.read_csv(cls.train_path)
        cls.test = pd.read_csv(cls.test_path)
        cls.features = [column for column in cls.train if column != "Item_Outlet_Sales"]

    def test_existing_real_datasets_pass_schema_and_value_validation(self):
        train_report = MONITORING.validate_sales_dataset(
            self.train,
            self.features,
            require_target=True,
        )
        test_report = MONITORING.validate_sales_dataset(
            self.test,
            self.features,
            require_target=False,
        )

        self.assertEqual(train_report["rows"], 8523)
        self.assertEqual(test_report["rows"], 5681)
        self.assertEqual(train_report["duplicate_rows"], 0)
        self.assertFalse(test_report["has_target"])
        self.assertGreater(train_report["missing_value_count"], 0)

    def test_actual_train_to_test_drift_report_covers_numeric_and_categorical(self):
        result = MONITORING.analyze_data_drift(
            self.train.loc[:, self.features],
            self.test.loc[:, self.features],
            self.features,
            psi_threshold=0.2,
        )

        kinds = {item["kind"] for item in result["features"]}
        self.assertIn("numeric", kinds)
        self.assertIn("categorical", kinds)
        self.assertEqual(result["tested_feature_count"], len(self.features) - 1)
        self.assertIsInstance(result["drift_detected"], bool)
        self.assertTrue(
            all(
                item["psi"] is None
                or pd.notna(item["psi"])
                for item in result["features"]
            )
        )

    def test_identical_real_feature_distribution_has_zero_psi(self):
        values = self.train["Item_MRP"].head(200)
        self.assertAlmostEqual(
            MONITORING.population_stability_index(
                values,
                values,
                numeric=True,
            ),
            0.0,
            places=10,
        )

    def test_prediction_distribution_from_saved_real_evaluation_is_stable(self):
        evaluation = pd.read_csv(PROJECT_ROOT / "outputs" / "predictions.csv")
        result = MONITORING.compare_predictions(
            evaluation["predicted"],
            evaluation["predicted"],
            psi_threshold=0.2,
            minimum_sample_size=30,
        )

        self.assertEqual(result["status"], "EVALUATED")
        self.assertFalse(result["drift_detected"])
        self.assertAlmostEqual(result["psi"], 0.0, places=10)

    def test_regression_performance_uses_real_saved_holdout_predictions(self):
        evaluation = pd.read_csv(PROJECT_ROOT / "outputs" / "predictions.csv")
        metrics = MONITORING.regression_metrics(
            evaluation["actual"],
            evaluation["predicted"],
        )
        report = MONITORING.monitor_labeled_performance(
            evaluation["actual"],
            evaluation["predicted"],
            baseline=metrics,
            mae_relative_increase=0.2,
            rmse_relative_increase=0.2,
            r2_absolute_drop=0.05,
        )

        self.assertEqual(set(metrics), {"MAE", "RMSE", "R2"})
        self.assertEqual(report["status"], "NO_DEGRADATION_DETECTED")

    def test_legacy_api_log_counts_batch_members_as_one_request(self):
        result = MONITORING.summarize_api_events(
            PROJECT_ROOT / "logs" / "inference" / "inference_log.jsonl"
        )

        self.assertEqual(result["prediction_count"], 8)
        self.assertEqual(result["request_count"], 6)
        self.assertGreater(result["latency"]["p95_ms"], 0)
        self.assertTrue(result["prediction_trend_by_day"])

    def test_invalid_values_and_duplicate_rows_are_rejected(self):
        invalid = self.train.head(20).copy()
        invalid.loc[invalid.index[0], "Item_MRP"] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            MONITORING.validate_sales_dataset(
                invalid,
                self.features,
                require_target=True,
            )

        duplicate = pd.concat(
            [self.train.head(20), self.train.head(1)],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate rows"):
            MONITORING.validate_sales_dataset(
                duplicate,
                self.features,
                require_target=True,
            )

    def test_dashboard_contains_monitoring_sections(self):
        drift = MONITORING.analyze_data_drift(
            self.train.loc[:, self.features].head(100),
            self.test.loc[:, self.features].head(100),
            self.features,
            psi_threshold=0.2,
        )
        html = MONITORING.render_dashboard(
            {
                "generated_at_utc": "2026-11-06T00:00:00+00:00",
                "model": {"name": "BigMart_Sales_Model", "version": 3},
                "data_drift": drift,
                "prediction_drift": {
                    "status": "EVALUATED",
                    "reference_mean": 1.0,
                    "current_mean": 1.0,
                    "psi": 0.0,
                },
                "concept_drift": {
                    "status": "NOT_AVAILABLE_NO_NEW_LABELS",
                    "interpretation": "Awaiting labels.",
                },
                "historical_validation": {"status": "NOT_AVAILABLE"},
                "api_performance": {
                    "request_count": 0,
                    "failed_request_count": 0,
                    "latency": {
                        "p50_ms": None,
                        "p95_ms": None,
                        "max_ms": None,
                    },
                    "prediction_trend_by_day": [],
                },
            }
        )

        self.assertIn("Feature drift", html)
        self.assertIn("Prediction trends by day", html)
        self.assertIn("NOT_AVAILABLE_NO_NEW_LABELS", html)
        json.dumps({"html": html})


if __name__ == "__main__":
    unittest.main()

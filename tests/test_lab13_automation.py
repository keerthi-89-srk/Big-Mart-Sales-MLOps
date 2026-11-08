
from __future__ import annotations

import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "mlops_pipeline"
    / "13_end_to_end_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location(
    "lab13_end_to_end_pipeline", MODULE_PATH
)

if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load automation module from {MODULE_PATH}")

PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class EndToEndAutomationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_path = PROJECT_ROOT / "data" / "raw" / "Train.csv"
        cls.test_path = PROJECT_ROOT / "data" / "raw" / "Test.csv"

        cls.features = [
            column
            for column in pd.read_csv(cls.train_path, nrows=0).columns
            if column != "Item_Outlet_Sales"
        ]

        cls.deployment_metadata = json.loads(
            (
                PROJECT_ROOT
                / "deployment"
                / "bigmart_sales_model_v3"
                / "model_metadata.json"
            ).read_text(encoding="utf-8")
        )

        cls.gate_config = json.loads(
            (
                PROJECT_ROOT / "config" / "lab7_quality_gate.json"
            ).read_text(encoding="utf-8")
        )

    @staticmethod
    def dispose_mlflow_clients(*clients):
        """Dispose SQLAlchemy engines held by the test clients."""
        engines = {}

        for client in clients:
            if client is None:
                continue

            tracking_client = getattr(client, "_tracking_client", None)
            tracking_store = getattr(tracking_client, "store", None)

            registry_client = getattr(client, "_registry_client", None)
            get_registry_client = getattr(client, "_get_registry_client", None)
            if callable(get_registry_client):
                registry_client = get_registry_client()
            registry_store = getattr(registry_client, "store", None)

            for store in (tracking_store, registry_store):
                engine = getattr(store, "engine", None)
                if engine is not None:
                    engines[id(engine)] = engine

        for engine in engines.values():
            engine.dispose()

    def test_real_candidate_metrics_must_improve_all_production_metrics(self):
        current = self.gate_config["baseline"]["metrics"]
        candidate = self.deployment_metadata["metrics"]

        decision = PIPELINE.compare_to_production(candidate, current)

        self.assertTrue(decision["passed"])
        self.assertTrue(all(decision["metric_checks"].values()))

    def test_identical_model_metrics_are_rejected_as_no_improvement(self):
        metrics = self.deployment_metadata["metrics"]

        decision = PIPELINE.compare_to_production(metrics, metrics)

        self.assertFalse(decision["passed"])
        self.assertFalse(decision["strict_improvement"])

    def test_unlabeled_real_test_data_cannot_enter_retraining(self):
        monitoring = PIPELINE._load_module(
            "lab12_advanced_monitoring",
            PROJECT_ROOT
            / "src"
            / "mlops_pipeline"
            / "12_advanced_monitoring.py",
        )

        incoming = pd.read_csv(self.test_path)

        self.assertNotIn("Item_Outlet_Sales", incoming.columns)

        result = monitoring.validate_sales_dataset(
            incoming,
            self.features,
            require_target=False,
        )
        self.assertFalse(result["has_target"])
        with self.assertRaisesRegex(ValueError, "must contain Item_Outlet_Sales"):
            monitoring.validate_sales_dataset(
                incoming,
                self.features,
                require_target=True,
            )

    def test_ingestion_ledger_blocks_reprocessing_the_same_real_file(self):
        fingerprint = PIPELINE.sha256_file(self.train_path)

        with tempfile.TemporaryDirectory() as temporary_directory:
            ledger_path = Path(temporary_directory) / "ingestion.json"

            ledger = PIPELINE.read_ingestion_ledger(ledger_path)
            PIPELINE.mark_ingested(
                ledger,
                self.train_path,
                fingerprint,
                "CANDIDATE_REJECTED",
            )
            PIPELINE.write_ingestion_ledger(ledger_path, ledger)
            restored = PIPELINE.read_ingestion_ledger(ledger_path)

            self.assertIn(fingerprint, restored["files"])
            with self.assertRaisesRegex(ValueError, "already been ingested"):
                PIPELINE.mark_ingested(
                    restored,
                    self.train_path,
                    fingerprint,
                    "CANDIDATE_REJECTED",
                )

    def test_failed_quality_gate_does_not_change_production_registry(self):
        import mlflow
        from mlflow.tracking import MlflowClient

        source_database = PROJECT_ROOT / "mlflow.db"

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = root / "isolated_mlflow.db"

            source = sqlite3.connect(
                f"file:{source_database.as_posix()}?mode=ro",
                uri=True,
            )
            destination = sqlite3.connect(database)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()

            tracking_uri = f"sqlite:///{database.as_posix()}"
            previous_tracking_uri = mlflow.get_tracking_uri()
            previous_registry_uri = mlflow.get_registry_uri()

            client = None
            try:
                mlflow.set_tracking_uri(tracking_uri)
                client = MlflowClient(
                    tracking_uri=tracking_uri,
                    registry_uri=tracking_uri,
                )

                production_before = client.get_model_version_by_alias(
                    "BigMart_Sales_Model",
                    "production",
                )

                strict_gate = json.loads(json.dumps(self.gate_config))
                strict_gate["quality_gate"]["metrics"]["MAE"]["threshold"] = 0.0
                strict_gate_path = root / "strict_gate.json"
                strict_gate_path.write_text(
                    json.dumps(strict_gate),
                    encoding="utf-8",
                )
                lifecycle_config = {
                    "model_name": "BigMart_Sales_Model",
                    "experiment_name": "Big_Mart_Sales",
                    "tracking_uri": tracking_uri,
                    "production_alias": "production",
                    "quality_gate_config": str(strict_gate_path),
                    "report_directory": str(root / "reports"),
                    "promotion_requires_validation_tags": True,
                }
                lifecycle_config_path = root / "lifecycle.json"
                lifecycle_config_path.write_text(
                    json.dumps(lifecycle_config),
                    encoding="utf-8",
                )

                lifecycle = PIPELINE._load_module(
                    "lab8_model_lifecycle",
                    PROJECT_ROOT
                    / "src"
                    / "mlops_pipeline"
                    / "08_automate_lifecycle.py",
                )
                exit_code, report_path = lifecycle.run_lifecycle(
                    lifecycle_config_path,
                    apply=True,
                )

                production_after = client.get_model_version_by_alias(
                    "BigMart_Sales_Model",
                    "production",
                )
                report = json.loads(report_path.read_text(encoding="utf-8"))

                self.assertEqual(exit_code, 1)
                self.assertEqual(
                    production_after.version,
                    production_before.version,
                )
                self.assertEqual(report["status"], "NO_QUALIFYING_MODEL")

            finally:
                self.dispose_mlflow_clients(client)
                mlflow.set_tracking_uri(previous_tracking_uri)
                mlflow.set_registry_uri(previous_registry_uri)

    def test_successful_promotion_is_confined_to_an_isolated_registry_copy(self):
        import mlflow
        from mlflow.tracking import MlflowClient

        source_database = PROJECT_ROOT / "mlflow.db"

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            database = root / "isolated_mlflow.db"

            source = sqlite3.connect(
                f"file:{source_database.as_posix()}?mode=ro",
                uri=True,
            )
            destination = sqlite3.connect(database)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()

            tracking_uri = f"sqlite:///{database.as_posix()}"
            previous_tracking_uri = mlflow.get_tracking_uri()
            previous_registry_uri = mlflow.get_registry_uri()

            client = None
            original_client = None

            try:
                mlflow.set_tracking_uri(tracking_uri)
                client = MlflowClient(
                    tracking_uri=tracking_uri,
                    registry_uri=tracking_uri,
                )

                original_client = MlflowClient(
                    tracking_uri=f"sqlite:///{source_database.as_posix()}",
                    registry_uri=f"sqlite:///{source_database.as_posix()}",
                )

                original_version = original_client.get_model_version_by_alias(
                    "BigMart_Sales_Model",
                    "production",
                )
                lifecycle_config = {
                    "model_name": "BigMart_Sales_Model",
                    "experiment_name": "Big_Mart_Sales",
                    "tracking_uri": tracking_uri,
                    "production_alias": "production",
                    "quality_gate_config": str(
                        PROJECT_ROOT / "config" / "lab7_quality_gate.json"
                    ),
                    "report_directory": str(root / "reports"),
                    "promotion_requires_validation_tags": True,
                }
                lifecycle_config_path = root / "lifecycle.json"
                lifecycle_config_path.write_text(
                    json.dumps(lifecycle_config),
                    encoding="utf-8",
                )
                lifecycle = PIPELINE._load_module(
                    "lab8_model_lifecycle",
                    PROJECT_ROOT
                    / "src"
                    / "mlops_pipeline"
                    / "08_automate_lifecycle.py",
                )
                exit_code, report_path = lifecycle.run_lifecycle(
                    lifecycle_config_path,
                    apply=True,
                )

                isolated_version = client.get_model_version_by_alias(
                    "BigMart_Sales_Model",
                    "production",
                )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                original_after = original_client.get_model_version_by_alias(
                    "BigMart_Sales_Model",
                    "production",
                )

                self.assertEqual(exit_code, 0)
                self.assertEqual(report["status"], "PROMOTED")
                self.assertNotEqual(
                    isolated_version.version,
                    original_version.version,
                )
                self.assertEqual(original_after.version, original_version.version)

            finally:
                self.dispose_mlflow_clients(client, original_client)
                mlflow.set_tracking_uri(previous_tracking_uri)
                mlflow.set_registry_uri(previous_registry_uri)


if __name__ == "__main__":
    unittest.main()

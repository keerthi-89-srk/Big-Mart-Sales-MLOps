from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "mlops_pipeline" / "08_automate_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("lab8_model_lifecycle", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load lifecycle module from {MODULE_PATH}")
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


def candidate(version: int, mae: float, rmse: float, r2: float, **overrides):
    value = {
        "model_version": version,
        "run_id": f"run-{version}",
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "eligible": True,
    }
    value.update(overrides)
    return value


class FakeRegistryClient:
    def __init__(self, versions, metrics_by_run):
        self.versions = versions
        self.metrics_by_run = metrics_by_run
        self.production_version = None
        self.alias_changes = []
        self.tag_changes = []

    def search_model_versions(self, filter_string):
        return self.versions

    def get_model_version_by_alias(self, name, alias):
        if self.production_version is None:
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import ErrorCode

            raise MlflowException(
                "Registered model alias does not exist.",
                error_code=ErrorCode.RESOURCE_DOES_NOT_EXIST,
            )
        return self.production_version

    def get_run(self, run_id):
        return SimpleNamespace(
            data=SimpleNamespace(metrics=self.metrics_by_run[run_id])
        )

    def set_registered_model_alias(self, name, alias, version):
        self.alias_changes.append((name, alias, version))

    def set_model_version_tag(self, name, version, key, value):
        self.tag_changes.append((name, version, key, value))


class ModelLifecycleTests(unittest.TestCase):
    def create_lifecycle_config(self, root: Path) -> Path:
        gate_path = root / "gate.json"
        gate_path.write_text(
            json.dumps(
                {
                    "quality_gate": {
                        "metrics": {
                            "MAE": {"operator": "<=", "threshold": 740},
                            "RMSE": {"operator": "<=", "threshold": 1060},
                            "R2": {"operator": ">=", "threshold": 0.59},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        config_path = root / "lifecycle.json"
        config_path.write_text(
            json.dumps(
                {
                    "model_name": "BigMart_Sales_Model",
                    "production_alias": "production",
                    "quality_gate_config": str(gate_path),
                    "report_directory": str(root / "reports"),
                    "promotion_requires_validation_tags": True,
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_candidate_passes_or_fails_using_all_regression_metrics(self):
        gates = {
            "quality_gate": {
                "metrics": {
                    "MAE": {"operator": "<=", "threshold": 740},
                    "RMSE": {"operator": "<=", "threshold": 1060},
                    "R2": {"operator": ">=", "threshold": 0.59},
                }
            }
        }

        passed = LIFECYCLE.evaluate_candidate(
            {"MAE": 720, "RMSE": 1030, "R2": 0.61},
            gates,
        )
        failed = LIFECYCLE.evaluate_candidate(
            {"MAE": 750, "RMSE": 1030, "R2": 0.61},
            gates,
        )

        self.assertTrue(passed["eligible"])
        self.assertFalse(failed["eligible"])

    def test_candidate_with_negative_error_metric_is_rejected(self):
        gates = {
            "quality_gate": {
                "metrics": {
                    "MAE": {"operator": "<=", "threshold": 740},
                    "RMSE": {"operator": "<=", "threshold": 1060},
                    "R2": {"operator": ">=", "threshold": 0.59},
                }
            }
        }

        result = LIFECYCLE.evaluate_candidate(
            {"MAE": -1, "RMSE": 1030, "R2": 0.61},
            gates,
        )

        self.assertFalse(result["eligible"])
        self.assertEqual(
            result["metric_results"]["MAE"]["status"],
            "FAILED",
        )

    def test_selects_best_eligible_candidate_without_downgrade(self):
        current = candidate(1, 737.5, 1053.9, 0.591)
        weaker = candidate(2, 730, 1070, 0.60)
        stronger = candidate(3, 719, 1028, 0.611)

        selected, decision = LIFECYCLE.choose_candidate(
            [current, weaker, stronger],
            current,
        )

        self.assertEqual(selected["model_version"], 3)
        self.assertEqual(decision, "IMPROVEMENT_FOUND")

    def test_reads_production_alias_directly_from_registry(self):
        production = SimpleNamespace(
            version="1",
            run_id="run-1",
            tags={},
            aliases=[],
        )
        client = FakeRegistryClient(
            [production],
            {"run-1": {"MAE": 737, "RMSE": 1053, "R2": 0.592}},
        )
        candidates = LIFECYCLE.collect_candidates(
            client,
            "BigMart_Sales_Model",
            {
                "quality_gate": {
                    "metrics": {
                        "MAE": {"operator": "<=", "threshold": 740},
                        "RMSE": {"operator": "<=", "threshold": 1060},
                        "R2": {"operator": ">=", "threshold": 0.59},
                    }
                }
            },
            require_validation_tags=False,
        )
        client.production_version = production

        current = LIFECYCLE.get_current_production(
            client,
            "BigMart_Sales_Model",
            candidates,
        )

        self.assertEqual(current["model_version"], 1)
        self.assertEqual(current["aliases"], ["production"])

    def test_returns_no_qualifying_candidate_when_gate_fails(self):
        rejected = candidate(2, 800, 1200, 0.4, eligible=False)

        selected, decision = LIFECYCLE.choose_candidate([rejected], None)

        self.assertIsNone(selected)
        self.assertEqual(decision, "NO_QUALIFYING_MODEL")

    def test_dry_run_does_not_change_registry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.create_lifecycle_config(root)
            production = SimpleNamespace(
                version="1",
                run_id="run-1",
                tags={
                    "validation_status": "PASSED",
                    "deployment_readiness": "VALIDATED",
                },
                aliases=[],
            )
            challenger = SimpleNamespace(
                version="2",
                run_id="run-2",
                tags={
                    "validation_status": "PASSED",
                    "deployment_readiness": "VALIDATED",
                },
                aliases=[],
            )
            client = FakeRegistryClient(
                [production, challenger],
                {
                    "run-1": {"MAE": 737, "RMSE": 1053, "R2": 0.592},
                    "run-2": {"MAE": 720, "RMSE": 1030, "R2": 0.61},
                },
            )
            client.production_version = production

            exit_code, report_path = LIFECYCLE.run_lifecycle(
                config_path,
                client=client,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(client.alias_changes, [])
            self.assertEqual(client.tag_changes, [])
            self.assertTrue(report_path.exists())

    def test_apply_changes_only_selected_registry_version(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self.create_lifecycle_config(root)
            production = SimpleNamespace(
                version="1",
                run_id="run-1",
                tags={
                    "validation_status": "PASSED",
                    "deployment_readiness": "VALIDATED",
                },
                aliases=[],
            )
            challenger = SimpleNamespace(
                version="2",
                run_id="run-2",
                tags={
                    "validation_status": "PASSED",
                    "deployment_readiness": "VALIDATED",
                },
                aliases=[],
            )
            client = FakeRegistryClient(
                [production, challenger],
                {
                    "run-1": {"MAE": 737, "RMSE": 1053, "R2": 0.592},
                    "run-2": {"MAE": 720, "RMSE": 1030, "R2": 0.61},
                },
            )
            client.production_version = production

            exit_code, _ = LIFECYCLE.run_lifecycle(
                config_path,
                apply=True,
                client=client,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                client.alias_changes,
                [("BigMart_Sales_Model", "production", "2")],
            )
            self.assertEqual(
                client.tag_changes,
                [
                    (
                        "BigMart_Sales_Model",
                        "2",
                        "lifecycle_state",
                        "PRODUCTION",
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()

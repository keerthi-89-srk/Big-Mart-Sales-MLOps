from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MONITORING = _load_module(
    "lab12_advanced_monitoring",
    "src/mlops_pipeline/12_advanced_monitoring.py",
)
PIPELINE = _load_module(
    "lab13_end_to_end_pipeline",
    "src/mlops_pipeline/13_end_to_end_pipeline.py",
)
FAILURE_RECOVERY = _load_module(
    "lab9_failure_recovery",
    "src/mlops_pipeline/09_failure_recovery.py",
)
API_BUNDLE = _load_module(
    "lab10_prepare_api_bundle",
    "src/mlops_pipeline/10_prepare_api_bundle.py",
)


@pytest.mark.parametrize(
    "module",
    [FAILURE_RECOVERY, API_BUNDLE],
)
def test_mlflow_cleanup_disposes_tracking_and_registry_engines_once(module):
    tracking_engine = SimpleNamespace(dispose=Mock())
    registry_engine = SimpleNamespace(dispose=Mock())
    registry_component = SimpleNamespace(
        store=SimpleNamespace(engine=registry_engine),
    )
    client = SimpleNamespace(
        _tracking_client=SimpleNamespace(
            store=SimpleNamespace(engine=tracking_engine),
        ),
        _registry_client=None,
        _get_registry_client=lambda: registry_component,
    )

    module._dispose_mlflow_client(client)

    tracking_engine.dispose.assert_called_once_with()
    registry_engine.dispose.assert_called_once_with()


@pytest.mark.parametrize(
    "module",
    [FAILURE_RECOVERY, API_BUNDLE],
)
def test_registered_model_runs_uri_uses_owned_client_for_artifact_download(
    module,
    tmp_path,
):
    downloaded_model = tmp_path / "downloaded_model"
    client = SimpleNamespace(
        get_model_version_download_uri=Mock(
            return_value="runs:/run-123/model",
        ),
        download_artifacts=Mock(return_value=str(downloaded_model)),
    )
    loader = SimpleNamespace(load_model=Mock(return_value=object()))
    mlflow = SimpleNamespace(sklearn=loader)

    model = module._load_registered_model(
        mlflow,
        client,
        "BigMart_Sales_Model",
        3,
        "run-123",
    )

    assert model is loader.load_model.return_value
    client.download_artifacts.assert_called_once()
    assert client.download_artifacts.call_args.args[:2] == (
        "run-123",
        "model",
    )
    loader.load_model.assert_called_once_with(str(downloaded_model))


def test_lab12_historical_validation_uses_preprocessor_output_paths():
    config = json.loads(
        (PROJECT_ROOT / "config" / "lab12_monitoring.json").read_text(
            encoding="utf-8",
        )
    )

    assert config["validation_features"] == "data/processed/X_train_processed.csv"
    assert config["validation_targets"] == "data/processed/y_train_processed.csv"


def test_missing_lab11_deployment_dependency_is_reported_as_skipped(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(PIPELINE, "PROJECT_ROOT", tmp_path)

    result = PIPELINE._deploy_promoted_model(tmp_path / "config.json")

    assert result["status"] == "SKIPPED_MISSING_DEPENDENCY"
    assert "deploy_lab11.ps1" in result["reason"]


def test_psi_is_zero_for_identical_values_and_detects_a_distribution_shift():
    reference = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    shifted = pd.Series([30.0, 31.0, 32.0, 33.0, 34.0, 35.0])

    assert MONITORING.population_stability_index(
        reference,
        reference,
        numeric=True,
    ) == pytest.approx(0.0)
    assert MONITORING.population_stability_index(
        reference,
        shifted,
        numeric=True,
    ) > 0.2


def test_categorical_psi_detects_new_categories():
    reference = pd.Series(["Small", "Medium", "Large", "Small"])
    current = pd.Series(["New", "New", "Medium", "New"])

    assert MONITORING.population_stability_index(
        reference,
        current,
        numeric=False,
    ) > 0.2


def test_validation_rejects_missing_target_for_retraining():
    frame = pd.DataFrame(
        {
            "Item_MRP": [1.0, 2.0],
            "Item_Outlet_Sales": [10.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="must contain Item_Outlet_Sales"):
        MONITORING.validate_sales_dataset(
            frame.drop(columns=["Item_Outlet_Sales"]),
            ["Item_MRP"],
            require_target=True,
        )


def test_production_gate_rejects_a_candidate_that_does_not_improve():
    metrics = {"MAE": 7.0, "RMSE": 9.0, "R2": 0.6}

    result = PIPELINE.compare_to_production(metrics, metrics)

    assert result["passed"] is False
    assert result["strict_improvement"] is False

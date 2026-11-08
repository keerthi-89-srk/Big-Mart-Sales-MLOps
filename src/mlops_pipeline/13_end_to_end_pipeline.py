from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab13_automation.json"
TARGET_COLUMN = "Item_Outlet_Sales"
METRIC_NAMES = ("MAE", "RMSE", "R2")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ingestion_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "files": {}}
    value = _read_json(path)
    entries = value.get("files")
    if value.get("version") != 1 or not isinstance(entries, dict):
        raise ValueError(f"Ingestion ledger has an invalid structure: {path}")
    return value


def write_ingestion_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            json.dump(ledger, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def mark_ingested(
    ledger: dict[str, Any],
    path: Path,
    fingerprint: str,
    outcome: str,
    report_path: str | None = None,
) -> None:
    if fingerprint in ledger["files"]:
        raise ValueError(f"Dataset has already been ingested: {path.name}")
    entry: dict[str, Any] = {
        "file_name": path.name,
        "sha256": fingerprint,
        "processed_at_utc": _timestamp(),
        "outcome": outcome,
    }
    if report_path:
        entry["report_path"] = report_path
    ledger["files"][fingerprint] = entry


def _dispose_mlflow_client(client: Any) -> None:
    engines: dict[int, Any] = {}
    components = [
        getattr(client, "_tracking_client", None),
        getattr(client, "_registry_client", None),
    ]
    get_registry_client = getattr(client, "_get_registry_client", None)
    if callable(get_registry_client):
        components.append(get_registry_client())
    for component in components:
        store = getattr(component, "store", None)
        engine = getattr(store, "engine", None)
        if engine is not None:
            engines[id(engine)] = engine
    for engine in engines.values():
        engine.dispose()


def compare_to_production(
    candidate: dict[str, Any],
    production: dict[str, Any],
    *,
    minimum_improvement: float = 0.0,
) -> dict[str, Any]:
    if minimum_improvement < 0:
        raise ValueError("minimum_improvement cannot be negative.")
    for metric in METRIC_NAMES:
        if metric not in candidate or metric not in production:
            raise ValueError(f"Both models must include the {metric} metric.")
        values = (float(candidate[metric]), float(production[metric]))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{metric} metrics must be finite.")
        if metric in {"MAE", "RMSE"} and min(values) < 0:
            raise ValueError(f"{metric} cannot be negative.")

    checks = {
        "MAE": candidate["MAE"] <= production["MAE"] * (1 - minimum_improvement),
        "RMSE": candidate["RMSE"]
        <= production["RMSE"] * (1 - minimum_improvement),
        "R2": candidate["R2"] >= production["R2"] + minimum_improvement,
    }
    strictly_better = (
        candidate["MAE"] < production["MAE"]
        or candidate["RMSE"] < production["RMSE"]
        or candidate["R2"] > production["R2"]
    )
    return {
        "passed": all(checks.values()) and strictly_better,
        "metric_checks": checks,
        "strict_improvement": strictly_better,
        "minimum_improvement": minimum_improvement,
    }


def _metric_values(actual: Any, predicted: Any) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_values.size < 2 or actual_values.size != predicted_values.size:
        raise ValueError("Candidate evaluation needs at least two matching rows.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Candidate evaluation contains non-finite sales values.")
    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(math.sqrt(mean_squared_error(actual_values, predicted_values))),
        "R2": float(r2_score(actual_values, predicted_values)),
    }


def _prepare_candidate_data(
    reference_path: Path,
    incoming_frames: list[pd.DataFrame],
    feature_columns: list[str],
    *,
    max_missing_rate: float,
) -> tuple[pd.DataFrame, pd.Series, int]:
    monitoring = _load_module(
        "lab12_advanced_monitoring",
        PROJECT_ROOT / "src" / "mlops_pipeline" / "12_advanced_monitoring.py",
    )
    reference = pd.read_csv(reference_path)
    monitoring.validate_sales_dataset(
        reference,
        feature_columns,
        require_target=True,
        max_missing_rate=max_missing_rate,
    )
    for frame in incoming_frames:
        monitoring.validate_sales_dataset(
            frame,
            feature_columns,
            require_target=True,
            max_missing_rate=max_missing_rate,
        )

    combined = pd.concat([reference, *incoming_frames], ignore_index=True)
    duplicate_count = int(combined.duplicated().sum())
    if duplicate_count:
        combined = combined.drop_duplicates(ignore_index=True)
    target = pd.to_numeric(combined[TARGET_COLUMN], errors="raise")
    return combined.loc[:, feature_columns], target, duplicate_count


def _get_production(
    client: Any,
    model_name: str,
) -> tuple[Any, int]:
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import ErrorCode

    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except MlflowException as exc:
        missing_code = str(ErrorCode.RESOURCE_DOES_NOT_EXIST)
        if str(exc.error_code) not in {missing_code, "RESOURCE_DOES_NOT_EXIST"}:
            raise
        raise ValueError(
            f"Registered model {model_name!r} has no Production alias."
        ) from exc
    if not version.run_id:
        raise ValueError("Production model version has no source run ID.")
    return version, int(version.version)


def _quality_gate(
    metrics: dict[str, float],
    gate_config_path: Path,
) -> dict[str, Any]:
    gate_module = _load_module(
        "lab7_quality_gate",
        PROJECT_ROOT / "src" / "mlops_pipeline" / "07_quality_gate.py",
    )
    config = gate_module._read_json(gate_config_path)
    evaluation = {
        "target_column": TARGET_COLUMN,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2_Score": metrics["R2"],
    }
    result = gate_module.evaluate_quality_gate(config, evaluation)
    return {
        "status": result["status"],
        "metric_results": result["metric_results"],
    }


def _prepare_deployment_bundle(
    model: Any,
    preprocessor: Any,
    config: dict[str, Any],
    *,
    model_name: str,
    model_version: int,
    run_id: str,
    metrics: dict[str, float],
) -> tuple[Path, Path]:
    import joblib

    bundle_name = f"bigmart_sales_model_v{model_version}"
    bundle_path = _resolve_path(config["deployment_directory"]) / bundle_name
    config_path = PROJECT_ROOT / "config" / f"lab10_api_v{model_version}.json"
    if bundle_path.exists() or config_path.exists():
        raise FileExistsError(
            "Versioned deployment bundle/configuration already exists; "
            "refusing to overwrite."
        )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{bundle_name}_",
        dir=bundle_path.parent,
    ) as temporary:
        staging = Path(temporary)
        model_path = staging / "model.joblib"
        preprocessor_path = staging / "preprocessor.json"
        joblib.dump(model, model_path)
        preprocessor_path.write_text(
            json.dumps(preprocessor.to_metadata(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "lab": 13,
            "model_name": model_name,
            "model_version": model_version,
            "run_id": run_id,
            "model_uri": f"models:/{model_name}/{model_version}",
            "target_column": TARGET_COLUMN,
            "feature_count": len(preprocessor.feature_names),
            "metrics": metrics,
            "generated_at_utc": _timestamp(),
            "sha256": {
                "model.joblib": sha256_file(model_path),
                "preprocessor.json": sha256_file(preprocessor_path),
            },
        }
        (staging / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, bundle_path)

    api_config = dict(config["api_config"])
    api_config["model_name"] = model_name
    api_config["model_version"] = model_version
    api_config["bundle_directory"] = str(
        bundle_path.relative_to(PROJECT_ROOT).as_posix()
    )
    with config_path.open("x", encoding="utf-8") as file:
        json.dump(api_config, file, indent=2)
        file.write("\n")
    return bundle_path, config_path


def _validate_promoted_api(config_path: Path, reference_path: Path) -> dict[str, Any]:
    from api.model_loader import ModelLoader

    loader = ModelLoader(config_path=config_path)
    loader.load()
    if loader.model is None or loader.preprocessor is None:
        raise RuntimeError("Promoted API bundle did not load.")
    sample = pd.read_csv(reference_path, nrows=1).drop(columns=[TARGET_COLUMN])
    features = loader.preprocessor.transform(sample.to_dict(orient="records"))
    predictions = np.asarray(loader.model.predict(features), dtype=float).reshape(-1)
    if predictions.size != 1 or not np.isfinite(predictions).all():
        raise ValueError("Promoted API smoke check did not return finite sales.")
    return {
        "status": "PASSED",
        "model_version": loader.model_version,
        "sample_prediction": float(predictions[0]),
    }


def _deploy_promoted_model(config_path: Path) -> dict[str, Any]:
    script = PROJECT_ROOT / "deploy_lab11.ps1"
    if not script.is_file():
        return {
            "status": "SKIPPED_MISSING_DEPENDENCY",
            "reason": (
                "The optional Lab 11 deployment script is not included in this "
                f"upload: {script.name}"
            ),
        }
    relative_config = config_path.relative_to(PROJECT_ROOT)
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-File",
            str(script),
            "-Apply",
            "-ExpectedContext",
            "docker-desktop",
            "-ApiConfig",
            str(relative_config),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    deployment = {
        "status": "PASSED" if result.returncode == 0 else "FAILED",
        "return_code": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }
    if result.returncode:
        raise RuntimeError(
            "Promoted model deployment failed after the Production alias changed. "
            f"See deployment output: {result.stderr[-1000:]}"
        )
    return deployment


def _write_report(report: dict[str, Any], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"lab13_execution_{_suffix()}.json"
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    return path


def _run_pipeline(
    config_path: Path,
    *,
    input_paths: list[Path] | None = None,
    monitoring_input: Path | None = None,
    apply: bool = False,
    deploy: bool = False,
    mlflow_clients: list[Any],
) -> tuple[dict[str, Any], Path]:
    if deploy and not apply:
        raise ValueError("Deployment requires explicit --apply promotion.")

    config = _read_json(config_path)
    lab12_config_path = _resolve_path(config["monitoring_config"])
    monitoring_config = _read_json(lab12_config_path)
    reference_path = _resolve_path(config["reference_train"])
    incoming_directory = _resolve_path(config["incoming_directory"])
    ledger_path = _resolve_path(config["ingestion_ledger"])
    report_directory = _resolve_path(config["report_directory"])
    max_missing_rate = float(config["max_missing_rate"])

    from api.model_loader import ModelLoader

    api_loader = ModelLoader()
    api_loader.load()
    if api_loader.preprocessor is None:
        raise RuntimeError("The configured Big Mart preprocessor did not load.")
    feature_columns = list(api_loader.preprocessor.raw_columns)

    if input_paths is None:
        files = sorted(incoming_directory.glob("*.csv")) if incoming_directory.is_dir() else []
    else:
        files = [path.resolve() for path in input_paths]

    observed_input = monitoring_input
    if observed_input is None:
        observed_input = (
            files[0]
            if files
            else _resolve_path(monitoring_config["incoming_data"])
        )
    monitoring = _load_module(
        "lab12_advanced_monitoring",
        PROJECT_ROOT / "src" / "mlops_pipeline" / "12_advanced_monitoring.py",
    )
    monitoring_report = monitoring.build_monitoring_report(
        lab12_config_path,
        incoming_path=observed_input,
    )

    report: dict[str, Any] = {
        "lab": 13,
        "generated_at_utc": _timestamp(),
        "mode": "APPLY" if apply else "DRY_RUN",
        "production_model_name": config["model_name"],
        "monitoring_input": str(observed_input),
        "monitoring_summary": {
            "data_drift_detected": monitoring_report["data_drift"]["drift_detected"],
            "data_drifted_features": monitoring_report["data_drift"][
                "drifted_features"
            ],
            "prediction_drift": monitoring_report["prediction_drift"]["status"],
            "concept_drift": monitoring_report["concept_drift"]["status"],
            "api_request_count": monitoring_report["api_performance"][
                "request_count"
            ],
        },
        "stages": [],
        "incoming_files": [],
        "promotion_applied": False,
        "deployment": {"status": "NOT_REQUESTED"},
    }
    report["stages"].append(
        {
            "stage": "incoming_data_detection",
            "status": "FOUND" if files else "NO_NEW_DATA",
            "file_count": len(files),
        }
    )
    report["stages"].append(
        {
            "stage": "data_validation_and_drift",
            "status": "PASSED",
            "monitoring_report": monitoring_report,
        }
    )

    ledger = read_ingestion_ledger(ledger_path)
    pending: list[tuple[Path, str, pd.DataFrame]] = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Incoming data file does not exist: {path}")
        fingerprint = sha256_file(path)
        item: dict[str, Any] = {
            "path": str(path),
            "sha256": fingerprint,
            "status": "VALIDATING",
        }
        report["incoming_files"].append(item)
        if fingerprint in ledger["files"]:
            item["status"] = "DUPLICATE_INGESTION_SKIPPED"
            item["previous_outcome"] = ledger["files"][fingerprint]["outcome"]
            continue
        frame = pd.read_csv(path)
        try:
            validation = monitoring.validate_sales_dataset(
                frame,
                feature_columns,
                require_target=False,
                max_missing_rate=max_missing_rate,
            )
        except (ValueError, TypeError) as exc:
            item["status"] = "REJECTED_INVALID_DATA"
            item["error"] = str(exc)
            continue
        item["validation"] = validation
        if not validation["has_target"]:
            item["status"] = "SKIPPED_UNLABELED"
            continue
        item["status"] = "VALIDATED_LABELED"
        pending.append((path, fingerprint, frame))

    if not pending:
        report["retraining"] = {
            "status": "SKIPPED_NO_NEW_LABELED_DATA",
            "reason": (
                "No new, validated, labeled sales rows were found. Existing "
                "unlabeled test data was monitored but not ingested for training."
            ),
        }
        report["quality_gate"] = {"status": "NOT_RUN"}
        report["production_comparison"] = {"status": "NOT_RUN"}
        report["promotion"] = {
            "status": "NOT_RUN",
            "reason": "A candidate cannot be trained without new labeled data.",
        }
        report["deployment"] = {
            "status": "NOT_RUN",
            "reason": "No model was promoted.",
        }
        report["status"] = "SKIPPED_NO_NEW_LABELED_DATA"
        report_path = _write_report(report, report_directory)
        print(f"[INFO] No new labeled data; retraining was skipped.")
        print(f"[INFO] Lab 13 report: {report_path}")
        return report, report_path

    from api.model_loader import ModelLoader

    X, y, cross_dataset_duplicates = _prepare_candidate_data(
        reference_path,
        [item[2] for item in pending],
        feature_columns,
        max_missing_rate=max_missing_rate,
    )
    api_loader = ModelLoader()
    api_loader.load()
    if api_loader.preprocessor is None:
        raise RuntimeError("The configured Big Mart preprocessor did not load.")
    feature_frame = api_loader.preprocessor.transform(
        X.to_dict(orient="records")
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        feature_frame,
        y,
        test_size=float(config["validation_fraction"]),
        random_state=int(config["random_state"]),
    )
    candidate_model = RandomForestRegressor(
        n_estimators=int(config["n_estimators"]),
        max_depth=(
            int(config["max_depth"])
            if config.get("max_depth") is not None
            else None
        ),
        min_samples_split=int(config["min_samples_split"]),
        random_state=int(config["random_state"]),
        n_jobs=1,
    )

    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    mlflow_clients.append(client)
    mlflow.set_experiment(config["experiment_name"])
    candidate_model.fit(X_train, y_train)
    candidate_metrics = _metric_values(
        y_validation,
        candidate_model.predict(X_validation),
    )
    production_version, current_version_number = _get_production(
        client,
        config["model_name"],
    )
    production_model = mlflow.sklearn.load_model(
        f"models:/{config['model_name']}/{current_version_number}"
    )
    production_feature_count = getattr(production_model, "n_features_in_", None)
    if (
        production_feature_count is not None
        and int(production_feature_count) != X_validation.shape[1]
    ):
        raise ValueError(
            "Production model feature count differs from the candidate pipeline."
        )
    production_feature_names = getattr(
        production_model,
        "feature_names_in_",
        None,
    )
    if production_feature_names is not None and list(
        production_feature_names
    ) != X_validation.columns.tolist():
        raise ValueError(
            "Production model feature names differ from the candidate pipeline."
        )
    production_metrics = _metric_values(
        y_validation,
        production_model.predict(X_validation),
    )
    comparison = compare_to_production(
        candidate_metrics,
        production_metrics,
        minimum_improvement=float(config["minimum_improvement"]),
    )
    quality_gate = _quality_gate(
        candidate_metrics,
        _resolve_path(config["quality_gate_config"]),
    )

    with mlflow.start_run(run_name="Lab13_Random_Forest_Candidate") as run:
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "target_column": TARGET_COLUMN,
                "n_estimators": int(config["n_estimators"]),
                "max_depth": config.get("max_depth") or "None",
                "min_samples_split": int(config["min_samples_split"]),
                "random_state": int(config["random_state"]),
                "reference_rows": int(len(pd.read_csv(reference_path))),
                "new_labeled_rows": int(sum(len(item[2]) for item in pending)),
                "training_rows_after_deduplication": int(len(X)),
                "validation_rows": int(len(X_validation)),
                "feature_count": int(X.shape[1]),
                "production_model_version": current_version_number,
            }
        )
        mlflow.log_metrics(candidate_metrics)
        mlflow.sklearn.log_model(
            sk_model=candidate_model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        mlflow.log_dict(
            {
                "candidate_metrics": candidate_metrics,
                "production_metrics": production_metrics,
                "production_comparison": comparison,
                "quality_gate": quality_gate,
                "target_column": TARGET_COLUMN,
            },
            "candidate_evaluation.json",
        )
        run_id = run.info.run_id

    report["retraining"] = {
        "status": "COMPLETED",
        "run_id": run_id,
        "model_type": "RandomForestRegressor",
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "feature_count": int(X.shape[1]),
        "cross_dataset_duplicate_rows_removed": cross_dataset_duplicates,
    }
    report["candidate_metrics"] = candidate_metrics
    report["production_metrics"] = production_metrics
    report["production_comparison"] = {
        "current_version": current_version_number,
        **comparison,
    }
    report["quality_gate"] = quality_gate

    can_register = (
        quality_gate["status"] == "PASSED" and comparison["passed"]
    )
    if not can_register:
        report["promotion"] = {
            "status": "REJECTED",
            "reason": (
                "Candidate failed the configured quality gate or did not "
                "strictly improve every Production metric."
            ),
            "production_alias_changed": False,
        }
        report["deployment"] = {
            "status": "NOT_RUN",
            "reason": "Rejected candidates are not deployable.",
        }
        report["status"] = "CANDIDATE_REJECTED"
        for path, fingerprint, _ in pending:
            mark_ingested(
                ledger,
                path,
                fingerprint,
                "CANDIDATE_REJECTED",
            )
        write_ingestion_ledger(ledger_path, ledger)
        report_path = _write_report(report, report_directory)
        print(f"[INFO] Candidate rejected; Production remains version {current_version_number}.")
        print(f"[INFO] Lab 13 report: {report_path}")
        return report, report_path

    registration = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=config["model_name"],
    )
    candidate_version = int(registration.version)
    client.set_model_version_tag(
        config["model_name"],
        str(candidate_version),
        "validation_status",
        "PASSED",
    )
    client.set_model_version_tag(
        config["model_name"],
        str(candidate_version),
        "deployment_readiness",
        "VALIDATED",
    )
    client.set_model_version_tag(
        config["model_name"],
        str(candidate_version),
        "lifecycle_state",
        "VALIDATION",
    )
    report["candidate_version"] = candidate_version
    report["promotion"] = {
        "status": "DRY_RUN_CANDIDATE_READY",
        "production_alias_changed": False,
        "candidate_version": candidate_version,
    }
    report["deployment"] = {
        "status": "NOT_RUN",
        "reason": "Promotion requires explicit --apply.",
    }
    report["status"] = "CANDIDATE_READY"

    if apply:
        client.set_registered_model_alias(
            config["model_name"],
            "production",
            str(candidate_version),
        )
        client.set_model_version_tag(
            config["model_name"],
            str(candidate_version),
            "lifecycle_state",
            "PRODUCTION",
        )
        bundle_path, api_config_path = _prepare_deployment_bundle(
            candidate_model,
            api_loader.preprocessor,
            config,
            model_name=config["model_name"],
            model_version=candidate_version,
            run_id=run_id,
            metrics=candidate_metrics,
        )
        api_validation = _validate_promoted_api(api_config_path, reference_path)
        report["promotion"] = {
            "status": "PROMOTED",
            "production_alias_changed": True,
            "previous_version": current_version_number,
            "candidate_version": candidate_version,
        }
        report["deployment"] = {
            "status": "READY_FOR_DEPLOYMENT",
            "bundle_directory": str(bundle_path),
            "api_config": str(api_config_path),
            "api_validation": api_validation,
            "deployment_requested": deploy,
        }
        report["promotion_applied"] = True
        report["status"] = "PROMOTED"
        if deploy:
            execution = _deploy_promoted_model(api_config_path)
            report["deployment"]["execution"] = execution
            report["deployment"]["status"] = execution["status"]
            if execution["status"] == "SKIPPED_MISSING_DEPENDENCY":
                print(f"[WARNING] Deployment skipped: {execution['reason']}")

    for path, fingerprint, _ in pending:
        mark_ingested(
            ledger,
            path,
            fingerprint,
            report["status"],
        )
    write_ingestion_ledger(ledger_path, ledger)
    report_path = _write_report(report, report_directory)
    print(f"[INFO] Candidate status: {report['status']}")
    print(f"[INFO] Quality gate: {quality_gate['status']}")
    print(f"[INFO] Production comparison: {comparison['passed']}")
    print(f"[INFO] Lab 13 report: {report_path}")
    return report, report_path


def run_pipeline(
    config_path: Path,
    *,
    input_paths: list[Path] | None = None,
    monitoring_input: Path | None = None,
    apply: bool = False,
    deploy: bool = False,
) -> tuple[dict[str, Any], Path]:
    mlflow_clients: list[Any] = []
    try:
        return _run_pipeline(
            config_path,
            input_paths=input_paths,
            monitoring_input=monitoring_input,
            apply=apply,
            deploy=deploy,
            mlflow_clients=mlflow_clients,
        )
    finally:
        for client in mlflow_clients:
            _dispose_mlflow_client(client)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Big Mart data, monitor drift, and run gated retraining."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Labeled incoming CSV; may be supplied more than once.",
    )
    parser.add_argument("--monitoring-input", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Promote only a candidate that passes gates and improves Production.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy a successfully promoted model to the explicitly gated local cluster.",
    )
    args = parser.parse_args(argv)
    try:
        report, _ = run_pipeline(
            _resolve_path(args.config),
            input_paths=(
                [_resolve_path(path) for path in args.input]
                if args.input
                else None
            ),
            monitoring_input=(
                _resolve_path(args.monitoring_input)
                if args.monitoring_input
                else None
            ),
            apply=args.apply,
            deploy=args.deploy,
        )
        return (
            0
            if report["status"]
            in {
                "SKIPPED_NO_NEW_LABELED_DATA",
                "CANDIDATE_READY",
                "CANDIDATE_REJECTED",
                "PROMOTED",
            }
            else 1
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
    ) as exc:
        print(f"[ERROR] Lab 13 pipeline failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

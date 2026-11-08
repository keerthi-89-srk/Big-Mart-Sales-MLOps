from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab10_api.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.preprocessing import BigMartPreprocessor, TARGET_COLUMN


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_names(model: Any, processed_path: Path) -> list[str]:
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        names = [str(value) for value in model_features]
    else:
        if not processed_path.is_file():
            raise FileNotFoundError(
                "Model has no feature_names_in_; processed Big Mart features "
                f"are required to verify its input schema: {processed_path}"
            )
        names = pd.read_csv(processed_path, nrows=0).columns.tolist()
    if not names or len(set(names)) != len(names):
        raise ValueError("Model input feature names are missing or duplicated.")
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and int(expected_count) != len(names):
        raise ValueError(
            f"Model expects {expected_count} features, but {len(names)} names "
            "were found."
        )
    return names


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


def _load_registered_model(
    mlflow: Any,
    client: Any,
    model_name: str,
    model_version: int,
    run_id: str,
) -> Any:
    download_uri = client.get_model_version_download_uri(
        model_name,
        str(model_version),
    )
    if download_uri.startswith("runs:/"):
        run_prefix = f"runs:/{run_id}/"
        if not download_uri.startswith(run_prefix):
            raise ValueError(
                "Registered model download URI does not match its source run."
            )
        artifact_path = download_uri[len(run_prefix):]
        if not artifact_path:
            raise ValueError("Registered model download URI has no artifact path.")
        with tempfile.TemporaryDirectory(prefix="bigmart_lab10_model_") as directory:
            model_path = client.download_artifacts(
                run_id,
                artifact_path,
                directory,
            )
            return mlflow.sklearn.load_model(model_path)
    return mlflow.sklearn.load_model(download_uri)


def prepare_api_bundle(config_path: Path) -> Path:
    config = _read_config(config_path)
    model_name = config.get("model_name")
    model_version = config.get("model_version")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("tracking_uri")
    bundle_setting = config.get("bundle_directory")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Configuration must define model_name.")
    if not isinstance(model_version, int) or model_version < 1:
        raise ValueError("Configuration must define a positive model_version.")
    if not isinstance(tracking_uri, str) or not tracking_uri:
        raise ValueError("Configuration must define tracking_uri.")
    if not isinstance(bundle_setting, str) or not bundle_setting:
        raise ValueError("Configuration must define bundle_directory.")

    bundle_path = _resolve_path(PROJECT_ROOT, bundle_setting).resolve()
    try:
        bundle_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Bundle directory must be inside the project.") from exc
    if bundle_path.exists():
        raise FileExistsError(
            f"API bundle already exists and will not be overwritten: {bundle_path}"
        )

    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    try:
        return _prepare_api_bundle(
            config,
            bundle_path,
            model_name,
            model_version,
            mlflow,
            client,
        )
    finally:
        _dispose_mlflow_client(client)


def _prepare_api_bundle(
    config: dict[str, Any],
    bundle_path: Path,
    model_name: str,
    model_version: int,
    mlflow: Any,
    client: Any,
) -> Path:
    import mlflow.sklearn

    version = client.get_model_version(model_name, str(model_version))
    if version.status != "READY":
        raise RuntimeError(
            f"Registered model version is not ready: {version.status}"
        )
    if not version.run_id:
        raise ValueError("Registered model version has no source run ID.")

    model_uri = f"models:/{model_name}/{model_version}"
    model = _load_registered_model(
        mlflow,
        client,
        model_name,
        model_version,
        version.run_id,
    )
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("Registered model does not provide predict().")

    train_path = _resolve_path(
        PROJECT_ROOT,
        config.get("train_data", "data/raw/Train.csv"),
    )
    test_path = _resolve_path(
        PROJECT_ROOT,
        config.get("test_data", "data/raw/Test.csv"),
    )
    processed_path = _resolve_path(
        PROJECT_ROOT,
        config.get("processed_features", "data/processed/X_train_processed.csv"),
    )
    feature_names = _feature_names(model, processed_path)
    preprocessor = BigMartPreprocessor.fit_from_files(
        train_path,
        test_path,
        feature_names,
    )

    train_sample = pd.read_csv(train_path, nrows=1)
    if train_sample.empty or TARGET_COLUMN not in train_sample.columns:
        raise ValueError("Cannot verify the model against a Big Mart sample.")
    transformed = preprocessor.transform(
        train_sample.drop(columns=[TARGET_COLUMN]).to_dict(orient="records")
    )
    if processed_path.is_file():
        reference = pd.read_csv(processed_path, nrows=1)
        if reference.columns.tolist() != feature_names:
            raise ValueError(
                "Saved Big Mart feature columns differ from the registered model."
            )
        if not np.allclose(
            transformed.to_numpy(dtype=float),
            reference.to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                "API preprocessing does not reproduce the saved Big Mart "
                "training features for the verification row."
            )
    prediction = float(np.asarray(predict(transformed)).reshape(-1)[0])
    if not math.isfinite(prediction):
        raise ValueError("Registered model returned a non-finite sales prediction.")

    run = client.get_run(version.run_id)
    source = getattr(version, "source", "")
    model_id = source.rsplit("/", 1)[-1] if source else None
    parent = bundle_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".lab10_bundle_stage_",
        dir=parent,
    ) as stage_directory:
        stage = Path(stage_directory)
        model_path = stage / "model.joblib"
        preprocessor_path = stage / "preprocessor.json"
        import joblib

        joblib.dump(model, model_path)
        preprocessor_path.write_text(
            json.dumps(preprocessor.to_metadata(), indent=2) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "lab": 10,
            "model_name": model_name,
            "model_version": int(model_version),
            "model_id": model_id,
            "run_id": version.run_id,
            "model_uri": model_uri,
            "target_column": TARGET_COLUMN,
            "feature_count": len(feature_names),
            "metrics": {
                name: float(value)
                for name, value in run.data.metrics.items()
                if math.isfinite(float(value))
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verification": {
                "test_rows": 1,
                "finite_sales_prediction": prediction,
                "preprocessing_matches_saved_features": processed_path.is_file(),
            },
            "sha256": {
                "model.joblib": _sha256(model_path),
                "preprocessor.json": _sha256(preprocessor_path),
            },
        }
        (stage / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        if bundle_path.exists():
            raise FileExistsError(
                f"API bundle appeared while preparing; no files were changed: "
                f"{bundle_path}"
            )
        stage.rename(bundle_path)

    print(f"[SUCCESS] Prepared {model_name} version {model_version}.")
    print(f"[INFO] Target: {TARGET_COLUMN}")
    print(f"[INFO] Feature count: {len(feature_names)}")
    print(f"[INFO] Sample predicted sales: {prediction:.6f}")
    print(f"[INFO] Bundle saved to: {bundle_path}")
    print("[INFO] MLflow model versions and aliases were not changed.")
    return bundle_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a verified Big Mart API model bundle from MLflow."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    try:
        prepare_api_bundle(config_path.resolve())
        return 0
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(
            f"[ERROR] Could not prepare the Lab 10 API bundle "
            f"({type(exc).__name__}): {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import contextlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab9_failure_recovery.json"
PROJECT_FILES = (
    Path("src") / "validate_data.py",
    Path("src") / "preprocess.py",
    Path("src") / "validate_outputs.py",
    Path("data") / "raw" / "Train.csv",
    Path("data") / "raw" / "Test.csv",
)


def ensure_sandbox_path(sandbox_root: Path, candidate: Path) -> Path:
    root = sandbox_root.resolve()
    path = candidate.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the temporary sandbox: {path}") from exc
    if not relative.parts:
        raise ValueError("A sandbox path must identify a file or subdirectory.")
    return path


def make_scenario_result(
    name: str,
    expected: str,
    actual: str,
    passed: bool,
    **details: Any,
) -> dict[str, Any]:
    result = {
        "scenario": name,
        "expected": expected,
        "actual": actual,
        "passed": bool(passed),
        "status": "PASSED" if passed else "FAILED",
    }
    if details:
        result["details"] = details
    return result


def build_report(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    all_results = [
        result
        for repetition in repetitions
        for result in repetition.get("results", [])
    ]
    passed = bool(all_results) and all(
        result.get("passed") is True for result in all_results
    )
    return {
        "lab": 9,
        "stage": "Failure Handling and Recovery",
        "repetitions": len(repetitions),
        "runs": repetitions,
        "scenario_count": len(all_results),
        "passed_scenarios": sum(
            result.get("passed") is True for result in all_results
        ),
        "overall_status": "PASSED" if passed else "FAILED",
    }


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


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
        with tempfile.TemporaryDirectory(prefix="bigmart_lab9_model_") as directory:
            model_path = client.download_artifacts(
                run_id,
                artifact_path,
                directory,
            )
            return mlflow.sklearn.load_model(model_path)
    return mlflow.sklearn.load_model(download_uri)


def _copy_sandbox_project(sandbox_root: Path) -> None:
    for relative_path in PROJECT_FILES:
        source = PROJECT_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(
                f"Required Lab 9 fixture is missing: {source}"
            )
        destination = ensure_sandbox_path(
            sandbox_root,
            sandbox_root / relative_path,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    processed_dir = ensure_sandbox_path(
        sandbox_root,
        sandbox_root / "data" / "processed",
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    source_features = PROJECT_ROOT / "data" / "processed" / "X_train_processed.csv"
    if not source_features.is_file():
        raise FileNotFoundError(f"Required model feature fixture is missing: {source_features}")
    with source_features.open("r", encoding="utf-8", newline="") as file:
        rows = csv.reader(file)
        header = next(rows, None)
        first_row = next(rows, None)
    if not header or not first_row:
        raise ValueError(f"Model feature fixture is empty: {source_features}")
    with (processed_dir / "X_train_processed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(first_row)

    compact_features = ["fixture_feature_1", "fixture_feature_2"]
    for file_name, fixture_rows in (
        ("X_train.csv", [["1", "2"], ["3", "4"]]),
        ("X_test.csv", [["5", "6"]]),
        ("y_train.csv", [["100"], ["200"]]),
        ("y_test.csv", [["300"]]),
    ):
        with (processed_dir / file_name).open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Item_Outlet_Sales"] if file_name.startswith("y_") else compact_features
            )
            writer.writerows(fixture_rows)


def _run_script(sandbox_root: Path, relative_script: Path) -> dict[str, Any]:
    script = ensure_sandbox_path(sandbox_root, sandbox_root / relative_script)
    spec = importlib.util.spec_from_file_location(
        f"lab9_sandbox_{script.stem}_{id(script)}",
        script,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sandboxed script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project_paths = {
        "TRAIN_FILE": sandbox_root / "data" / "raw" / "Train.csv",
        "TEST_FILE": sandbox_root / "data" / "raw" / "Test.csv",
        "TRAIN_PATH": sandbox_root / "data" / "raw" / "Train.csv",
        "TEST_PATH": sandbox_root / "data" / "raw" / "Test.csv",
        "PROCESSED_DIR": sandbox_root / "data" / "processed",
        "ARTIFACTS_DIR": sandbox_root / "artifacts",
    }
    for name, path in project_paths.items():
        if hasattr(module, name):
            setattr(module, name, ensure_sandbox_path(sandbox_root, path))

    standard_output = io.StringIO()
    error_output = io.StringIO()
    return_code = 0
    with contextlib.redirect_stdout(standard_output), contextlib.redirect_stderr(
        error_output
    ):
        try:
            module.main()
        except (FileNotFoundError, ValueError, TypeError, KeyError, OSError) as exc:
            return_code = 1
            error_output.write(f"{type(exc).__name__}: {exc}\n")
    return {
        "return_code": return_code,
        "stdout": standard_output.getvalue()[-4000:],
        "stderr": error_output.getvalue()[-4000:],
    }


def _run_isolated_scenario(
    scenario: str,
    inject_and_run: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"bigmart_lab9_{scenario}_") as directory:
        sandbox_root = Path(directory).resolve()
        _copy_sandbox_project(sandbox_root)
        return inject_and_run(sandbox_root)


def _missing_data_scenario() -> dict[str, Any]:
    def run(root: Path) -> dict[str, Any]:
        data_path = ensure_sandbox_path(root, root / "data" / "raw" / "Train.csv")
        data_path.rename(data_path.with_suffix(".csv.hidden"))
        execution = _run_script(root, Path("src") / "validate_data.py")
        halted = execution["return_code"] != 0
        return make_scenario_result(
            "missing_data",
            "HALT",
            "HALT" if halted else "CONTINUED",
            halted,
            execution=execution,
        )

    return _run_isolated_scenario("missing_data", run)


def _invalid_datatype_scenario() -> dict[str, Any]:
    def run(root: Path) -> dict[str, Any]:
        import pandas as pd

        data_path = ensure_sandbox_path(root, root / "data" / "raw" / "Train.csv")
        frame = pd.read_csv(data_path)
        if "Item_Visibility" not in frame.columns or frame.empty:
            raise ValueError("Training data lacks Item_Visibility rows to test.")
        frame["Item_Visibility"] = frame["Item_Visibility"].astype(object)
        frame.loc[frame.index[0], "Item_Visibility"] = "INVALID_NUMBER"
        frame.to_csv(data_path, index=False)
        execution = _run_script(root, Path("src") / "preprocess.py")
        halted = execution["return_code"] != 0
        return make_scenario_result(
            "invalid_datatype",
            "HALT",
            "HALT" if halted else "CONTINUED",
            halted,
            execution=execution,
        )

    return _run_isolated_scenario("invalid_datatype", run)


def _schema_mismatch_scenario() -> dict[str, Any]:
    def run(root: Path) -> dict[str, Any]:
        import pandas as pd

        data_path = ensure_sandbox_path(root, root / "data" / "raw" / "Train.csv")
        frame = pd.read_csv(data_path)
        target = "Item_Outlet_Sales"
        if target not in frame.columns:
            raise ValueError(f"Training fixture lacks target column {target}.")
        frame.drop(columns=[target]).to_csv(data_path, index=False)
        execution = _run_script(root, Path("src") / "validate_data.py")
        halted = execution["return_code"] != 0
        return make_scenario_result(
            "schema_mismatch",
            "HALT",
            "HALT" if halted else "CONTINUED",
            halted,
            execution=execution,
        )

    return _run_isolated_scenario("schema_mismatch", run)


def _preprocessing_inconsistency_scenario() -> dict[str, Any]:
    def run(root: Path) -> dict[str, Any]:
        import pandas as pd

        data_path = ensure_sandbox_path(root, root / "data" / "processed" / "X_train.csv")
        frame = pd.read_csv(data_path)
        if frame.empty or frame.shape[1] == 0:
            raise ValueError("Processed training fixture is empty.")
        frame.iloc[0, 0] = float("nan")
        frame.to_csv(data_path, index=False)
        execution = _run_script(root, Path("src") / "validate_outputs.py")
        halted = execution["return_code"] != 0
        return make_scenario_result(
            "preprocessing_inconsistency",
            "HALT",
            "HALT" if halted else "CONTINUED",
            halted,
            execution=execution,
        )

    return _run_isolated_scenario("preprocessing_inconsistency", run)


def _model_recovery_scenario(config: dict[str, Any]) -> dict[str, Any]:
    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ImportError as exc:
        return make_scenario_result(
            "missing_model_artifact",
            "RECOVER",
            "RECOVERY_FAILED",
            False,
            error=f"{type(exc).__name__}: {exc}",
            registry_modified=False,
            project_model_files_modified=False,
        )

    def run(root: Path) -> dict[str, Any]:
        import joblib
        import pandas as pd

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config.get(
            "tracking_uri"
        )
        if not isinstance(tracking_uri, str) or not tracking_uri:
            raise ValueError("Lab 9 config must define an MLflow tracking URI.")
        model_name = config.get("model_name")
        model_version = config.get("recovery_model_version")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Lab 9 config must define model_name.")
        if not isinstance(model_version, int) or model_version <= 0:
            raise ValueError("recovery_model_version must be a positive integer.")

        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        client = MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
        model_uri = f"models:/{model_name}/{model_version}"
        try:
            version = client.get_model_version(model_name, str(model_version))
            model = _load_registered_model(
                mlflow,
                client,
                model_name,
                model_version,
                version.run_id or "",
            )
        finally:
            _dispose_mlflow_client(client)

        feature_path = ensure_sandbox_path(
            root,
            root / "data" / "processed" / "X_train_processed.csv",
        )
        features = pd.read_csv(feature_path)
        expected_features = getattr(model, "n_features_in_", None)
        if (
            expected_features is not None
            and int(expected_features) != features.shape[1]
        ):
            raise ValueError(
                f"Model version {model_version} expects {expected_features} "
                f"features, but Big Mart preprocessing produced {features.shape[1]}."
            )
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None and list(feature_names) != list(features.columns):
            raise ValueError(
                f"Model version {model_version} feature names do not match "
                "the current Big Mart preprocessing output."
            )

        model_path = ensure_sandbox_path(
            root,
            root / "models" / "lab9_recovered_model.joblib",
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        recovered_model = joblib.load(model_path)
        sample = features.head(1)
        predictions = recovered_model.predict(sample)
        if (
            len(predictions) != 1
            or not math.isfinite(float(predictions[0]))
        ):
            raise ValueError(
                "Recovered model did not return one finite sales prediction."
            )

        return make_scenario_result(
            "missing_model_artifact",
            "RECOVER",
            "RECOVERED",
            True,
            model_uri=model_uri,
            model_version=model_version,
            target_column="Item_Outlet_Sales",
            feature_count=int(features.shape[1]),
            prediction=float(predictions[0]),
            recovered_artifact="models/lab9_recovered_model.joblib",
            registry_modified=False,
            project_model_files_modified=False,
        )

    try:
        return _run_isolated_scenario("model_recovery", run)
    except (
        MlflowException,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        return make_scenario_result(
            "missing_model_artifact",
            "RECOVER",
            "RECOVERY_FAILED",
            False,
            error=f"{type(exc).__name__}: {exc}",
            registry_modified=False,
            project_model_files_modified=False,
        )


def _new_report_path(report_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return report_dir / f"lab9_failure_recovery_{timestamp}.json"


def run_failure_tests(
    config_path: Path,
    report_dir: Path | None = None,
) -> tuple[bool, Path]:
    config = _read_config(config_path)
    repetitions = config.get("repetitions", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer.")
    configured_report_dir = config.get("report_directory")
    if not isinstance(configured_report_dir, str):
        raise ValueError("Configuration must define report_directory.")
    output_dir = report_dir or _resolve_path(configured_report_dir)

    scenarios: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
        ("missing_data", _missing_data_scenario),
        ("invalid_datatype", _invalid_datatype_scenario),
        ("schema_mismatch", _schema_mismatch_scenario),
        ("preprocessing_inconsistency", _preprocessing_inconsistency_scenario),
        ("missing_model_artifact", lambda: _model_recovery_scenario(config)),
    )
    runs: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        print(f"[INFO] Lab 9 repetition {repetition}/{repetitions}")
        results = []
        for name, scenario in scenarios:
            print(f"[INFO] Running scenario: {name}")
            try:
                result = scenario()
            except (
                ImportError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                result = make_scenario_result(
                    name,
                    "HALT",
                    "NOT_TESTED",
                    False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            print(f"[{result['status']}] {name}: {result['actual']}")
        runs.append(
            {
                "repetition": repetition,
                "results": results,
                "passed": all(item["passed"] for item in results),
            }
        )

    report = build_report(runs)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI") or config.get(
        "tracking_uri"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = _new_report_path(output_dir)
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(f"[INFO] Lab 9 status: {report['overall_status']}")
    print(f"[INFO] Report saved to: {report_path}")
    return report["overall_status"] == "PASSED", report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exercise Big Mart failure handling and model recovery."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        passed, _ = run_failure_tests(_resolve_path(args.config))
        return 0 if passed else 1
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        print(
            f"[ERROR] Lab 9 failure-recovery suite could not run: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

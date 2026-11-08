from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab8_model_lifecycle.json"
METRIC_NAMES = ("MAE", "RMSE", "R2")


class RegistryClient(Protocol):
    def search_model_versions(self, filter_string: str) -> list[Any]: ...

    def get_run(self, run_id: str) -> Any: ...

    def get_model_version_by_alias(self, name: str, alias: str) -> Any: ...

    def set_registered_model_alias(
        self, name: str, alias: str, version: str
    ) -> None: ...

    def set_model_version_tag(
        self, name: str, version: str, key: str, value: str
    ) -> None: ...


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    if name not in metrics:
        return None
    try:
        value = float(metrics[name])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def evaluate_candidate(
    metrics: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    rules = gate_config.get("quality_gate", {}).get("metrics")
    if not isinstance(rules, dict) or set(rules) != set(METRIC_NAMES):
        raise ValueError("Lab 7 configuration must define MAE, RMSE, and R2.")

    results: dict[str, dict[str, Any]] = {}
    eligible = True
    for metric in METRIC_NAMES:
        actual = _metric_value(metrics, metric)
        rule = rules[metric]
        if (
            actual is None
            or (metric in {"MAE", "RMSE"} and actual < 0)
            or not isinstance(rule, dict)
        ):
            eligible = False
            results[metric] = {
                "actual": actual,
                "operator": rule.get("operator") if isinstance(rule, dict) else None,
                "threshold": rule.get("threshold") if isinstance(rule, dict) else None,
                "status": "FAILED",
                "reason": (
                    "Error metrics cannot be negative."
                    if metric in {"MAE", "RMSE"} and actual is not None and actual < 0
                    else "Metric missing or invalid."
                ),
            }
            continue

        try:
            threshold = float(rule["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid threshold for {metric}.") from exc
        if not math.isfinite(threshold):
            raise ValueError(f"Threshold for {metric} must be finite.")
        operator = rule.get("operator")
        if operator == "<=":
            passed = actual <= threshold
        elif operator == ">=":
            passed = actual >= threshold
        else:
            raise ValueError(f"Unsupported operator for {metric}: {operator!r}")

        eligible = eligible and passed
        results[metric] = {
            "actual": actual,
            "operator": operator,
            "threshold": threshold,
            "status": "PASSED" if passed else "FAILED",
        }

    return {"eligible": eligible, "metric_results": results}


def _version_number(candidate: dict[str, Any]) -> int:
    try:
        return int(candidate["model_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Registered model version must be an integer.") from exc


def ranking_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    metrics = candidate["metrics"]
    rmse = _metric_value(metrics, "RMSE")
    mae = _metric_value(metrics, "MAE")
    r2 = _metric_value(metrics, "R2")
    if rmse is None or mae is None or r2 is None:
        raise ValueError("Cannot rank a model with missing or invalid metrics.")
    return (rmse, mae, -r2, _version_number(candidate))


def choose_candidate(
    candidates: list[dict[str, Any]],
    current_production: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return None, "NO_QUALIFYING_MODEL"

    best = min(eligible, key=ranking_key)
    if current_production is None:
        return best, "NO_CURRENT_PRODUCTION"
    if _version_number(best) == _version_number(current_production):
        return None, "PRODUCTION_ALREADY_BEST"
    if ranking_key(best) >= ranking_key(current_production):
        return None, "NO_IMPROVEMENT"
    return best, "IMPROVEMENT_FOUND"


def collect_candidates(
    client: RegistryClient,
    model_name: str,
    gate_config: dict[str, Any],
    require_validation_tags: bool,
) -> list[dict[str, Any]]:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for {model_name!r}.")

    candidates: list[dict[str, Any]] = []
    for version in versions:
        run_id = getattr(version, "run_id", None)
        if not run_id:
            raise ValueError(
                f"Registered version {getattr(version, 'version', '?')} "
                "does not have a source run ID."
            )
        run = client.get_run(run_id)
        metrics = dict(run.data.metrics)
        gate = evaluate_candidate(metrics, gate_config)
        tags = dict(getattr(version, "tags", {}) or {})
        aliases = list(getattr(version, "aliases", []) or [])
        validation_ready = (
            tags.get("validation_status") == "PASSED"
            and tags.get("deployment_readiness") == "VALIDATED"
        )
        eligible = gate["eligible"] and (
            validation_ready or not require_validation_tags
        )

        candidate = {
            "model_version": int(version.version),
            "run_id": run_id,
            "metrics": {name: _metric_value(metrics, name) for name in METRIC_NAMES},
            "aliases": aliases,
            "validation_status": tags.get("validation_status"),
            "deployment_readiness": tags.get("deployment_readiness"),
            "gate_status": "PASSED" if gate["eligible"] else "FAILED",
            "validation_tags_passed": validation_ready,
            "eligible": eligible,
            "metric_results": gate["metric_results"],
        }
        candidates.append(candidate)
    return candidates


def get_current_production(
    client: RegistryClient,
    model_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        production_version = client.get_model_version_by_alias(
            model_name,
            "production",
        )
    except Exception as exc:
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import ErrorCode

        missing_alias_code = str(ErrorCode.RESOURCE_DOES_NOT_EXIST)
        if not isinstance(exc, MlflowException) or str(exc.error_code) not in {
            missing_alias_code,
            "RESOURCE_DOES_NOT_EXIST",
        }:
            raise
        return None

    version_number = int(production_version.version)
    for candidate in candidates:
        if candidate["model_version"] == version_number:
            candidate["aliases"] = sorted(
                set(candidate["aliases"]) | {"production"}
            )
            return candidate

    raise ValueError(
        "Production alias points to a version that was not returned by "
        "search_model_versions."
    )


def _new_report_path(report_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return report_dir / f"lab8_model_lifecycle_{timestamp}.json"


def write_report(report_dir: Path, report: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = _new_report_path(report_dir)
    with path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return path


def _run_lifecycle(
    config_path: Path,
    apply: bool = False,
    report_dir: Path | None = None,
    client: RegistryClient | None = None,
) -> tuple[int, Path]:
    config = _read_json(config_path)
    model_name = config.get("model_name")
    alias = config.get("production_alias")
    gate_config_path = config.get("quality_gate_config")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Configuration must define model_name.")
    if not isinstance(alias, str) or alias.lower() != "production":
        raise ValueError("Only the existing 'production' alias is supported.")
    if not isinstance(gate_config_path, str):
        raise ValueError("Configuration must define quality_gate_config.")

    gate_config = _read_json(_resolve_path(gate_config_path))
    if client is None:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config.get(
            "tracking_uri"
        )
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

    candidates = collect_candidates(
        client,
        model_name,
        gate_config,
        bool(config.get("promotion_requires_validation_tags", True)),
    )
    current = get_current_production(client, model_name, candidates)
    selected, decision = choose_candidate(candidates, current)

    if selected is None:
        report_status = decision
    elif apply:
        client.set_registered_model_alias(
            model_name,
            alias,
            str(selected["model_version"]),
        )
        client.set_model_version_tag(
            model_name,
            str(selected["model_version"]),
            "lifecycle_state",
            "PRODUCTION",
        )
        report_status = "PROMOTED"
    else:
        report_status = "DRY_RUN_PROMOTION"

    report = {
        "lab": 8,
        "registered_model_name": model_name,
        "production_alias": alias,
        "mode": "APPLY" if apply else "DRY_RUN",
        "status": report_status,
        "decision": decision,
        "current_production": current,
        "selected_candidate": selected,
        "candidates": candidates,
        "promotion_applied": bool(apply and selected is not None),
        "quality_gate_config": str(_resolve_path(gate_config_path)),
    }
    configured_report_dir = config.get("report_directory")
    if not isinstance(configured_report_dir, str):
        raise ValueError("Configuration must define report_directory.")
    output_dir = report_dir or _resolve_path(configured_report_dir)
    report_path = write_report(output_dir, report)

    print(f"[INFO] Registry model: {model_name}")
    print(f"[INFO] Current production: {current['model_version'] if current else 'none'}")
    print(f"[INFO] Decision: {decision}")
    if selected:
        print(
            f"[INFO] Candidate version {selected['model_version']} | "
            f"MAE={selected['metrics']['MAE']:.6f} | "
            f"RMSE={selected['metrics']['RMSE']:.6f} | "
            f"R2={selected['metrics']['R2']:.6f}"
        )
    if apply and selected:
        print(f"[SUCCESS] Production alias assigned to version {selected['model_version']}.")
    elif selected:
        print("[INFO] Dry run only; no MLflow alias or tags were changed.")
    else:
        print("[INFO] No model promotion was needed or allowed.")
    print(f"[INFO] Report saved to: {report_path}")
    return (0 if selected or decision in {"PRODUCTION_ALREADY_BEST", "NO_IMPROVEMENT"} else 1), report_path


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


def run_lifecycle(
    config_path: Path,
    apply: bool = False,
    report_dir: Path | None = None,
    client: RegistryClient | None = None,
) -> tuple[int, Path]:
    if client is not None:
        return _run_lifecycle(
            config_path,
            apply=apply,
            report_dir=report_dir,
            client=client,
        )

    import mlflow
    from mlflow.tracking import MlflowClient

    config = _read_json(config_path)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        owned_client = MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )
    else:
        owned_client = MlflowClient()
    try:
        return _run_lifecycle(
            config_path,
            apply=apply,
            report_dir=report_dir,
            client=owned_client,
        )
    finally:
        _dispose_mlflow_client(owned_client)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Big Mart registry versions and optionally promote."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Assign the production alias. This changes MLflow registry state.",
    )
    args = parser.parse_args(argv)
    try:
        exit_code, _ = run_lifecycle(_resolve_path(args.config), apply=args.apply)
        return exit_code
    except ImportError as exc:
        print(
            "[ERROR] MLflow is required for registry operations. "
            "Install the project's MLflow dependency in this environment.",
            file=sys.stderr,
        )
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[ERROR] Lab 8 lifecycle could not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

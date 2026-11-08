from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab7_quality_gate.json"
REQUIRED_METRICS = ("MAE", "RMSE", "R2")
ALLOWED_OPERATORS = {"<=", ">="}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def evaluate_quality_gate(
    config: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    target_column = config.get("target_column")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("Configuration must define a non-empty target_column.")
    if evaluation.get("target_column") != target_column:
        raise ValueError(
            f"Evaluation target must be {target_column!r}; "
            f"found {evaluation.get('target_column')!r}."
        )

    gate_config = config.get("quality_gate")
    if not isinstance(gate_config, dict):
        raise ValueError("Configuration must define quality_gate.")
    rules = gate_config.get("metrics")
    if not isinstance(rules, dict) or set(rules) != set(REQUIRED_METRICS):
        raise ValueError(
            "Quality gate must define exactly MAE, RMSE, and R2 rules."
        )

    actual_metrics: dict[str, float] = {}
    metric_results: dict[str, dict[str, Any]] = {}
    all_passed = True

    for metric in REQUIRED_METRICS:
        source_key = "R2_Score" if metric == "R2" else metric
        if source_key not in evaluation:
            raise ValueError(
                f"Evaluation report is missing metric {source_key!r}."
            )
        actual = _finite_number(evaluation[source_key], source_key)
        if metric in {"MAE", "RMSE"} and actual < 0:
            raise ValueError(f"{metric} cannot be negative.")

        rule = rules[metric]
        if not isinstance(rule, dict):
            raise ValueError(f"Quality rule for {metric} must be an object.")
        operator = rule.get("operator")
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Quality rule for {metric} must use <= or >=."
            )
        threshold = _finite_number(
            rule.get("threshold"),
            f"{metric} threshold",
        )

        passed = actual <= threshold if operator == "<=" else actual >= threshold
        actual_metrics[metric] = actual
        metric_results[metric] = {
            "actual": actual,
            "operator": operator,
            "threshold": threshold,
            "status": "PASSED" if passed else "FAILED",
        }
        all_passed = all_passed and passed

    baseline = config.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("Configuration must define baseline provenance.")

    return {
        "lab": 7,
        "target_column": target_column,
        "evaluation_metrics": actual_metrics,
        "baseline": baseline,
        "metric_results": metric_results,
        "status": "PASSED" if all_passed else "FAILED",
    }


def _new_report_path(report_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return report_dir / f"lab7_quality_gate_{timestamp}.json"


def write_report(report_dir: Path, report: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    base_path = _new_report_path(report_dir)
    report_path = base_path
    suffix = 0
    while True:
        try:
            file = report_path.open("x", encoding="utf-8")
        except FileExistsError:
            suffix += 1
            report_path = base_path.with_name(
                f"{base_path.stem}_{suffix}{base_path.suffix}"
            )
            continue
        with file:
            json.dump(report, file, indent=2)
            file.write("\n")
        return report_path


def run_quality_gate(
    config_path: Path,
    evaluation_path: Path | None = None,
    report_dir: Path | None = None,
) -> int:
    config = _read_json(config_path)
    configured_evaluation_path = config.get("evaluation_report")
    if not isinstance(configured_evaluation_path, str):
        raise ValueError("Configuration must define evaluation_report.")

    evaluation_file = evaluation_path or _resolve_project_path(
        configured_evaluation_path
    )
    evaluation = _read_json(evaluation_file)
    result = evaluate_quality_gate(config, evaluation)

    configured_report_dir = config.get("report_directory")
    if not isinstance(configured_report_dir, str):
        raise ValueError("Configuration must define report_directory.")
    output_dir = report_dir or _resolve_project_path(configured_report_dir)
    result["evaluation_report"] = str(evaluation_file)
    report_path = write_report(output_dir, result)

    print(f"[INFO] Target: {result['target_column']}")
    for metric, details in result["metric_results"].items():
        print(
            f"[{details['status']}] {metric}: {details['actual']:.6f} "
            f"{details['operator']} {details['threshold']:.6f}"
        )
    print(f"[{result['status']}] Lab 7 regression quality gate.")
    print(f"[INFO] Report saved to: {report_path}")
    return 0 if result["status"] == "PASSED" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the Big Mart regression model quality gate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the JSON quality-gate configuration.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        help="Override the evaluation JSON path from the configuration.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Override the directory for a new timestamped gate report.",
    )
    args = parser.parse_args(argv)

    config_path = _resolve_project_path(args.config)
    evaluation_path = (
        _resolve_project_path(args.evaluation_report)
        if args.evaluation_report
        else None
    )
    report_dir = (
        _resolve_project_path(args.report_dir) if args.report_dir else None
    )

    try:
        return run_quality_gate(config_path, evaluation_path, report_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[ERROR] Lab 7 quality gate could not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

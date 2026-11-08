from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab12_monitoring.json"
TARGET_COLUMN = "Item_Outlet_Sales"
IDENTIFIER_COLUMNS = {"Item_Identifier"}

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


def _timestamp_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _finite_array(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{label} must contain at least one value.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains a non-finite value.")
    return array


def validate_sales_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    require_target: bool,
    max_missing_rate: float = 0.6,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Incoming Big Mart dataset is empty.")
    if frame.columns.has_duplicates:
        raise ValueError("Incoming Big Mart dataset has duplicate column names.")
    if not 0 <= max_missing_rate < 1:
        raise ValueError("max_missing_rate must be in [0, 1).")

    has_target = TARGET_COLUMN in frame.columns
    if require_target and not has_target:
        raise ValueError(f"Labeled sales data must contain {TARGET_COLUMN}.")

    actual_features = [name for name in frame.columns if name != TARGET_COLUMN]
    missing_columns = sorted(set(feature_columns) - set(actual_features))
    extra_columns = sorted(set(actual_features) - set(feature_columns))
    if missing_columns or extra_columns:
        raise ValueError(
            "Big Mart feature schema mismatch. "
            f"Missing columns: {missing_columns}; extra columns: {extra_columns}."
        )
    if frame.duplicated().any():
        duplicate_count = int(frame.duplicated().sum())
        raise ValueError(
            f"Incoming Big Mart dataset contains {duplicate_count} duplicate rows."
        )

    ordered = frame.loc[
        :, feature_columns + ([TARGET_COLUMN] if has_target else [])
    ].copy()
    numeric_features = [
        name
        for name in feature_columns
        if pd.api.types.is_numeric_dtype(ordered[name].dtype)
    ]
    for column in numeric_features:
        values = pd.to_numeric(ordered[column], errors="coerce")
        invalid = ordered[column].notna() & values.isna()
        if invalid.any():
            raise ValueError(f"Numeric feature {column} contains invalid values.")
        observed = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"Numeric feature {column} contains non-finite values.")
        if observed.size == 0:
            raise ValueError(f"Numeric feature {column} contains only missing values.")
        if column in {"Item_Weight", "Item_Visibility", "Item_MRP"} and (
            observed < 0
        ).any():
            raise ValueError(f"Numeric feature {column} cannot be negative.")
        if column == "Item_Visibility" and (observed > 1).any():
            raise ValueError("Item_Visibility must be between 0 and 1.")
        if column == "Outlet_Establishment_Year" and (
            ((observed < 1900) | (observed > 2100)).any()
            or not np.equal(observed, np.floor(observed)).all()
        ):
            raise ValueError(
                "Outlet_Establishment_Year must be an integer from 1900 to 2100."
            )
        ordered[column] = values

    if has_target:
        target = pd.to_numeric(ordered[TARGET_COLUMN], errors="coerce")
        if target.isna().any():
            raise ValueError(f"{TARGET_COLUMN} contains missing or invalid values.")
        target_values = target.to_numpy(dtype=float)
        if not np.isfinite(target_values).all():
            raise ValueError(f"{TARGET_COLUMN} contains non-finite values.")
        if (target_values < 0).any():
            raise ValueError(f"{TARGET_COLUMN} cannot be negative.")
        ordered[TARGET_COLUMN] = target

    missing_counts = {
        column: int(count)
        for column, count in ordered.isna().sum().items()
        if count
    }
    missing_rates = ordered.isna().mean()
    over_limit = {
        column: float(rate)
        for column, rate in missing_rates.items()
        if rate > max_missing_rate
    }
    if over_limit:
        raise ValueError(
            "Missing-value rate exceeds the configured limit: "
            f"{over_limit}."
        )

    return {
        "rows": int(len(ordered)),
        "feature_count": len(feature_columns),
        "has_target": has_target,
        "duplicate_rows": 0,
        "missing_values_by_column": missing_counts,
        "missing_value_count": int(sum(missing_counts.values())),
    }


def _psi_from_counts(
    reference_counts: np.ndarray,
    current_counts: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    reference = reference_counts.astype(float)
    current = current_counts.astype(float)
    if reference.sum() <= 0 or current.sum() <= 0:
        raise ValueError("Both distributions must contain observed values.")
    reference = (reference + epsilon) / (
        reference.sum() + epsilon * len(reference)
    )
    current = (current + epsilon) / (current.sum() + epsilon * len(current))
    return float(np.sum((current - reference) * np.log(current / reference)))


def population_stability_index(
    reference: pd.Series | np.ndarray | list[float],
    current: pd.Series | np.ndarray | list[float],
    *,
    numeric: bool,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    reference_series = pd.Series(reference)
    current_series = pd.Series(current)

    if numeric:
        reference_values = pd.to_numeric(
            reference_series, errors="raise"
        ).to_numpy(dtype=float, na_value=np.nan)
        current_values = pd.to_numeric(
            current_series, errors="raise"
        ).to_numpy(dtype=float, na_value=np.nan)
        reference_observed = reference_values[np.isfinite(reference_values)]
        if not np.isfinite(reference_values[~np.isnan(reference_values)]).all():
            raise ValueError("Reference numeric feature contains non-finite values.")
        if reference_observed.size == 0:
            raise ValueError("Reference numeric feature has no observed values.")
        if not np.isfinite(current_values[~np.isnan(current_values)]).all():
            raise ValueError("Current numeric feature contains non-finite values.")
        quantiles = np.linspace(0, 1, min(bins, reference_observed.size) + 1)
        edges = np.unique(np.quantile(reference_observed, quantiles))
        if edges.size < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf

        reference_observed_mask = np.isfinite(reference_values)
        current_observed_mask = np.isfinite(current_values)
        bucket_count = len(edges) - 1
        reference_counts = np.zeros(bucket_count + 1, dtype=int)
        current_counts = np.zeros(bucket_count + 1, dtype=int)
        reference_indices = np.searchsorted(
            edges[1:-1], reference_values[reference_observed_mask], side="right"
        )
        current_indices = np.searchsorted(
            edges[1:-1], current_values[current_observed_mask], side="right"
        )
        reference_counts[:bucket_count] = np.bincount(
            reference_indices, minlength=bucket_count
        )
        current_counts[:bucket_count] = np.bincount(
            current_indices, minlength=bucket_count
        )
        reference_counts[-1] = int((~reference_observed_mask).sum())
        current_counts[-1] = int((~current_observed_mask).sum())
        return _psi_from_counts(reference_counts, current_counts)

    def as_key(value: Any) -> tuple[str, str]:
        if pd.isna(value):
            return ("missing", "")
        return ("value", str(value))

    reference_keys = [as_key(value) for value in reference_series.tolist()]
    current_keys = [as_key(value) for value in current_series.tolist()]
    categories = sorted(set(reference_keys) | set(current_keys))
    reference_counts = Counter(reference_keys)
    current_counts = Counter(current_keys)
    return _psi_from_counts(
        np.array([reference_counts[key] for key in categories]),
        np.array([current_counts[key] for key in categories]),
    )


def analyze_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str],
    *,
    psi_threshold: float,
    bins: int = 10,
) -> dict[str, Any]:
    if not 0 <= psi_threshold:
        raise ValueError("psi_threshold cannot be negative.")
    results: list[dict[str, Any]] = []
    for column in feature_columns:
        if column in IDENTIFIER_COLUMNS:
            results.append(
                {
                    "feature": column,
                    "kind": "identifier",
                    "psi": None,
                    "drift_detected": False,
                    "note": "Unique identifiers are excluded from distribution tests.",
                }
            )
            continue
        numeric = pd.api.types.is_numeric_dtype(reference[column].dtype)
        psi = population_stability_index(
            reference[column],
            current[column],
            numeric=numeric,
            bins=bins,
        )
        results.append(
            {
                "feature": column,
                "kind": "numeric" if numeric else "categorical",
                "psi": psi,
                "threshold": psi_threshold,
                "drift_detected": psi >= psi_threshold,
            }
        )

    drifted = [item["feature"] for item in results if item["drift_detected"]]
    return {
        "method": "Population Stability Index (PSI)",
        "threshold": psi_threshold,
        "feature_count": len(feature_columns),
        "tested_feature_count": sum(item["psi"] is not None for item in results),
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
        "features": results,
    }


def compare_predictions(
    reference_predictions: Any,
    current_predictions: Any,
    *,
    psi_threshold: float,
    minimum_sample_size: int,
) -> dict[str, Any]:
    reference = _finite_array(reference_predictions, "Reference predictions")
    current = _finite_array(current_predictions, "Current predictions")
    if minimum_sample_size < 1:
        raise ValueError("minimum_sample_size must be positive.")
    summary = {
        "reference_count": int(reference.size),
        "current_count": int(current.size),
        "reference_mean": float(np.mean(reference)),
        "current_mean": float(np.mean(current)),
        "reference_median": float(np.median(reference)),
        "current_median": float(np.median(current)),
        "reference_p95": float(np.quantile(reference, 0.95)),
        "current_p95": float(np.quantile(current, 0.95)),
    }
    if min(reference.size, current.size) < minimum_sample_size:
        return {
            **summary,
            "status": "INSUFFICIENT_SAMPLE",
            "psi": None,
            "threshold": psi_threshold,
            "drift_detected": False,
        }
    psi = population_stability_index(
        reference,
        current,
        numeric=True,
    )
    return {
        **summary,
        "status": "EVALUATED",
        "psi": psi,
        "threshold": psi_threshold,
        "drift_detected": psi >= psi_threshold,
    }


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    actual_values = _finite_array(actual, "Actual sales")
    predicted_values = _finite_array(predicted, "Predicted sales")
    if actual_values.size < 2 or actual_values.size != predicted_values.size:
        raise ValueError(
            "At least two matching actual and predicted sales values are required."
        )
    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(math.sqrt(mean_squared_error(actual_values, predicted_values))),
        "R2": float(r2_score(actual_values, predicted_values)),
    }


def monitor_labeled_performance(
    actual: Any,
    predicted: Any,
    *,
    baseline: dict[str, Any],
    mae_relative_increase: float,
    rmse_relative_increase: float,
    r2_absolute_drop: float,
) -> dict[str, Any]:
    metrics = regression_metrics(actual, predicted)
    required = ("MAE", "RMSE", "R2")
    if any(name not in baseline for name in required):
        raise ValueError("Performance baseline must define MAE, RMSE, and R2.")
    baseline_metrics = {
        name: float(baseline[name]) for name in required
    }
    if not all(math.isfinite(value) for value in baseline_metrics.values()):
        raise ValueError("Performance baseline metrics must be finite.")
    if min(mae_relative_increase, rmse_relative_increase, r2_absolute_drop) < 0:
        raise ValueError("Concept-drift degradation thresholds cannot be negative.")
    degraded = (
        metrics["MAE"] > baseline_metrics["MAE"] * (1 + mae_relative_increase)
        or metrics["RMSE"]
        > baseline_metrics["RMSE"] * (1 + rmse_relative_increase)
        or metrics["R2"] < baseline_metrics["R2"] - r2_absolute_drop
    )
    return {
        "status": "CONCEPT_DRIFT" if degraded else "NO_DEGRADATION_DETECTED",
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "thresholds": {
            "mae_relative_increase": mae_relative_increase,
            "rmse_relative_increase": rmse_relative_increase,
            "r2_absolute_drop": r2_absolute_drop,
        },
        "interpretation": (
            "Concept/performance drift is evaluated from labeled actual-vs-predicted sales; "
            "it is independent of feature distribution drift."
        ),
    }


def summarize_api_events(log_path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid inference log JSON at line {line_number}."
                    ) from exc
                if not isinstance(event, dict):
                    raise ValueError(
                        f"Inference log line {line_number} is not a JSON object."
                    )
                events.append(event)

    requests = [event for event in events if event.get("event_type") == "request"]
    inference_requests = [
        event
        for event in requests
        if event.get("path") in {"/predict", "/predict/batch"}
    ]
    predictions = [
        event
        for event in events
        if event.get("event_type", "prediction") == "prediction"
        and event.get("predicted_sales") is not None
    ]
    failures = [
        event
        for event in inference_requests
        if event.get("status") == "failure"
    ]
    legacy = not inference_requests and bool(predictions)
    if legacy:
        request_ids = {
            str(event.get("request_id", "")).rsplit("-", 1)[0]
            if str(event.get("request_id", "")).rsplit("-", 1)[-1].isdigit()
            else str(event.get("request_id", ""))
            for event in predictions
        }
        request_count = len(request_ids)
        failures = [
            event for event in predictions if event.get("status") == "failure"
        ]
        latency_events = predictions
    else:
        request_count = len(inference_requests)
        latency_events = inference_requests

    latencies: list[float] = []
    for event in latency_events:
        value = event.get("latency_ms")
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("API monitoring log has an invalid latency value.")
        latencies.append(number)

    error_counts = Counter(
        str(event.get("error"))
        for event in failures
        if event.get("error")
    )
    trends: dict[str, list[float]] = defaultdict(list)
    for event in predictions:
        value = float(event["predicted_sales"])
        if not math.isfinite(value):
            raise ValueError("API monitoring log has a non-finite prediction.")
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str):
            raise ValueError("API monitoring log is missing a timestamp.")
        try:
            day = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()
        except ValueError as exc:
            raise ValueError(
                f"API monitoring log has an invalid timestamp: {timestamp!r}."
            ) from exc
        trends[day.isoformat()].append(value)

    latency_metrics = {
        "count": len(latencies),
        "mean_ms": float(np.mean(latencies)) if latencies else None,
        "p50_ms": float(np.quantile(latencies, 0.50)) if latencies else None,
        "p95_ms": float(np.quantile(latencies, 0.95)) if latencies else None,
        "max_ms": float(max(latencies)) if latencies else None,
    }
    trend_rows = [
        {
            "date": date,
            "prediction_count": len(values),
            "mean_sales": float(np.mean(values)),
            "median_sales": float(np.median(values)),
        }
        for date, values in sorted(trends.items())
    ]
    return {
        "request_count": request_count,
        "prediction_count": len(predictions),
        "failed_request_count": len(failures),
        "error_counts": dict(error_counts),
        "latency": latency_metrics,
        "prediction_trend_by_day": trend_rows,
        "request_count_source": (
            "inference events; batch members are grouped by request id"
            if legacy
            else "request-level API events"
        ),
    }


def _predict(
    model: Any,
    preprocessor: Any,
    frame: pd.DataFrame,
    chunk_size: int,
) -> np.ndarray:
    if chunk_size < 1:
        raise ValueError("prediction_chunk_size must be positive.")
    raw = frame.drop(columns=[TARGET_COLUMN], errors="ignore")
    outputs: list[np.ndarray] = []
    for start in range(0, len(raw), chunk_size):
        records = raw.iloc[start : start + chunk_size].to_dict(orient="records")
        features = preprocessor.transform(records)
        predictions = _finite_array(model.predict(features), "Model predictions")
        if predictions.size != len(records):
            raise ValueError("Model returned an unexpected number of predictions.")
        outputs.append(predictions)
    return np.concatenate(outputs)


def _historical_validation(
    model: Any,
    feature_path: Path,
    target_path: Path,
) -> dict[str, Any]:
    if not feature_path.is_file() or not target_path.is_file():
        return {
            "status": "NOT_AVAILABLE",
            "reason": "Configured historical validation files are missing.",
        }
    features = pd.read_csv(feature_path)
    target_frame = pd.read_csv(target_path)
    if features.empty or target_frame.empty or target_frame.shape[1] != 1:
        raise ValueError("Historical validation feature/target files are invalid.")
    if len(features) != len(target_frame):
        raise ValueError("Historical validation feature and target row counts differ.")
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and int(expected_count) != features.shape[1]:
        raise ValueError(
            "Historical validation feature count does not match the deployed model."
        )
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != features.columns.tolist():
        raise ValueError(
            "Historical validation feature names do not match the deployed model."
        )
    actual = pd.to_numeric(target_frame.iloc[:, 0], errors="raise")
    predicted = _finite_array(model.predict(features), "Historical predictions")
    return {
        "status": "HISTORICAL_VALIDATION",
        "rows": len(features),
        "source_features": str(feature_path),
        "source_targets": str(target_path),
        "metrics": regression_metrics(actual, predicted),
        "interpretation": (
            "A held-out historical regression check, not a live concept-drift signal."
        ),
    }


def build_monitoring_report(
    config_path: Path,
    *,
    incoming_path: Path | None = None,
    labels_path: Path | None = None,
) -> dict[str, Any]:
    config = _read_json(config_path)
    reference_path = _resolve_path(config["reference_train"])
    incoming_file = incoming_path or _resolve_path(config["incoming_data"])
    log_path = _resolve_path(config["inference_log"])
    reference_frame = pd.read_csv(reference_path)
    incoming_frame = pd.read_csv(incoming_file)

    from api.model_loader import ModelLoader

    loader = ModelLoader()
    loader.load()
    if loader.preprocessor is None or loader.model is None:
        raise RuntimeError("The configured Big Mart API model did not load.")
    feature_columns = list(loader.preprocessor.raw_columns)
    max_missing_rate = float(config["max_missing_rate"])
    reference_validation = validate_sales_dataset(
        reference_frame,
        feature_columns,
        require_target=True,
        max_missing_rate=max_missing_rate,
    )
    incoming_validation = validate_sales_dataset(
        incoming_frame,
        feature_columns,
        require_target=False,
        max_missing_rate=max_missing_rate,
    )
    reference_features = reference_frame.loc[:, feature_columns]
    incoming_features = incoming_frame.loc[:, feature_columns]
    data_drift = analyze_data_drift(
        reference_features,
        incoming_features,
        feature_columns,
        psi_threshold=float(config["data_psi_threshold"]),
        bins=int(config["numeric_bins"]),
    )

    reference_predictions = _predict(
        loader.model,
        loader.preprocessor,
        reference_frame.head(int(config["prediction_reference_rows"])),
        int(config["prediction_chunk_size"]),
    )
    incoming_predictions = _predict(
        loader.model,
        loader.preprocessor,
        incoming_frame,
        int(config["prediction_chunk_size"]),
    )
    prediction_drift = compare_predictions(
        reference_predictions,
        incoming_predictions,
        psi_threshold=float(config["prediction_psi_threshold"]),
        minimum_sample_size=int(config["minimum_prediction_sample"]),
    )
    prediction_drift["comparison"] = (
        "The same deployed model applied to reference and incoming feature rows."
    )
    prediction_summary = {
        "count": int(incoming_predictions.size),
        "mean": float(np.mean(incoming_predictions)),
        "median": float(np.median(incoming_predictions)),
        "p05": float(np.quantile(incoming_predictions, 0.05)),
        "p95": float(np.quantile(incoming_predictions, 0.95)),
        "minimum": float(np.min(incoming_predictions)),
        "maximum": float(np.max(incoming_predictions)),
    }

    if TARGET_COLUMN in incoming_frame.columns:
        labeled_predictions = _predict(
            loader.model,
            loader.preprocessor,
            incoming_frame,
            int(config["prediction_chunk_size"]),
        )
        actual = pd.to_numeric(incoming_frame[TARGET_COLUMN], errors="raise")
        concept_drift = monitor_labeled_performance(
            actual,
            labeled_predictions,
            baseline=loader.metadata["metrics"],
            mae_relative_increase=float(config["mae_relative_increase"]),
            rmse_relative_increase=float(config["rmse_relative_increase"]),
            r2_absolute_drop=float(config["r2_absolute_drop"]),
        )
        concept_drift["source"] = str(incoming_file)
    elif labels_path is not None:
        labels_frame = pd.read_csv(labels_path)
        if not {"request_id", "actual_sales"}.issubset(labels_frame.columns):
            raise ValueError(
                "Prediction feedback CSV must contain request_id and actual_sales."
            )
        log_events = _read_prediction_events(log_path)
        by_key = {
            (
                str(event["request_id"]),
                int(event.get("prediction_index", 0)),
            ): event
            for event in log_events
            if event.get("predicted_sales") is not None
        }
        feedback_keys = [
            (
                str(row["request_id"]),
                int(row.get("prediction_index", 0)),
            )
            for row in labels_frame.to_dict(orient="records")
        ]
        if len(set(feedback_keys)) != len(feedback_keys):
            raise ValueError("Prediction feedback contains duplicate request keys.")
        matched = [by_key[key] for key in feedback_keys if key in by_key]
        if len(matched) != len(feedback_keys):
            raise ValueError("Prediction feedback contains unknown request ids.")
        actual = pd.to_numeric(labels_frame["actual_sales"], errors="raise")
        predicted = [float(event["predicted_sales"]) for event in matched]
        concept_drift = monitor_labeled_performance(
            actual,
            predicted,
            baseline=loader.metadata["metrics"],
            mae_relative_increase=float(config["mae_relative_increase"]),
            rmse_relative_increase=float(config["rmse_relative_increase"]),
            r2_absolute_drop=float(config["r2_absolute_drop"]),
        )
        concept_drift["source"] = str(labels_path)
    else:
        concept_drift = {
            "status": "NOT_AVAILABLE_NO_NEW_LABELS",
            "metrics": None,
            "interpretation": (
                "No incoming actual sales labels are available. Data drift is "
                "reported separately and is not labeled concept drift."
            ),
        }

    validation_features = _resolve_path(config["validation_features"])
    validation_targets = _resolve_path(config["validation_targets"])
    historical_validation = _historical_validation(
        loader.model,
        validation_features,
        validation_targets,
    )
    return {
        "lab": 12,
        "generated_at_utc": _timestamp(),
        "model": {
            "name": loader.model_name,
            "version": loader.model_version,
            "run_id": loader.run_id,
            "target_column": TARGET_COLUMN,
        },
        "reference_data": {
            "path": str(reference_path),
            **reference_validation,
        },
        "incoming_data": {
            "path": str(incoming_file),
            **incoming_validation,
        },
        "data_drift": data_drift,
        "prediction_drift": prediction_drift,
        "prediction_summary": prediction_summary,
        "concept_drift": concept_drift,
        "historical_validation": historical_validation,
        "api_performance": summarize_api_events(log_path),
        "thresholds": {
            "data_psi": float(config["data_psi_threshold"]),
            "prediction_psi": float(config["prediction_psi_threshold"]),
            "concept_drift": {
                "mae_relative_increase": float(config["mae_relative_increase"]),
                "rmse_relative_increase": float(config["rmse_relative_increase"]),
                "r2_absolute_drop": float(config["r2_absolute_drop"]),
            },
        },
    }


def _read_prediction_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid inference log JSON at line {line_number}."
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"Inference log line {line_number} is not a JSON object."
                )
            events.append(event)
    return events


def render_dashboard(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    data_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['feature']))}</td>"
        f"<td>{html.escape(str(item['kind']))}</td>"
        f"<td>{cell(item['psi'])}</td>"
        f"<td>{'DRIFT' if item['drift_detected'] else 'OK'}</td>"
        "</tr>"
        for item in report["data_drift"]["features"]
    )
    trend_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['date']))}</td>"
        f"<td>{item['prediction_count']}</td>"
        f"<td>{cell(item['mean_sales'])}</td>"
        f"<td>{cell(item['median_sales'])}</td>"
        "</tr>"
        for item in report["api_performance"]["prediction_trend_by_day"]
    )
    api = report["api_performance"]
    history = report["historical_validation"]
    historical_metrics = history.get("metrics", {})
    summary_cards = [
        ("Data drift", "DETECTED" if report["data_drift"]["drift_detected"] else "NOT DETECTED"),
        ("Prediction drift", report["prediction_drift"]["status"]),
        ("Labeled performance", report["concept_drift"]["status"]),
        ("API requests", str(api["request_count"])),
        ("Failed requests", str(api["failed_request_count"])),
        ("P95 latency (ms)", cell(api["latency"]["p95_ms"])),
        ("Historical holdout MAE", cell(historical_metrics.get("MAE"))),
        ("Historical holdout RMSE", cell(historical_metrics.get("RMSE"))),
        ("Historical holdout R²", cell(historical_metrics.get("R2"))),
    ]
    cards = "".join(
        f"<article><h2>{html.escape(title)}</h2><p>{html.escape(value)}</p></article>"
        for title, value in summary_cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Big Mart Lab 12 Monitoring</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;margin:2rem;color:#1f2937;background:#f8fafc}}
h1,h2{{margin:0 0 .6rem}} section{{margin:2rem 0}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}}
article,table{{background:white;border:1px solid #dbe3ec;border-radius:8px;padding:1rem}}
article p{{font-size:1.2rem;font-weight:650;margin:.3rem 0}} table{{width:100%;border-collapse:collapse;padding:0}}
th,td{{padding:.65rem;text-align:left;border-bottom:1px solid #e5e7eb}} th{{background:#eef2f7}}
.note{{color:#475569}} code{{overflow-wrap:anywhere}}
</style>
</head>
<body>
<h1>Big Mart Sales — Monitoring</h1>
<p class="note">Generated {html.escape(str(report['generated_at_utc']))} ·
Model {html.escape(str(report['model']['name']))} v{report['model']['version']} ·
Target {TARGET_COLUMN}</p>
<section><h2>Overview</h2><div class="cards">{cards}</div></section>
<section><h2>Feature drift (PSI threshold {report['data_drift']['threshold']})</h2>
<table><thead><tr><th>Feature</th><th>Type</th><th>PSI</th><th>Status</th></tr></thead>
<tbody>{data_rows}</tbody></table></section>
<section><h2>Prediction trends by day</h2>
<table><thead><tr><th>Date</th><th>Count</th><th>Mean sales</th><th>Median sales</th></tr></thead>
<tbody>{trend_rows}</tbody></table></section>
<section><h2>Prediction distribution</h2>
<p>Reference mean: {cell(report['prediction_drift']['reference_mean'])} ·
Incoming mean: {cell(report['prediction_drift']['current_mean'])} ·
PSI: {cell(report['prediction_drift']['psi'])}</p></section>
<section><h2>Performance interpretation</h2>
<p>{html.escape(str(report['concept_drift']['interpretation']))}</p>
<p>{html.escape(str(history.get('interpretation', history.get('reason', ''))))}</p>
<p>API latency P50/P95/max: {cell(api['latency']['p50_ms'])} /
{cell(api['latency']['p95_ms'])} / {cell(api['latency']['max_ms'])} ms.</p></section>
</body></html>
"""


def write_monitoring_artifacts(
    report: dict[str, Any],
    report_dir: Path,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = _timestamp_suffix()
    json_path = report_dir / f"lab12_monitoring_{suffix}.json"
    html_path = report_dir / f"lab12_monitoring_dashboard_{suffix}.html"
    with json_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    with html_path.open("x", encoding="utf-8") as file:
        file.write(render_dashboard(report))
    return json_path, html_path


def run_monitoring(
    config_path: Path,
    *,
    incoming_path: Path | None = None,
    labels_path: Path | None = None,
    report_dir: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    config = _read_json(config_path)
    report = build_monitoring_report(
        config_path,
        incoming_path=incoming_path,
        labels_path=labels_path,
    )
    output_dir = report_dir or _resolve_path(config["report_directory"])
    json_path, html_path = write_monitoring_artifacts(report, output_dir)
    print(f"[INFO] Data drift: {report['data_drift']['drift_detected']}")
    print(f"[INFO] Prediction drift: {report['prediction_drift']['status']}")
    print(f"[INFO] Labeled performance: {report['concept_drift']['status']}")
    print(f"[INFO] API requests: {report['api_performance']['request_count']}")
    print(f"[INFO] JSON report: {json_path}")
    print(f"[INFO] Dashboard: {html_path}")
    return report, json_path, html_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monitor Big Mart data, predictions, labeled performance, and API events."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--incoming", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        help="CSV with request_id and actual_sales for delayed prediction feedback.",
    )
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        run_monitoring(
            _resolve_path(args.config),
            incoming_path=_resolve_path(args.incoming) if args.incoming else None,
            labels_path=_resolve_path(args.labels) if args.labels else None,
            report_dir=_resolve_path(args.report_dir) if args.report_dir else None,
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"[ERROR] Lab 12 monitoring failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

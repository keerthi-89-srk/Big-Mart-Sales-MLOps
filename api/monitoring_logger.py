from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = Path(
    os.getenv(
        "BIGMART_INFERENCE_LOG",
        str(PROJECT_ROOT / "logs" / "inference" / "inference_log.jsonl"),
    )
)
_LOCK = threading.Lock()


def log_inference(
    input_field_count: int,
    predicted_sales: float | None = None,
    latency_ms: float | None = None,
    status: str = "success",
    error: str | None = None,
    model_name: str | None = None,
    model_version: int | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
    prediction_index: int | None = None,
    prediction_count: int | None = None,
) -> str:
    request_id = request_id or uuid.uuid4().hex[:12]
    event: dict[str, Any] = {
        "event_type": "prediction",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_field_count": input_field_count,
        "predicted_sales": predicted_sales,
        "latency_ms": latency_ms,
        "status": status,
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
        "prediction_index": prediction_index,
        "prediction_count": prediction_count,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, allow_nan=False) + "\n")
    return request_id


def log_api_request(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    model_name: str | None = None,
    model_version: int | None = None,
    run_id: str | None = None,
    error: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "event_type": "request",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "http_status": status_code,
        "latency_ms": latency_ms,
        "status": "failure" if status_code >= 400 else "success",
        "error": error,
        "model_name": model_name,
        "model_version": model_version,
        "run_id": run_id,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, allow_nan=False) + "\n")

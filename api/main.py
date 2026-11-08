from __future__ import annotations

import math
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from api.model_loader import ModelLoader
from api.monitoring_logger import log_api_request, log_inference
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BigMartInput,
    SalesPrediction,
)


model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_loader.load()
    yield


app = FastAPI(
    title="Big Mart Sales Prediction API",
    description="Lab 10 - regression model serving and API testing",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def monitor_api_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    error: str | None = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        if status_code >= 400:
            error = f"HTTP {status_code}"
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        log_api_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            error=error,
        )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    errors = []
    for error in exc.errors():
        locations = [str(value) for value in error["loc"] if value != "body"]
        errors.append(
            {
                "field": ".".join(locations),
                "message": error["msg"],
            }
        )
    log_inference(
        input_field_count=0,
        status="failure",
        error="Request validation failed",
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
        run_id=model_loader.run_id,
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Request validation failed",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    _: Request,
    exc: HTTPException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service unavailable",
            "details": [
                {"field": "prediction", "message": str(exc.detail)}
            ],
        },
    )


def _record_count(value: BigMartInput) -> int:
    return len(value.model_dump())


def _predict_records(
    records: list[BigMartInput],
    request_id: str,
) -> list[SalesPrediction]:
    started = time.perf_counter()
    input_field_count = _record_count(records[0]) if records else 0
    if not model_loader.is_ready():
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        log_inference(
            input_field_count=input_field_count,
            latency_ms=latency_ms,
            status="failure",
            error="Model or preprocessor is not available",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
            prediction_count=0,
        )
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:
        record_dicts = [record.model_dump() for record in records]
        assert model_loader.preprocessor is not None
        features = model_loader.preprocessor.transform(record_dicts)
        predictions = model_loader.model.predict(features)
        if len(predictions) != len(records):
            raise ValueError("Model returned an unexpected number of predictions.")
        outputs = [float(value) for value in predictions]
        if not all(math.isfinite(value) for value in outputs):
            raise ValueError("Model returned a non-finite sales prediction.")
    except (ValueError, TypeError, RuntimeError) as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        log_inference(
            input_field_count=input_field_count,
            latency_ms=latency_ms,
            status="failure",
            error=type(exc).__name__,
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
            prediction_count=0,
        )
        raise HTTPException(
            status_code=500,
            detail="Big Mart sales prediction failed",
        ) from exc

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    for index, output in enumerate(outputs):
        log_inference(
            input_field_count=input_field_count,
            predicted_sales=output,
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
            prediction_index=index,
            prediction_count=len(outputs),
        )
    return [SalesPrediction(predicted_sales=value) for value in outputs]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Big Mart Sales Prediction API is running",
        "lab": "Lab 10",
        "status": "success",
    }


@app.get("/health")
def health() -> dict[str, str]:
    if not model_loader.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return {
        "service": "Big Mart Sales Prediction API",
        "version": app.version,
        "target_column": "Item_Outlet_Sales",
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "run_id": model_loader.run_id,
        "model_uri": model_loader.model_uri,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


@app.post("/predict", response_model=SalesPrediction)
def predict(data: BigMartInput, request: Request) -> SalesPrediction:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    return _predict_records([data], request_id)[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    data: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    predictions = _predict_records(data.items, request_id)
    return BatchPredictionResponse(
        count=len(predictions),
        predictions=predictions,
    )

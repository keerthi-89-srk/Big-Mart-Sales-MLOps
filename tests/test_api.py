from __future__ import annotations

import json

from api import main, monitoring_logger


def test_root_and_health(api_client):
    assert api_client.get("/").status_code == 200
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_returns_unavailable_without_model(api_client, monkeypatch):
    monkeypatch.setattr(main.model_loader, "is_ready", lambda: False)
    response = api_client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"


def test_metadata_includes_regression_model_provenance(api_client):
    response = api_client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Big Mart Sales Prediction API"
    assert data["target_column"] == "Item_Outlet_Sales"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["model_version"] == 3


def test_predict_returns_sales_not_classification_values(
    api_client,
    valid_big_mart_item,
):
    response = api_client.post("/predict", json=valid_big_mart_item)
    assert response.status_code == 200
    assert response.json() == {
        "predicted_sales": valid_big_mart_item["Item_MRP"] * 10
    }


def test_batch_prediction_returns_one_sales_value_per_item(
    api_client,
    valid_big_mart_item,
):
    second = valid_big_mart_item | {"Item_MRP": 100.0}
    response = api_client.post(
        "/predict/batch",
        json={"items": [valid_big_mart_item, second]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "count": 2,
        "predictions": [
            {"predicted_sales": valid_big_mart_item["Item_MRP"] * 10},
            {"predicted_sales": 1000.0},
        ],
    }


def test_invalid_category_is_rejected(api_client, valid_big_mart_item):
    payload = valid_big_mart_item | {"Item_Type": "Unknown"}
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["details"][0]["field"] == "Item_Type"


def test_missing_and_extra_features_are_rejected(api_client, valid_big_mart_item):
    missing = valid_big_mart_item.copy()
    del missing["Outlet_Type"]
    extra = valid_big_mart_item | {"Item_Outlet_Sales": 500.0}

    missing_response = api_client.post("/predict", json=missing)
    extra_response = api_client.post("/predict", json=extra)
    assert missing_response.status_code == 422
    assert extra_response.status_code == 422


def test_negative_and_non_finite_features_are_rejected(
    api_client,
    valid_big_mart_item,
):
    negative = valid_big_mart_item | {"Item_MRP": -1}
    infinite = valid_big_mart_item | {"Item_Visibility": float("inf")}

    assert api_client.post("/predict", json=negative).status_code == 422
    response = api_client.post(
        "/predict",
        content=json.dumps(infinite),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_batch_size_is_limited_to_twenty(api_client, valid_big_mart_item):
    response = api_client.post(
        "/predict/batch",
        json={"items": [valid_big_mart_item] * 21},
    )
    assert response.status_code == 422


def test_empty_batch_is_rejected(api_client):
    response = api_client.post("/predict/batch", json={"items": []})
    assert response.status_code == 422


def test_malformed_json_is_rejected(api_client):
    response = api_client.post(
        "/predict",
        content='{"Item_Identifier":',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_inference_log_excludes_raw_item_values(
    api_client,
    valid_big_mart_item,
    tmp_path,
):
    log_path = tmp_path / "inference_log.jsonl"
    from api import monitoring_logger

    monitoring_logger.LOG_PATH = log_path
    response = api_client.post("/predict", json=valid_big_mart_item)
    assert response.status_code == 200

    contents = log_path.read_text(encoding="utf-8")
    assert valid_big_mart_item["Item_Identifier"] not in contents
    assert str(valid_big_mart_item["Item_MRP"]) not in contents
    events = [json.loads(line) for line in contents.splitlines()]
    prediction = next(event for event in events if event["event_type"] == "prediction")
    request = next(event for event in events if event["event_type"] == "request")
    assert prediction["input_field_count"] == len(valid_big_mart_item)
    assert prediction["model_version"] == 3
    assert prediction["status"] == "success"
    assert request["status"] == "success"
    assert request["request_id"] == response.headers["X-Request-ID"]
    assert prediction["request_id"] == request["request_id"]


def test_preprocessor_failure_returns_logged_server_error(
    api_client,
    valid_big_mart_item,
    monkeypatch,
):
    def fail_transform(records):
        raise ValueError("internal detail")

    monkeypatch.setattr(
        main.model_loader.preprocessor,
        "transform",
        fail_transform,
    )
    response = api_client.post("/predict", json=valid_big_mart_item)
    assert response.status_code == 500
    assert "internal detail" not in response.text

from __future__ import annotations

import statistics
import time


def test_repeated_prediction_latency_and_consistency(
    api_client,
    valid_big_mart_item,
):
    latencies = []
    predictions = []
    for _ in range(20):
        started = time.perf_counter()
        response = api_client.post("/predict", json=valid_big_mart_item)
        latencies.append((time.perf_counter() - started) * 1000)
        assert response.status_code == 200
        predictions.append(response.json()["predicted_sales"])

    assert len(set(predictions)) == 1
    print(
        "Single prediction latency (ms): "
        f"average={statistics.mean(latencies):.3f}, "
        f"median={statistics.median(latencies):.3f}, "
        f"max={max(latencies):.3f}"
    )


def test_batch_prediction_performance(api_client, valid_big_mart_item):
    payload = {"items": [valid_big_mart_item.copy() for _ in range(20)]}
    started = time.perf_counter()
    response = api_client.post("/predict/batch", json=payload)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert response.status_code == 200
    assert response.json()["count"] == 20
    assert len(response.json()["predictions"]) == 20
    print(f"20-item batch latency (ms): {elapsed_ms:.3f}")

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main, monitoring_logger


class FakePreprocessor:
    feature_names = ["Item_MRP"]

    def transform(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(
            {"Item_MRP": [float(record["Item_MRP"]) for record in records]}
        )


class FakeRegressor:
    n_features_in_ = 1
    feature_names_in_ = np.array(["Item_MRP"])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["Item_MRP"].to_numpy(dtype=float) * 10


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    loader = main.model_loader
    monkeypatch.setattr(loader, "load", lambda: None)
    monkeypatch.setattr(loader, "model", FakeRegressor())
    monkeypatch.setattr(loader, "preprocessor", FakePreprocessor())
    monkeypatch.setattr(loader, "model_name", "BigMart_Sales_Model")
    monkeypatch.setattr(loader, "model_version", 3)
    monkeypatch.setattr(loader, "run_id", "test-run")
    monkeypatch.setattr(loader, "model_uri", "models:/BigMart_Sales_Model/3")
    monkeypatch.setattr(
        monitoring_logger,
        "LOG_PATH",
        tmp_path / "inference_log.jsonl",
    )
    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def valid_big_mart_item() -> dict[str, Any]:
    return {
        "Item_Identifier": "FDA15",
        "Item_Weight": 9.3,
        "Item_Fat_Content": "Low Fat",
        "Item_Visibility": 0.016047301,
        "Item_Type": "Dairy",
        "Item_MRP": 249.8092,
        "Outlet_Identifier": "OUT049",
        "Outlet_Establishment_Year": 1999,
        "Outlet_Size": "Medium",
        "Outlet_Location_Type": "Tier 1",
        "Outlet_Type": "Supermarket Type1",
    }

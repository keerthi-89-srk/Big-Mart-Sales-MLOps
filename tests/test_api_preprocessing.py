from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from api.preprocessing import BigMartPreprocessor, TARGET_COLUMN


def test_transform_matches_existing_processed_training_row():
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "raw" / "Train.csv"
    test_path = project_root / "data" / "raw" / "Test.csv"
    processed_path = (
        project_root / "data" / "processed" / "X_train_processed.csv"
    )
    expected = pd.read_csv(processed_path, nrows=1)
    preprocessor = BigMartPreprocessor.fit_from_files(
        train_path,
        test_path,
        expected.columns.tolist(),
    )
    raw = pd.read_csv(train_path, nrows=1).drop(columns=[TARGET_COLUMN])
    actual = preprocessor.transform(raw.to_dict(orient="records"))

    assert actual.columns.tolist() == expected.columns.tolist()
    assert np.allclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


def test_nullable_raw_fields_are_imputed_using_training_statistics():
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "raw" / "Train.csv"
    test_path = project_root / "data" / "raw" / "Test.csv"
    processed_path = (
        project_root / "data" / "processed" / "X_train_processed.csv"
    )
    features = pd.read_csv(processed_path, nrows=0).columns.tolist()
    preprocessor = BigMartPreprocessor.fit_from_files(
        train_path,
        test_path,
        features,
    )
    raw = pd.read_csv(train_path, nrows=1).drop(columns=[TARGET_COLUMN])
    raw.loc[raw.index[0], ["Item_Weight", "Outlet_Size"]] = None
    transformed = preprocessor.transform(raw.to_dict(orient="records"))

    assert transformed.shape == (1, len(features))
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()

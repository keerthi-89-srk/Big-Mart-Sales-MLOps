from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TEST_PATH, TRAIN_PATH


def load_datasets(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def clean_train_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the missing-value and feature-engineering logic from the notebook."""
    cleaned = train_df.copy()

    cleaned["Item_Fat_Content"] = cleaned["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    )
    cleaned["Item_Weight"] = cleaned["Item_Weight"].fillna(cleaned["Item_Weight"].median())
    cleaned["Outlet_Size"] = cleaned["Outlet_Size"].fillna(cleaned["Outlet_Size"].mode()[0])
    cleaned["Outlet_Age"] = 2026 - cleaned["Outlet_Establishment_Year"]

    return cleaned


def prepare_features(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the model-ready feature matrix and target vector."""
    features = train_df.drop(columns=["Item_Outlet_Sales"], errors="ignore").copy()
    target = train_df["Item_Outlet_Sales"].copy()

    feature_columns_to_drop = [
        column for column in ["Item_Identifier", "Outlet_Identifier", "Outlet_Establishment_Year"]
        if column in features.columns
    ]
    if feature_columns_to_drop:
        features = features.drop(columns=feature_columns_to_drop)

    categorical_columns = features.select_dtypes(include=["object"]).columns.tolist()
    features = pd.get_dummies(features, columns=categorical_columns, drop_first=True)

    return features, target

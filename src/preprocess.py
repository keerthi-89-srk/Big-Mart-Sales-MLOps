from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
TRAIN_PATH = PROJECT_ROOT / "Train.csv"
TEST_PATH = PROJECT_ROOT / "Test.csv"
TARGET_COLUMN = "Item_Outlet_Sales"


def load_datasets(train_path: str | Path = TRAIN_PATH, test_path: str | Path = TEST_PATH):
    """Load the notebook's original raw train and test CSV files."""
    print("[INFO] Loading raw datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"[INFO] Train rows: {train_df.shape[0]}, columns: {train_df.shape[1]}")
    print(f"[INFO] Test rows: {test_df.shape[0]}, columns: {test_df.shape[1]}")
    return train_df, test_df


def clean_train_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the actual preprocessing logic used in the notebook."""
    cleaned = train_df.copy()

    cleaned["Item_Fat_Content"] = cleaned["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    )

    cleaned["Item_Weight"] = cleaned["Item_Weight"].fillna(cleaned["Item_Weight"].median())
    cleaned["Outlet_Size"] = cleaned["Outlet_Size"].fillna(cleaned["Outlet_Size"].mode()[0])
    cleaned["Outlet_Age"] = 2026 - cleaned["Outlet_Establishment_Year"]

    return cleaned


def prepare_features(train_df: pd.DataFrame):
    """Split into feature matrix and target column using the notebook logic."""
    X = train_df.drop(TARGET_COLUMN, axis=1)
    y = train_df[TARGET_COLUMN]

    X = X.drop(
        ["Item_Identifier", "Outlet_Identifier", "Outlet_Establishment_Year"],
        axis=1,
        errors="ignore",
    )

    categorical_columns = X.select_dtypes(include="object").columns
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    return X, y


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        train_df, test_df = load_datasets(TRAIN_PATH, TEST_PATH)

        cleaned_train = clean_train_data(train_df)
        X, y = prepare_features(cleaned_train)

        processed_summary = {
            "target_column": TARGET_COLUMN,
            "raw_train_shape": list(train_df.shape),
            "raw_test_shape": list(test_df.shape),
            "cleaned_train_shape": list(cleaned_train.shape),
            "feature_matrix_shape": list(X.shape),
            "target_shape": list(y.shape),
            "categorical_columns": list(X.select_dtypes(include="object").columns),
            "missing_item_weight_after_fill": int(cleaned_train["Item_Weight"].isnull().sum()),
            "missing_outlet_size_after_fill": int(cleaned_train["Outlet_Size"].isnull().sum()),
        }

        cleaned_train.to_csv(PROCESSED_DIR / "train_cleaned.csv", index=False)
        X.to_csv(PROCESSED_DIR / "X_train_processed.csv", index=False)
        y.to_csv(PROCESSED_DIR / "y_train_processed.csv", index=False)
        (PROCESSED_DIR / "preprocess_summary.json").write_text(json.dumps(processed_summary, indent=2), encoding="utf-8")

        print("[SUCCESS] Preprocessing completed successfully.")
        print(f"[INFO] Processed files saved to: {PROCESSED_DIR}")

    except Exception as exc:
        print("[ERROR] Preprocessing failed.")
        print(f"[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()

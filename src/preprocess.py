from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw dataset paths
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "Train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "Test.csv"

# Processed output directory
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_datasets(
    train_path: Path,
    test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[INFO] Loading raw datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(
        f"[INFO] Train rows: {len(train_df)}, "
        f"columns: {train_df.shape[1]}"
    )

    print(
        f"[INFO] Test rows: {len(test_df)}, "
        f"columns: {test_df.shape[1]}"
    )

    return train_df, test_df


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    target_column = "Item_Outlet_Sales"

    if train_df.empty:
        raise ValueError("Training dataset is empty.")
    if test_df.empty:
        raise ValueError("Test dataset is empty.")
    if train_df.columns.has_duplicates:
        raise ValueError("Training dataset contains duplicate column names.")
    if test_df.columns.has_duplicates:
        raise ValueError("Test dataset contains duplicate column names.")
    if target_column not in train_df.columns:
        raise ValueError(
            f"Training dataset is missing target column: {target_column}"
        )
    if target_column in test_df.columns:
        raise ValueError(
            f"Test dataset must not contain target column: {target_column}"
        )

    # Separate target from training features
    train_features = train_df.drop(
        columns=[target_column]
    )
    missing_columns = train_features.columns.difference(test_df.columns)
    extra_columns = test_df.columns.difference(train_features.columns)
    if not missing_columns.empty or not extra_columns.empty:
        raise ValueError(
            "Training and test feature columns do not match. "
            f"Missing from test: {missing_columns.tolist()}; "
            f"extra in test: {extra_columns.tolist()}."
        )
    test_features = test_df.loc[:, train_features.columns].copy()

    try:
        y_train = pd.to_numeric(train_df[target_column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Target column {target_column} must contain numeric values."
        ) from exc

    if y_train.isna().any():
        raise ValueError(f"Target column {target_column} contains missing values.")
    if not np.isfinite(y_train.to_numpy(dtype=float)).all():
        raise ValueError(f"Target column {target_column} contains non-finite values.")

    # Combine train and test features so that
    # categorical encoding is consistent
    combined = pd.concat(
        [train_features, test_features],
        axis=0,
        ignore_index=True
    )

    categorical_columns = combined.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    # Handle missing values
    for column in combined.columns:
        if column in categorical_columns:
            values = combined[column].astype(object)
            mode_value = values.mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            combined[column] = values.fillna(fill_value)
        elif pd.api.types.is_numeric_dtype(combined[column].dtype):
            if np.isinf(combined[column].dropna().to_numpy(dtype=float)).any():
                raise ValueError(
                    f"Numeric feature column {column} contains non-finite values."
                )
            if combined[column].isna().all():
                raise ValueError(
                    f"Numeric feature column {column} contains only missing values."
                )
            median_value = combined[column].median()
            combined[column] = combined[column].fillna(median_value)
        else:
            raise TypeError(
                f"Feature column {column} has unsupported dtype "
                f"{combined[column].dtype}."
            )

    # Standardize Item_Fat_Content values
    if "Item_Fat_Content" in combined.columns:

        combined["Item_Fat_Content"] = combined[
            "Item_Fat_Content"
        ].replace(
            {
                "LF": "Low Fat",
                "low fat": "Low Fat",
                "reg": "Regular"
            }
        )

    # One-hot encode categorical columns
    categorical_columns = combined.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    combined_encoded = pd.get_dummies(
        combined,
        columns=categorical_columns,
        drop_first=False
    )

    # Convert boolean columns to integers
    boolean_columns = combined_encoded.select_dtypes(
        include=["bool"]
    ).columns

    if len(boolean_columns) > 0:
        combined_encoded[boolean_columns] = (
            combined_encoded[boolean_columns].astype(int)
        )

    # Split back into train and test
    X_train = combined_encoded.iloc[
        :len(train_df)
    ].copy()

    X_test = combined_encoded.iloc[
        len(train_df):
    ].copy()

    # Ensure numeric data
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test, y_train


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series
) -> None:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed training features
    X_train.to_csv(
        PROCESSED_DIR / "X_train_processed.csv",
        index=False
    )

    # Save processed test features
    X_test.to_csv(
        PROCESSED_DIR / "X_test_processed.csv",
        index=False
    )

    # Save target
    y_train.to_csv(
        PROCESSED_DIR / "y_train_processed.csv",
        index=False
    )

    print(
        "[SUCCESS] Preprocessing completed successfully."
    )

    print(
        f"[INFO] Processed files saved to: "
        f"{PROCESSED_DIR}"
    )


def main() -> None:

    try:

        train_df, test_df = load_datasets(
            TRAIN_PATH,
            TEST_PATH
        )

        X_train, X_test, y_train = preprocess_data(
            train_df,
            test_df
        )

        save_processed_data(
            X_train,
            X_test,
            y_train
        )

    except Exception as exc:

        print("[ERROR] Preprocessing failed.")
        print(f"[ERROR] {exc}")

        raise


if __name__ == "__main__":
    main()
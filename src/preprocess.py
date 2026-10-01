from __future__ import annotations

from pathlib import Path

import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw dataset paths
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "Train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "Test.csv"

# Processed output directory
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_datasets(train_path: Path, test_path: Path):
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
):
    target_column = "Item_Outlet_Sales"

    # Separate target from training features
    train_features = train_df.drop(
        columns=[target_column]
    )

    test_features = test_df.copy()

    # Combine train and test features so that
    # categorical encoding is consistent
    combined = pd.concat(
        [train_features, test_features],
        axis=0,
        ignore_index=True
    )

    # Handle missing values
    for column in combined.columns:

        if combined[column].dtype == "object":

            mode_value = combined[column].mode()

            if not mode_value.empty:
                combined[column] = combined[column].fillna(
                    mode_value.iloc[0]
                )
            else:
                combined[column] = combined[column].fillna(
                    "Unknown"
                )

        else:

            combined[column] = combined[column].fillna(
                combined[column].median()
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
        include=["object"]
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

    y_train = train_df[
        target_column
    ].copy()

    # Ensure numeric data
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test, y_train


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series
):

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


def main():

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
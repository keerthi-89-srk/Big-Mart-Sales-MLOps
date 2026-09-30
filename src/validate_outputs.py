from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():

    print("==========================================")
    print("Lab 5 - Validate Preprocessing Outputs")
    print("==========================================")

    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv"
    ]

    for file_name in required_files:

        file_path = PROCESSED_DIR / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required processed file not found: {file_path}"
            )

        print(f"[SUCCESS] Found: {file_name}")

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    )

    print("\n[INFO] Checking dataset dimensions...")

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_train shape: {y_train.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train row counts do not match."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test row counts do not match."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "X_train and X_test feature counts do not match."
        )

    print("[SUCCESS] Row and feature dimensions are consistent.")

    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()

    print(f"[INFO] Missing values in X_train: {missing_train}")
    print(f"[INFO] Missing values in X_test: {missing_test}")

    if missing_train > 0 or missing_test > 0:
        raise ValueError(
            "Missing values detected in processed feature data."
        )

    print("[SUCCESS] No missing values found.")

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {
        "validation_status": "PASSED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape)
        },
        "data_quality": {
            "missing_values_X_train": int(missing_train),
            "missing_values_X_test": int(missing_test)
        },
        "errors": []
    }

    report_path = (
        ARTIFACTS_DIR /
        "preprocessing_summary_report.json"
    )

    report_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8"
    )

    print(
        f"[SUCCESS] Summary report saved to: {report_path}"
    )

    print("\n[SUCCESS] Output Validation PASSED.")

    print(
        "[SUCCESS] Processed data is clean "
        "and dimensionally consistent."
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 2,
    "random_state": 42,
    "n_jobs": -1
}


def load_data():

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).iloc[:, 0]

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).iloc[:, 0]

    return X_train, y_train, X_test, y_test


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test
):

    model = RandomForestRegressor(
        **MODEL_PARAMS
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions
    )

    return r2


def main():

    print("==========================================")
    print("Lab 5 - Reproducibility Validation")
    print("==========================================")

    print("[INFO] Loading processed data...")

    X_train, y_train, X_test, y_test = load_data()

    print(
        f"[INFO] X_train shape: {X_train.shape}"
    )

    print(
        f"[INFO] X_test shape: {X_test.shape}"
    )

    print("\n[INFO] Running execution 1...")

    r2_execution_1 = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print(
        f"[INFO] Execution 1 R2: "
        f"{r2_execution_1:.6f}"
    )

    print("\n[INFO] Running execution 2...")

    r2_execution_2 = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print(
        f"[INFO] Execution 2 R2: "
        f"{r2_execution_2:.6f}"
    )

    is_reproducible = (
        r2_execution_1 == r2_execution_2
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "parameters": MODEL_PARAMS,
        "execution_1_r2": float(r2_execution_1),
        "execution_2_r2": float(r2_execution_2),
        "is_strictly_reproducible": is_reproducible,
        "status": "PASSED" if is_reproducible else "FAILED"
    }

    report_path = (
        ARTIFACTS_DIR /
        "reproducibility_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=4
        ),
        encoding="utf-8"
    )

    if is_reproducible:

        print(
            "\n[SUCCESS] Pipeline is 100% reproducible."
        )

        print(
            f"[SUCCESS] Report saved to: "
            f"{report_path}"
        )

    else:

        print(
            "\n[ERROR] Pipeline is NOT reproducible."
        )

        print(
            f"[ERROR] Report saved to: "
            f"{report_path}"
        )


if __name__ == "__main__":
    main()
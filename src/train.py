from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TARGET_COLUMN = "Item_Outlet_Sales"


def load_training_data():
    print("[INFO] Loading processed training data...")
    X = pd.read_csv(PROCESSED_DIR / "X_train_processed.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train_processed.csv").iloc[:, 0]
    return X, y


def train_best_model(X: pd.DataFrame, y: pd.Series):
    print("[INFO] Splitting data into train and validation sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    print("[INFO] Training tuned Random Forest model using GridSearchCV...")
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        parameter_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    model_path = MODELS_DIR / "best_random_forest_model.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    training_summary = {
        "target_column": TARGET_COLUMN,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "best_parameters": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
        "model_path": str(model_path),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    print("[SUCCESS] Tuned Random Forest model trained successfully.")
    print(f"[INFO] Best parameters: {grid_search.best_params_}")
    print(f"[SUCCESS] Model saved to: {model_path}")

    return best_model, X_test, y_test


def main():
    try:
        X, y = load_training_data()
        train_best_model(X, y)
    except Exception as exc:
        print("[ERROR] Training failed.")
        print(f"[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()

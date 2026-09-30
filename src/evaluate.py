from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TARGET_COLUMN = "Item_Outlet_Sales"


def load_model_and_data():
    model_path = MODELS_DIR / "best_random_forest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run the training step first.")

    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    model = joblib.load(model_path)
    return model, X_test, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        "target_column": TARGET_COLUMN,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2_Score": float(r2),
    }

    print("[INFO] Evaluation metrics:")
    print(f"[INFO] MAE: {mae:.6f}")
    print(f"[INFO] RMSE: {rmse:.6f}")
    print(f"[INFO] R2 Score: {r2:.6f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "model_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions_df = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": predictions,
            "absolute_error": (y_test.reset_index(drop=True) - predictions).abs(),
        }
    )
    predictions_df.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    print(f"[SUCCESS] Evaluation results saved to: {OUTPUTS_DIR}")
    return metrics


def main():
    try:
        model, X_test, y_test = load_model_and_data()
        evaluate_model(model, X_test, y_test)
    except Exception as exc:
        print("[ERROR] Evaluation failed.")
        print(f"[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()

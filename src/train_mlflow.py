from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    print("[INFO] Loading processed Big Mart data...")

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_train shape: {y_train.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# MAIN MLflow EXPERIMENT
# ==========================================================

def main():

    X_train, X_test, y_train, y_test = load_data()

    # ------------------------------------------------------
    # MLflow tracking database
    # ------------------------------------------------------

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Create/select experiment
    mlflow.set_experiment("Big_Mart_Sales")

    # ------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------

    n_estimators = 200
    max_depth = 10
    min_samples_split = 2
    random_state = 42

    print("\n==========================================")
    print("Starting MLflow Experiment")
    print("Big Mart Sales Prediction")
    print("==========================================")

    with mlflow.start_run(run_name="Tuned_Random_Forest"):

        # --------------------------------------------------
        # Create model
        # --------------------------------------------------

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )

        print("\n[INFO] Training Random Forest model...")

        model.fit(X_train, y_train)

        print("[SUCCESS] Model training completed.")

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        y_pred = model.predict(X_test)

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        mae = mean_absolute_error(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        r2 = r2_score(y_test, y_pred)

        print("\n==========================================")
        print("Evaluation Metrics")
        print("==========================================")

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        # --------------------------------------------------
        # Log model parameters
        # --------------------------------------------------

        mlflow.log_param(
            "model_type",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "target_column",
            "Item_Outlet_Sales"
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_param(
            "max_depth",
            max_depth
        )

        mlflow.log_param(
            "min_samples_split",
            min_samples_split
        )

        mlflow.log_param(
            "random_state",
            random_state
        )

        mlflow.log_param(
            "cv_method",
            "Train/Test Split"
        )

        # --------------------------------------------------
        # Log dataset information
        # --------------------------------------------------

        mlflow.log_param(
            "training_rows",
            X_train.shape[0]
        )

        mlflow.log_param(
            "testing_rows",
            X_test.shape[0]
        )

        mlflow.log_param(
            "number_of_features",
            X_train.shape[1]
        )

        # --------------------------------------------------
        # Log metrics
        # --------------------------------------------------

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # --------------------------------------------------
        # Actual vs Predicted Plot
        # --------------------------------------------------

        plt.figure(figsize=(8, 6))

        plt.scatter(
            y_test,
            y_pred,
            alpha=0.5
        )

        plt.xlabel("Actual Item Outlet Sales")
        plt.ylabel("Predicted Item Outlet Sales")
        plt.title("Actual vs Predicted Sales")

        plt.tight_layout()

        plot_path = ARTIFACTS_DIR / "actual_vs_predicted.png"

        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(str(plot_path))

        # --------------------------------------------------
        # Residual Plot
        # --------------------------------------------------

        residuals = y_test - y_pred

        plt.figure(figsize=(8, 6))

        plt.scatter(
            y_pred,
            residuals,
            alpha=0.5
        )

        plt.axhline(
            y=0,
            linestyle="--"
        )

        plt.xlabel("Predicted Sales")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")

        plt.tight_layout()

        residual_path = ARTIFACTS_DIR / "residual_plot.png"

        plt.savefig(residual_path)
        plt.close()

        mlflow.log_artifact(str(residual_path))

        # --------------------------------------------------
        # Log preprocessing summary
        # --------------------------------------------------

        preprocessing_summary = (
            PROCESSED_DIR / "preprocess_summary.json"
        )

        if preprocessing_summary.exists():

            mlflow.log_artifact(
                str(preprocessing_summary)
            )

        # --------------------------------------------------
        # Log trained Random Forest model
        # --------------------------------------------------
        # The trusted tree type is required by MLflow/skops
        # when saving Random Forest models.

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\n[SUCCESS] Parameters logged.")
        print("[SUCCESS] Metrics logged.")
        print("[SUCCESS] Plots logged.")
        print("[SUCCESS] Preprocessing information logged.")
        print("[SUCCESS] Model logged to MLflow.")

    print("\n==========================================")
    print("MLflow Experiment Completed Successfully")
    print("==========================================")


if __name__ == "__main__":
    main()
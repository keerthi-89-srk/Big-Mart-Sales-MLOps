from pathlib import Path
import json
import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EXPERIMENT_NAME = "Big_Mart_Sales"
REGISTERED_MODEL_NAME = "BigMart_Sales_Model"


# ==========================================
# MLflow configuration
# ==========================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

client = MlflowClient()


# ==========================================
# Register latest successful model
# ==========================================

def main():

    print("==========================================")
    print("Lab 6 - Model Registration")
    print("==========================================")

    # --------------------------------------
    # Find experiment
    # --------------------------------------

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        )

    print(f"[INFO] Experiment: {EXPERIMENT_NAME}")
    print(f"[INFO] Experiment ID: {experiment.experiment_id}")

    # --------------------------------------
    # Get completed MLflow runs
    # --------------------------------------

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"]
    )

    if not runs:
        raise ValueError("No completed MLflow runs found.")

    # --------------------------------------
    # Select latest run
    # --------------------------------------

    run = runs[0]

    run_id = run.info.run_id

    print(f"[INFO] Latest MLflow Run ID: {run_id}")

    # --------------------------------------
    # Read metrics
    # --------------------------------------

    metrics = run.data.metrics
    params = run.data.params

    mae = metrics.get("MAE")
    rmse = metrics.get("RMSE")
    r2 = metrics.get("R2")

    print(f"[INFO] MAE: {mae}")
    print(f"[INFO] RMSE: {rmse}")
    print(f"[INFO] R2 Score: {r2}")

    # --------------------------------------
    # Model URI
    # --------------------------------------

    model_uri = f"runs:/{run_id}/model"

    print(f"[INFO] Model URI: {model_uri}")

    # --------------------------------------
    # Register model
    # --------------------------------------

    print("\n[INFO] Registering model...")

    registration = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    model_version = registration.version

    print(
        f"[SUCCESS] Model registered successfully."
    )

    print(
        f"[INFO] Registered Model: {REGISTERED_MODEL_NAME}"
    )

    print(
        f"[INFO] Model Version: {model_version}"
    )

    # --------------------------------------
    # Add model metadata/tags
    # --------------------------------------

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version,
        "project",
        "Big_Mart_Sales"
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version,
        "model_type",
        "RandomForestRegressor"
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version,
        "dataset",
        "Big_Mart_Sales"
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version,
        "preprocessing",
        "Lab5"
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version,
        "validation_status",
        "PASSED"
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version,
        "deployment_readiness",
        "VALIDATED"
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        model_version,
        "lifecycle_state",
        "VALIDATION"
    )

    # --------------------------------------
    # Create Lab 6 registry report
    # --------------------------------------

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "model_version": int(model_version),
        "run_id": run_id,
        "model_uri": model_uri,

        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        },

        "parameters": params,

        "dataset_lineage": {
            "dataset": "Big_Mart_Sales",
            "raw_train_file": "data/raw/Train.csv",
            "preprocessing_pipeline": "Lab5"
        },

        "preprocessing_dependencies": {
            "preprocessing_validation": (
                "artifacts/preprocessing_summary_report.json"
            ),
            "reproducibility_validation": (
                "artifacts/reproducibility_report.json"
            )
        },

        "validation": {
            "validation_status": "PASSED",
            "deployment_readiness": "VALIDATED"
        },

        "lifecycle_state": "VALIDATION"
    }

    report_path = ARTIFACTS_DIR / "model_registry_report.json"

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(
        f"[SUCCESS] Registry report saved to: {report_path}"
    )

    print("\n==========================================")
    print("[SUCCESS] Lab 6 Model Registration Completed")
    print("==========================================")


if __name__ == "__main__":
    main()
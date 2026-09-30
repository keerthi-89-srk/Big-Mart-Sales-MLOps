from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# Project configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

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
# Create Version 2
# ==========================================

def main():

    print("==========================================")
    print("Lab 6 - Create Model Version 2")
    print("==========================================")

    # --------------------------------------
    # Find experiment
    # --------------------------------------

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' not found."
        )

    print(
        f"[INFO] Experiment: {EXPERIMENT_NAME}"
    )

    # --------------------------------------
    # Get latest completed run
    # --------------------------------------

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"]
    )

    if not runs:
        raise ValueError(
            "No completed MLflow runs found."
        )

    run = runs[0]

    run_id = run.info.run_id

    print(
        f"[INFO] Using MLflow Run ID: {run_id}"
    )

    # --------------------------------------
    # Get existing model artifact
    # --------------------------------------

    model_uri = f"runs:/{run_id}/model"

    print(
        f"[INFO] Model URI: {model_uri}"
    )

    # --------------------------------------
    # Register model
    # --------------------------------------

    print(
        "\n[INFO] Registering another model version..."
    )

    registration = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    version = registration.version

    print(
        f"[SUCCESS] Model Version {version} created."
    )

    # --------------------------------------
    # Add metadata
    # --------------------------------------

    metadata_tags = {
        "project": "Big_Mart_Sales",
        "model_type": "RandomForestRegressor",
        "dataset": "Big_Mart_Sales",
        "preprocessing": "Lab5",
        "validation_status": "PASSED",
        "deployment_readiness": "VALIDATED",
        "lifecycle_state": "VALIDATION"
    }

    for key, value in metadata_tags.items():

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            key,
            value
        )

    # --------------------------------------
    # Display metrics
    # --------------------------------------

    metrics = run.data.metrics

    print("\n[INFO] Model Metrics")
    print("------------------------------------------")
    print(f"MAE:  {metrics.get('MAE')}")
    print(f"RMSE: {metrics.get('RMSE')}")
    print(f"R2:   {metrics.get('R2')}")

    print("\n==========================================")
    print("[SUCCESS] Model Version 2 Creation Completed")
    print("==========================================")


if __name__ == "__main__":
    main()
from pathlib import Path
import json

import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

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
# Compare registered model versions
# ==========================================

def main():

    print("==========================================")
    print("Lab 6 - Model Version Comparison")
    print("==========================================")

    # --------------------------------------
    # Get registered model
    # --------------------------------------

    try:
        registered_model = client.get_registered_model(
            REGISTERED_MODEL_NAME
        )
    except Exception as exc:
        raise ValueError(
            f"Registered model '{REGISTERED_MODEL_NAME}' "
            f"was not found. Run register_model.py first."
        ) from exc

    print(
        f"[INFO] Registered Model: "
        f"{registered_model.name}"
    )

    # --------------------------------------
    # Get all model versions
    # --------------------------------------

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not versions:
        raise ValueError(
            "No model versions found."
        )

    print(
        f"[INFO] Number of registered versions: "
        f"{len(versions)}"
    )

    # --------------------------------------
    # Collect version information
    # --------------------------------------

    comparison = []

    for version in versions:

        run_id = version.run_id

        metrics = {}

        if run_id:
            try:
                run = client.get_run(run_id)
                metrics = run.data.metrics
            except Exception:
                metrics = {}

        mae = metrics.get("MAE")
        rmse = metrics.get("RMSE")
        r2 = metrics.get("R2")

        tags = version.tags

        deployment_readiness = tags.get(
            "deployment_readiness",
            "NOT_AVAILABLE"
        )

        lifecycle_state = tags.get(
            "lifecycle_state",
            "NOT_AVAILABLE"
        )

        comparison.append(
            {
                "model_version": int(version.version),
                "run_id": run_id,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "lifecycle_state": lifecycle_state,
                "deployment_readiness": deployment_readiness
            }
        )

    # --------------------------------------
    # Sort versions
    # --------------------------------------

    comparison.sort(
        key=lambda item: item["model_version"]
    )

    # --------------------------------------
    # Display comparison
    # --------------------------------------

    print("\n[INFO] Model Version Comparison")
    print("------------------------------------------")

    for item in comparison:

        print(
            f"Version {item['model_version']} | "
            f"MAE: {item['MAE']} | "
            f"RMSE: {item['RMSE']} | "
            f"R2: {item['R2']} | "
            f"State: {item['lifecycle_state']}"
        )

    # --------------------------------------
    # Find best version
    # --------------------------------------

    valid_versions = [
        item
        for item in comparison
        if item["MAE"] is not None
        and item["RMSE"] is not None
        and item["R2"] is not None
    ]

    if valid_versions:

        best_version = min(
            valid_versions,
            key=lambda item: (
                item["RMSE"],
                item["MAE"],
                -item["R2"]
            )
        )

        print("\n[INFO] Best candidate based on metrics:")
        print(
            f"[INFO] Version: "
            f"{best_version['model_version']}"
        )
        print(
            f"[INFO] MAE: {best_version['MAE']}"
        )
        print(
            f"[INFO] RMSE: {best_version['RMSE']}"
        )
        print(
            f"[INFO] R2: {best_version['R2']}"
        )

    else:
        best_version = None

        print(
            "\n[WARNING] No complete evaluation metrics "
            "were available."
        )

    # --------------------------------------
    # Create comparison report
    # --------------------------------------

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "total_versions": len(comparison),
        "versions": comparison,
        "best_candidate": best_version,
        "comparison_criteria": {
            "MAE": "Lower is better",
            "RMSE": "Lower is better",
            "R2": "Higher is better"
        }
    }

    report_path = (
        ARTIFACTS_DIR /
        "model_comparison_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(
        f"\n[SUCCESS] Comparison report saved to:"
    )
    print(report_path)

    print("\n==========================================")
    print(
        "[SUCCESS] Lab 6 Model Comparison Completed"
    )
    print("==========================================")


if __name__ == "__main__":
    main()
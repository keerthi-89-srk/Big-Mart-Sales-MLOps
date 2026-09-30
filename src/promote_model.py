from pathlib import Path
import argparse
import json

import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# Project configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REGISTERED_MODEL_NAME = "BigMart_Sales_Model"


# ==========================================
# MLflow configuration
# ==========================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

client = MlflowClient()


# ==========================================
# Promote model
# ==========================================

def promote_model(version, target_state):

    print("==========================================")
    print("Lab 6 - Model Promotion")
    print("==========================================")

    version = str(version)
    target_state = target_state.upper()

    allowed_states = [
        "VALIDATION",
        "STAGING",
        "PRODUCTION"
    ]

    if target_state not in allowed_states:
        raise ValueError(
            f"Invalid state: {target_state}. "
            f"Allowed states: {allowed_states}"
        )

    # --------------------------------------
    # Check model version
    # --------------------------------------

    print(
        f"[INFO] Model: {REGISTERED_MODEL_NAME}"
    )

    print(
        f"[INFO] Requested Version: {version}"
    )

    model_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version
    )

    print(
        f"[INFO] Run ID: {model_version.run_id}"
    )

    # --------------------------------------
    # Check deployment readiness
    # --------------------------------------

    deployment_readiness = model_version.tags.get(
        "deployment_readiness",
        "NOT_AVAILABLE"
    )

    validation_status = model_version.tags.get(
        "validation_status",
        "NOT_AVAILABLE"
    )

    print(
        f"[INFO] Validation Status: "
        f"{validation_status}"
    )

    print(
        f"[INFO] Deployment Readiness: "
        f"{deployment_readiness}"
    )

    if validation_status != "PASSED":
        raise ValueError(
            "Model validation has not PASSED."
        )

    if deployment_readiness != "VALIDATED":
        raise ValueError(
            "Model is not deployment-ready."
        )

    # --------------------------------------
    # Assign lifecycle alias
    # --------------------------------------

    alias = target_state.lower()

    print(
        f"[INFO] Assigning MLflow alias: "
        f"{alias}"
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        alias,
        version
    )

    # --------------------------------------
    # Update lifecycle tag
    # --------------------------------------

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        version,
        "lifecycle_state",
        target_state
    )

    # --------------------------------------
    # Read metrics
    # --------------------------------------

    run = client.get_run(
        model_version.run_id
    )

    metrics = run.data.metrics

    # --------------------------------------
    # Save promotion report
    # --------------------------------------

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "registered_model_name":
            REGISTERED_MODEL_NAME,

        "model_version":
            int(version),

        "run_id":
            model_version.run_id,

        "lifecycle_state":
            target_state,

        "mlflow_alias":
            alias,

        "validation_status":
            validation_status,

        "deployment_readiness":
            deployment_readiness,

        "metrics": {
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("R2")
        },

        "promotion_status":
            "SUCCESS"
    }

    report_path = (
        ARTIFACTS_DIR /
        "model_promotion_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(
        f"[SUCCESS] Model Version {version} "
        f"promoted to {target_state}."
    )

    print(
        f"[SUCCESS] Alias '{alias}' assigned."
    )

    print(
        f"[SUCCESS] Promotion report saved to:"
    )

    print(report_path)

    print("\n==========================================")
    print(
        "[SUCCESS] Model Promotion Completed"
    )
    print("==========================================")


# ==========================================
# Main
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Promote Big Mart model version."
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Model version to promote"
    )

    parser.add_argument(
        "--state",
        required=True,
        choices=[
            "VALIDATION",
            "STAGING",
            "PRODUCTION"
        ],
        help="Target lifecycle state"
    )

    args = parser.parse_args()

    promote_model(
        args.version,
        args.state
    )


if __name__ == "__main__":
    main()
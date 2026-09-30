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
# Rollback production model
# ==========================================

def rollback_model(version):

    print("==========================================")
    print("Lab 6 - Model Rollback")
    print("==========================================")

    version = str(version)

    # --------------------------------------
    # Check requested version
    # --------------------------------------

    print(
        f"[INFO] Registered Model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"[INFO] Rollback target version: {version}"
    )

    target_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version
    )

    print(
        f"[INFO] Target Run ID: "
        f"{target_version.run_id}"
    )

    # --------------------------------------
    # Check target model readiness
    # --------------------------------------

    validation_status = target_version.tags.get(
        "validation_status",
        "NOT_AVAILABLE"
    )

    deployment_readiness = target_version.tags.get(
        "deployment_readiness",
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
            "Rollback target has not passed validation."
        )

    if deployment_readiness != "VALIDATED":
        raise ValueError(
            "Rollback target is not deployment-ready."
        )

    # --------------------------------------
    # Get current production version
    # --------------------------------------

    current_production_version = None

    try:

        current_production = (
            client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                "production"
            )
        )

        current_production_version = (
            str(current_production.version)
        )

        print(
            f"[INFO] Current production version: "
            f"{current_production_version}"
        )

    except Exception:

        print(
            "[WARNING] No production alias currently exists."
        )

    # --------------------------------------
    # Prevent meaningless rollback
    # --------------------------------------

    if (
        current_production_version is not None
        and current_production_version == version
    ):

        print(
            "[WARNING] Target version is already "
            "the production version."
        )

        rollback_status = "NO_CHANGE"

    else:

        # ----------------------------------
        # Assign production alias
        # ----------------------------------

        print(
            "[INFO] Reassigning production alias..."
        )

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "production",
            version
        )

        # ----------------------------------
        # Update lifecycle tag
        # ----------------------------------

        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            version,
            "lifecycle_state",
            "PRODUCTION"
        )

        rollback_status = "SUCCESS"

        print(
            f"[SUCCESS] Production rolled back "
            f"to Version {version}."
        )

    # --------------------------------------
    # Get target metrics
    # --------------------------------------

    run = client.get_run(
        target_version.run_id
    )

    metrics = run.data.metrics

    # --------------------------------------
    # Create rollback report
    # --------------------------------------

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {

        "registered_model_name":
            REGISTERED_MODEL_NAME,

        "previous_production_version":
            current_production_version,

        "rollback_target_version":
            int(version),

        "target_run_id":
            target_version.run_id,

        "lifecycle_state":
            "PRODUCTION",

        "production_alias":
            "production",

        "validation_status":
            validation_status,

        "deployment_readiness":
            deployment_readiness,

        "metrics": {
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("R2")
        },

        "rollback_status":
            rollback_status
    }

    report_path = (
        ARTIFACTS_DIR /
        "model_rollback_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print(
        f"[SUCCESS] Rollback report saved to:"
    )

    print(report_path)

    print("\n==========================================")
    print("[SUCCESS] Model Rollback Completed")
    print("==========================================")


# ==========================================
# Main
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="Rollback Big Mart production model."
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Version to assign to production"
    )

    args = parser.parse_args()

    rollback_model(args.version)


if __name__ == "__main__":
    main()
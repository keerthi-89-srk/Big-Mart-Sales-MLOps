from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from api.preprocessing import BigMartPreprocessor, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lab10_api.json"


class ModelLoader:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        self.config_path = config_path
        self.model: Any = None
        self.preprocessor: BigMartPreprocessor | None = None
        self.metadata: dict[str, Any] = {}
        self.model_name: str | None = None
        self.model_version: int | None = None
        self.run_id: str | None = None
        self.model_uri: str | None = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load(self) -> None:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"API configuration not found: {self.config_path}")
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        bundle_setting = config.get("bundle_directory")
        if not isinstance(bundle_setting, str) or not bundle_setting:
            raise ValueError("API configuration must define bundle_directory.")
        bundle_path = Path(bundle_setting)
        if not bundle_path.is_absolute():
            bundle_path = PROJECT_ROOT / bundle_path
        bundle_path = bundle_path.resolve()

        metadata_path = bundle_path / "model_metadata.json"
        model_path = bundle_path / "model.joblib"
        preprocessor_path = bundle_path / "preprocessor.json"
        for path in (metadata_path, model_path, preprocessor_path):
            if not path.is_file():
                raise FileNotFoundError(f"API bundle file is missing: {path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("target_column") != TARGET_COLUMN:
            raise ValueError("API bundle target must be Item_Outlet_Sales.")
        if metadata.get("model_name") != config.get("model_name"):
            raise ValueError("API bundle model name does not match configuration.")
        if metadata.get("model_version") != config.get("model_version"):
            raise ValueError("API bundle model version does not match configuration.")

        hashes = metadata.get("sha256")
        if not isinstance(hashes, dict):
            raise ValueError("API bundle metadata must include artifact checksums.")
        for filename, path in (
            ("model.joblib", model_path),
            ("preprocessor.json", preprocessor_path),
        ):
            expected = hashes.get(filename)
            if not isinstance(expected, str) or self._sha256(path) != expected:
                raise ValueError(f"API bundle checksum failed: {filename}")

        preprocessor_data = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        preprocessor = BigMartPreprocessor.from_metadata(preprocessor_data)
        model = joblib.load(model_path)
        if not callable(getattr(model, "predict", None)):
            raise TypeError("API bundle model does not provide predict().")
        expected_features = getattr(model, "n_features_in_", None)
        if (
            expected_features is not None
            and int(expected_features) != len(preprocessor.feature_names)
        ):
            raise ValueError(
                "API bundle model and preprocessor feature counts are inconsistent."
            )
        model_features = getattr(model, "feature_names_in_", None)
        if (
            model_features is not None
            and list(model_features) != preprocessor.feature_names
        ):
            raise ValueError(
                "API bundle model and preprocessor feature names are inconsistent."
            )

        self.metadata = metadata
        self.model = model
        self.preprocessor = preprocessor
        self.model_name = metadata["model_name"]
        self.model_version = int(metadata["model_version"])
        self.run_id = metadata["run_id"]
        self.model_uri = metadata["model_uri"]

        log_setting = config.get("inference_log")
        if isinstance(log_setting, str) and log_setting:
            log_path = Path(log_setting)
            if not log_path.is_absolute():
                log_path = PROJECT_ROOT / log_path
            from api import monitoring_logger

            monitoring_logger.LOG_PATH = log_path

    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None

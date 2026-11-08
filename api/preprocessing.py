from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_COLUMN = "Item_Outlet_Sales"
FAT_CONTENT_MAPPING = {
    "LF": "Low Fat",
    "low fat": "Low Fat",
    "reg": "Regular",
}


class BigMartPreprocessor:
    def __init__(
        self,
        feature_names: list[str],
        raw_columns: list[str],
        fill_values: dict[str, Any],
    ) -> None:
        if not feature_names or not raw_columns:
            raise ValueError("Preprocessor feature and raw-column lists cannot be empty.")
        if set(raw_columns) != set(fill_values):
            raise ValueError("Preprocessor imputation values do not match raw columns.")
        self.feature_names = list(feature_names)
        self.raw_columns = list(raw_columns)
        self.fill_values = dict(fill_values)

    @classmethod
    def fit_from_files(
        cls,
        train_path: Path,
        test_path: Path,
        feature_names: list[str],
    ) -> BigMartPreprocessor:
        if not train_path.is_file():
            raise FileNotFoundError(f"Big Mart training data not found: {train_path}")
        if not test_path.is_file():
            raise FileNotFoundError(f"Big Mart test data not found: {test_path}")

        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        if TARGET_COLUMN not in train.columns:
            raise ValueError(f"Training data must include {TARGET_COLUMN}.")

        raw_columns = train.drop(columns=[TARGET_COLUMN]).columns.tolist()
        if test.columns.tolist() != raw_columns:
            raise ValueError("Train and test raw feature columns are inconsistent.")

        combined = pd.concat(
            [train.drop(columns=[TARGET_COLUMN]), test],
            ignore_index=True,
        )
        fill_values: dict[str, Any] = {}
        for column in raw_columns:
            if combined[column].dtype == "object":
                modes = combined[column].mode()
                fill_values[column] = (
                    modes.iloc[0] if not modes.empty else "Unknown"
                )
            else:
                median = combined[column].median()
                if pd.isna(median) or not math.isfinite(float(median)):
                    raise ValueError(
                        f"Cannot determine a finite imputation value for {column}."
                    )
                fill_values[column] = float(median)

        return cls(feature_names, raw_columns, fill_values)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> BigMartPreprocessor:
        feature_names = metadata.get("feature_names")
        raw_columns = metadata.get("raw_columns")
        fill_values = metadata.get("fill_values")
        if (
            not isinstance(feature_names, list)
            or not all(isinstance(item, str) for item in feature_names)
            or not isinstance(raw_columns, list)
            or not all(isinstance(item, str) for item in raw_columns)
            or not isinstance(fill_values, dict)
        ):
            raise ValueError("Preprocessing metadata has an invalid structure.")
        return cls(feature_names, raw_columns, fill_values)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "raw_columns": self.raw_columns,
            "fill_values": self.fill_values,
        }

    def transform(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        if not records:
            raise ValueError("At least one Big Mart record is required.")

        frame = pd.DataFrame(records)
        unexpected = set(frame.columns) - set(self.raw_columns)
        if unexpected:
            raise ValueError(
                f"Unexpected Big Mart feature(s): {', '.join(sorted(unexpected))}"
            )
        for column in self.raw_columns:
            if column not in frame.columns:
                frame[column] = None
            frame[column] = frame[column].fillna(self.fill_values[column])

        if "Item_Fat_Content" in frame.columns:
            frame["Item_Fat_Content"] = frame["Item_Fat_Content"].replace(
                FAT_CONTENT_MAPPING
            )

        categorical = frame.select_dtypes(include=["object"]).columns.tolist()
        encoded = pd.get_dummies(
            frame,
            columns=categorical,
            drop_first=False,
        )
        encoded = encoded.reindex(columns=self.feature_names, fill_value=0)
        encoded = encoded.astype(float)
        if encoded.isna().any().any() or not all(
            math.isfinite(float(value))
            for value in encoded.to_numpy().ravel()
        ):
            raise ValueError("Preprocessed Big Mart features must be finite.")
        return encoded

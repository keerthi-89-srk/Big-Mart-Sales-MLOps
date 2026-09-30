from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ARTIFACTS_DIR, TEST_PATH, TRAIN_PATH
from .data_preprocessing import clean_train_data, load_datasets, prepare_features
from .modeling import train_and_compare


def run_training(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
    output_dir: str | Path = ARTIFACTS_DIR,
):
    train_df, _ = load_datasets(train_path=train_path, test_path=test_path)
    clean_df = clean_train_data(train_df)
    X, y = prepare_features(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    comparison, best_model_name, best_model = train_and_compare(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "model_metrics.csv")
    with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(comparison.to_dict(orient="index"), file, indent=2)

    joblib.dump(best_model, output_dir / "final_model.pkl")

    print("Best model:", best_model_name)
    print(comparison.sort_values("R2 Score", ascending=False))

    return comparison, best_model_name, best_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Big Mart sales prediction models.")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH, help="Path to the training CSV file.")
    parser.add_argument("--test-path", type=Path, default=TEST_PATH, help="Path to the test CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory where model metrics and artifacts are saved.",
    )
    args = parser.parse_args()

    run_training(train_path=args.train_path, test_path=args.test_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

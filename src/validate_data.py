from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "Train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "raw" / "Test.csv"


def validate_file(path):
    print(f"[INFO] Validating: {path.name}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print(f"[INFO] Rows: {df.shape[0]}")
    print(f"[INFO] Columns: {df.shape[1]}")

    if df.empty:
        raise ValueError(f"{path.name} is empty.")

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(f"[WARNING] Duplicate rows found: {duplicate_count}")
    else:
        print("[SUCCESS] No duplicate rows found.")

    missing_count = df.isnull().sum().sum()

    print(f"[INFO] Total missing values: {missing_count}")

    print("[SUCCESS] Dataset validation passed.")

    return df


def main():

    print("==========================================")
    print("Lab 5 - Big Mart Data Validation")
    print("==========================================")

    train_df = validate_file(TRAIN_FILE)
    test_df = validate_file(TEST_FILE)

    required_train_columns = [
        "Item_Outlet_Sales"
    ]

    for column in required_train_columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Required column missing from Train.csv: {column}"
            )

    print("\n[SUCCESS] Train.csv validation PASSED.")
    print("[SUCCESS] Test.csv validation PASSED.")
    print("[SUCCESS] Big Mart data validation completed.")


if __name__ == "__main__":
    main()
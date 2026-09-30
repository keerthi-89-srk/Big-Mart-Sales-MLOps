from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    print("==========================================")
    print("Lab 5 - Big Mart Production Preprocessing")
    print("==========================================")

    preprocess_script = PROJECT_ROOT / "src" / "preprocess.py"

    if not preprocess_script.exists():
        raise FileNotFoundError(
            f"Preprocessing script not found: {preprocess_script}"
        )

    print("[INFO] Running existing preprocessing pipeline...")
    print("[INFO] Reusing preprocess.py to keep Lab 3, Lab 4 and Lab 5 consistent.")

    result = subprocess.run(
        [sys.executable, str(preprocess_script)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("[ERROR] Preprocessing failed.")
        sys.exit(result.returncode)

    print("[SUCCESS] Production preprocessing completed successfully.")
    print("[SUCCESS] Processed datasets are ready.")


if __name__ == "__main__":
    main()
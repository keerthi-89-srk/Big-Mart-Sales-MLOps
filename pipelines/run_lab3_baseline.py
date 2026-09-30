from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(script_name: str):
    script_path = PROJECT_ROOT / script_name
    print(f"[INFO] Running: {script_path.name}")
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT), check=True)
        print(f"[SUCCESS] Completed: {script_path.name}")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Failed while running: {script_path.name}")
        raise SystemExit(exc.returncode) from exc


def main():
    print("[INFO] Starting Lab 3 baseline pipeline...")
    run_step("src/preprocess.py")
    run_step("src/train.py")
    run_step("src/evaluate.py")
    print("[SUCCESS] Lab 3 baseline pipeline finished successfully.")


if __name__ == "__main__":
    main()

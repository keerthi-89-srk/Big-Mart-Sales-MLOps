from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = BASE_DIR / "Train.csv"
TEST_PATH = BASE_DIR / "Test.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "LSWMD.pkl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DATA_PATH = PROCESSED_DIR / "wafer_features.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PHASE2_OUTPUT_DIR = OUTPUT_DIR / "phase2"
PHASE2_FEATURE_DIR = PHASE2_OUTPUT_DIR / "features"
PHASE2_ANALYSIS_DIR = PHASE2_OUTPUT_DIR / "feature_analysis"

MODEL_DIR = PROJECT_ROOT / "models" / "classical_ml"
SPLIT_DIR = PROCESSED_DIR / "phase3_splits"
RESULT_DIR = PROJECT_ROOT / "results" / "phase3_ml" / "baselines"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.dl_imbalance import save_imbalance_artifacts
from wm811k.paths import DL_IMAGE_DIR, DL_IMBALANCE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DL_IMAGE_DIR,
        help="Directory produced by 04_prepare_dl_image_dataset.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=DL_IMBALANCE_DIR)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9999,
        help="Effective-number weighting beta. Larger values emphasize rare classes.",
    )
    parser.add_argument(
        "--sampler-weight-column",
        choices=["effective_number_weight", "balanced_weight", "balanced_weight_normalized"],
        default="effective_number_weight",
    )
    return parser.parse_args()


def load_metadata(image_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "train": image_dir / "train_metadata.csv",
        "validation": image_dir / "validation_metadata.csv",
        "test": image_dir / "test_metadata.csv",
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing DL metadata files. Run 04_prepare_dl_image_dataset.py first:\n"
            f"{missing_text}"
        )

    return (
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["validation"]),
        pd.read_csv(paths["test"]),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 4.1 - DEEP LEARNING CLASS IMBALANCE HANDLING")
    print("=" * 70)

    train_metadata, validation_metadata, test_metadata = load_metadata(args.image_dir)
    paths = save_imbalance_artifacts(
        train_metadata=train_metadata,
        validation_metadata=validation_metadata,
        test_metadata=test_metadata,
        output_dir=args.output_dir,
        beta=args.beta,
        sampler_weight_column=args.sampler_weight_column,
    )

    class_weights = pd.read_csv(paths["class_weights"])
    print("\nTrain class weights:")
    print(class_weights.round(6).to_string(index=False))
    print("\nSaved artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\nPhase 4.1 completed successfully.")


if __name__ == "__main__":
    main()

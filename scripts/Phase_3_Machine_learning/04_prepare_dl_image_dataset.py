import argparse
from pathlib import Path
import sys

import cv2
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wm811k.labels import LABEL_COLUMN, add_label_column, filter_labeled, stratified_head
from wm811k.masks import wafer_to_grayscale
from wm811k.paths import (
    DL_IMAGE_DIR,
    DL_IMAGE_METADATA_PATH,
    MODEL_DIR,
    RAW_DATA_PATH,
    ensure_dir,
)


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
IMAGE_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DL_IMAGE_DIR)
    parser.add_argument("--metadata", type=Path, default=DL_IMAGE_METADATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--validation-size", type=float, default=VALIDATION_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help=(
            "Optional stratified cap for quick experiments. "
            "Omit this option to export every labeled WM-811K wafer."
        ),
    )
    return parser.parse_args()


def create_split_table(
    dataframe: pd.DataFrame,
    label_column: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> pd.DataFrame:
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1.")

    split_frame = dataframe[[label_column]].copy()
    split_frame["wafer_id"] = split_frame.index.astype(int)

    train_frame, temporary_frame = train_test_split(
        split_frame,
        test_size=test_size + validation_size,
        random_state=random_state,
        stratify=split_frame[label_column],
    )

    relative_test_size = test_size / (test_size + validation_size)
    validation_frame, test_frame = train_test_split(
        temporary_frame,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temporary_frame[label_column],
    )

    train_frame = train_frame.copy()
    validation_frame = validation_frame.copy()
    test_frame = test_frame.copy()

    train_frame["split"] = "train"
    validation_frame["split"] = "validation"
    test_frame["split"] = "test"

    return pd.concat(
        [train_frame, validation_frame, test_frame],
        axis=0,
    ).sort_index()


def wafer_to_dl_image(wafer, image_size: int):
    grayscale = wafer_to_grayscale(wafer)
    resized = cv2.resize(
        grayscale,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def export_images(
    dataframe: pd.DataFrame,
    split_table: pd.DataFrame,
    label_encoder: LabelEncoder,
    output_dir: Path,
    image_size: int,
) -> pd.DataFrame:
    rows = []
    ensure_dir(output_dir)

    for wafer_id, split_row in tqdm(
        split_table.iterrows(),
        total=len(split_table),
        desc="Exporting DL images",
    ):
        wafer = dataframe.at[wafer_id, "waferMap"]
        label = str(split_row[LABEL_COLUMN])
        split = split_row["split"]

        image = wafer_to_dl_image(wafer, image_size=image_size)
        image_dir = ensure_dir(output_dir / split / label)
        image_name = f"{label}_{wafer_id}.png"
        image_path = image_dir / image_name

        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Failed to write image: {image_path}")

        rows.append(
            {
                "wafer_id": int(wafer_id),
                "label": label,
                "label_encoded": int(label_encoder.transform([label])[0]),
                "split": split,
                "source_height": int(wafer.shape[0]),
                "source_width": int(wafer.shape[1]),
                "image_height": image_size,
                "image_width": image_size,
                "channels": 3,
                "relative_path": image_path.relative_to(output_dir).as_posix(),
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "label", "wafer_id"])


def save_split_manifests(metadata: pd.DataFrame, output_dir: Path) -> None:
    for split, split_metadata in metadata.groupby("split", sort=False):
        split_metadata.to_csv(output_dir / f"{split}_metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.metadata.parent)
    ensure_dir(args.model_dir)

    print("=" * 70)
    print("PHASE 3.4 - DEEP LEARNING IMAGE DATASET PREPARATION")
    print("=" * 70)

    dataframe = pd.read_pickle(args.raw_data)
    dataframe = filter_labeled(add_label_column(dataframe), label_column=LABEL_COLUMN)
    dataframe = stratified_head(
        dataframe,
        label_column=LABEL_COLUMN,
        max_per_class=args.max_per_class,
    )

    print(f"Raw dataset: {args.raw_data}")
    print(f"Labeled wafers selected: {len(dataframe):,}")
    print("\nClass distribution:")
    print(dataframe[LABEL_COLUMN].value_counts())

    split_table = create_split_table(
        dataframe=dataframe,
        label_column=LABEL_COLUMN,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(dataframe[LABEL_COLUMN].astype(str))

    metadata = export_images(
        dataframe=dataframe,
        split_table=split_table,
        label_encoder=label_encoder,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )

    metadata.to_csv(args.metadata, index=False)
    save_split_manifests(metadata, args.output_dir)

    class_mapping = pd.DataFrame(
        {
            "label": label_encoder.classes_,
            "label_encoded": range(len(label_encoder.classes_)),
        }
    )
    class_mapping.to_csv(args.output_dir / "class_mapping.csv", index=False)
    joblib.dump(label_encoder, args.model_dir / "dl_label_encoder.joblib")

    print("\nSplit summary:")
    print(pd.crosstab(metadata["split"], metadata["label"]))
    print(f"\nSaved images to: {args.output_dir}")
    print(f"Saved metadata to: {args.metadata}")
    print(f"Saved class mapping to: {args.output_dir / 'class_mapping.csv'}")
    print("Phase 3.4 completed successfully.")


if __name__ == "__main__":
    main()

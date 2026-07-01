import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import models, transforms
except ImportError as error:
    raise ImportError(
        "PyTorch and torchvision are required for ResNet-50 training. "
        "Install them in .venv before running this script."
    ) from error

from wm811k.dl_data import (
    FocalLoss,
    WaferImageDataset,
    create_weighted_sampler,
    load_class_weight_tensor,
)
from wm811k.paths import DL_IMAGE_DIR, DL_IMBALANCE_DIR, DL_MODEL_DIR, DL_RESULT_DIR, ensure_dir
from wm811k.plots import save_confusion_matrix_image


RANDOM_STATE = 42
NUM_CLASSES = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DL_IMAGE_DIR)
    parser.add_argument("--imbalance-dir", type=Path, default=DL_IMBALANCE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DL_MODEL_DIR)
    parser.add_argument("--result-dir", type=Path, default=DL_RESULT_DIR / "resnet50_baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet pretrained ResNet-50 weights or random initialization.",
    )
    parser.add_argument(
        "--no-sampler",
        action="store_true",
        help="Disable WeightedRandomSampler and train on the natural distribution.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def set_reproducibility(random_state: int) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def get_transforms(weights_name: str):
    if weights_name == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, evaluation_transform


def maybe_subset(dataset, max_samples: int | None, random_state: int):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def create_dataloaders(args: argparse.Namespace):
    train_transform, evaluation_transform = get_transforms(args.weights)

    train_dataset = WaferImageDataset(
        metadata_csv=args.image_dir / "train_metadata.csv",
        image_root=args.image_dir,
        transform=train_transform,
    )
    validation_dataset = WaferImageDataset(
        metadata_csv=args.image_dir / "validation_metadata.csv",
        image_root=args.image_dir,
        transform=evaluation_transform,
    )

    train_dataset = maybe_subset(
        train_dataset,
        max_samples=args.max_train_samples,
        random_state=args.random_state,
    )
    validation_dataset = maybe_subset(
        validation_dataset,
        max_samples=args.max_validation_samples,
        random_state=args.random_state,
    )

    sampler = None
    shuffle = True
    if not args.no_sampler and args.max_train_samples is None:
        sampler = create_weighted_sampler(args.imbalance_dir / "train_sample_weights.csv")
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def create_model(args: argparse.Namespace, device):
    if args.weights == "imagenet":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        weights = None

    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def calculate_metrics(y_true: np.ndarray, y_prediction: np.ndarray) -> dict[str, float]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_prediction,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_prediction,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_prediction),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        prediction = logits.argmax(dim=1)
        correct += int((prediction == labels).sum().item())
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    predictions = []
    targets = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)

        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(labels.cpu().numpy())

    y_true = np.concatenate(targets)
    y_prediction = np.concatenate(predictions)
    metrics = calculate_metrics(y_true, y_prediction)
    metrics["loss"] = total_loss / total
    return metrics, y_true, y_prediction


def save_evaluation_outputs(
    result_dir: Path,
    image_dir: Path,
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> None:
    class_mapping = pd.read_csv(image_dir / "class_mapping.csv").sort_values(
        "label_encoded"
    )
    class_names = class_mapping["label"].tolist()
    class_labels = np.arange(len(class_names))

    report = classification_report(
        y_true,
        y_prediction,
        labels=class_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(result_dir / "classification_report.csv")

    raw_matrix = confusion_matrix(y_true, y_prediction, labels=class_labels)
    normalized_matrix = confusion_matrix(
        y_true,
        y_prediction,
        labels=class_labels,
        normalize="true",
    )
    pd.DataFrame(raw_matrix, index=class_names, columns=class_names).to_csv(
        result_dir / "confusion_matrix_raw.csv"
    )
    pd.DataFrame(normalized_matrix, index=class_names, columns=class_names).to_csv(
        result_dir / "confusion_matrix_normalized.csv"
    )
    save_confusion_matrix_image(
        raw_matrix,
        class_names,
        "ResNet-50 Baseline - Raw Confusion Matrix",
        result_dir / "confusion_matrix_raw.png",
        "d",
    )
    save_confusion_matrix_image(
        normalized_matrix,
        class_names,
        "ResNet-50 Baseline - Normalized Confusion Matrix",
        result_dir / "confusion_matrix_normalized.png",
        ".2f",
    )

    pd.DataFrame(
        {
            "true_label_encoded": y_true,
            "predicted_label_encoded": y_prediction,
            "correct_prediction": y_true == y_prediction,
        }
    ).to_csv(result_dir / "validation_predictions.csv", index=False)


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_reproducibility(args.random_state)

    result_dir = ensure_dir(args.result_dir)
    model_dir = ensure_dir(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PHASE 4.2 - RESNET-50 BASELINE TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")
    print(f"Image directory: {args.image_dir}")
    print(f"Result directory: {result_dir}")

    train_loader, validation_loader = create_dataloaders(args)
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")

    model = create_model(args, device=device)
    class_weights = load_class_weight_tensor(
        args.imbalance_dir / "class_weights.csv",
        weight_column="effective_number_weight",
        device=device,
    )
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    history = []
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print("\n" + "-" * 70)
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 70)

        start_time = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_metrics, y_true, y_prediction = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        scheduler.step()
        epoch_time = time.perf_counter() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_weighted_f1": validation_metrics["weighted_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_seconds": epoch_time,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)

        print(
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}, "
            f"time={epoch_time:.2f}s"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model_dir / "resnet50_baseline_best.pt",
                model,
                optimizer,
                epoch,
                validation_metrics,
                args,
            )
            save_evaluation_outputs(result_dir, args.image_dir, y_true, y_prediction)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    save_checkpoint(
        model_dir / "resnet50_baseline_last.pt",
        model,
        optimizer,
        history[-1]["epoch"],
        validation_metrics,
        args,
    )

    summary_lines = [
        "RESNET-50 BASELINE SUMMARY",
        "=" * 32,
        f"Device: {device}",
        f"Weights: {args.weights}",
        f"Epochs completed: {history[-1]['epoch']}",
        f"Best epoch: {best_epoch}",
        f"Best validation macro-F1: {best_macro_f1:.4f}",
        f"Batch size: {args.batch_size}",
        f"Weighted sampler: {not args.no_sampler and args.max_train_samples is None}",
        f"Focal gamma: {args.focal_gamma}",
        f"Learning rate: {args.learning_rate}",
        f"Weight decay: {args.weight_decay}",
    ]
    (result_dir / "resnet50_baseline_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nTraining completed.")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print(f"Saved best checkpoint to: {model_dir / 'resnet50_baseline_best.pt'}")
    print(f"Saved outputs to: {result_dir}")


if __name__ == "__main__":
    main()

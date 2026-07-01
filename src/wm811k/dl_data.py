from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, WeightedRandomSampler
except ImportError:  # pragma: no cover - used when DL dependencies are absent.
    torch = None
    nn = None
    Dataset = object
    WeightedRandomSampler = None


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for this helper. Install torch and torchvision "
            "before running deep-learning training scripts."
        )


class WaferImageDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Path,
        image_root: Path,
        transform=None,
    ) -> None:
        require_torch()
        self.metadata = pd.read_csv(metadata_csv)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self.image_root / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["label_encoded"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_class_weight_tensor(
    class_weights_csv: Path,
    weight_column: str = "effective_number_weight",
    device=None,
):
    require_torch()
    class_weights = pd.read_csv(class_weights_csv).sort_values("label")
    weights = class_weights[weight_column].to_numpy(dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_weighted_sampler(
    train_sample_weights_csv: Path,
    replacement: bool = True,
):
    require_torch()
    sample_weights = pd.read_csv(train_sample_weights_csv)["sample_weight"].to_numpy(
        dtype=np.float64
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=replacement,
    )


class FocalLoss(nn.Module if nn is not None else object):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean",
    ) -> None:
        require_torch()
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, targets):
        cross_entropy = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
        )
        probabilities = torch.exp(-cross_entropy)
        loss = (1.0 - probabilities) ** self.gamma * cross_entropy

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unknown reduction: {self.reduction}")

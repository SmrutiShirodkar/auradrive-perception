"""torchvision-compatible detection Dataset over the Gold-layer CAM_FRONT
label table produced by `auradrive.training.labels`.

Returns (image_tensor, target_dict) pairs in the format torchvision's
detection models expect: target["boxes"] is [N,4] xyxy in pixel space,
target["labels"] is [N] class indices.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from auradrive.training.labels import CLASS_NAMES

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class CamFrontDetectionDataset(Dataset):
    """One item per key-frame sample; all boxes for that frame in one target."""

    def __init__(self, nuscenes_root: str | Path, labels_df: pd.DataFrame):
        self.nuscenes_root = Path(nuscenes_root)
        self.samples = list(labels_df.groupby("sample_token"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        sample_token, rows = self.samples[idx]
        image_path = self.nuscenes_root / rows["filename"].iloc[0]
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image)

        boxes = torch.tensor(
            rows[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype="float32")
        )
        labels = torch.tensor(
            [CLASS_TO_IDX[c] for c in rows["train_class"]], dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "sample_token": sample_token,
        }
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list, list]:
    """torchvision detection models take lists of variable-sized images/targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)

"""
Thin model-loading layer for the serving app.

Kept intentionally simple: loads a torch state_dict checkpoint from a local
path into the same architecture used in training. The checkpoint path is
the seam where this would point at an Azure ML registered model / Blob URI
instead of a local file — no other code changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from auradrive.training.labels import CLASS_NAMES
from auradrive.training.train import build_model

DEFAULT_CHECKPOINT = Path("data/models/cam_front_fasterrcnn.pt")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    version: str
    device: torch.device


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    version: str = "local-dev",
) -> LoadedModel:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run "
            f"`python -m auradrive.training.train` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return LoadedModel(model=model, version=version, device=device)

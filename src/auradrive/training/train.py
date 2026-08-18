"""
Fine-tunes a lightweight, COCO-pretrained Faster R-CNN (MobileNetV3-320
backbone) on the AuraDrive Gold-layer CAM_FRONT detection labels.

Chosen deliberately for footprint, not accuracy: this backbone is ~20MB
and the 320px input keeps CPU training tractable on nuScenes-mini's ~98
labeled key frames, while the code path is identical to what you'd run
unchanged on an Azure ML GPU compute instance for a larger fleet dataset.

    python -m auradrive.training.train --nuscenes-root v1.0-mini \
        --labels data/gold/cam_front_labels.parquet --epochs 5

Tracks params/metrics/model with MLflow (local file store by default;
point --mlflow-uri at an Azure ML tracking URI to log there instead —
same code, no branching).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from auradrive.training.dataset import CamFrontDetectionDataset, collate_fn
from auradrive.training.labels import CLASS_NAMES


def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _run_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items() if k != "sample_token"} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _evaluate_loss(model, loader, device) -> float:
    """Approximate validation signal: keep the model in train() mode (so
    torchvision still returns the loss dict) but disable gradients, since
    the tiny nuScenes-mini split doesn't warrant a full COCO-mAP evaluator."""
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items() if k != "sample_token"} for t in targets]
        loss_dict = model(images, targets)
        total_loss += float(sum(loss_dict.values()))
    return total_loss / max(len(loader), 1)


def train(
    nuscenes_root: str | Path,
    labels_path: str | Path,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 0.005,
    val_fraction: float = 0.2,
    mlflow_uri: str | None = None,
    mlflow_experiment: str = "auradrive-cam-front-detection",
    seed: int = 42,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_df = pd.read_parquet(labels_path)
    dataset = CamFrontDetectionDataset(nuscenes_root, labels_df)

    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=5e-4
    )

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment)

    out_dir = Path("data") / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "cam_front_fasterrcnn.pt"

    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "backbone": "fasterrcnn_mobilenet_v3_large_320_fpn",
                "num_classes": len(CLASS_NAMES),
                "train_samples": n_train,
                "val_samples": n_val,
                "device": str(device),
            }
        )

        best_val_loss = float("inf")
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = _run_epoch(model, train_loader, optimizer, device)
            val_loss = _evaluate_loss(model, val_loader, device)
            elapsed = time.perf_counter() - t0

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
            )
            print(
                f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  ({elapsed:.1f}s)"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)

        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_metric("best_val_loss", best_val_loss)

    print(f"best checkpoint -> {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CAM_FRONT detector on AuraDrive Gold labels.")
    parser.add_argument("--nuscenes-root", required=True)
    parser.add_argument("--labels", default="data/gold/cam_front_labels.parquet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--mlflow-uri", default=None, help="e.g. an Azure ML tracking URI")
    args = parser.parse_args()

    train(
        nuscenes_root=args.nuscenes_root,
        labels_path=args.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mlflow_uri=args.mlflow_uri,
    )


if __name__ == "__main__":
    main()

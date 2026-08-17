"""
Builds a Gold-layer 2D object-detection label table: for every key-frame
CAM_FRONT image in the Silver fused world model, projects every 3D
annotation visible in that sample into a 2D box, using box_projection.

Categories are collapsed to a small set of coarse classes (the report's
data covers cars, pedestrians, barriers, cones, trucks, etc. at wildly
uneven frequency; a coarse label set keeps the training problem
tractable on nuScenes-mini's 404 CAM_FRONT key frames).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from auradrive.training.box_projection import project_annotation_to_2d

# Coarse category collapse: nuScenes fine-grained name -> training class.
CATEGORY_MAP = {
    "vehicle.car": "vehicle",
    "vehicle.truck": "vehicle",
    "vehicle.bus.rigid": "vehicle",
    "vehicle.bus.bendy": "vehicle",
    "vehicle.construction": "vehicle",
    "vehicle.trailer": "vehicle",
    "vehicle.motorcycle": "two_wheeler",
    "vehicle.bicycle": "two_wheeler",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}
CLASS_NAMES = ["background", "vehicle", "pedestrian", "two_wheeler", "barrier", "traffic_cone"]


def _load_annotation_context(nuscenes_root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(nuscenes_root) / "v1.0-mini"
    ann = pd.DataFrame(json.loads((root / "sample_annotation.json").read_text()))
    instance = pd.DataFrame(json.loads((root / "instance.json").read_text()))
    category = pd.DataFrame(json.loads((root / "category.json").read_text()))

    instance = instance.merge(
        category.rename(columns={"token": "category_token", "name": "category_name"}),
        on="category_token",
        how="left",
    )
    ann = ann.merge(
        instance[["token", "category_name"]].rename(columns={"token": "instance_token"}),
        on="instance_token",
        how="left",
    )
    ann["train_class"] = ann["category_name"].map(CATEGORY_MAP)
    return {"annotations": ann}


def build_cam_front_detection_labels(
    nuscenes_root: str | Path,
    fused_wide: pd.DataFrame,
    bronze_sweeps: pd.DataFrame,
    image_width: int = 1600,
    image_height: int = 900,
) -> pd.DataFrame:
    """
    Returns one row per (sample_token, box) with pixel-space [x_min, y_min,
    x_max, y_max] and a coarse `train_class`, restricted to samples that
    have a quality-gate-passed CAM_FRONT sweep (i.e. present in fused_wide).
    """
    ann = _load_annotation_context(nuscenes_root)["annotations"]
    ann = ann.dropna(subset=["train_class"])

    cam_context = (
        bronze_sweeps.loc[bronze_sweeps["channel"] == "CAM_FRONT"]
        .drop_duplicates("sample_token")
        .set_index("sample_token")
    )

    valid_samples = fused_wide.loc[fused_wide["CAM_FRONT"].notna(), "sample_token"]

    rows = []
    for sample_token in valid_samples:
        if sample_token not in cam_context.index:
            continue
        ctx = cam_context.loc[sample_token]
        sample_anns = ann.loc[ann["sample_token"] == sample_token]

        for _, a in sample_anns.iterrows():
            box = project_annotation_to_2d(
                box_translation=a["translation"],
                box_size=a["size"],
                box_rotation=a["rotation"],
                category=a["train_class"],
                ego_translation=ctx["ego_pose.translation"],
                ego_rotation=ctx["ego_pose.rotation"],
                cam_translation=ctx["calibrated_sensor.translation"],
                cam_rotation=ctx["calibrated_sensor.rotation"],
                camera_intrinsic=ctx["calibrated_sensor.camera_intrinsic"],
                image_width=image_width,
                image_height=image_height,
            )
            if box is None:
                continue
            rows.append(
                {
                    "sample_token": sample_token,
                    "filename": ctx["filename"],
                    "x_min": box.x_min,
                    "y_min": box.y_min,
                    "x_max": box.x_max,
                    "y_max": box.y_max,
                    "train_class": box.category,
                }
            )

    return pd.DataFrame(rows)

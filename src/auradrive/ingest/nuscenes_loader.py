"""
Stage 1 (Ingest) + Stage 2 (Extraction) of the AuraDrive pipeline.

Reads the raw nuScenes-mini relational JSON tables (the "Tier 2: Relational
Metadata" and "Tier 3: Spatial-Temporal Calibration Logs" described in the
AuraDrive report) and produces a single flat, typed table of sensor sweeps
joined to their sample, scene, ego-pose and calibration context.

This is the code equivalent of the report's "Landing and Ingest" +
"Extraction and Demultiplexing" stages, minus the Docker/Azure Batch
container orchestration (not needed at nuScenes-mini scale).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_TABLES = (
    "sample",
    "sample_data",
    "sample_annotation",
    "ego_pose",
    "calibrated_sensor",
    "sensor",
    "scene",
    "category",
    "instance",
    "log",
    "map",
    "visibility",
    "attribute",
)


@dataclass(frozen=True)
class NuScenesMiniPaths:
    """Resolves the on-disk layout of an extracted nuScenes-mini archive."""

    root: Path
    version: str = "v1.0-mini"

    @property
    def table_dir(self) -> Path:
        return self.root / self.version

    @property
    def data_dir(self) -> Path:
        return self.root

    def table_path(self, table: str) -> Path:
        return self.table_dir / f"{table}.json"


def load_table(paths: NuScenesMiniPaths, table: str) -> pd.DataFrame:
    """Load a single nuScenes JSON table into a DataFrame, token-indexed."""
    if table not in RAW_TABLES:
        raise ValueError(f"Unknown nuScenes table: {table!r}")
    records = json.loads(paths.table_path(table).read_text())
    return pd.DataFrame.from_records(records)


def load_all_tables(paths: NuScenesMiniPaths) -> dict[str, pd.DataFrame]:
    """Load every raw relational table. Equivalent to landing all 13 JSON
    metadata files described in the report's Tier 2/Tier 3 data inventory."""
    return {table: load_table(paths, table) for table in RAW_TABLES}


def build_sensor_sweep_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Joins sample_data -> ego_pose -> calibrated_sensor -> sensor -> sample -> scene
    into one flat "fused metadata" table: one row per sensor sweep file, with
    full spatial-temporal context attached.

    This is the Bronze-layer output: raw sweep-level granularity, no fusion
    or quality filtering applied yet.
    """
    sample_data = tables["sample_data"]
    ego_pose = tables["ego_pose"].add_prefix("ego_pose.")
    calibrated_sensor = tables["calibrated_sensor"].add_prefix("calibrated_sensor.")
    sensor = tables["sensor"].add_prefix("sensor.")
    sample = tables["sample"].add_prefix("sample.")
    scene = tables["scene"].add_prefix("scene.")

    df = sample_data.merge(
        ego_pose, left_on="ego_pose_token", right_on="ego_pose.token", how="left"
    )
    df = df.merge(
        calibrated_sensor,
        left_on="calibrated_sensor_token",
        right_on="calibrated_sensor.token",
        how="left",
    )
    df = df.merge(
        sensor,
        left_on="calibrated_sensor.sensor_token",
        right_on="sensor.token",
        how="left",
    )
    df = df.merge(sample, left_on="sample_token", right_on="sample.token", how="left")
    df = df.merge(
        scene, left_on="sample.scene_token", right_on="scene.token", how="left"
    )

    df = df.rename(
        columns={
            "token": "sample_data_token",
            "sensor.channel": "channel",
            "sensor.modality": "modality",
            "sample.scene_token": "scene_token",
            "scene.name": "scene_name",
        }
    )

    keep = [
        "sample_data_token",
        "sample_token",
        "scene_token",
        "scene_name",
        "channel",
        "modality",
        "filename",
        "fileformat",
        "is_key_frame",
        "timestamp",
        "ego_pose.translation",
        "ego_pose.rotation",
        "ego_pose.timestamp",
        "calibrated_sensor.translation",
        "calibrated_sensor.rotation",
        "calibrated_sensor.camera_intrinsic",
    ]
    return df[keep].reset_index(drop=True)


def load_bronze_sweeps(nuscenes_root: str | Path) -> pd.DataFrame:
    """Convenience entrypoint: raw nuScenes-mini root -> Bronze sweep table."""
    paths = NuScenesMiniPaths(root=Path(nuscenes_root))
    tables = load_all_tables(paths)
    return build_sensor_sweep_table(tables)

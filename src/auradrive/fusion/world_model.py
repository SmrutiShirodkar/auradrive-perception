"""
Stage 4 (Integrated Sensor Fusion) output: the "Fused World Model".

Combines the Bronze sweep table with the temporal quality gate result to
produce one Silver-layer row per (sample, channel): the sweep file that
passed the residual check, plus its pose/calibration context. Rows that
failed the gate are dropped from the fused output and kept in a separate
quarantine table for audit, matching the report's quarantine-partition
design (section 3.2.4).

Kept lean on purpose: everything here is a small number of vectorized
pandas merges over the already-computed residual table — no per-row
Python loops, no extra full-table scans of the Bronze data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from auradrive.quality.temporal_gate import (
    DEFAULT_RESIDUAL_THRESHOLD_MS,
    QualityGateResult,
    apply_temporal_quality_gate,
    compute_temporal_residuals,
)

SWEEP_CONTEXT_COLUMNS = [
    "sample_data_token",
    "filename",
    "fileformat",
    "modality",
    "ego_pose.translation",
    "ego_pose.rotation",
    "calibrated_sensor.translation",
    "calibrated_sensor.rotation",
    "calibrated_sensor.camera_intrinsic",
]


@dataclass(frozen=True)
class FusedWorldModel:
    """Silver-layer output of the fusion stage."""

    fused_long: pd.DataFrame       # one row per (sample_token, channel)
    quarantined: pd.DataFrame      # residual rows that failed the gate
    threshold_ms: float

    @property
    def pass_rate(self) -> float:
        total = len(self.fused_long) + len(self.quarantined)
        return len(self.fused_long) / total if total else 0.0

    def to_wide(self) -> pd.DataFrame:
        """Pivots to one row per sample_token, one column group per channel
        (channel-prefixed filename), i.e. the flat "fused scene" record used
        for perception training."""
        return self.fused_long.pivot(
            index="sample_token", columns="channel", values="filename"
        ).reset_index()


def build_fused_world_model(
    bronze_sweeps: pd.DataFrame,
    threshold_ms: float = DEFAULT_RESIDUAL_THRESHOLD_MS,
) -> FusedWorldModel:
    """
    Bronze sweeps -> temporal residuals -> quality gate -> fused Silver table.

    For every key-frame sample and sensor channel, attaches the nearest
    quality-passing sweep's file path and calibration/pose context.
    """
    residuals = compute_temporal_residuals(bronze_sweeps)
    gate: QualityGateResult = apply_temporal_quality_gate(residuals, threshold_ms)

    sweep_context = bronze_sweeps[SWEEP_CONTEXT_COLUMNS].drop_duplicates(
        "sample_data_token"
    )

    fused_long = gate.passed.merge(
        sweep_context,
        left_on="matched_sample_data_token",
        right_on="sample_data_token",
        how="left",
    ).drop(columns=["matched_sample_data_token"])

    return FusedWorldModel(
        fused_long=fused_long,
        quarantined=gate.quarantined,
        threshold_ms=threshold_ms,
    )

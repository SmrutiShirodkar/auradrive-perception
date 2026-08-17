"""
Stage 4 (Integrated Sensor Fusion and Quality Control) of the AuraDrive
pipeline, "temporal residual" data quality gate.

The report (section 3.2.4) defines the gate as:

    "the pipeline calculates the Temporal Residual, which is the delta in
    milliseconds between the target timestamp and the nearest available
    sensor frame. If the modality's residual exceeds a 10ms threshold, the
    record is written to a quarantine partition with a diagnostic tag."

This module implements that gate directly against the nuScenes-mini Bronze
sweep table: for every key-frame sample (the fusion target timestamp), it
finds the nearest sweep per sensor channel and computes the residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_RESIDUAL_THRESHOLD_MS = 10.0


@dataclass(frozen=True)
class QualityGateResult:
    passed: pd.DataFrame
    quarantined: pd.DataFrame
    threshold_ms: float

    @property
    def pass_rate(self) -> float:
        total = len(self.passed) + len(self.quarantined)
        return len(self.passed) / total if total else 0.0


def _nearest_match(
    target_ts: np.ndarray, candidate_ts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each target timestamp, find the index into `candidate_ts` of the
    nearest value and the absolute delta (microseconds), via a sorted
    searchsorted — O((n+m) log m) instead of an O(n*m) cross join."""
    order = np.argsort(candidate_ts)
    sorted_ts = candidate_ts[order]

    idx = np.searchsorted(sorted_ts, target_ts)
    idx_lo = np.clip(idx - 1, 0, len(sorted_ts) - 1)
    idx_hi = np.clip(idx, 0, len(sorted_ts) - 1)

    delta_lo = np.abs(target_ts - sorted_ts[idx_lo])
    delta_hi = np.abs(target_ts - sorted_ts[idx_hi])
    use_hi = delta_hi < delta_lo

    nearest_idx = order[np.where(use_hi, idx_hi, idx_lo)]
    residual = np.where(use_hi, delta_hi, delta_lo)
    return nearest_idx, residual


def compute_temporal_residuals(
    bronze_sweeps: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every (scene, key-frame sample) target timestamp, and for every
    sensor channel present in that scene, find the nearest sweep on that
    channel and compute the temporal residual to it.

    Returns one row per (sample_token, channel) with the matched
    `sample_data_token` and a `residual_ms` column, so downstream fusion
    can pull the exact sweep file that passed the quality gate.
    """
    key_frames = (
        bronze_sweeps.loc[bronze_sweeps["is_key_frame"], ["sample_token", "scene_token", "timestamp"]]
        .drop_duplicates("sample_token")
        .rename(columns={"timestamp": "target_timestamp"})
    )

    rows = []
    for scene_token, scene_group in bronze_sweeps.groupby("scene_token"):
        targets = key_frames[key_frames["scene_token"] == scene_token]
        if targets.empty:
            continue
        target_ts = targets["target_timestamp"].to_numpy()

        for channel, channel_group in scene_group.groupby("channel"):
            candidate_ts = channel_group["timestamp"].to_numpy()
            if len(candidate_ts) == 0:
                continue
            candidate_tokens = channel_group["sample_data_token"].to_numpy()
            nearest_idx, residual_us = _nearest_match(target_ts, candidate_ts)
            rows.append(
                pd.DataFrame(
                    {
                        "sample_token": targets["sample_token"].to_numpy(),
                        "scene_token": scene_token,
                        "channel": channel,
                        "matched_sample_data_token": candidate_tokens[nearest_idx],
                        "target_timestamp": target_ts,
                        "residual_ms": residual_us / 1000.0,
                    }
                )
            )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[
            "sample_token",
            "scene_token",
            "channel",
            "matched_sample_data_token",
            "target_timestamp",
            "residual_ms",
        ]
    )


def apply_temporal_quality_gate(
    residuals: pd.DataFrame,
    threshold_ms: float = DEFAULT_RESIDUAL_THRESHOLD_MS,
) -> QualityGateResult:
    """Splits residual records into passed / quarantined based on the
    10ms (default) threshold described in the AuraDrive report."""
    mask = residuals["residual_ms"] <= threshold_ms
    passed = residuals.loc[mask].copy()
    quarantined = residuals.loc[~mask].copy()
    quarantined["diagnostic_tag"] = "TEMPORAL_RESIDUAL_EXCEEDED"
    return QualityGateResult(passed=passed, quarantined=quarantined, threshold_ms=threshold_ms)

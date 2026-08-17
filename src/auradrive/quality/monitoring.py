"""
Lightweight data-quality and drift monitoring for the AuraDrive pipeline —
the code equivalent of the report's AVOps layer claim (section 3.2.7:
"Application Insights and Azure Managed Grafana provide real-time tracking
of pipeline performance and safety KPIs").

Two concerns, kept separate on purpose:

1. Pipeline health metrics: quarantine rate per sensor channel from the
   temporal quality gate (auradrive.quality.temporal_gate) — flags a
   sensor whose calibration/sync has drifted (the report's "Calibration
   Integrity and Sensor Drift" veracity risk, section 1.4).

2. Model input drift: compares a batch of incoming detection confidence
   scores / class distribution against a reference (training-time)
   distribution, using population stability index (PSI) — flags when
   live traffic looks statistically different from what the model was
   trained on, a standard MLOps drift signal.

No external monitoring service dependency: this produces plain
dict/DataFrame summaries that Grafana, Application Insights, or a notebook
could all consume identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUARANTINE_RATE_WARN_THRESHOLD = 0.5


@dataclass(frozen=True)
class ChannelHealth:
    channel: str
    total: int
    quarantined: int

    @property
    def quarantine_rate(self) -> float:
        return self.quarantined / self.total if self.total else 0.0

    @property
    def is_unhealthy(self) -> bool:
        return self.quarantine_rate > QUARANTINE_RATE_WARN_THRESHOLD


def summarize_channel_health(
    passed: pd.DataFrame, quarantined: pd.DataFrame
) -> list[ChannelHealth]:
    """
    Per-channel quarantine rate from the temporal quality gate output.
    A channel with a rate above threshold suggests sensor drift or a
    calibration issue on that specific sensor, not a one-off bad frame.
    """
    passed_counts = passed.groupby("channel").size()
    quarantined_counts = quarantined.groupby("channel").size()
    all_channels = sorted(set(passed_counts.index) | set(quarantined_counts.index))

    results = []
    for channel in all_channels:
        n_pass = int(passed_counts.get(channel, 0))
        n_quarantine = int(quarantined_counts.get(channel, 0))
        results.append(
            ChannelHealth(channel=channel, total=n_pass + n_quarantine, quarantined=n_quarantine)
        )
    return results


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1D distributions.

    PSI < 0.1: no significant shift. 0.1-0.25: moderate shift, monitor.
    > 0.25: significant shift, investigate (common industry rule of thumb).
    """
    edges = np.histogram_bin_edges(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass(frozen=True)
class DriftReport:
    score_psi: float
    class_distribution_reference: dict[str, float]
    class_distribution_current: dict[str, float]

    @property
    def severity(self) -> str:
        if self.score_psi < 0.1:
            return "stable"
        if self.score_psi < 0.25:
            return "moderate_shift"
        return "significant_shift"


def compute_detection_drift(
    reference_scores: pd.Series,
    current_scores: pd.Series,
    reference_labels: pd.Series,
    current_labels: pd.Series,
) -> DriftReport:
    """Compares a live batch of model detections against a reference batch
    (e.g. validation-set predictions captured at training time)."""
    psi = _psi(reference_scores.to_numpy(), current_scores.to_numpy())

    ref_dist = (reference_labels.value_counts(normalize=True)).to_dict()
    cur_dist = (current_labels.value_counts(normalize=True)).to_dict()

    return DriftReport(
        score_psi=psi,
        class_distribution_reference=ref_dist,
        class_distribution_current=cur_dist,
    )

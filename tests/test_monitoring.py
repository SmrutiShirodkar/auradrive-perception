import pandas as pd
import pytest

from auradrive.quality.monitoring import (
    QUARANTINE_RATE_WARN_THRESHOLD,
    compute_detection_drift,
    summarize_channel_health,
)


def test_summarize_channel_health_flags_high_quarantine_rate():
    passed = pd.DataFrame({"channel": ["CAM_FRONT"] * 2 + ["LIDAR_TOP"] * 8})
    quarantined = pd.DataFrame({"channel": ["CAM_FRONT"] * 8 + ["LIDAR_TOP"] * 2})

    health = {h.channel: h for h in summarize_channel_health(passed, quarantined)}

    assert health["CAM_FRONT"].quarantine_rate == pytest.approx(0.8)
    assert health["CAM_FRONT"].is_unhealthy is True
    assert health["LIDAR_TOP"].quarantine_rate == pytest.approx(0.2)
    assert health["LIDAR_TOP"].is_unhealthy is False


def test_summarize_channel_health_handles_channel_with_zero_quarantine():
    passed = pd.DataFrame({"channel": ["RADAR_FRONT"] * 5})
    quarantined = pd.DataFrame({"channel": []}, dtype=object)

    health = {h.channel: h for h in summarize_channel_health(passed, quarantined)}

    assert health["RADAR_FRONT"].quarantine_rate == 0.0
    assert health["RADAR_FRONT"].total == 5


def test_compute_detection_drift_identical_distributions_is_stable():
    scores = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9] * 20)
    labels = pd.Series(["vehicle", "pedestrian"] * 50)

    report = compute_detection_drift(scores, scores, labels, labels)

    assert report.score_psi == pytest.approx(0.0, abs=1e-9)
    assert report.severity == "stable"


def test_compute_detection_drift_shifted_distribution_flagged():
    reference_scores = pd.Series([0.9] * 100)  # all high-confidence
    current_scores = pd.Series([0.1] * 100)    # all low-confidence, big shift
    labels = pd.Series(["vehicle"] * 100)

    report = compute_detection_drift(reference_scores, current_scores, labels, labels)

    assert report.severity == "significant_shift"
    assert report.score_psi > 0.25

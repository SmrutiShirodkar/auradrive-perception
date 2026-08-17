import numpy as np
import pandas as pd

from auradrive.quality.temporal_gate import (
    apply_temporal_quality_gate,
    compute_temporal_residuals,
)


def _make_bronze_sweeps():
    # One scene, one key-frame sample at t=1000 (us), two channels.
    # CAM: sweep at t=1005 -> residual 5us = 0.005ms -> passes
    # LIDAR: sweep at t=1050000 -> residual 50000us = 50ms -> quarantined
    rows = [
        {"sample_data_token": "sd1", "sample_token": "s1", "scene_token": "sc1", "channel": "CAM_FRONT",
         "timestamp": 1000, "is_key_frame": True},
        {"sample_data_token": "sd2", "sample_token": "s1", "scene_token": "sc1", "channel": "CAM_FRONT",
         "timestamp": 1005, "is_key_frame": False},
        {"sample_data_token": "sd3", "sample_token": "s1", "scene_token": "sc1", "channel": "LIDAR_TOP",
         "timestamp": 1000, "is_key_frame": True},
        {"sample_data_token": "sd4", "sample_token": "s1", "scene_token": "sc1", "channel": "LIDAR_TOP",
         "timestamp": 51000, "is_key_frame": False},
    ]
    return pd.DataFrame(rows)


def test_compute_temporal_residuals_nearest_neighbour():
    bronze = _make_bronze_sweeps()
    residuals = compute_temporal_residuals(bronze)

    cam_row = residuals.loc[residuals["channel"] == "CAM_FRONT"].iloc[0]
    lidar_row = residuals.loc[residuals["channel"] == "LIDAR_TOP"].iloc[0]

    # key frame itself is the nearest candidate (residual to self is 0)
    assert cam_row["residual_ms"] == 0.0
    assert cam_row["matched_sample_data_token"] == "sd1"
    assert lidar_row["residual_ms"] == 0.0
    assert lidar_row["matched_sample_data_token"] == "sd3"


def test_apply_temporal_quality_gate_splits_on_threshold():
    residuals = pd.DataFrame(
        {
            "sample_token": ["a", "b", "c"],
            "scene_token": ["sc1"] * 3,
            "channel": ["CAM_FRONT"] * 3,
            "target_timestamp": [1, 2, 3],
            "residual_ms": [5.0, 10.0, 15.0],
        }
    )
    result = apply_temporal_quality_gate(residuals, threshold_ms=10.0)

    assert len(result.passed) == 2  # 5.0 and 10.0 (<=)
    assert len(result.quarantined) == 1  # 15.0
    assert "diagnostic_tag" in result.quarantined.columns
    assert result.pass_rate == 2 / 3


def test_nearest_match_handles_boundaries():
    from auradrive.quality.temporal_gate import _nearest_match

    candidates = np.array([100, 200, 300])
    targets = np.array([50, 350])  # unambiguous: nearest to 100 and 300 respectively
    nearest_idx, residuals = _nearest_match(targets, candidates)

    np.testing.assert_array_equal(residuals, [50, 50])
    np.testing.assert_array_equal(candidates[nearest_idx], [100, 300])


def test_nearest_match_ties_pick_a_consistent_side():
    from auradrive.quality.temporal_gate import _nearest_match

    candidates = np.array([100, 200])
    targets = np.array([150])  # equidistant from both candidates
    nearest_idx, residuals = _nearest_match(targets, candidates)

    assert residuals[0] == 50
    assert candidates[nearest_idx[0]] in (100, 200)

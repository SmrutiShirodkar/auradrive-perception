from pathlib import Path

import pytest

from auradrive.ingest.nuscenes_loader import load_bronze_sweeps

NUSCENES_ROOT = Path(__file__).resolve().parents[1] / "v1.0-mini"


@pytest.mark.skipif(
    not NUSCENES_ROOT.exists(), reason="nuScenes-mini dataset not present locally"
)
def test_load_bronze_sweeps_shape_and_channels():
    df = load_bronze_sweeps(NUSCENES_ROOT)

    assert len(df) == 31206
    assert df["scene_name"].nunique() == 10

    expected_channels = {
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        "LIDAR_TOP",
        "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
        "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
    }
    assert set(df["channel"].unique()) == expected_channels

    assert df["sample_data_token"].is_unique

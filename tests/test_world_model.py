import pandas as pd

from auradrive.fusion.world_model import build_fused_world_model


def _make_bronze_sweeps():
    # Two channels, one key frame at t=1000.
    # CAM_FRONT: exact match at t=1000 -> residual 0ms -> passes
    # LIDAR_TOP: nearest sweep at t=51000 -> residual 50ms -> quarantined
    rows = [
        {
            "sample_data_token": "sd_cam_key", "sample_token": "s1", "scene_token": "sc1",
            "channel": "CAM_FRONT", "filename": "cam_key.jpg", "fileformat": "jpg",
            "modality": "camera", "is_key_frame": True, "timestamp": 1000,
            "ego_pose.translation": [0, 0, 0], "ego_pose.rotation": [1, 0, 0, 0],
            "calibrated_sensor.translation": [0, 0, 0], "calibrated_sensor.rotation": [1, 0, 0, 0],
            "calibrated_sensor.camera_intrinsic": [],
        },
        {
            "sample_data_token": "sd_lidar_key", "sample_token": "s1", "scene_token": "sc1",
            "channel": "LIDAR_TOP", "filename": "lidar_key.pcd.bin", "fileformat": "pcd",
            "modality": "lidar", "is_key_frame": True, "timestamp": 1000,
            "ego_pose.translation": [0, 0, 0], "ego_pose.rotation": [1, 0, 0, 0],
            "calibrated_sensor.translation": [0, 0, 0], "calibrated_sensor.rotation": [1, 0, 0, 0],
            "calibrated_sensor.camera_intrinsic": [],
        },
        {
            "sample_data_token": "sd_lidar_far", "sample_token": "s1", "scene_token": "sc1",
            "channel": "LIDAR_TOP", "filename": "lidar_far.pcd.bin", "fileformat": "pcd",
            "modality": "lidar", "is_key_frame": False, "timestamp": 51000,
            "ego_pose.translation": [0, 0, 0], "ego_pose.rotation": [1, 0, 0, 0],
            "calibrated_sensor.translation": [0, 0, 0], "calibrated_sensor.rotation": [1, 0, 0, 0],
            "calibrated_sensor.camera_intrinsic": [],
        },
    ]
    return pd.DataFrame(rows)


def test_build_fused_world_model_keeps_passing_and_quarantines_failing():
    bronze = _make_bronze_sweeps()
    fused = build_fused_world_model(bronze, threshold_ms=10.0)

    assert set(fused.fused_long["channel"]) == {"CAM_FRONT", "LIDAR_TOP"}
    lidar_row = fused.fused_long.loc[fused.fused_long["channel"] == "LIDAR_TOP"].iloc[0]
    assert lidar_row["filename"] == "lidar_key.pcd.bin"  # nearest match, itself, not the far one

    assert len(fused.quarantined) == 0
    assert fused.pass_rate == 1.0


def test_to_wide_pivots_one_row_per_sample():
    bronze = _make_bronze_sweeps()
    fused = build_fused_world_model(bronze, threshold_ms=10.0)
    wide = fused.to_wide()

    assert len(wide) == 1
    assert wide.loc[0, "CAM_FRONT"] == "cam_key.jpg"
    assert wide.loc[0, "LIDAR_TOP"] == "lidar_key.pcd.bin"

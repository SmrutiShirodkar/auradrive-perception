import pytest

from auradrive.training.box_projection import project_annotation_to_2d


def test_project_annotation_directly_in_front_of_camera():
    # Ego and camera both at world origin, no rotation (identity quaternion).
    # A 2x2x2 box centered 10m in front (camera looks down +z after this setup)
    # should project to a roughly centered box in the image.
    identity_q = [1, 0, 0, 0]
    intrinsic = [[1000, 0, 400], [0, 1000, 300], [0, 0, 1]]

    box = project_annotation_to_2d(
        box_translation=[0, 0, 10],
        box_size=[2, 2, 2],
        box_rotation=identity_q,
        category="vehicle",
        ego_translation=[0, 0, 0],
        ego_rotation=identity_q,
        cam_translation=[0, 0, 0],
        cam_rotation=identity_q,
        camera_intrinsic=intrinsic,
        image_width=800,
        image_height=600,
    )

    assert box is not None
    # A 2x2x2 box at 10m depth, principal point (400, 300): box should be
    # centered on the principal point with a positive, finite extent.
    center_x = (box.x_min + box.x_max) / 2
    center_y = (box.y_min + box.y_max) / 2
    assert center_x == pytest.approx(400, abs=1e-6)
    assert center_y == pytest.approx(300, abs=1e-6)
    # Near face of the box is at z=9m (10m center - 1m half-depth), so its
    # 2m width projects widest: 2m * 1000 focal / 9m depth.
    assert box.x_max - box.x_min == pytest.approx(2 * 1000 / 9, abs=1e-6)
    assert box.category == "vehicle"


def test_project_annotation_behind_camera_returns_none():
    identity_q = [1, 0, 0, 0]
    intrinsic = [[1000, 0, 400], [0, 1000, 300], [0, 0, 1]]

    box = project_annotation_to_2d(
        box_translation=[0, 0, -10],  # behind the camera
        box_size=[2, 2, 2],
        box_rotation=identity_q,
        category="vehicle",
        ego_translation=[0, 0, 0],
        ego_rotation=identity_q,
        cam_translation=[0, 0, 0],
        cam_rotation=identity_q,
        camera_intrinsic=intrinsic,
        image_width=800,
        image_height=600,
    )

    assert box is None


def test_project_annotation_outside_image_returns_none():
    identity_q = [1, 0, 0, 0]
    intrinsic = [[1000, 0, 400], [0, 1000, 300], [0, 0, 1]]

    box = project_annotation_to_2d(
        box_translation=[1000, 0, 10],  # far off to the side
        box_size=[2, 2, 2],
        box_rotation=identity_q,
        category="vehicle",
        ego_translation=[0, 0, 0],
        ego_rotation=identity_q,
        cam_translation=[0, 0, 0],
        cam_rotation=identity_q,
        camera_intrinsic=intrinsic,
        image_width=800,
        image_height=600,
    )

    assert box is None

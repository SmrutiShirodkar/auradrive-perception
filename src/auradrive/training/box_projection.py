"""
Projects nuScenes 3D box annotations into 2D image-plane boxes for a given
camera channel, using the calibration/pose context already carried through
the Bronze/Silver tables (ego_pose + calibrated_sensor).

This turns the Silver "Fused World Model" into 2D object-detection training
labels without needing a separate annotation tool: nuScenes ships 3D boxes
in the global frame; we transform box corners global -> ego -> camera ->
pixel and take the axis-aligned bounding rectangle of the visible corners.

Kept dependency-light: numpy only, no nuscenes-devkit box/geometry classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class Box2D:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    category: str

    def clip(self, width: int, height: int) -> "Box2D | None":
        x_min, x_max = np.clip([self.x_min, self.x_max], 0, width)
        y_min, y_max = np.clip([self.y_min, self.y_max], 0, height)
        if x_max <= x_min or y_max <= y_min:
            return None
        return Box2D(x_min, y_min, x_max, y_max, self.category)


def _quaternion_to_rotation_matrix(q: list[float]) -> np.ndarray:
    """nuScenes stores quaternions as [w, x, y, z]."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def _box_corners(translation: list[float], size: list[float], rotation: list[float]) -> np.ndarray:
    """8 corners of a 3D box in its own local frame, then rotated + translated
    into the global frame. Size is [width, length, height] per nuScenes convention."""
    w, l, h = size
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack([x_corners, y_corners, z_corners])  # 3x8

    R = _quaternion_to_rotation_matrix(rotation)
    corners = R @ corners + np.array(translation).reshape(3, 1)
    return corners  # 3x8, global frame


def _global_to_camera(
    points_global: np.ndarray,
    ego_translation: list[float],
    ego_rotation: list[float],
    cam_translation: list[float],
    cam_rotation: list[float],
) -> np.ndarray:
    """global -> ego frame -> camera frame."""
    R_ego = _quaternion_to_rotation_matrix(ego_rotation)
    points_ego = R_ego.T @ (points_global - np.array(ego_translation).reshape(3, 1))

    R_cam = _quaternion_to_rotation_matrix(cam_rotation)
    points_cam = R_cam.T @ (points_ego - np.array(cam_translation).reshape(3, 1))
    return points_cam  # 3x8, camera frame (z = depth)


def project_annotation_to_2d(
    *,
    box_translation: list[float],
    box_size: list[float],
    box_rotation: list[float],
    category: str,
    ego_translation: list[float],
    ego_rotation: list[float],
    cam_translation: list[float],
    cam_rotation: list[float],
    camera_intrinsic: list[list[float]],
    image_width: int,
    image_height: int,
    min_visible_corners: int = 1,
) -> Box2D | None:
    """
    Projects one 3D annotation box into a 2D pixel-space bounding box for the
    given camera. Returns None if the box is entirely behind the camera or
    falls outside the image after clipping.
    """
    corners_global = _box_corners(box_translation, box_size, box_rotation)
    corners_cam = _global_to_camera(
        corners_global, ego_translation, ego_rotation, cam_translation, cam_rotation
    )

    in_front = corners_cam[2, :] > 0.1
    if in_front.sum() < min_visible_corners:
        return None

    visible = corners_cam[:, in_front]
    K = np.array(camera_intrinsic)
    pixels = K @ visible
    pixels = pixels[:2, :] / pixels[2, :]

    box = Box2D(
        x_min=float(pixels[0, :].min()),
        y_min=float(pixels[1, :].min()),
        x_max=float(pixels[0, :].max()),
        y_max=float(pixels[1, :].max()),
        category=category,
    )
    return box.clip(image_width, image_height)

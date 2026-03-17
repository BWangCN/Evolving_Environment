"""Compute wrist camera poses from EE trajectory."""

from __future__ import annotations

import numpy as np

from src.config import franka
from src.trajectory.generator import Trajectory
from src.trajectory.interpolation import quat_multiply, quat_normalize
from src.trajectory.collision import CollisionChecker  # for rotation matrix utility


def compute_camera_poses(traj: Trajectory) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute wrist camera pose at each waypoint.

    Camera is rigidly attached to the end-effector with a fixed transform
    defined in franka.CAMERA_TO_EE_POSITION and franka.CAMERA_TO_EE_ORIENTATION.

    Returns:
        List of (cam_position, cam_orientation) tuples, same length as trajectory.
        cam_position: (3,) world frame.
        cam_orientation: (4,) quaternion wxyz in world frame.
    """
    cam_poses = []

    for wp in traj.waypoints:
        ee_pos = wp.position
        ee_ori = wp.orientation

        # Camera position: rotate cam-to-EE offset by EE orientation, then add EE position
        R_ee = CollisionChecker._quat_to_rotation_matrix(ee_ori)
        cam_offset_world = R_ee @ franka.CAMERA_TO_EE_POSITION
        cam_pos = ee_pos + cam_offset_world

        # Camera orientation: compose EE orientation with cam-to-EE orientation
        cam_ori = quat_multiply(ee_ori, franka.CAMERA_TO_EE_ORIENTATION)
        cam_ori = quat_normalize(cam_ori)

        cam_poses.append((cam_pos.copy(), cam_ori.copy()))

    return cam_poses


def camera_intrinsic_matrix() -> np.ndarray:
    """Return the 3x3 camera intrinsic matrix K."""
    return np.array([
        [franka.CAMERA_FX, 0.0, franka.CAMERA_CX],
        [0.0, franka.CAMERA_FY, franka.CAMERA_CY],
        [0.0, 0.0, 1.0],
    ])

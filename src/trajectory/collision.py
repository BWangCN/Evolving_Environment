"""Collision detection with inflated safety volumes."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from typing import Optional


class CollisionChecker:
    """Check trajectory waypoints against scene point cloud with inflated margins.

    Uses KDTree for fast nearest-neighbor distance queries.
    Different phases of manipulation use different safety margins.
    """

    def __init__(
        self,
        scene_cloud: np.ndarray,
        gripper_margin: float = 0.03,
        object_margin: float = 0.05,
        place_margin: float = 0.01,
    ):
        """
        Args:
            scene_cloud: (N, 3) point cloud of obstacle objects.
            gripper_margin: Safety margin around gripper (meters).
            object_margin: Safety margin around grasped object during transport (meters).
            place_margin: Reduced margin when approaching place target (meters).
        """
        self.gripper_margin = gripper_margin
        self.object_margin = object_margin
        self.place_margin = place_margin

        if len(scene_cloud) == 0:
            self._tree = None
        else:
            self._tree = KDTree(scene_cloud)

    def check_point(self, point: np.ndarray, margin: float) -> bool:
        """Check if a single 3D point is in collision (within margin of any obstacle).

        Returns True if collision detected.
        """
        if self._tree is None:
            return False
        dist, _ = self._tree.query(point)
        return dist < margin

    def check_trajectory(
        self,
        waypoints: list[tuple[np.ndarray, np.ndarray]],
        gripper_phase: bool = True,
        grasped_object_cloud: Optional[np.ndarray] = None,
        place_phase_start: Optional[int] = None,
    ) -> tuple[bool, Optional[int]]:
        """Check an entire trajectory for collisions.

        Args:
            waypoints: List of (position, orientation) tuples.
            gripper_phase: If True, check gripper collision at each waypoint.
            grasped_object_cloud: (M, 3) point cloud of grasped object in local frame.
                If provided, transform it to each waypoint and check collision.
            place_phase_start: Index from which to use reduced place_margin.

        Returns:
            (is_valid, first_collision_idx):
                is_valid=True if no collision, first_collision_idx=None.
                is_valid=False if collision, first_collision_idx=index of first collision.
        """
        if self._tree is None:
            return True, None

        for i, (pos, ori) in enumerate(waypoints):
            # Determine margin for this waypoint
            if place_phase_start is not None and i >= place_phase_start:
                margin = self.place_margin
            else:
                margin = self.gripper_margin

            # Check gripper center position
            if self.check_point(pos, margin):
                return False, i

            # Check grasped object if present
            if grasped_object_cloud is not None and len(grasped_object_cloud) > 0:
                # Transform object cloud to current EE frame
                transformed = self._transform_cloud(grasped_object_cloud, pos, ori)
                obj_margin = self.place_margin if (place_phase_start and i >= place_phase_start) else self.object_margin
                dists, _ = self._tree.query(transformed)
                if np.any(dists < obj_margin):
                    return False, i

        return True, None

    @staticmethod
    def _transform_cloud(
        cloud: np.ndarray,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """Transform a point cloud from local frame to world frame.

        Args:
            cloud: (M, 3) points in local (EE) frame.
            position: (3,) EE position in world frame.
            orientation: (4,) quaternion (wxyz) EE orientation.

        Returns:
            (M, 3) points in world frame.
        """
        R = CollisionChecker._quat_to_rotation_matrix(orientation)
        return (R @ cloud.T).T + position

    @staticmethod
    def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion (wxyz) to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])

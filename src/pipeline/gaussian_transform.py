"""
SE(3) transform for Gaussian clusters.

Moves and rotates a group of 3D Gaussians rigidly, preserving their
relative geometry. Used to animate grasped objects along trajectories.

Each Gaussian has:
  - position (xyz): translated + rotated around anchor
  - quaternion (wxyz): rotated by the delta rotation
  - scale: unchanged (rigid transform)
  - opacity: unchanged
  - SH/color: unchanged (DC-only; higher-order SH would need rotation too)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (wxyz convention). Supports batched q2 (N, 4)."""
    w1, x1, y1, z1 = q1
    if q2.ndim == 1:
        w2, x2, y2, z2 = q2
    else:
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (wxyz). For unit quaternions, this is the conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


@dataclass
class GaussianCluster:
    """A group of Gaussians representing one object or scene element.

    All arrays have first dimension N (number of Gaussians).
    """
    positions: np.ndarray      # (N, 3) float32
    quaternions: np.ndarray    # (N, 4) float32, wxyz
    scales: np.ndarray         # (N, 3) float32 (log-space from PLY)
    opacities: np.ndarray      # (N,) float32 (logit-space from PLY)
    sh_dc: np.ndarray          # (N, 3) float32, DC SH coefficients
    sh_rest: Optional[np.ndarray] = None  # (N, K, 3) higher-order SH, optional

    @property
    def n(self) -> int:
        return len(self.positions)

    @property
    def centroid(self) -> np.ndarray:
        return self.positions.mean(axis=0)

    def copy(self) -> "GaussianCluster":
        return GaussianCluster(
            positions=self.positions.copy(),
            quaternions=self.quaternions.copy(),
            scales=self.scales.copy(),
            opacities=self.opacities.copy(),
            sh_dc=self.sh_dc.copy(),
            sh_rest=self.sh_rest.copy() if self.sh_rest is not None else None,
        )


def transform_gaussians(
    cluster: GaussianCluster,
    anchor: np.ndarray,
    new_anchor: np.ndarray,
    rotation_quat: Optional[np.ndarray] = None,
) -> GaussianCluster:
    """Apply rigid SE(3) transform to a Gaussian cluster.

    The transform is defined by:
      - anchor: (3,) the pivot point (grasp contact position)
      - new_anchor: (3,) where the anchor moves to (current EE position)
      - rotation_quat: (4,) wxyz quaternion for the rotation delta.
            If None, pure translation (no rotation).

    Returns a new GaussianCluster with transformed positions and quaternions.
    Scales, opacities, and SH coefficients are copied unchanged.
    """
    result = cluster.copy()

    if rotation_quat is None or np.allclose(rotation_quat, [1, 0, 0, 0], atol=1e-6):
        # Pure translation — fast path
        delta = new_anchor - anchor
        result.positions = cluster.positions + delta
    else:
        # Full SE(3): rotate around anchor, then translate
        R = quat_to_rotation_matrix(rotation_quat)

        # Translate to anchor-centered coordinates, rotate, translate back
        centered = cluster.positions - anchor  # (N, 3)
        rotated = (R @ centered.T).T           # (N, 3)
        result.positions = rotated + new_anchor

        # Rotate each Gaussian's own quaternion
        result.quaternions = quat_multiply(rotation_quat, cluster.quaternions)

        # Normalize quaternions
        norms = np.linalg.norm(result.quaternions, axis=-1, keepdims=True)
        result.quaternions = result.quaternions / norms

    return result


def compute_grasp_offset(
    gripper_pos: np.ndarray,
    gripper_quat: np.ndarray,
    object_centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the rigid offset from gripper frame to object centroid.

    At the moment of grasping, this offset is locked and stays constant
    throughout the carry phase.

    Args:
        gripper_pos: (3,) gripper/EE position at grasp.
        gripper_quat: (4,) gripper/EE quaternion (wxyz) at grasp.
        object_centroid: (3,) centroid of the object Gaussian cluster.

    Returns:
        offset_pos: (3,) position offset in gripper frame.
        offset_quat: (4,) orientation offset (identity for now, since we
            assume the object's local frame is axis-aligned at grasp time).
    """
    R_grip = quat_to_rotation_matrix(gripper_quat)
    # Object centroid in gripper-local frame
    offset_pos = R_grip.T @ (object_centroid - gripper_pos)
    # For simplicity, assume object orientation = world frame at grasp time
    offset_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return offset_pos, offset_quat


def apply_grasp_offset(
    gripper_pos: np.ndarray,
    gripper_quat: np.ndarray,
    offset_pos: np.ndarray,
    offset_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the object's world-frame pose from current gripper pose + offset.

    Args:
        gripper_pos: (3,) current EE position.
        gripper_quat: (4,) current EE quaternion (wxyz).
        offset_pos: (3,) position offset in gripper frame (from compute_grasp_offset).
        offset_quat: (4,) orientation offset (from compute_grasp_offset).

    Returns:
        object_pos: (3,) object centroid in world frame.
        rotation_delta: (4,) quaternion to rotate the object relative to its original pose.
    """
    R_grip = quat_to_rotation_matrix(gripper_quat)
    object_pos = gripper_pos + R_grip @ offset_pos

    # Rotation delta: how much the gripper has rotated since grasp
    # For our trajectories with constant orientation, this is identity
    rotation_delta = quat_multiply(gripper_quat, offset_quat)

    return object_pos, rotation_delta

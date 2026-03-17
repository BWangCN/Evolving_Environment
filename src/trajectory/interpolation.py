"""Cartesian trajectory interpolation: linear position + slerp orientation."""

from __future__ import annotations

import numpy as np


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation between two vectors."""
    return a + t * (b - a)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (wxyz convention)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (wxyz convention)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (wxyz).

    Handles the double-cover: always takes the shorter path.
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    dot = np.dot(q1, q2)

    # Ensure shortest path
    if dot < 0:
        q2 = -q2
        dot = -dot

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return quat_normalize(result)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return quat_normalize(s1 * q1 + s2 * q2)


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (wxyz) to euler angles (roll, pitch, yaw)."""
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Convert euler angles (roll, pitch, yaw) to quaternion (wxyz)."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def interpolate_segment(
    pos_start: np.ndarray,
    ori_start: np.ndarray,
    pos_end: np.ndarray,
    ori_end: np.ndarray,
    n_steps: int,
    include_start: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Interpolate between two EE poses (position + quaternion).

    Returns list of (position, orientation) tuples.
    """
    waypoints = []
    start_idx = 0 if include_start else 1
    for i in range(start_idx, n_steps + 1):
        t = i / n_steps
        pos = lerp(pos_start, pos_end, t)
        ori = slerp(ori_start, ori_end, t)
        waypoints.append((pos.copy(), ori.copy()))
    return waypoints

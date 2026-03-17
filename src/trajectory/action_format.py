"""Convert trajectory waypoints to π0.5-compatible delta actions."""

from __future__ import annotations

import numpy as np

from src.config import franka
from src.trajectory.generator import Trajectory
from src.trajectory.interpolation import quat_to_euler


def trajectory_to_actions(traj: Trajectory) -> np.ndarray:
    """Convert a Trajectory to π0.5 delta action format.

    π0.5 action: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_state]

    Position deltas are normalized by ACTION_POS_SCALE.
    Orientation deltas are normalized by ACTION_ROT_SCALE.
    Gripper state: 0.0=closed, 1.0=open.

    Returns:
        (T-1, 7) array of actions. One fewer than waypoints (delta between consecutive).
    """
    if len(traj) < 2:
        return np.empty((0, franka.ACTION_DIM))

    positions = traj.positions       # (T, 3)
    orientations = traj.orientations  # (T, 4) wxyz
    grippers = traj.gripper_states   # (T,)

    # Convert quaternions to euler for delta computation
    eulers = np.array([quat_to_euler(q) for q in orientations])  # (T, 3)

    # Compute deltas
    pos_deltas = np.diff(positions, axis=0)          # (T-1, 3)
    euler_deltas = np.diff(eulers, axis=0)            # (T-1, 3)

    # Wrap euler deltas to [-pi, pi]
    euler_deltas = (euler_deltas + np.pi) % (2 * np.pi) - np.pi

    # Normalize
    pos_deltas_norm = pos_deltas / franka.ACTION_POS_SCALE
    euler_deltas_norm = euler_deltas / franka.ACTION_ROT_SCALE

    # Gripper state at each action step (use the target waypoint's gripper)
    gripper_actions = grippers[1:]  # (T-1,)

    # Stack: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
    actions = np.column_stack([
        pos_deltas_norm,
        euler_deltas_norm,
        gripper_actions[:, np.newaxis],
    ])

    return actions


def actions_to_raw(actions: np.ndarray) -> np.ndarray:
    """Denormalize actions back to raw deltas (meters, radians).

    Returns:
        (T-1, 7) array with [Δx_m, Δy_m, Δz_m, Δroll_rad, Δpitch_rad, Δyaw_rad, gripper].
    """
    raw = actions.copy()
    raw[:, :3] *= franka.ACTION_POS_SCALE
    raw[:, 3:6] *= franka.ACTION_ROT_SCALE
    return raw

"""Compute place target positions from scene object geometry + semantics."""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.scene.object import SceneObject, TaskType


def compute_place_target(
    task_type: TaskType,
    target_object: SceneObject,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute a 3D place target position given task type and target object geometry.

    Args:
        task_type: The type of placement task.
        target_object: The object serving as placement target.
        rng: Optional random generator for small perturbations.

    Returns:
        (3,) array: [x, y, z] place target in world frame.
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_object.position is None:
        raise ValueError(f"Target object '{target_object.object_id}' has no geometry.")

    if task_type == TaskType.PLACE_ON:
        return _place_on(target_object, rng)
    elif task_type == TaskType.PLACE_IN:
        return _place_in(target_object, rng)
    elif task_type == TaskType.PLACE_NEXT_TO:
        return _place_next_to(target_object, rng)
    elif task_type == TaskType.STACK_ON:
        return _stack_on(target_object, rng)
    else:
        raise ValueError(f"Task type '{task_type}' does not have a place target.")


def _place_on(target: SceneObject, rng: np.random.Generator) -> np.ndarray:
    """Place on top of a surface (plate, tray, table)."""
    # Target: centroid XY of top surface, slightly above top Z
    top_pts = target.get_top_surface_points(threshold=0.01)
    if top_pts is not None and len(top_pts) > 0:
        center_xy = top_pts[:, :2].mean(axis=0)
    else:
        center_xy = target.position[:2]

    # Small random offset to avoid exact center every time
    offset = rng.uniform(-0.02, 0.02, size=2)
    z = target.top_z + 0.005  # 5mm above surface

    return np.array([center_xy[0] + offset[0], center_xy[1] + offset[1], z])


def _place_in(target: SceneObject, rng: np.random.Generator) -> np.ndarray:
    """Place inside a container (bowl, cup, box).

    Strategy: target the opening center, z at ~50% of container depth.
    """
    top_pts = target.get_top_surface_points(threshold=0.02)
    if top_pts is not None and len(top_pts) > 0:
        opening_center = top_pts[:, :2].mean(axis=0)
    else:
        opening_center = target.position[:2]

    # Place at 50% depth inside container
    depth = target.top_z - target.bottom_z
    place_z = target.top_z - depth * 0.5

    # Small random offset within opening
    max_offset = target.container_opening_radius * 0.3
    offset = rng.uniform(-max_offset, max_offset, size=2)

    return np.array([
        opening_center[0] + offset[0],
        opening_center[1] + offset[1],
        place_z,
    ])


def _place_next_to(target: SceneObject, rng: np.random.Generator) -> np.ndarray:
    """Place next to a reference object.

    Strategy: pick a random direction (left/right/front/back), offset by ~12cm.
    """
    direction = rng.choice(4)
    dist = rng.uniform(0.10, 0.15)

    offsets = [
        np.array([dist, 0, 0]),     # right
        np.array([-dist, 0, 0]),    # left
        np.array([0, dist, 0]),     # forward
        np.array([0, -dist, 0]),    # backward
    ]

    place_pos = target.position.copy()
    place_pos += offsets[direction]
    # Place at same z as target's bottom (on the table surface)
    place_pos[2] = target.bottom_z + 0.005

    return place_pos


def _stack_on(target: SceneObject, rng: np.random.Generator) -> np.ndarray:
    """Stack on top of another object.

    Strategy: directly above target's top surface center.
    """
    top_pts = target.get_top_surface_points(threshold=0.01)
    if top_pts is not None and len(top_pts) > 0:
        center_xy = top_pts[:, :2].mean(axis=0)
    else:
        center_xy = target.position[:2]

    # Tiny offset for realism
    offset = rng.uniform(-0.005, 0.005, size=2)
    z = target.top_z + 0.005

    return np.array([center_xy[0] + offset[0], center_xy[1] + offset[1], z])

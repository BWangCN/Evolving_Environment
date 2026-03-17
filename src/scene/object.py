"""Scene object representation combining geometry and semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TaskType(str, Enum):
    """Supported manipulation task types."""
    PICK = "pick"
    PLACE_ON = "place_on"
    PLACE_IN = "place_in"
    PLACE_NEXT_TO = "place_next_to"
    STACK_ON = "stack_on"
    PUSH = "push"


@dataclass
class SceneObject:
    """A single object in the scene with geometry, semantics, and affordances.

    Geometry comes from 3DGS Gaussians (or simulation ground truth).
    Semantics come from OmniGibson synsets (sim) or VLM/CLIP (real).
    Affordances are inferred from category via `infer_affordances()`.
    """

    # --- Identity ---
    object_id: str                     # Unique ID, e.g., "bowl_01"
    category: str                      # Object category, e.g., "bowl"
    description: str = ""              # Free-text, e.g., "white ceramic bowl"

    # --- Geometry (populated from 3DGS or sim) ---
    point_cloud: Optional[np.ndarray] = None   # (N, 3) object surface points
    gaussian_indices: Optional[np.ndarray] = None  # indices into scene Gaussian array
    position: Optional[np.ndarray] = None      # (3,) centroid in world frame
    bbox_min: Optional[np.ndarray] = None      # (3,) axis-aligned bounding box min
    bbox_max: Optional[np.ndarray] = None      # (3,) axis-aligned bounding box max

    # --- Semantics / Affordances ---
    graspable: bool = True
    is_container: bool = False         # Can hold objects inside (bowl, cup, box)
    is_surface: bool = False           # Can support objects on top (plate, table)
    container_opening_radius: float = 0.0  # Radius of opening if container (meters)
    valid_as_target: list[TaskType] = field(default_factory=list)
    valid_as_object: list[TaskType] = field(default_factory=list)

    def __post_init__(self):
        if self.point_cloud is not None:
            self._compute_geometry()

    def _compute_geometry(self):
        """Derive bounding box and centroid from point cloud."""
        pc = self.point_cloud
        if pc is None or len(pc) == 0:
            return
        self.bbox_min = pc.min(axis=0)
        self.bbox_max = pc.max(axis=0)
        self.position = pc.mean(axis=0)

    @property
    def bbox_size(self) -> Optional[np.ndarray]:
        """(3,) bounding box dimensions [width, depth, height]."""
        if self.bbox_min is None or self.bbox_max is None:
            return None
        return self.bbox_max - self.bbox_min

    @property
    def top_z(self) -> Optional[float]:
        """Z-coordinate of the object's top surface."""
        if self.bbox_max is None:
            return None
        return float(self.bbox_max[2])

    @property
    def bottom_z(self) -> Optional[float]:
        """Z-coordinate of the object's bottom."""
        if self.bbox_min is None:
            return None
        return float(self.bbox_min[2])

    def get_top_surface_points(self, threshold: float = 0.02) -> Optional[np.ndarray]:
        """Return points near the top surface (within threshold of max z)."""
        if self.point_cloud is None or self.top_z is None:
            return None
        mask = self.point_cloud[:, 2] > (self.top_z - threshold)
        return self.point_cloud[mask]

    def update_point_cloud(self, point_cloud: np.ndarray):
        """Update point cloud and recompute derived geometry."""
        self.point_cloud = point_cloud
        self._compute_geometry()

    def __repr__(self) -> str:
        pos_str = ""
        if self.position is not None:
            pos_str = f" pos=[{self.position[0]:.3f},{self.position[1]:.3f},{self.position[2]:.3f}]"
        return f"SceneObject('{self.object_id}' [{self.category}]{pos_str})"

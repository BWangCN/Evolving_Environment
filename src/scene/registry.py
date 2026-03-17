"""Scene object registry — manages all objects in a scene."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .affordance import infer_affordances
from .object import SceneObject, TaskType


class SceneObjectRegistry:
    """Manages all objects in a single environment.

    Provides lookup by ID, category, affordance, and spatial queries.
    This is the central data structure that feeds into TaskPlanner,
    PlaceTargetComputer, CollisionChecker, and LanguageGenerator.
    """

    def __init__(self, environment_id: str):
        self.environment_id = environment_id
        self._objects: dict[str, SceneObject] = {}

    def add_object(
        self,
        object_id: str,
        category: str,
        description: str = "",
        point_cloud: Optional[np.ndarray] = None,
        gaussian_indices: Optional[np.ndarray] = None,
    ) -> SceneObject:
        """Create, register, and return a new SceneObject with auto-inferred affordances."""
        obj = SceneObject(
            object_id=object_id,
            category=category,
            description=description,
            point_cloud=point_cloud,
            gaussian_indices=gaussian_indices,
        )
        infer_affordances(obj)
        self._objects[object_id] = obj
        return obj

    def add_existing(self, obj: SceneObject) -> SceneObject:
        """Register an already-constructed SceneObject (affordances inferred if not set)."""
        if not obj.valid_as_object:
            infer_affordances(obj)
        self._objects[obj.object_id] = obj
        return obj

    def remove_object(self, object_id: str) -> Optional[SceneObject]:
        """Remove and return an object, or None if not found."""
        return self._objects.pop(object_id, None)

    def get(self, object_id: str) -> Optional[SceneObject]:
        """Get object by ID."""
        return self._objects.get(object_id)

    def get_by_category(self, category: str) -> list[SceneObject]:
        """Get all objects of a given category."""
        return [o for o in self._objects.values() if o.category == category]

    def get_graspable(self) -> list[SceneObject]:
        """Get all graspable objects."""
        return [o for o in self._objects.values() if o.graspable]

    def get_containers(self) -> list[SceneObject]:
        """Get all container objects (bowl, cup, etc.)."""
        return [o for o in self._objects.values() if o.is_container]

    def get_surfaces(self) -> list[SceneObject]:
        """Get all surface objects (plate, table, etc.)."""
        return [o for o in self._objects.values() if o.is_surface]

    def get_valid_targets(self, task_type: TaskType) -> list[SceneObject]:
        """Get all objects that can serve as target for a given task type."""
        return [
            o for o in self._objects.values()
            if task_type in o.valid_as_target
        ]

    def get_valid_objects(self, task_type: TaskType) -> list[SceneObject]:
        """Get all objects that can be manipulated in a given task type."""
        return [
            o for o in self._objects.values()
            if task_type in o.valid_as_object
        ]

    def get_scene_point_cloud(self, exclude: Optional[list[str]] = None) -> np.ndarray:
        """Get combined point cloud of all objects, optionally excluding some.

        Used for collision checking: exclude the grasped object and place target.
        """
        exclude = set(exclude or [])
        clouds = []
        for obj in self._objects.values():
            if obj.object_id in exclude:
                continue
            if obj.point_cloud is not None:
                clouds.append(obj.point_cloud)
        if not clouds:
            return np.empty((0, 3))
        return np.concatenate(clouds, axis=0)

    @property
    def all_objects(self) -> list[SceneObject]:
        return list(self._objects.values())

    def __len__(self) -> int:
        return len(self._objects)

    def __repr__(self) -> str:
        obj_summary = ", ".join(
            f"{o.object_id}({o.category})" for o in self._objects.values()
        )
        return f"SceneObjectRegistry(env='{self.environment_id}', objects=[{obj_summary}])"

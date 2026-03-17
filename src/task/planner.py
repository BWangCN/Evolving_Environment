"""Task planner — enumerates valid manipulation tasks from a scene registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.scene.object import SceneObject, TaskType
from src.scene.registry import SceneObjectRegistry


@dataclass
class ManipulationTask:
    """A single manipulation task: manipulate `obj` with `task_type`, targeting `target`."""
    task_type: TaskType
    obj: SceneObject           # The object being manipulated (grasped/pushed)
    target: Optional[SceneObject] = None  # The target object (for place/stack tasks)

    @property
    def task_id(self) -> str:
        target_part = f"_{self.target.object_id}" if self.target else ""
        return f"{self.task_type.value}__{self.obj.object_id}{target_part}"

    def __repr__(self) -> str:
        if self.target:
            return f"Task({self.task_type.value}: {self.obj.object_id} → {self.target.object_id})"
        return f"Task({self.task_type.value}: {self.obj.object_id})"


class TaskPlanner:
    """Enumerates all valid manipulation tasks given a scene registry.

    Rules:
    - PICK: any graspable object
    - PLACE_ON: graspable object → surface object
    - PLACE_IN: graspable object → container (obj must fit in container opening)
    - PLACE_NEXT_TO: graspable object → any other object as spatial reference
    - STACK_ON: graspable object → surface or another graspable object
    - PUSH: any object (including non-graspable, excluding environment fixtures)

    Size filtering:
    - PLACE_IN: object's smallest bbox dimension must be < container opening diameter
    - STACK_ON: object's bbox footprint must be smaller than target's top surface
    """

    def __init__(self, registry: SceneObjectRegistry):
        self.registry = registry

    def enumerate_tasks(self, task_types: Optional[list[TaskType]] = None) -> list[ManipulationTask]:
        """Enumerate all valid tasks. Optionally filter by task type."""
        if task_types is None:
            task_types = list(TaskType)

        tasks = []
        for tt in task_types:
            tasks.extend(self._enumerate_for_type(tt))
        return tasks

    def _enumerate_for_type(self, task_type: TaskType) -> list[ManipulationTask]:
        if task_type == TaskType.PICK:
            return self._enumerate_pick()
        elif task_type == TaskType.PLACE_ON:
            return self._enumerate_place_on()
        elif task_type == TaskType.PLACE_IN:
            return self._enumerate_place_in()
        elif task_type == TaskType.PLACE_NEXT_TO:
            return self._enumerate_place_next_to()
        elif task_type == TaskType.STACK_ON:
            return self._enumerate_stack_on()
        elif task_type == TaskType.PUSH:
            return self._enumerate_push()
        return []

    def _enumerate_pick(self) -> list[ManipulationTask]:
        return [
            ManipulationTask(TaskType.PICK, obj)
            for obj in self.registry.get_graspable()
        ]

    def _enumerate_place_on(self) -> list[ManipulationTask]:
        tasks = []
        surfaces = self.registry.get_valid_targets(TaskType.PLACE_ON)
        for obj in self.registry.get_graspable():
            for surface in surfaces:
                if obj.object_id == surface.object_id:
                    continue
                tasks.append(ManipulationTask(TaskType.PLACE_ON, obj, surface))
        return tasks

    def _enumerate_place_in(self) -> list[ManipulationTask]:
        tasks = []
        containers = self.registry.get_containers()
        for obj in self.registry.get_graspable():
            for container in containers:
                if obj.object_id == container.object_id:
                    continue
                if not self._fits_in_container(obj, container):
                    continue
                tasks.append(ManipulationTask(TaskType.PLACE_IN, obj, container))
        return tasks

    def _enumerate_place_next_to(self) -> list[ManipulationTask]:
        tasks = []
        all_objects = self.registry.all_objects
        for obj in self.registry.get_graspable():
            for ref in all_objects:
                if obj.object_id == ref.object_id:
                    continue
                tasks.append(ManipulationTask(TaskType.PLACE_NEXT_TO, obj, ref))
        return tasks

    def _enumerate_stack_on(self) -> list[ManipulationTask]:
        tasks = []
        targets = self.registry.get_valid_targets(TaskType.STACK_ON)
        for obj in self.registry.get_graspable():
            for target in targets:
                if obj.object_id == target.object_id:
                    continue
                tasks.append(ManipulationTask(TaskType.STACK_ON, obj, target))
        return tasks

    def _enumerate_push(self) -> list[ManipulationTask]:
        return [
            ManipulationTask(TaskType.PUSH, obj)
            for obj in self.registry.all_objects
            if obj.graspable  # only push movable objects, not tables
        ]

    @staticmethod
    def _fits_in_container(obj: SceneObject, container: SceneObject) -> bool:
        """Check if obj's smallest dimension fits through container opening."""
        if container.container_opening_radius <= 0:
            return False
        if obj.bbox_size is None:
            return True  # no geometry yet, assume it fits
        smallest_dim = float(min(obj.bbox_size[:2]))  # XY footprint
        return smallest_dim < container.container_opening_radius * 2

"""Tests for scene object system."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scene import SceneObject, SceneObjectRegistry, infer_affordances
from src.scene.object import TaskType


def _make_bowl_cloud(center=(0.3, 0.1, 0.05), radius=0.07, depth=0.05, n=500):
    """Generate a mock bowl-shaped point cloud."""
    rng = np.random.default_rng(42)
    # Bowl bottom (disk)
    r = rng.uniform(0, radius * 0.8, n // 2)
    theta = rng.uniform(0, 2 * np.pi, n // 2)
    bottom = np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        np.full(n // 2, center[2]),
    ])
    # Bowl walls (cylinder ring)
    theta2 = rng.uniform(0, 2 * np.pi, n // 2)
    z = rng.uniform(center[2], center[2] + depth, n // 2)
    walls = np.column_stack([
        center[0] + radius * np.cos(theta2),
        center[1] + radius * np.sin(theta2),
        z,
    ])
    return np.vstack([bottom, walls])


def _make_plate_cloud(center=(0.0, 0.0, 0.02), radius=0.10, n=300):
    """Generate a mock plate-shaped point cloud (flat disk)."""
    rng = np.random.default_rng(42)
    r = rng.uniform(0, radius, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        np.full(n, center[2]) + rng.normal(0, 0.002, n),
    ])


def _make_mug_cloud(center=(0.15, -0.1, 0.05), radius=0.04, height=0.10, n=400):
    """Generate a mock mug-shaped point cloud (cylinder)."""
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(center[2], center[2] + height, n)
    r = rng.uniform(radius * 0.9, radius, n)
    return np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        z,
    ])


class TestSceneObject:
    def test_basic_creation(self):
        obj = SceneObject(object_id="mug_01", category="mug", description="red mug")
        assert obj.object_id == "mug_01"
        assert obj.category == "mug"
        assert obj.position is None

    def test_geometry_from_point_cloud(self):
        cloud = _make_mug_cloud()
        obj = SceneObject(
            object_id="mug_01", category="mug",
            description="red mug", point_cloud=cloud,
        )
        assert obj.position is not None
        assert obj.bbox_min is not None
        assert obj.bbox_max is not None
        assert obj.top_z > obj.bottom_z
        assert obj.bbox_size is not None
        assert all(s > 0 for s in obj.bbox_size)

    def test_top_surface_points(self):
        cloud = _make_mug_cloud(height=0.10)
        obj = SceneObject(object_id="mug_01", category="mug", point_cloud=cloud)
        top_pts = obj.get_top_surface_points(threshold=0.02)
        assert top_pts is not None
        assert len(top_pts) > 0
        assert all(top_pts[:, 2] > obj.top_z - 0.02)


class TestAffordance:
    def test_bowl_affordances(self):
        obj = SceneObject(object_id="bowl_01", category="bowl")
        infer_affordances(obj)
        assert obj.is_container is True
        assert obj.is_surface is False
        assert obj.graspable is True
        assert TaskType.PLACE_IN in obj.valid_as_target
        assert TaskType.PICK in obj.valid_as_object

    def test_plate_affordances(self):
        obj = SceneObject(object_id="plate_01", category="plate")
        infer_affordances(obj)
        assert obj.is_container is False
        assert obj.is_surface is True
        assert TaskType.PLACE_ON in obj.valid_as_target
        assert TaskType.STACK_ON in obj.valid_as_target

    def test_table_not_graspable(self):
        obj = SceneObject(object_id="table_01", category="table")
        infer_affordances(obj)
        assert obj.graspable is False
        assert TaskType.PICK not in obj.valid_as_object

    def test_unknown_category_defaults(self):
        obj = SceneObject(object_id="mystery_01", category="alien_artifact")
        infer_affordances(obj)
        assert obj.graspable is True  # default: graspable
        assert obj.is_container is False


class TestRegistry:
    def _make_e1_registry(self):
        """Create a registry mimicking E1 (Base Kitchen)."""
        reg = SceneObjectRegistry("E1")
        reg.add_object("mug_01", "mug", "red mug",
                        point_cloud=_make_mug_cloud(center=(0.15, -0.1, 0.05)))
        reg.add_object("bowl_01", "bowl", "white bowl",
                        point_cloud=_make_bowl_cloud(center=(0.3, 0.1, 0.05)))
        reg.add_object("plate_01", "plate", "ceramic plate",
                        point_cloud=_make_plate_cloud(center=(0.0, 0.0, 0.02)))
        reg.add_object("bottle_01", "bottle", "water bottle",
                        point_cloud=_make_mug_cloud(center=(-0.15, 0.0, 0.05),
                                                     radius=0.03, height=0.20))
        reg.add_object("spoon_01", "spoon", "metal spoon",
                        point_cloud=np.random.default_rng(42).uniform(
                            [-0.2, 0.15, 0.02], [-0.1, 0.17, 0.03], (100, 3)))
        return reg

    def test_add_and_retrieve(self):
        reg = self._make_e1_registry()
        assert len(reg) == 5
        assert reg.get("mug_01") is not None
        assert reg.get("nonexistent") is None

    def test_category_lookup(self):
        reg = self._make_e1_registry()
        mugs = reg.get_by_category("mug")
        assert len(mugs) == 1
        assert mugs[0].object_id == "mug_01"

    def test_graspable_filter(self):
        reg = self._make_e1_registry()
        # Add a non-graspable table
        reg.add_object("table_01", "table", "kitchen table")
        graspable = reg.get_graspable()
        assert len(graspable) == 5  # table excluded
        assert all(o.graspable for o in graspable)

    def test_containers(self):
        reg = self._make_e1_registry()
        containers = reg.get_containers()
        categories = {o.category for o in containers}
        assert "bowl" in categories
        assert "mug" in categories
        assert "plate" not in categories

    def test_valid_targets(self):
        reg = self._make_e1_registry()
        place_in_targets = reg.get_valid_targets(TaskType.PLACE_IN)
        categories = {o.category for o in place_in_targets}
        assert "bowl" in categories
        assert "mug" in categories
        assert "bottle" not in categories

    def test_scene_point_cloud(self):
        reg = self._make_e1_registry()
        full_cloud = reg.get_scene_point_cloud()
        assert full_cloud.shape[1] == 3
        assert full_cloud.shape[0] > 0

        # Exclude bowl → should have fewer points
        partial_cloud = reg.get_scene_point_cloud(exclude=["bowl_01"])
        assert partial_cloud.shape[0] < full_cloud.shape[0]

    def test_remove_object(self):
        reg = self._make_e1_registry()
        removed = reg.remove_object("mug_01")
        assert removed is not None
        assert removed.object_id == "mug_01"
        assert len(reg) == 4
        assert reg.get("mug_01") is None

    def test_repr(self):
        reg = self._make_e1_registry()
        s = repr(reg)
        assert "E1" in s
        assert "mug_01" in s


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

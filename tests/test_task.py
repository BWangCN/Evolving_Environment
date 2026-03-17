"""Tests for task planning and language generation."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scene import SceneObjectRegistry
from src.scene.object import TaskType
from src.task import TaskPlanner, ManipulationTask, LanguageGenerator


def _make_e1_registry():
    """E1 Base Kitchen: mug, bowl, plate, bottle, spoon."""
    rng = np.random.default_rng(42)
    reg = SceneObjectRegistry("E1")

    # Mug (container, r=0.04)
    theta = rng.uniform(0, 2 * np.pi, 400)
    z = rng.uniform(0.05, 0.15, 400)
    mug_cloud = np.column_stack([
        0.15 + 0.04 * np.cos(theta), -0.1 + 0.04 * np.sin(theta), z
    ])
    reg.add_object("mug_01", "mug", "red mug", point_cloud=mug_cloud)

    # Bowl (container, r=0.07)
    r = rng.uniform(0, 0.07, 300)
    theta2 = rng.uniform(0, 2 * np.pi, 300)
    bowl_cloud = np.column_stack([
        0.3 + r * np.cos(theta2), 0.1 + r * np.sin(theta2),
        rng.uniform(0.02, 0.07, 300),
    ])
    reg.add_object("bowl_01", "bowl", "white bowl", point_cloud=bowl_cloud)

    # Plate (surface, r=0.10)
    r2 = rng.uniform(0, 0.10, 200)
    theta3 = rng.uniform(0, 2 * np.pi, 200)
    plate_cloud = np.column_stack([
        r2 * np.cos(theta3), r2 * np.sin(theta3),
        np.full(200, 0.02) + rng.normal(0, 0.001, 200),
    ])
    reg.add_object("plate_01", "plate", "ceramic plate", point_cloud=plate_cloud)

    # Bottle (tall cylinder, r=0.03)
    theta4 = rng.uniform(0, 2 * np.pi, 300)
    z4 = rng.uniform(0.05, 0.25, 300)
    bottle_cloud = np.column_stack([
        -0.15 + 0.03 * np.cos(theta4), 0.03 * np.sin(theta4), z4
    ])
    reg.add_object("bottle_01", "bottle", "water bottle", point_cloud=bottle_cloud)

    # Spoon (small elongated)
    spoon_cloud = rng.uniform([-0.2, 0.15, 0.02], [-0.1, 0.17, 0.03], (100, 3))
    reg.add_object("spoon_01", "spoon", "metal spoon", point_cloud=spoon_cloud)

    return reg


class TestTaskPlanner:
    def test_pick_tasks(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        tasks = planner.enumerate_tasks([TaskType.PICK])
        # All 5 graspable objects
        assert len(tasks) == 5
        assert all(t.task_type == TaskType.PICK for t in tasks)
        assert all(t.target is None for t in tasks)

    def test_place_on_tasks(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        tasks = planner.enumerate_tasks([TaskType.PLACE_ON])
        # plate is the only surface, 4 other graspable objects can be placed on it
        # (plate itself can also be placed on plate? no, self-pair excluded)
        assert len(tasks) == 4
        assert all(t.target.category == "plate" for t in tasks)

    def test_place_in_tasks(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        tasks = planner.enumerate_tasks([TaskType.PLACE_IN])
        # Containers: mug (opening r=0.04, diam=0.08), bowl (opening r=0.07, diam=0.14)
        # spoon should fit in both (small)
        # bottle is tall (min XY dim ~0.06) — fits in bowl but not mug
        target_ids = {t.target.object_id for t in tasks}
        assert "bowl_01" in target_ids or "mug_01" in target_ids
        # No object places itself in itself
        for t in tasks:
            assert t.obj.object_id != t.target.object_id

    def test_place_next_to_tasks(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        tasks = planner.enumerate_tasks([TaskType.PLACE_NEXT_TO])
        # 5 graspable × 4 others = 20
        assert len(tasks) == 20

    def test_all_tasks(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        all_tasks = planner.enumerate_tasks()
        assert len(all_tasks) > 0
        task_types = {t.task_type for t in all_tasks}
        assert TaskType.PICK in task_types
        assert TaskType.PLACE_ON in task_types

    def test_task_id_unique(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        all_tasks = planner.enumerate_tasks()
        ids = [t.task_id for t in all_tasks]
        assert len(ids) == len(set(ids)), "Task IDs should be unique"

    def test_no_self_tasks(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        all_tasks = planner.enumerate_tasks()
        for t in all_tasks:
            if t.target is not None:
                assert t.obj.object_id != t.target.object_id


class TestLanguageGenerator:
    def test_template_generation(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        gen = LanguageGenerator(backend="template")

        pick_tasks = planner.enumerate_tasks([TaskType.PICK])
        instructions = gen.generate(pick_tasks[0], n_variants=3, seed=42)
        assert len(instructions) == 3
        assert all(isinstance(s, str) for s in instructions)
        assert all(len(s) > 0 for s in instructions)

    def test_place_in_instruction(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        gen = LanguageGenerator(backend="template")

        place_in_tasks = planner.enumerate_tasks([TaskType.PLACE_IN])
        if place_in_tasks:
            instructions = gen.generate(place_in_tasks[0], n_variants=3, seed=42)
            # Should mention both object and target
            for instr in instructions:
                assert len(instr) > 10

    def test_single_instruction(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        gen = LanguageGenerator(backend="template")

        tasks = planner.enumerate_tasks([TaskType.PICK])
        instr = gen.generate_single(tasks[0], seed=42)
        assert isinstance(instr, str)
        assert len(instr) > 0

    def test_deterministic_with_seed(self):
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        gen = LanguageGenerator(backend="template")

        task = planner.enumerate_tasks([TaskType.PICK])[0]
        r1 = gen.generate(task, n_variants=3, seed=123)
        r2 = gen.generate(task, n_variants=3, seed=123)
        assert r1 == r2

    def test_llm_backend_not_implemented(self):
        gen = LanguageGenerator(backend="llm")
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        task = planner.enumerate_tasks([TaskType.PICK])[0]
        try:
            gen.generate(task)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass

    def test_all_task_types_have_templates(self):
        """Every task type should produce non-empty instructions."""
        reg = _make_e1_registry()
        planner = TaskPlanner(reg)
        gen = LanguageGenerator(backend="template")

        for tt in TaskType:
            tasks = planner.enumerate_tasks([tt])
            if tasks:
                instr = gen.generate(tasks[0], n_variants=1, seed=42)
                assert len(instr) == 1
                assert len(instr[0]) > 0, f"Empty instruction for {tt}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

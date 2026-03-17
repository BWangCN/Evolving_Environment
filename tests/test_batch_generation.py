"""Batch trajectory generation stress test.

Generates hundreds of trajectories with randomized grasp poses across all
E1 objects and task types. Reports statistics on:
- Collision rejection rate
- Trajectory lengths
- Action delta distributions
- Per-task-type success rates
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import franka
from src.scene import SceneObjectRegistry
from src.scene.object import TaskType
from src.task import TaskPlanner, LanguageGenerator
from src.trajectory import (
    TrajectoryGenerator, GraspPose, CollisionChecker,
    compute_place_target, trajectory_to_actions, compute_camera_poses,
)


def _make_full_e1_registry():
    """E1 with 5 objects placed realistically on a table."""
    rng = np.random.default_rng(42)
    reg = SceneObjectRegistry("E1")

    def _cylinder(cx, cy, cz, r, h, n=400):
        theta = rng.uniform(0, 2 * np.pi, n)
        z = rng.uniform(cz, cz + h, n)
        rad = rng.uniform(r * 0.85, r, n)
        return np.column_stack([cx + rad * np.cos(theta), cy + rad * np.sin(theta), z])

    def _bowl(cx, cy, cz, r, depth, n=500):
        # Bottom disk
        r1 = rng.uniform(0, r * 0.8, n // 2)
        t1 = rng.uniform(0, 2 * np.pi, n // 2)
        bottom = np.column_stack([cx + r1 * np.cos(t1), cy + r1 * np.sin(t1), np.full(n // 2, cz)])
        # Walls
        t2 = rng.uniform(0, 2 * np.pi, n // 2)
        z2 = rng.uniform(cz, cz + depth, n // 2)
        walls = np.column_stack([cx + r * np.cos(t2), cy + r * np.sin(t2), z2])
        return np.vstack([bottom, walls])

    def _disk(cx, cy, cz, r, n=300):
        r1 = rng.uniform(0, r, n)
        t1 = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([cx + r1 * np.cos(t1), cy + r1 * np.sin(t1),
                                np.full(n, cz) + rng.normal(0, 0.002, n)])

    def _elongated(x_range, y_range, z_range, n=150):
        return rng.uniform([x_range[0], y_range[0], z_range[0]],
                           [x_range[1], y_range[1], z_range[1]], (n, 3))

    reg.add_object("mug_01", "mug", "red ceramic mug",
                   point_cloud=_cylinder(0.20, -0.10, 0.02, 0.04, 0.10))
    reg.add_object("bowl_01", "bowl", "white porcelain bowl",
                   point_cloud=_bowl(0.35, 0.12, 0.02, 0.07, 0.05))
    reg.add_object("plate_01", "plate", "round ceramic plate",
                   point_cloud=_disk(0.0, 0.0, 0.02, 0.10))
    reg.add_object("bottle_01", "bottle", "clear water bottle",
                   point_cloud=_cylinder(-0.15, 0.05, 0.02, 0.03, 0.22))
    reg.add_object("spoon_01", "spoon", "stainless steel spoon",
                   point_cloud=_elongated((-0.25, -0.12), (0.18, 0.20), (0.02, 0.035)))

    return reg


def _random_grasp_for_object(obj, rng, n_grasps=10):
    """Generate random grasp poses around an object's bounding box."""
    grasps = []
    if obj.position is None:
        return grasps

    for _ in range(n_grasps):
        # Random position near the object center, at a graspable height
        pos = obj.position.copy()
        pos[:2] += rng.uniform(-0.02, 0.02, 2)
        pos[2] = max(pos[2], 0.03)  # ensure above table

        # Random approach angle: mostly top-down with some variation
        roll = rng.uniform(-0.3, 0.3)
        pitch = rng.uniform(-0.3, 0.3)
        yaw = rng.uniform(-np.pi, np.pi)

        from src.trajectory.interpolation import euler_to_quat
        ori = euler_to_quat(np.array([roll, pitch, yaw]))

        width = rng.uniform(0.03, franka.GRIPPER_MAX_OPENING)
        score = rng.uniform(0.5, 1.0)

        grasps.append(GraspPose(position=pos, orientation=ori, score=score, width=width))

    return grasps


class TestBatchGeneration:
    """Stress test: generate many trajectories and collect statistics."""

    def test_batch_pick_trajectories(self):
        """Generate 50 pick trajectories with random grasps across all objects."""
        rng = np.random.default_rng(123)
        reg = _make_full_e1_registry()
        gen = TrajectoryGenerator()

        total = 0
        valid = 0
        lengths = []
        t_start = time.time()

        for obj in reg.get_graspable():
            grasps = _random_grasp_for_object(obj, rng, n_grasps=10)
            scene_cloud = reg.get_scene_point_cloud(exclude=[obj.object_id])
            checker = CollisionChecker(scene_cloud)

            for grasp in grasps:
                traj = gen.generate_pick(grasp)
                is_valid = gen.validate_trajectory(traj, checker)
                total += 1
                if is_valid:
                    valid += 1
                    lengths.append(len(traj))

        elapsed = time.time() - t_start

        print(f"\n=== Batch PICK Test ===")
        print(f"Total: {total}, Valid: {valid}, Rejected: {total - valid}")
        print(f"Acceptance rate: {valid/total:.1%}")
        print(f"Trajectory length: mean={np.mean(lengths):.0f}, "
              f"std={np.std(lengths):.0f}, range=[{min(lengths)}, {max(lengths)}]")
        print(f"Time: {elapsed:.2f}s ({elapsed/total*1000:.1f}ms per trajectory)")

        assert valid > 0, "At least some trajectories should be valid"
        assert valid / total > 0.3, "Acceptance rate should be reasonable"

    def test_batch_pick_place_all_task_types(self):
        """Generate pick-place trajectories for every valid task in E1."""
        rng = np.random.default_rng(456)
        reg = _make_full_e1_registry()
        gen = TrajectoryGenerator()
        planner = TaskPlanner(reg)
        lang_gen = LanguageGenerator()

        task_types_to_test = [TaskType.PLACE_ON, TaskType.PLACE_IN,
                              TaskType.PLACE_NEXT_TO, TaskType.STACK_ON]

        stats: dict[str, dict] = {}
        all_actions = []
        t_start = time.time()

        for tt in task_types_to_test:
            tasks = planner.enumerate_tasks([tt])
            total = 0
            valid = 0
            lengths = []

            for task in tasks:
                # Generate multiple grasps per task
                grasps = _random_grasp_for_object(task.obj, rng, n_grasps=5)
                exclude = [task.obj.object_id]
                if task.target:
                    exclude.append(task.target.object_id)
                scene_cloud = reg.get_scene_point_cloud(exclude=exclude)
                checker = CollisionChecker(scene_cloud)

                for grasp in grasps:
                    try:
                        place_pos = compute_place_target(tt, task.target, rng=rng)
                    except (ValueError, AttributeError):
                        continue

                    traj = gen.generate_pick_place(grasp, place_pos)
                    is_valid = gen.validate_trajectory(traj, checker)
                    total += 1

                    if is_valid:
                        valid += 1
                        lengths.append(len(traj))
                        actions = trajectory_to_actions(traj)
                        all_actions.append(actions)

                        # Also generate language (quick check it doesn't crash)
                        lang_gen.generate(task, n_variants=1, seed=42)

            stats[tt.value] = {
                "total": total,
                "valid": valid,
                "acceptance": valid / total if total > 0 else 0,
                "mean_length": np.mean(lengths) if lengths else 0,
            }

        elapsed = time.time() - t_start

        # Print results
        print(f"\n=== Batch PICK-PLACE Test (all task types) ===")
        print(f"{'Task Type':<18} {'Total':>6} {'Valid':>6} {'Accept':>8} {'Avg Len':>8}")
        print("-" * 50)
        total_all = 0
        valid_all = 0
        for tt_name, s in stats.items():
            print(f"{tt_name:<18} {s['total']:>6} {s['valid']:>6} "
                  f"{s['acceptance']:>7.1%} {s['mean_length']:>8.0f}")
            total_all += s["total"]
            valid_all += s["valid"]
        print("-" * 50)
        print(f"{'TOTAL':<18} {total_all:>6} {valid_all:>6} "
              f"{valid_all/total_all:>7.1%}")
        print(f"Time: {elapsed:.2f}s")

        # Action statistics
        if all_actions:
            all_act = np.concatenate(all_actions, axis=0)
            print(f"\n=== Action Statistics ({all_act.shape[0]} action steps) ===")
            labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
            print(f"{'Dim':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
            for i, label in enumerate(labels):
                col = all_act[:, i]
                print(f"{label:<10} {np.mean(col):>8.3f} {np.std(col):>8.3f} "
                      f"{np.min(col):>8.3f} {np.max(col):>8.3f}")

        assert valid_all > 0, "Should have some valid trajectories"

    def test_camera_pose_consistency(self):
        """Verify camera poses are consistent across a batch of trajectories."""
        rng = np.random.default_rng(789)
        reg = _make_full_e1_registry()
        gen = TrajectoryGenerator()

        mug = reg.get("mug_01")
        grasps = _random_grasp_for_object(mug, rng, n_grasps=20)

        cam_distances = []  # distance between cam and EE at each waypoint

        for grasp in grasps:
            traj = gen.generate_pick(grasp)
            cam_poses = compute_camera_poses(traj)

            for (cam_pos, _), wp in zip(cam_poses, traj.waypoints):
                dist = np.linalg.norm(cam_pos - wp.position)
                cam_distances.append(dist)

        cam_distances = np.array(cam_distances)
        print(f"\n=== Camera-EE Distance ===")
        print(f"Mean: {cam_distances.mean():.4f}m, Std: {cam_distances.std():.6f}m")
        print(f"Range: [{cam_distances.min():.4f}, {cam_distances.max():.4f}]m")

        # Camera offset should be consistent (fixed transform)
        assert cam_distances.std() < 0.01, "Camera-EE distance should be nearly constant"

    def test_action_normalization_range(self):
        """Verify actions stay within reasonable normalized ranges."""
        rng = np.random.default_rng(321)
        reg = _make_full_e1_registry()
        gen = TrajectoryGenerator()

        all_actions = []
        for obj in reg.get_graspable():
            grasps = _random_grasp_for_object(obj, rng, n_grasps=10)
            for grasp in grasps:
                traj = gen.generate_pick(grasp)
                actions = trajectory_to_actions(traj)
                all_actions.append(actions)

        all_act = np.concatenate(all_actions, axis=0)

        # Position deltas should be within ±20 normalized units (±40cm)
        assert np.all(np.abs(all_act[:, :3]) < 20), \
            f"Position deltas too large: max={np.abs(all_act[:, :3]).max():.1f}"

        # Orientation deltas should be within ±20 normalized units (±2 rad)
        assert np.all(np.abs(all_act[:, 3:6]) < 20), \
            f"Orientation deltas too large: max={np.abs(all_act[:, 3:6]).max():.1f}"

        # Gripper should only be 0 or 1
        gripper_vals = set(all_act[:, 6])
        assert gripper_vals.issubset({0.0, 1.0}), f"Unexpected gripper values: {gripper_vals}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])

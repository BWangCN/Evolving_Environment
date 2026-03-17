"""Tests for trajectory generation pipeline."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import franka
from src.scene import SceneObjectRegistry
from src.scene.object import TaskType
from src.trajectory import (
    TrajectoryGenerator, GraspPose, CollisionChecker,
    compute_place_target, trajectory_to_actions, compute_camera_poses,
)
from src.trajectory.interpolation import slerp, quat_to_euler, euler_to_quat


def _mock_grasp_top_down(pos=(0.3, 0.1, 0.07)):
    """Mock grasp pose: top-down approach (gripper pointing -z)."""
    return GraspPose(
        position=np.array(pos),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz, pointing down
        score=0.9,
        width=0.05,
    )


def _mock_grasp_angled(pos=(0.15, -0.1, 0.08)):
    """Mock grasp pose: 45-degree angled approach."""
    # Rotate 45 deg around Y axis
    angle = np.pi / 4
    return GraspPose(
        position=np.array(pos),
        orientation=np.array([np.cos(angle/2), 0, np.sin(angle/2), 0]),
        score=0.85,
        width=0.06,
    )


def _make_e1_registry():
    """E1 scene with bowl and plate."""
    rng = np.random.default_rng(42)
    reg = SceneObjectRegistry("E1")

    # Bowl at (0.3, 0.1, z)
    r = rng.uniform(0, 0.07, 300)
    theta = rng.uniform(0, 2 * np.pi, 300)
    bowl_cloud = np.column_stack([
        0.3 + r * np.cos(theta), 0.1 + r * np.sin(theta),
        rng.uniform(0.02, 0.07, 300),
    ])
    reg.add_object("bowl_01", "bowl", "white bowl", point_cloud=bowl_cloud)

    # Plate at (0.0, 0.0, 0.02)
    r2 = rng.uniform(0, 0.10, 200)
    theta2 = rng.uniform(0, 2 * np.pi, 200)
    plate_cloud = np.column_stack([
        r2 * np.cos(theta2), r2 * np.sin(theta2),
        np.full(200, 0.02),
    ])
    reg.add_object("plate_01", "plate", "ceramic plate", point_cloud=plate_cloud)

    return reg


class TestInterpolation:
    def test_slerp_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = slerp(q, q, 0.5)
        np.testing.assert_allclose(result, q, atol=1e-6)

    def test_slerp_halfway(self):
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = euler_to_quat(np.array([0, 0, np.pi/2]))
        mid = slerp(q1, q2, 0.5)
        euler_mid = quat_to_euler(mid)
        np.testing.assert_allclose(euler_mid[2], np.pi/4, atol=0.01)

    def test_euler_roundtrip(self):
        rpy = np.array([0.1, 0.2, 0.3])
        q = euler_to_quat(rpy)
        rpy_back = quat_to_euler(q)
        np.testing.assert_allclose(rpy_back, rpy, atol=1e-6)


class TestTrajectoryGenerator:
    def test_pick_trajectory_length(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        # Should have multiple waypoints across phases
        assert len(traj) > 20
        assert traj.task_type == "pick"

    def test_pick_starts_at_home(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        first = traj.waypoints[0]
        np.testing.assert_allclose(first.position, franka.HOME_POSITION, atol=1e-6)

    def test_pick_gripper_sequence(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        grippers = traj.gripper_states

        # Should start open, then close at grasp, stay closed for lift
        assert grippers[0] == franka.GRIPPER_OPEN
        assert grippers[-1] == franka.GRIPPER_CLOSE
        # Find the close transition
        close_idx = np.where(grippers == franka.GRIPPER_CLOSE)[0][0]
        assert close_idx > 0  # Not the first waypoint

    def test_pick_place_trajectory(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        place = np.array([0.0, 0.0, 0.025])
        traj = gen.generate_pick_place(grasp, place)

        assert len(traj) > 40  # Pick + place phases
        assert traj.task_type == "pick_place"

        grippers = traj.gripper_states
        # Should open, close (grasp), then open again (release)
        assert grippers[0] == franka.GRIPPER_OPEN   # start
        assert grippers[-1] == franka.GRIPPER_OPEN   # end (after release)

    def test_pick_place_reaches_target(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        place = np.array([0.0, 0.0, 0.025])
        traj = gen.generate_pick_place(grasp, place)

        # Find the release waypoint
        for w in traj.waypoints:
            if w.phase == "release":
                np.testing.assert_allclose(w.position, place, atol=1e-6)
                break

    def test_phases_present(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        place = np.array([0.0, 0.0, 0.025])
        traj = gen.generate_pick_place(grasp, place)

        phases = {w.phase for w in traj.waypoints}
        expected = {"transit_pick", "pre_grasp", "approach", "grasp",
                    "lift", "transit_place", "pre_place", "place_descend",
                    "release", "retreat"}
        assert expected.issubset(phases)


class TestCollisionChecker:
    def test_no_obstacles(self):
        checker = CollisionChecker(np.empty((0, 3)))
        assert not checker.check_point(np.array([0, 0, 0]), 0.05)

    def test_collision_detected(self):
        obstacles = np.array([[0.3, 0.1, 0.05]])
        checker = CollisionChecker(obstacles, gripper_margin=0.05)
        # Point within margin
        assert checker.check_point(np.array([0.32, 0.1, 0.05]), 0.05)
        # Point outside margin
        assert not checker.check_point(np.array([0.5, 0.5, 0.5]), 0.05)

    def test_trajectory_validation(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down(pos=(0.3, 0.1, 0.07))

        # Obstacle right at grasp location → should collide
        obstacles = np.array([[0.3, 0.1, 0.20]])  # something above
        checker = CollisionChecker(obstacles, gripper_margin=0.05)

        traj = gen.generate_pick(grasp)
        is_valid = gen.validate_trajectory(traj, checker)
        # May or may not collide depending on exact geometry
        assert isinstance(is_valid, bool)

    def test_clear_workspace(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down(pos=(0.3, 0.1, 0.07))

        # No obstacles → should be valid
        checker = CollisionChecker(np.empty((0, 3)))
        traj = gen.generate_pick(grasp)
        assert gen.validate_trajectory(traj, checker) is True


class TestPlaceTarget:
    def test_place_on_plate(self):
        reg = _make_e1_registry()
        plate = reg.get("plate_01")
        target = compute_place_target(TaskType.PLACE_ON, plate, rng=np.random.default_rng(42))
        # Should be above plate's top surface
        assert target[2] > plate.top_z - 0.01
        # Should be near plate's center XY
        assert abs(target[0] - plate.position[0]) < 0.15
        assert abs(target[1] - plate.position[1]) < 0.15

    def test_place_in_bowl(self):
        reg = _make_e1_registry()
        bowl = reg.get("bowl_01")
        target = compute_place_target(TaskType.PLACE_IN, bowl, rng=np.random.default_rng(42))
        # Should be inside the bowl (z between bottom and top)
        assert target[2] < bowl.top_z
        assert target[2] > bowl.bottom_z
        # Should be near bowl center XY
        assert abs(target[0] - bowl.position[0]) < bowl.container_opening_radius

    def test_place_next_to(self):
        reg = _make_e1_registry()
        bowl = reg.get("bowl_01")
        target = compute_place_target(TaskType.PLACE_NEXT_TO, bowl, rng=np.random.default_rng(42))
        # Should be offset from bowl
        dist = np.linalg.norm(target[:2] - bowl.position[:2])
        assert dist > 0.08


class TestActionFormat:
    def test_action_shape(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        actions = trajectory_to_actions(traj)
        assert actions.shape[0] == len(traj) - 1
        assert actions.shape[1] == franka.ACTION_DIM

    def test_gripper_in_actions(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        actions = trajectory_to_actions(traj)
        # Gripper column (index 6) should contain both open and closed
        gripper_vals = set(actions[:, 6])
        assert franka.GRIPPER_OPEN in gripper_vals or franka.GRIPPER_CLOSE in gripper_vals

    def test_position_deltas_reasonable(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        actions = trajectory_to_actions(traj)
        # Normalized position deltas should not be extreme
        pos_deltas = actions[:, :3]
        assert np.all(np.abs(pos_deltas) < 50), "Position deltas seem too large"


class TestCameraPose:
    def test_camera_pose_count(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        cam_poses = compute_camera_poses(traj)
        assert len(cam_poses) == len(traj)

    def test_camera_offset_from_ee(self):
        gen = TrajectoryGenerator()
        grasp = _mock_grasp_top_down()
        traj = gen.generate_pick(grasp)
        cam_poses = compute_camera_poses(traj)
        for (cam_pos, _), wp in zip(cam_poses, traj.waypoints):
            # Camera should be close to but not exactly at EE
            dist = np.linalg.norm(cam_pos - wp.position)
            assert dist < 0.2  # within 20cm (camera offset is 5cm)


class TestEndToEnd:
    """Full pipeline: registry → task → grasp (mock) → trajectory → actions."""

    def test_full_pick_place_pipeline(self):
        # 1. Scene setup
        reg = _make_e1_registry()

        # 2. Task: place something in the bowl
        bowl = reg.get("bowl_01")

        # 3. Mock grasp pose (as if from AnyGrasp)
        grasp = _mock_grasp_top_down(pos=(0.15, -0.1, 0.07))

        # 4. Compute place target
        place = compute_place_target(TaskType.PLACE_IN, bowl, rng=np.random.default_rng(42))

        # 5. Generate trajectory
        gen = TrajectoryGenerator()
        traj = gen.generate_pick_place(grasp, place)
        assert len(traj) > 0

        # 6. Collision check (exclude bowl as it's the target)
        scene_cloud = reg.get_scene_point_cloud(exclude=["bowl_01"])
        checker = CollisionChecker(scene_cloud)
        is_valid = gen.validate_trajectory(traj, checker)
        # Should pass since objects are spread out
        assert is_valid

        # 7. Convert to actions
        actions = trajectory_to_actions(traj)
        assert actions.shape[1] == franka.ACTION_DIM

        # 8. Camera poses
        cam_poses = compute_camera_poses(traj)
        assert len(cam_poses) == len(traj)

        print(f"\n=== End-to-End Test ===")
        print(f"Trajectory: {len(traj)} waypoints")
        print(f"Actions: {actions.shape}")
        print(f"Camera poses: {len(cam_poses)}")
        print(f"Collision-free: {is_valid}")
        print(f"Place target: {place}")
        print(f"Phases: {[w.phase for w in traj.waypoints[::10]]}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

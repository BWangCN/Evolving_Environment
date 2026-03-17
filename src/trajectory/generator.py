"""Trajectory generator: grasp pose + place target → phased waypoint sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.config import franka
from src.trajectory.interpolation import interpolate_segment, quat_normalize
from src.trajectory.collision import CollisionChecker


@dataclass
class GraspPose:
    """A 6-DoF grasp pose from AnyGrasp (or mock)."""
    position: np.ndarray       # (3,) grasp contact point
    orientation: np.ndarray    # (4,) quaternion wxyz — gripper approach direction
    score: float = 1.0        # Grasp quality score from AnyGrasp
    width: float = 0.05       # Predicted gripper width at grasp (meters)


@dataclass
class Waypoint:
    """A single waypoint in the trajectory."""
    position: np.ndarray       # (3,) EE position
    orientation: np.ndarray    # (4,) quaternion wxyz
    gripper: float             # 0.0=closed, 1.0=open
    phase: str                 # Phase label for debugging


@dataclass
class Trajectory:
    """A complete manipulation trajectory with metadata."""
    waypoints: list[Waypoint] = field(default_factory=list)
    grasp_pose: Optional[GraspPose] = None
    place_target: Optional[np.ndarray] = None
    task_type: str = ""
    is_valid: bool = True      # False if collision detected

    @property
    def positions(self) -> np.ndarray:
        """(T, 3) array of all waypoint positions."""
        return np.array([w.position for w in self.waypoints])

    @property
    def orientations(self) -> np.ndarray:
        """(T, 4) array of all waypoint orientations (wxyz)."""
        return np.array([w.orientation for w in self.waypoints])

    @property
    def gripper_states(self) -> np.ndarray:
        """(T,) array of gripper states."""
        return np.array([w.gripper for w in self.waypoints])

    def __len__(self) -> int:
        return len(self.waypoints)


class TrajectoryGenerator:
    """Generate phased pick-place trajectories from grasp pose and place target.

    Phases:
        0. home          → Start at home pose (gripper open)
        1. transit_pick   → Move above grasp position
        2. pre_grasp     → Descend to pre-grasp height above object
        3. approach      → Descend to grasp contact
        4. grasp         → Close gripper (single waypoint)
        5. lift          → Lift vertically
        6. transit_place → Move above place target (at transit height)
        7. pre_place     → Descend to pre-place height
        8. place_descend → Descend to place target
        9. release       → Open gripper (single waypoint)
       10. retreat       → Lift away from placed object

    For PICK-only tasks, phases 6-10 are omitted.
    """

    def __init__(
        self,
        home_position: np.ndarray = franka.HOME_POSITION,
        home_orientation: np.ndarray = franka.HOME_ORIENTATION,
        steps_per_segment: int = franka.WAYPOINTS_PER_SEGMENT,
    ):
        self.home_pos = home_position.copy()
        self.home_ori = quat_normalize(home_orientation.copy())
        self.steps = steps_per_segment

    def generate_pick(self, grasp: GraspPose) -> Trajectory:
        """Generate a pick-only trajectory (no placement)."""
        traj = Trajectory(grasp_pose=grasp, task_type="pick")
        g_pos = grasp.position
        g_ori = quat_normalize(grasp.orientation)

        # Approach direction: come from above the grasp (negative z of grasp frame)
        pre_grasp_pos = g_pos.copy()
        pre_grasp_pos[2] += franka.PRE_GRASP_HEIGHT

        transit_pos = g_pos.copy()
        transit_pos[2] = franka.TRANSIT_HEIGHT

        lift_pos = g_pos.copy()
        lift_pos[2] += franka.LIFT_HEIGHT

        # Phase 0→1: home → transit above object
        self._add_segment(traj, self.home_pos, self.home_ori,
                          transit_pos, g_ori, franka.GRIPPER_OPEN, "transit_pick")
        # Phase 1→2: transit → pre-grasp
        self._add_segment(traj, transit_pos, g_ori,
                          pre_grasp_pos, g_ori, franka.GRIPPER_OPEN, "pre_grasp",
                          include_start=False)
        # Phase 2→3: pre-grasp → grasp contact
        self._add_segment(traj, pre_grasp_pos, g_ori,
                          g_pos, g_ori, franka.GRIPPER_OPEN, "approach",
                          include_start=False)
        # Phase 4: close gripper
        traj.waypoints.append(Waypoint(g_pos.copy(), g_ori.copy(),
                                       franka.GRIPPER_CLOSE, "grasp"))
        # Phase 5: lift
        self._add_segment(traj, g_pos, g_ori,
                          lift_pos, g_ori, franka.GRIPPER_CLOSE, "lift",
                          include_start=False)
        return traj

    def generate_pick_place(
        self,
        grasp: GraspPose,
        place_target: np.ndarray,
        place_orientation: Optional[np.ndarray] = None,
    ) -> Trajectory:
        """Generate a full pick-and-place trajectory.

        Args:
            grasp: Grasp pose for picking.
            place_target: (3,) place position.
            place_orientation: (4,) quaternion wxyz for placement.
                If None, uses the grasp orientation (object keeps same orientation).
        """
        if place_orientation is None:
            place_orientation = quat_normalize(grasp.orientation)

        traj = self.generate_pick(grasp)
        traj.place_target = place_target.copy()
        traj.task_type = "pick_place"

        # Continue from the last waypoint (lift position)
        lift_pos = traj.waypoints[-1].position
        lift_ori = traj.waypoints[-1].orientation

        # Place phase waypoints
        transit_place_pos = place_target.copy()
        transit_place_pos[2] = franka.TRANSIT_HEIGHT

        pre_place_pos = place_target.copy()
        pre_place_pos[2] += franka.PRE_PLACE_HEIGHT

        retreat_pos = place_target.copy()
        retreat_pos[2] += franka.RETREAT_HEIGHT

        # Phase 6: lift → transit above place
        self._add_segment(traj, lift_pos, lift_ori,
                          transit_place_pos, place_orientation,
                          franka.GRIPPER_CLOSE, "transit_place", include_start=False)
        # Phase 7: transit → pre-place
        self._add_segment(traj, transit_place_pos, place_orientation,
                          pre_place_pos, place_orientation,
                          franka.GRIPPER_CLOSE, "pre_place", include_start=False)
        # Phase 8: pre-place → place target
        self._add_segment(traj, pre_place_pos, place_orientation,
                          place_target, place_orientation,
                          franka.GRIPPER_CLOSE, "place_descend", include_start=False)
        # Phase 9: release
        traj.waypoints.append(Waypoint(place_target.copy(), place_orientation.copy(),
                                       franka.GRIPPER_OPEN, "release"))
        # Phase 10: retreat
        self._add_segment(traj, place_target, place_orientation,
                          retreat_pos, place_orientation,
                          franka.GRIPPER_OPEN, "retreat", include_start=False)

        return traj

    def validate_trajectory(
        self,
        traj: Trajectory,
        collision_checker: CollisionChecker,
        grasped_object_cloud: Optional[np.ndarray] = None,
    ) -> bool:
        """Check trajectory for collisions. Sets traj.is_valid and returns it."""
        wp_list = [(w.position, w.orientation) for w in traj.waypoints]

        # Find where the place descent phase starts (for reduced margin)
        place_start = None
        for i, w in enumerate(traj.waypoints):
            if w.phase == "pre_place":
                place_start = i
                break

        # Only check grasped object collision after grasp phase
        grasp_idx = None
        for i, w in enumerate(traj.waypoints):
            if w.phase == "grasp":
                grasp_idx = i
                break

        # Before grasp: no grasped object to check
        if grasp_idx is not None and grasped_object_cloud is not None:
            # Check pre-grasp phases (no object in hand)
            is_valid_pre, fail_idx = collision_checker.check_trajectory(
                wp_list[:grasp_idx],
                place_phase_start=None,
            )
            if not is_valid_pre:
                traj.is_valid = False
                return False

            # Check post-grasp phases (object in hand)
            is_valid_post, fail_idx = collision_checker.check_trajectory(
                wp_list[grasp_idx:],
                grasped_object_cloud=grasped_object_cloud,
                place_phase_start=(place_start - grasp_idx) if place_start else None,
            )
            traj.is_valid = is_valid_post
            return is_valid_post
        else:
            is_valid, fail_idx = collision_checker.check_trajectory(
                wp_list,
                place_phase_start=place_start,
            )
            traj.is_valid = is_valid
            return is_valid

    def _add_segment(
        self,
        traj: Trajectory,
        pos_start: np.ndarray,
        ori_start: np.ndarray,
        pos_end: np.ndarray,
        ori_end: np.ndarray,
        gripper: float,
        phase: str,
        include_start: bool = True,
    ):
        """Interpolate a segment and append waypoints to trajectory."""
        points = interpolate_segment(
            pos_start, ori_start, pos_end, ori_end,
            self.steps, include_start=include_start,
        )
        for pos, ori in points:
            traj.waypoints.append(Waypoint(pos, ori, gripper, phase))

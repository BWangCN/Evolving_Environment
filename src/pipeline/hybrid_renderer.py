"""
Hybrid renderer: gsplat (3DGS scene + objects) + ManiSkill (robot arm).

Depth-based compositing produces the final training image with:
  - Correctly moving robot arm (from simulator)
  - Editable 3DGS environment (from gsplat)
  - Moving objects along trajectories (Gaussian SE(3) transform)
  - Proper occlusion in both directions

Usage:
    renderer = HybridRenderer(scene, arm_renderer, camera_params)
    for i, waypoint in enumerate(trajectory.waypoints):
        if i >= grasp_step:
            scene.set_object_pose("target", new_pos)
        image = renderer.render_frame(waypoint)
"""

import numpy as np
import torch
from typing import Optional

from src.pipeline.compositional_scene import CompositionalScene


def look_at_opencv(eye, target, up=np.array([0., 0., 1.])):
    """OpenCV convention W2C matrix: x-right, y-down, z-forward."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    nr = np.linalg.norm(right)
    if nr < 1e-6:
        right = np.cross(forward, np.array([0., 1., 0.]))
        nr = np.linalg.norm(right)
    right = right / nr
    down = np.cross(forward, right)
    R = np.eye(4)
    R[0, :3] = right
    R[1, :3] = down
    R[2, :3] = forward
    R[:3, 3] = R[:3, :3] @ (-eye)
    return R


class HybridRenderer:
    """Renders frames by compositing gsplat (scene) + ManiSkill (arm)."""

    def __init__(
        self,
        scene: CompositionalScene,
        arm_renderer,  # ManiSkillArmRenderer instance
        camera_eye: np.ndarray,
        camera_target: np.ndarray,
        resolution: int = 256,
        camera_fovy: float = 75.0,
    ):
        self.scene = scene
        self.arm_renderer = arm_renderer
        self.resolution = resolution

        # gsplat camera setup (OpenCV convention)
        self.viewmat = look_at_opencv(camera_eye, camera_target)
        fovy = np.radians(camera_fovy)
        fy = resolution / (2 * np.tan(fovy / 2))
        fx = fy
        self.K = np.array([[fx, 0, resolution / 2],
                           [0, fy, resolution / 2],
                           [0, 0, 1]])

    def render_gsplat(self) -> tuple[np.ndarray, np.ndarray]:
        """Render the 3DGS scene (environment + objects) via gsplat.

        Returns:
            rgb: (H, W, 3) uint8
            depth: (H, W) float32 (accumulated depth)
        """
        from gsplat import rasterization

        means, colors, opacities, scales, quats = self.scene.get_render_tensors("cuda")
        W = H = self.resolution

        vm = torch.tensor(self.viewmat, dtype=torch.float32, device="cuda").unsqueeze(0)
        Kt = torch.tensor(self.K, dtype=torch.float32, device="cuda").unsqueeze(0)

        renders, alphas, meta = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmats=vm, Ks=Kt, width=W, height=H,
            sh_degree=0, render_mode="RGB+D",
        )

        # renders shape: (1, H, W, 4) — last channel is depth
        out = renders[0].cpu().numpy()
        rgb = (out[:, :, :3].clip(0, 1) * 255).astype(np.uint8)
        depth = out[:, :, 3]  # accumulated depth
        depth[depth <= 0] = np.inf

        return rgb, depth

    def render_frame(
        self,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        gripper_state: float,
    ) -> np.ndarray:
        """Render one composited frame.

        Args:
            ee_position: (3,) EE position for robot arm.
            ee_orientation: (4,) EE quaternion (wxyz) for robot arm.
            gripper_state: 0=closed, 1=open.

        Returns:
            composited: (H, W, 3) uint8 final image.
        """
        # 1. Render robot arm via ManiSkill
        arm_result = self.arm_renderer.render_at_ee_pose(
            ee_position, ee_orientation, gripper_state
        )
        arm_rgb = arm_result["rgb"]
        arm_depth = arm_result["depth"]
        arm_mask = arm_result["robot_mask"]

        # 2. Render 3DGS scene + objects via gsplat
        scene_rgb, scene_depth = self.render_gsplat()

        # 3. Depth compositing
        composited = self.composite(
            arm_rgb, arm_depth, arm_mask,
            scene_rgb, scene_depth,
        )

        return composited

    @staticmethod
    def composite(
        arm_rgb: np.ndarray,
        arm_depth: np.ndarray,
        arm_mask: np.ndarray,
        scene_rgb: np.ndarray,
        scene_depth: np.ndarray,
    ) -> np.ndarray:
        """Per-pixel depth compositing.

        For each pixel:
          - If robot arm is present AND closer than 3DGS → show arm
          - Otherwise → show 3DGS scene

        Args:
            arm_rgb: (H, W, 3) uint8 from ManiSkill
            arm_depth: (H, W) float32 from ManiSkill
            arm_mask: (H, W) bool — True where robot is
            scene_rgb: (H, W, 3) uint8 from gsplat
            scene_depth: (H, W) float32 from gsplat

        Returns:
            composited: (H, W, 3) uint8
        """
        # Start with the 3DGS scene as base
        result = scene_rgb.copy()

        # Overlay robot arm where it's closer
        arm_closer = arm_mask & (arm_depth < scene_depth)
        result[arm_closer] = arm_rgb[arm_closer]

        return result

    def render_trajectory(
        self,
        trajectory,
        grasp_step: int,
        release_step: int,
        target_object: str,
        grasp_anchor: np.ndarray,
        grasp_offset_pos: np.ndarray,
        grasp_offset_quat: np.ndarray,
    ) -> list[np.ndarray]:
        """Render all frames for a pick-place trajectory.

        Args:
            trajectory: Trajectory object with waypoints.
            grasp_step: Index of the grasp waypoint.
            release_step: Index of the release waypoint.
            target_object: Name of the object in the scene.
            grasp_anchor: (3,) original object centroid (anchor for transform).
            grasp_offset_pos: (3,) position offset in gripper frame.
            grasp_offset_quat: (4,) orientation offset.

        Returns:
            List of (H, W, 3) uint8 composited images, one per waypoint.
        """
        from src.pipeline.gaussian_transform import apply_grasp_offset

        frames = []
        original_pos = self.scene.object_current_pos[target_object].copy()

        for i, wp in enumerate(trajectory.waypoints):
            # Update object position based on trajectory phase
            if grasp_step <= i < release_step:
                # Object is being carried — compute from gripper pose + offset
                obj_pos, rot_delta = apply_grasp_offset(
                    wp.position, wp.orientation,
                    grasp_offset_pos, grasp_offset_quat,
                )
                self.scene.set_object_pose(target_object, obj_pos, rot_delta)
            elif i >= release_step:
                # Object has been placed — stays at last carried position
                pass  # keep current pose
            else:
                # Pre-grasp — object at original position
                self.scene.set_object_pose(target_object, original_pos)

            # Render composited frame
            frame = self.render_frame(wp.position, wp.orientation, wp.gripper)
            frames.append(frame)

            if (i + 1) % 20 == 0 or i == 0:
                phase = wp.phase
                print(f"  [{i+1}/{len(trajectory)}] {phase}")

        return frames

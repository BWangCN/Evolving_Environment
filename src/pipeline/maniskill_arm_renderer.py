"""
ManiSkill robot arm renderer for hybrid compositing.

Given a sequence of EE poses (from our trajectory), uses ManiSkill's
IK controller to set robot joint positions and renders:
  - RGB image of the robot arm
  - Depth map
  - Binary robot mask (from segmentation)

The environment objects are hidden — only the robot arm and table are visible.
The table/floor rendering will be replaced by gsplat in the compositor.
"""

import numpy as np
from typing import Optional
from pathlib import Path


# Robot link actor IDs in ManiSkill PickCube env (IDs 2-14)
ROBOT_ACTOR_IDS = set(range(2, 15))


class ManiSkillArmRenderer:
    """Render robot arm at given EE poses using ManiSkill."""

    def __init__(
        self,
        resolution: int = 256,
        shader: str = "default",
        camera_eye: Optional[np.ndarray] = None,
        camera_target: Optional[np.ndarray] = None,
        camera_fovy: float = 75.0,
    ):
        """Initialize ManiSkill environment for arm rendering.

        Args:
            resolution: Render resolution (square).
            shader: ManiSkill shader ("default" supports segmentation, "rt" does not).
            camera_eye: (3,) camera position. Default: training view.
            camera_target: (3,) camera look-at target. Default: table center.
            camera_fovy: Camera field of view in degrees.
        """
        import mani_skill.envs  # noqa: register envs
        import gymnasium as gym
        import sapien

        self.resolution = resolution
        self.camera_fovy = camera_fovy

        if camera_eye is None:
            camera_eye = np.array([0.183, 0.326, 0.378])
        if camera_target is None:
            camera_target = np.array([0.0, 0.0, 0.08])
        self.camera_eye = camera_eye
        self.camera_target = camera_target

        # Create environment
        self.env = gym.make(
            "PickCube-v1",
            obs_mode="rgbd",
            num_envs=1,
            sim_backend="cpu",
            render_mode="rgb_array",
            control_mode="pd_ee_delta_pose",
            sensor_configs=dict(shader_pack=shader),
        )
        self.env.reset(seed=42)
        self.base_env = self.env.unwrapped
        self.scene = self.base_env.scene
        self.sub_scene = self.scene.sub_scenes[0]

        # Hide non-robot objects (cube, goal)
        if hasattr(self.base_env, "cube"):
            self.base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
        if hasattr(self.base_env, "goal_site"):
            self.base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
        for actor in self.base_env._hidden_objects:
            actor.set_pose(sapien.Pose(p=[0, 0, -10]))

        # Use the built-in base_camera (supports Segmentation texture)
        # and reposition it to our desired viewpoint
        from mani_skill.utils import sapien_utils
        self.cam = None
        for c in self.sub_scene.get_cameras():
            if "base_camera" in c.name:
                self.cam = c
                break
        assert self.cam is not None, "No base_camera found in ManiSkill scene"

        pose = sapien_utils.look_at(eye=camera_eye, target=camera_target)
        if hasattr(pose, "sp"):
            self.cam.entity.set_pose(pose.sp)
        elif hasattr(pose, "raw_pose"):
            rp = pose.raw_pose.squeeze().cpu().numpy()
            self.cam.entity.set_pose(sapien.Pose(p=rp[:3], q=rp[3:]))

        # Store exact camera parameters for gsplat alignment
        self._cam_w = self.cam.width    # 128
        self._cam_h = self.cam.height   # 128
        self.intrinsic_matrix = self.cam.get_intrinsic_matrix()  # (3, 3)

        # Compute W2C matrix matching COLMAP/GG convention for gsplat
        # GG/COLMAP convention: R_colmap = R_opencv.T, t = -R_colmap @ eye
        # OpenCV: z=forward (toward target), y=down, x=right
        forward = camera_target - camera_eye
        forward = forward / np.linalg.norm(forward)
        up_world = np.array([0., 0., 1.])
        right = np.cross(forward, up_world)
        nr = np.linalg.norm(right)
        if nr < 1e-6:
            right = np.cross(forward, np.array([0., 1., 0.]))
            nr = np.linalg.norm(right)
        right = right / nr
        down = np.cross(forward, right)

        R_opencv = np.stack([right, down, forward], axis=0)  # (3,3)
        R_colmap = R_opencv.T  # GG/COLMAP stores transposed
        t_colmap = -R_colmap @ camera_eye

        self.extrinsic_matrix = np.eye(4)
        self.extrinsic_matrix[:3, :3] = R_colmap
        self.extrinsic_matrix[:3, 3] = t_colmap

        # Step once to settle
        self.env.step(np.zeros(7))

    def render_at_ee_pose(
        self,
        ee_position: np.ndarray,
        ee_orientation_wxyz: np.ndarray,
        gripper_state: float,
    ) -> dict:
        """Render the robot arm at a given EE pose.

        Uses delta actions to move the robot toward the target EE pose.
        Since we're rendering static poses (not running physics), we
        directly set the robot's qpos via IK if available.

        Args:
            ee_position: (3,) desired EE position.
            ee_orientation_wxyz: (4,) desired EE quaternion (wxyz).
            gripper_state: 0.0 (closed) to 1.0 (open).

        Returns:
            dict with:
                "rgb": (H, W, 3) uint8
                "depth": (H, W) float32 (meters, 0=no hit)
                "robot_mask": (H, W) bool
        """
        # Move arm to target EE pose using multiple sub-steps
        # pd_ee_delta_pose clips actions to [-1, 1], so large moves need
        # multiple steps to converge
        grip_action = -1.0 if gripper_state < 0.5 else 1.0

        for _ in range(10):  # up to 10 sub-steps to converge
            tcp_pose = self.base_env.agent.tcp.pose
            current_pos = tcp_pose.p[0].cpu().numpy()

            delta_pos = ee_position - current_pos
            dist = np.linalg.norm(delta_pos)
            if dist < 0.005:  # within 5mm — close enough
                break

            delta_rot = np.zeros(3)
            action = np.concatenate([delta_pos, delta_rot, [grip_action]]).astype(np.float32)
            action[:3] = np.clip(action[:3], -1, 1)
            action[3:6] = np.clip(action[3:6], -1, 1)
            action[6] = grip_action

            self.env.step(action)

        # Render
        self.sub_scene.update_render()
        self.cam.take_picture()

        # RGB
        rgba = self.cam.get_picture("Color")
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

        # Segmentation → robot mask
        seg = self.cam.get_picture("Segmentation")[:, :, 1].astype(int)
        robot_mask = np.isin(seg, list(ROBOT_ACTOR_IDS))

        # Depth from Position texture
        pos = self.cam.get_picture("Position")  # (H, W, 4) in camera frame
        depth = -pos[:, :, 2]  # Z in camera frame (negative = in front)
        depth[depth <= 0] = np.inf  # no hit = infinity

        # No resize — render at native camera resolution (128×128)
        # Compositing and final upscale happen in HybridRenderer
        return {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "robot_mask": robot_mask,
        }

    def close(self):
        self.env.close()

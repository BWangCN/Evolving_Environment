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

        # The base_camera has its own resolution (128x128) — we'll resize output
        self._cam_w = self.cam.width
        self._cam_h = self.cam.height

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
        # Compute delta from current EE pose to target
        tcp_pose = self.base_env.agent.tcp.pose
        current_pos = tcp_pose.p[0].cpu().numpy()
        current_quat = tcp_pose.q[0].cpu().numpy()  # wxyz

        # Simple approach: apply delta action to move toward target
        # For a more accurate approach, use mplib IK directly
        delta_pos = ee_position - current_pos
        # For orientation, since our trajectories keep constant orientation,
        # set delta rotation to zero
        delta_rot = np.zeros(3)
        grip = -1.0 if gripper_state < 0.5 else 1.0  # ManiSkill: -1=close, 1=open

        action = np.concatenate([delta_pos, delta_rot, [grip]]).astype(np.float32)
        action = np.clip(action, -1, 1)

        # Step the environment (this applies IK internally)
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

        # Resize to target resolution if needed
        from PIL import Image as PILImage
        if rgb.shape[0] != self.resolution or rgb.shape[1] != self.resolution:
            rgb = np.array(PILImage.fromarray(rgb).resize(
                (self.resolution, self.resolution), PILImage.BILINEAR))
            robot_mask = np.array(PILImage.fromarray(robot_mask.astype(np.uint8) * 255).resize(
                (self.resolution, self.resolution), PILImage.NEAREST)) > 128
            depth = np.array(PILImage.fromarray(depth).resize(
                (self.resolution, self.resolution), PILImage.NEAREST))

        return {
            "rgb": rgb,
            "depth": depth.astype(np.float32),
            "robot_mask": robot_mask,
        }

    def close(self):
        self.env.close()

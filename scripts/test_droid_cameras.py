"""
Test DROID-aligned camera setup in ManiSkill.

Sets up two cameras matching DROID's configuration:
  1. Exterior camera: over-the-shoulder third-person view
  2. Wrist camera: ZED Mini mounted on panda_hand link, looking forward/down

Renders images from both views and saves them for visual inspection.

Usage:
    cd ~/Desktop/Manipulation
    conda activate gaussian_grouping
    python Evolving_Environment/scripts/test_droid_cameras.py
"""

import gymnasium as gym
import numpy as np
import sapien
from pathlib import Path

import mani_skill.envs  # noqa: F401 — register environments

OUTPUT_DIR = Path(__file__).parent.parent / "logs" / "camera_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_camera_image(cam, name: str, output_dir: Path):
    """Render and save RGB from a SAPIEN camera."""
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    # Save as PNG
    from PIL import Image
    img = Image.fromarray(rgb)
    path = output_dir / f"{name}.png"
    img.save(str(path))
    print(f"  Saved: {path} ({rgb.shape[1]}x{rgb.shape[0]})")
    return rgb


def main():
    # Create ManiSkill environment with Franka Panda
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        num_envs=1,
        sim_backend="cpu",
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        sensor_configs=dict(shader_pack="default"),
    )
    env.reset(seed=42)
    base_env = env.unwrapped
    scene = base_env.scene
    sub_scene = scene.sub_scenes[0]

    # --- Get robot link for wrist camera mount ---
    robot = base_env.agent.robot
    print("\nAvailable robot links:")
    for link_name in robot.links_map:
        print(f"  {link_name}")

    hand_link = robot.links_map.get("panda_hand")
    if hand_link is None:
        # Try alternative names
        for name in ["hand_tcp", "panda_link8", "right_panda_hand"]:
            hand_link = robot.links_map.get(name)
            if hand_link is not None:
                print(f"  Using link: {name}")
                break
    assert hand_link is not None, "Could not find hand link on robot"

    # Get the hand link entity for mounting
    if hasattr(hand_link, 'entity'):
        hand_entity = hand_link.entity
    else:
        hand_entity = hand_link

    # --- Create wrist camera (DROID ZED Mini style) ---
    # DROID: ZED Mini mounted on Franka flange/hand, looking forward along gripper
    # The camera looks in the -z direction in SAPIEN convention
    # We want it looking forward and slightly down (toward the workspace)

    # Rotation: look along the gripper's forward direction (toward objects)
    # SAPIEN camera convention: looks along -z axis
    # We rotate to look forward (along gripper z) and tilt down slightly

    # DROID ZED Mini mounting:
    # - Sits on top of the flange, behind the gripper
    # - ~40mm above flange surface, ~30mm behind gripper center
    # - ZED Mini FoV: 90°H x 60°V
    # - Tilted down ~30-45° so gripper fingers visible in lower frame
    #
    # panda_hand frame: Z- points toward fingertips (down in home pose)
    # SAPIEN camera convention: looks along -Z axis
    # So default camera (identity quat) already looks toward fingertips!
    # We need to pull it BACK (positive Z = away from fingers) and tilt slightly

    # panda_hand local frame (in home pose):
    #   +X = world right
    #   +Y = world back (away from table front)
    #   +Z = world up (AWAY from fingers)
    # SAPIEN camera looks along local -Z = toward fingers = toward table
    # Problem: arm body is above hand (in +Z direction), blocking camera
    # Solution: offset camera in -Y (forward, away from arm body) and +Z (up)
    #   then tilt to still see fingers

    wrist_camera_configs = {
        # Config A: Offset forward + up, looking at fingers
        "wrist_droid_a": dict(
            p=[-0.05, 0.0, 0.08],  # 5cm forward (X), 8cm up (Z) from hand
            # Tilt 20° forward (around Y axis) to see both fingers and table
            q=[np.cos(np.radians(10)), 0, np.sin(np.radians(10)), 0],
        ),
        # Config B: More overhead, less tilt
        "wrist_droid_b": dict(
            p=[0.0, 0.0, 0.12],  # 12cm above hand (clears arm body)
            q=[1, 0, 0, 0],  # Straight down at fingers
        ),
        # Config C: Offset sideways to clear arm body
        "wrist_droid_c": dict(
            p=[0.06, 0.0, 0.06],  # 6cm right + 6cm up
            # Tilt left 15° to center on fingers
            q=[np.cos(np.radians(7.5)), 0, 0, -np.sin(np.radians(7.5))],
        ),
        # Config D: DROID-realistic — camera on flange top-back, angled forward
        "wrist_droid_d": dict(
            p=[-0.03, 0.0, 0.10],  # 3cm forward, 10cm up
            # Tilt 15° forward
            q=[np.cos(np.radians(7.5)), 0, np.sin(np.radians(7.5)), 0],
        ),
    }

    # --- Create exterior cameras (DROID ZED 2 style) ---
    # DROID: Two exterior cameras on adjustable tripods, over-the-shoulder
    from mani_skill.utils import sapien_utils

    exterior_camera_configs = {
        # Over right shoulder (typical DROID angle)
        "exterior_right": dict(
            eye=[0.6, -0.4, 0.7],
            target=[0.0, 0.0, 0.1],
        ),
        # Over left shoulder
        "exterior_left": dict(
            eye=[0.6, 0.4, 0.7],
            target=[0.0, 0.0, 0.1],
        ),
        # Front-side view (like default ManiSkill)
        "exterior_front": dict(
            eye=[-0.5, 0.0, 0.6],
            target=[0.0, 0.0, 0.1],
        ),
    }

    print("\n=== Rendering exterior camera views ===")
    for name, cfg in exterior_camera_configs.items():
        cam = sub_scene.add_camera(
            name=name,
            width=256,
            height=256,
            fovy=np.radians(75),
            near=0.01,
            far=10.0,
        )
        pose = sapien_utils.look_at(eye=cfg["eye"], target=cfg["target"])
        if hasattr(pose, "sp"):
            cam.entity.set_pose(pose.sp)
        elif hasattr(pose, "raw_pose"):
            rp = pose.raw_pose.squeeze().cpu().numpy()
            cam.entity.set_pose(sapien.Pose(p=rp[:3], q=rp[3:]))
        else:
            cam.entity.set_pose(pose)
        sub_scene.update_render()
        save_camera_image(cam, name, OUTPUT_DIR)

    print("\n=== Rendering wrist camera views ===")
    for name, cfg in wrist_camera_configs.items():
        cam = sub_scene.add_camera(
            name=name,
            width=256,
            height=256,
            fovy=np.radians(90),  # ZED Mini has ~90° FoV
            near=0.01,
            far=10.0,
        )
        # Mount on hand link
        local_pose = sapien.Pose(p=cfg["p"], q=cfg["q"])
        # Get hand link's global pose from ManiSkill Link object
        rp = hand_link.pose.raw_pose.squeeze().cpu().numpy()
        hand_pose = sapien.Pose(p=rp[:3], q=rp[3:])
        global_pose = hand_pose * local_pose
        cam.entity.set_pose(global_pose)
        sub_scene.update_render()
        save_camera_image(cam, name, OUTPUT_DIR)

    # --- Also render the default ManiSkill camera for comparison ---
    print("\n=== Rendering default ManiSkill base_camera ===")
    for c in sub_scene.get_cameras():
        if "base_camera" in c.name:
            sub_scene.update_render()
            save_camera_image(c, "default_base_camera", OUTPUT_DIR)
            break

    print(f"\nAll images saved to: {OUTPUT_DIR}")
    print("Compare the views to choose the best DROID-aligned setup.")

    env.close()


if __name__ == "__main__":
    main()

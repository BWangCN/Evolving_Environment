"""
Record a smooth video of the full pipeline:
  SuGaR mesh objects → AnyGrasp grasp selection → motion-planned pick-and-place

Captures every physics step for smooth video output.

Usage:
    CUDA_HOME=/usr/local/cuda conda run -n gaussian_grouping python scripts/record_pipeline_video.py
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/objects")
OUTPUT_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs/pipeline_video")
RESOLUTION = 512

OBJECTS = {
    "005_tomato_soup_can": {"place": [0.00, 0.00, 0.02], "density": 800},
    "025_mug":             {"place": [-0.10, 0.08, 0.02], "density": 500},
    "024_bowl":            {"place": [0.10, -0.08, 0.02], "density": 400},
    "011_banana":          {"place": [-0.08, -0.10, 0.02], "density": 300},
    "013_apple":           {"place": [0.10, 0.08, 0.02], "density": 600},
}

TARGET_OBJ = "005_tomato_soup_can"
PLACE_TARGET = np.array([0.15, -0.12, 0.02], dtype=np.float32)
FPS = 20


def render_frame(env):
    f = env.render()
    if isinstance(f, torch.Tensor):
        f = f.cpu().numpy()
    if f.ndim == 4:
        f = f[0]
    if f.dtype != np.uint8:
        f = (np.clip(f, 0, 1) * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
    return f


def add_hud(frame, text, phase_color=(255, 255, 255)):
    """Add text overlay to frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    draw.rectangle([(0, 0), (w, 28)], fill=(0, 0, 0, 180))
    draw.text((8, 6), text, fill=phase_color)
    return np.array(img)


def move_to_pose_smooth(env, base_env, planner, ee_pos, ee_ori_wxyz, gripper_open,
                         phase_name="move", frames=None, n_settle=5):
    """Move robot to target pose, recording every step."""
    target_pose = np.concatenate([ee_pos, ee_ori_wxyz])
    qpos_current = base_env.agent.robot.get_qpos().cpu().numpy()[0]

    # Plan path
    result = planner.planner.plan_screw(target_pose, qpos_current, time_step=0.1)
    if result["status"] != "Success":
        result = planner.planner.plan_qpos_to_pose(
            target_pose, qpos_current, time_step=0.1, wrt_world=True,
        )

    grip_val = 1.0 if gripper_open else -1.0

    if result["status"] == "Success":
        # Execute full motion path, recording each step
        waypoints = result["position"]
        for i, qpos in enumerate(waypoints):
            action = np.zeros(8, dtype=np.float32)
            action[:7] = qpos[:7]
            action[7] = grip_val
            env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))

            if frames is not None and i % 2 == 0:  # Record every other step
                f = render_frame(env)
                grip_str = "open" if gripper_open else "closed"
                f = add_hud(f, f"{phase_name} | gripper: {grip_str}", get_phase_color(phase_name))
                frames.append(f)
    else:
        # Fallback: jump to position
        final_qpos = qpos_current[:7]
        action = np.zeros(8, dtype=np.float32)
        action[:7] = final_qpos
        action[7] = grip_val
        for _ in range(n_settle):
            env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))

    # Settle
    if result["status"] == "Success":
        action = np.zeros(8, dtype=np.float32)
        action[:7] = result["position"][-1][:7]
        action[7] = grip_val
    for _ in range(n_settle):
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
        if frames is not None:
            f = render_frame(env)
            grip_str = "open" if gripper_open else "closed"
            f = add_hud(f, f"{phase_name} (settle) | gripper: {grip_str}", get_phase_color(phase_name))
            frames.append(f)

    return result["status"] == "Success"


def get_phase_color(phase):
    colors = {
        "transit_to_pick": (51, 153, 255),
        "pre_grasp": (0, 204, 102),
        "approach": (255, 204, 0),
        "grasp (close)": (255, 50, 50),
        "lift": (204, 0, 204),
        "transit_to_place": (51, 153, 255),
        "pre_place": (0, 204, 102),
        "place (descend)": (255, 204, 0),
        "release (open)": (255, 50, 50),
        "retreat": (128, 128, 128),
    }
    for key, color in colors.items():
        if key in phase:
            return color
    return (200, 200, 200)


def main():
    import mani_skill.envs
    import gymnasium as gym
    import sapien
    import sapien.physx as physx

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Recording Pipeline Video ===\n")

    # Step 1: Create env with SuGaR meshes
    print("Step 1: Creating environment with reconstructed meshes...")
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd", num_envs=1, sim_backend="cpu",
        render_mode="rgb_array", control_mode="pd_joint_pos",
        sensor_configs=dict(shader_pack="default"),
    )
    env.reset(seed=42)
    base_env = env.unwrapped

    if hasattr(base_env, "cube"):
        base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, "goal_site"):
        base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
    for actor in base_env._hidden_objects:
        actor.set_pose(sapien.Pose(p=[0, 0, -10]))

    custom_actors = {}
    for obj_id, spec in OBJECTS.items():
        # Prefer textured visual mesh, fallback to plain mesh
        visual_mesh = DATA_DIR / obj_id / "mesh" / f"{obj_id}_visual.obj"
        collision_mesh = DATA_DIR / obj_id / "mesh" / f"{obj_id}_collision.obj"
        plain_mesh = DATA_DIR / obj_id / "mesh" / f"{obj_id}.obj"

        if not visual_mesh.exists() and not plain_mesh.exists():
            print(f"  {obj_id}: no mesh found, skipping")
            continue

        builder = base_env.scene.create_actor_builder()

        # Visual: use solid colors for clear visibility
        mat = sapien.render.RenderMaterial()
        color_map = {
            "005_tomato_soup_can": [0.85, 0.15, 0.1, 1.0],
            "025_mug":            [0.3, 0.2, 0.6, 1.0],
            "024_bowl":           [0.9, 0.85, 0.7, 1.0],
            "011_banana":         [0.95, 0.9, 0.2, 1.0],
            "013_apple":          [0.8, 0.1, 0.1, 1.0],
        }
        mat.base_color = color_map.get(obj_id, [0.7, 0.7, 0.7, 1.0])
        mesh_file = visual_mesh if visual_mesh.exists() else plain_mesh
        builder.add_visual_from_file(filename=str(mesh_file), material=mat)

        # Collision: use collision mesh if available, else plain
        col_mesh = collision_mesh if collision_mesh.exists() else plain_mesh
        phys_mat = physx.PhysxMaterial(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        builder.add_convex_collision_from_file(
            filename=str(col_mesh), material=phys_mat, density=spec["density"],
        )
        builder.initial_pose = sapien.Pose(p=spec["place"], q=[1, 0, 0, 0])
        builder.set_scene_idxs([0])
        actor = builder.build(name=obj_id)
        custom_actors[obj_id] = actor
        print(f"  {obj_id}: loaded")

    # Settle
    for _ in range(200):
        base_env.scene.step()

    # Motion planner
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False,
        base_pose=base_env.agent.robot.pose,
        print_env_info=False,
    )

    # Start recording
    frames = []

    # Record initial scene (hold for 1 second)
    print("\nRecording initial scene...")
    for _ in range(FPS):
        f = render_frame(env)
        f = add_hud(f, "Scene: reconstructed 3DGS meshes in ManiSkill", (200, 200, 200))
        frames.append(f)

    # Step 2: Grasp detection
    print("\nStep 2: AnyGrasp detection...")
    target_pos = custom_actors[TARGET_OBJ].pose.p[0].cpu().numpy()

    from run_sim_pipeline import mesh_to_pointcloud
    mesh_path = str(DATA_DIR / TARGET_OBJ / "mesh" / f"{TARGET_OBJ}.obj")
    pts, colors = mesh_to_pointcloud(mesh_path)
    pts_world = pts + target_pos.astype(np.float32)

    from src.pipeline.gs_to_grasp import detect_grasps
    grasps = detect_grasps(pts_world, colors, top_k=10, collision_detection=False)
    print(f"  Detected {len(grasps)} grasps")

    # Hold grasp detection frame
    for _ in range(FPS):
        f = render_frame(env)
        f = add_hud(f, f"AnyGrasp: {len(grasps)} grasps on {TARGET_OBJ}", (0, 255, 100))
        frames.append(f)

    # Step 3: Execute pick-and-place
    print("\nStep 3: Executing pick-and-place trajectory...")
    grasp = grasps[0]
    ee_ori = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    grasp_pos = grasp.position.copy()
    place_pos = PLACE_TARGET.copy()

    pre_grasp = grasp_pos.copy(); pre_grasp[2] += 0.10
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.15
    transit_h = 0.30
    pre_place = place_pos.copy(); pre_place[2] += 0.10
    retreat_pos = place_pos.copy(); retreat_pos[2] += 0.15
    transit_pick = grasp_pos.copy(); transit_pick[2] = transit_h
    transit_place = place_pos.copy(); transit_place[2] = transit_h

    phases = [
        ("transit_to_pick",  transit_pick, True),
        ("pre_grasp",        pre_grasp, True),
        ("approach",         grasp_pos, True),
        ("grasp (close)",    grasp_pos, False),
        ("lift",             lift_pos, False),
        ("transit_to_place", transit_place, False),
        ("pre_place",        pre_place, False),
        ("place (descend)",  place_pos, False),
        ("release (open)",   place_pos, True),
        ("retreat",          retreat_pos, True),
    ]

    for phase_name, target, grip_open in phases:
        tcp = base_env.agent.tcp.pose.p[0].cpu().numpy()
        print(f"  {phase_name:20s}: target={target.round(3)}, grip={'OPEN' if grip_open else 'CLOSED'}")
        move_to_pose_smooth(env, base_env, planner, target, ee_ori, grip_open,
                            phase_name=phase_name, frames=frames)

    # Hold final frame
    for _ in range(FPS):
        f = render_frame(env)
        f = add_hud(f, "Task complete!", (0, 255, 100))
        frames.append(f)

    # Save video
    print(f"\nSaving {len(frames)} frames as video...")
    frame_dir = OUTPUT_DIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(frame_dir / f"{i:05d}.png")

    import subprocess
    video_path = OUTPUT_DIR / "full_pipeline.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frame_dir / "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(video_path),
    ], capture_output=True)

    print(f"\n{'='*50}")
    print(f"Video saved: {video_path}")
    print(f"Frames: {len(frames)}, Duration: {len(frames)/FPS:.1f}s")
    print(f"{'='*50}")

    env.close()


if __name__ == "__main__":
    main()

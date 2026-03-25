"""
Full simulator pipeline: SuGaR meshes → ManiSkill → AnyGrasp → trajectory → execute → visualize.

Uses IK-based joint control for precise arm positioning.

Usage:
    CUDA_HOME=/usr/local/cuda conda run -n gaussian_grouping python scripts/run_sim_pipeline.py
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Configuration ─────────────────────────────────────────────────────

DATA_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/objects")
OUTPUT_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs/sim_pipeline")
RESOLUTION = 512

# Objects: id → table placement and physics
OBJECTS = {
    "005_tomato_soup_can": {"place": [0.00,  0.00, 0.02], "density": 800},
    "025_mug":             {"place": [-0.10, 0.08, 0.02], "density": 500},
    "024_bowl":            {"place": [0.10, -0.08, 0.02], "density": 400},
    "011_banana":          {"place": [-0.08, -0.10, 0.02], "density": 300},
    "013_apple":           {"place": [0.10,  0.08, 0.02], "density": 600},
}

TARGET_OBJ = "005_tomato_soup_can"
PLACE_TARGET = np.array([0.15, -0.12, 0.02], dtype=np.float32)


# ── Step 1: Create ManiSkill env with SuGaR meshes ───────────────────

def create_env():
    """Create ManiSkill env with custom mesh objects, using pd_joint_pos for IK control."""
    import mani_skill.envs
    import gymnasium as gym
    import sapien
    import sapien.physx as physx

    print("Step 1: Creating ManiSkill environment...")

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

    # Hide default objects
    if hasattr(base_env, "cube"):
        base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, "goal_site"):
        base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
    for actor in base_env._hidden_objects:
        actor.set_pose(sapien.Pose(p=[0, 0, -10]))

    # Load mesh objects
    custom_actors = {}
    for obj_id, spec in OBJECTS.items():
        mesh_path = DATA_DIR / obj_id / "mesh" / f"{obj_id}.obj"
        if not mesh_path.exists():
            print(f"  {obj_id}: mesh not found, skipping")
            continue

        builder = base_env.scene.create_actor_builder()

        # Visual mesh with material color hint
        mat = sapien.render.RenderMaterial()
        mat.base_color = [0.8, 0.3, 0.2, 1.0]  # reddish default
        if "banana" in obj_id:
            mat.base_color = [0.9, 0.85, 0.2, 1.0]
        elif "apple" in obj_id:
            mat.base_color = [0.8, 0.1, 0.1, 1.0]
        elif "bowl" in obj_id:
            mat.base_color = [0.6, 0.55, 0.5, 1.0]
        elif "mug" in obj_id:
            mat.base_color = [0.4, 0.2, 0.15, 1.0]
        elif "can" in obj_id or "soup" in obj_id:
            mat.base_color = [0.8, 0.2, 0.15, 1.0]

        builder.add_visual_from_file(filename=str(mesh_path), material=mat)

        # Collision
        phys_mat = physx.PhysxMaterial(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        builder.add_convex_collision_from_file(
            filename=str(mesh_path), material=phys_mat, density=spec["density"],
        )

        builder.initial_pose = sapien.Pose(p=spec["place"], q=[1, 0, 0, 0])
        builder.set_scene_idxs([0])
        actor = builder.build(name=obj_id)
        custom_actors[obj_id] = actor
        print(f"  {obj_id}: loaded at {spec['place']}")

    # Settle physics
    print("  Settling physics...")
    for _ in range(200):
        base_env.scene.step()

    for obj_id, actor in custom_actors.items():
        pos = actor.pose.p[0].cpu().numpy()
        print(f"  {obj_id} settled: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # Setup motion planner for IK
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    planner = PandaArmMotionPlanningSolver(
        env, debug=False, vis=False,
        base_pose=base_env.agent.robot.pose,
        print_env_info=False,
    )

    return env, base_env, custom_actors, planner


# ── Step 2: Point cloud from mesh ────────────────────────────────────

def mesh_to_pointcloud(mesh_path, n_points=10000):
    """Sample colored points from mesh surface."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.full((len(pts), 3), 0.5, dtype=np.float32)
    return pts, colors


# ── Step 3: Detect grasps ────────────────────────────────────────────

def detect_grasps_on_mesh(obj_id, obj_world_pos):
    """Detect grasps, transformed to world frame."""
    mesh_path = str(DATA_DIR / obj_id / "mesh" / f"{obj_id}.obj")
    pts, colors = mesh_to_pointcloud(mesh_path)

    # Transform to world frame
    pts_world = pts + obj_world_pos.astype(np.float32)
    print(f"  Points: {len(pts_world)}, bounds: {pts_world.min(0).round(3)} → {pts_world.max(0).round(3)}")

    from src.pipeline.gs_to_grasp import detect_grasps
    grasps = detect_grasps(pts_world, colors, top_k=10, collision_detection=False)
    return grasps


# ── Step 4: Execute trajectory via IK ────────────────────────────────

def move_to_pose(env, base_env, planner, ee_pos, ee_ori_wxyz, gripper_open, n_settle=10):
    """Move robot to target EE pose using motion planning + IK.

    Args:
        ee_pos: (3,) target EE position
        ee_ori_wxyz: (4,) target EE quaternion (wxyz)
        gripper_open: bool, True=open, False=closed
        n_settle: number of physics steps to settle

    Returns:
        success: bool
    """
    target_pose = np.concatenate([ee_pos, ee_ori_wxyz])
    qpos_current = base_env.agent.robot.get_qpos().cpu().numpy()[0]

    # Try IK via motion planner
    result = planner.planner.plan_screw(target_pose, qpos_current, time_step=0.1)
    if result["status"] != "Success":
        result = planner.planner.plan_qpos_to_pose(
            target_pose, qpos_current, time_step=0.1, wrt_world=True,
        )

    if result["status"] == "Success":
        final_qpos = result["position"][-1]
    else:
        # Fallback: just hold
        final_qpos = qpos_current[:7]

    grip_val = 1.0 if gripper_open else -1.0
    action = np.zeros(8, dtype=np.float32)
    action[:7] = final_qpos[:7]
    action[7] = grip_val

    for _ in range(n_settle):
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))

    return result["status"] == "Success"


def execute_pick_place(env, base_env, planner, grasp_pos, place_pos, ee_ori_wxyz):
    """Execute a full pick-place using IK. Returns frames at key steps."""
    frames = []
    labels = []

    def capture(label):
        f = env.render()
        if isinstance(f, torch.Tensor):
            f = f.cpu().numpy()
        if f.ndim == 4:
            f = f[0]
        frames.append(f.copy())
        labels.append(label)

    # Pre-grasp heights
    pre_grasp = grasp_pos.copy(); pre_grasp[2] += 0.10
    lift_pos = grasp_pos.copy(); lift_pos[2] += 0.15
    transit_h = 0.30
    pre_place = place_pos.copy(); pre_place[2] += 0.10
    retreat = place_pos.copy(); retreat[2] += 0.15

    transit_pick = grasp_pos.copy(); transit_pick[2] = transit_h
    transit_place = place_pos.copy(); transit_place[2] = transit_h

    steps = [
        ("start",          None, None, True),
        ("transit_pick",   transit_pick, ee_ori_wxyz, True),
        ("pre_grasp",      pre_grasp, ee_ori_wxyz, True),
        ("approach",       grasp_pos, ee_ori_wxyz, True),
        ("grasp",          grasp_pos, ee_ori_wxyz, False),   # close gripper
        ("lift",           lift_pos, ee_ori_wxyz, False),
        ("transit_place",  transit_place, ee_ori_wxyz, False),
        ("pre_place",      pre_place, ee_ori_wxyz, False),
        ("place_descend",  place_pos, ee_ori_wxyz, False),
        ("release",        place_pos, ee_ori_wxyz, True),    # open gripper
        ("retreat",        retreat, ee_ori_wxyz, True),
    ]

    for step_name, pos, ori, grip_open in steps:
        if pos is not None:
            ok = move_to_pose(env, base_env, planner, pos, ori, grip_open, n_settle=15)
            status = "OK" if ok else "IK_FAIL"
        else:
            status = "START"

        capture(step_name)
        tcp = base_env.agent.tcp.pose.p[0].cpu().numpy()
        print(f"  {step_name:18s}: TCP=[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}] grip={'OPEN' if grip_open else 'CLOSED':6s} [{status}]")

    return frames, labels


# ── Step 5: Save visualization ───────────────────────────────────────

def save_frames_and_video(frames, labels, prefix="traj"):
    """Save frames with HUD and create video."""
    out_dir = OUTPUT_DIR / f"{prefix}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_colors = {
        "start": (200, 200, 200), "transit_pick": (51, 153, 255),
        "pre_grasp": (0, 204, 102), "approach": (255, 153, 0),
        "grasp": (255, 0, 0), "lift": (204, 0, 204),
        "transit_place": (51, 153, 255), "pre_place": (0, 204, 102),
        "place_descend": (255, 153, 0), "release": (255, 0, 0),
        "retreat": (128, 128, 128),
    }

    for i, (frame, label) in enumerate(zip(frames, labels)):
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        h, w = frame.shape[:2]
        draw.rectangle([(0, 0), (w, 24)], fill=(0, 0, 0))
        color = phase_colors.get(label, (200, 200, 200))
        draw.text((4, 4), f"Step {i}/{len(frames)-1} | {label}", fill=color)
        img.save(out_dir / f"{i:04d}.png")

    # Video
    import subprocess
    video_path = OUTPUT_DIR / f"{prefix}.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "2",
        "-i", str(out_dir / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(video_path),
    ], capture_output=True)
    return video_path


# ── Main ──────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Create env
    env, base_env, custom_actors, planner = create_env()

    # Save initial scene
    init_frame = env.render()
    if isinstance(init_frame, torch.Tensor):
        init_frame = init_frame.cpu().numpy()
    if init_frame.ndim == 4:
        init_frame = init_frame[0]
    if init_frame.dtype != np.uint8:
        init_frame = (np.clip(init_frame, 0, 1) * 255).astype(np.uint8) if init_frame.max() <= 1.0 else init_frame.astype(np.uint8)
    Image.fromarray(init_frame).save(OUTPUT_DIR / "initial_scene.png")
    print(f"\n  Scene rendered: {OUTPUT_DIR / 'initial_scene.png'}")

    # Step 2: Get target object world position
    target_pos = custom_actors[TARGET_OBJ].pose.p[0].cpu().numpy()
    print(f"\nStep 2: Target '{TARGET_OBJ}' at {target_pos.round(4)}")

    # Step 3: Detect grasps
    print("\nStep 3: AnyGrasp detection...")
    grasps = detect_grasps_on_mesh(TARGET_OBJ, target_pos)
    if not grasps:
        print("ERROR: No grasps detected!")
        env.close()
        return

    for i, g in enumerate(grasps[:5]):
        print(f"  Grasp {i}: pos={g.position.round(3)}, score={g.score:.3f}")

    # EE orientation: pointing down (wxyz)
    # ManiSkill Panda EE home quat is roughly [0, 1, 0, 0] wxyz
    ee_ori = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # Step 4: Execute pick-place for top 3 grasps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_traj = min(3, len(grasps))
    place_targets = [
        PLACE_TARGET,
        np.array([-0.12, 0.12, 0.02], dtype=np.float32),
        np.array([0.05, -0.15, 0.02], dtype=np.float32),
    ]

    fig, axes = plt.subplots(n_traj, 6, figsize=(24, 4 * n_traj))
    if n_traj == 1:
        axes = axes[np.newaxis, :]

    import sapien
    for t_idx in range(n_traj):
        grasp = grasps[t_idx]
        place = place_targets[t_idx % len(place_targets)]

        print(f"\n{'='*50}")
        print(f"Trajectory {t_idx+1}/{n_traj}: grasp={grasp.position.round(3)}, place={place.round(3)}")
        print(f"{'='*50}")

        # Reset env
        env.reset(seed=42)
        if hasattr(base_env, "cube"):
            base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
        if hasattr(base_env, "goal_site"):
            base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
        for actor in base_env._hidden_objects:
            actor.set_pose(sapien.Pose(p=[0, 0, -10]))

        # Re-place objects
        for obj_id, spec in OBJECTS.items():
            if obj_id in custom_actors:
                custom_actors[obj_id].set_pose(sapien.Pose(p=spec["place"], q=[1, 0, 0, 0]))
        for _ in range(100):
            base_env.scene.step()

        # Use settled position for grasp
        settled_pos = custom_actors[TARGET_OBJ].pose.p[0].cpu().numpy()
        # Offset grasp from mesh-local to world
        grasp_world = grasp.position.copy()

        frames, labels = execute_pick_place(
            env, base_env, planner,
            grasp_pos=grasp_world,
            place_pos=place.copy(),
            ee_ori_wxyz=ee_ori,
        )

        # Save video for this trajectory
        video_path = save_frames_and_video(frames, labels, prefix=f"traj_{t_idx}")
        print(f"  Video: {video_path}")

        # Pick 6 key frames for overview (evenly spaced from the 11 steps)
        key_idx = [0, 2, 4, 5, 8, 10] if len(frames) >= 11 else list(range(min(6, len(frames))))
        for col, fi in enumerate(key_idx[:6]):
            frame = frames[fi]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            axes[t_idx, col].imshow(frame)
            axes[t_idx, col].set_title(labels[fi], fontsize=10)
            axes[t_idx, col].axis("off")
        axes[t_idx, 0].set_ylabel(f"Traj {t_idx+1}\nscore={grasp.score:.2f}", fontsize=11)

    plt.suptitle(f"ManiSkill Pick-Place: {TARGET_OBJ} (SuGaR mesh, IK control)", fontsize=14)
    plt.tight_layout()
    overview_path = OUTPUT_DIR / "trajectory_overview.png"
    plt.savefig(str(overview_path), dpi=120)
    plt.close()

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Objects: {len(custom_actors)}")
    print(f"  Grasps: {len(grasps)}")
    print(f"  Trajectories: {n_traj}")
    print(f"  Overview: {overview_path}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()

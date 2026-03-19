"""
Full hybrid rendering pipeline: 3DGS scene + ManiSkill arm → training data.

End-to-end: segment object → AnyGrasp → trajectory → hybrid render → video + (I,a,l) triplets.

Usage:
    CUDA_HOME=/usr/local/cuda conda run -n gaussian_grouping python scripts/run_hybrid_pipeline.py
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data")
OUTPUT_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs/hybrid_render_v2")

# Camera matching our verified good viewpoint
CAMERA_EYE = np.array([0.183, 0.326, 0.378])
CAMERA_TARGET = np.array([0.0, 0.0, 0.08])
RESOLUTION = 256


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    # ── Step 1: Build compositional scene ────────────────────────────
    print("Step 1: Building compositional scene...")
    from src.pipeline.compositional_scene import CompositionalScene, load_gaussian_cluster_from_ply, GaussianCluster
    from src.pipeline.gaussian_transform import compute_grasp_offset

    # Load empty table environment
    env_cluster = load_gaussian_cluster_from_ply(
        str(DATA_DIR / "empty_table/output/point_cloud/iteration_7000/point_cloud.ply"),
        max_points=999999999,
    )
    scene = CompositionalScene(env_cluster)

    # Object specs: bbox sizes and desired placements
    obj_specs = {
        "005_tomato_soup_can": {"bbox": (0.069, 0.069, 0.103), "place": [0.0, 0.0, 0.05]},
        "025_mug":             {"bbox": (0.107, 0.095, 0.081), "place": [-0.10, 0.08, 0.04]},
        "024_bowl":            {"bbox": (0.162, 0.162, 0.055), "place": [0.10, -0.08, 0.028]},
        "011_banana":          {"bbox": (0.070, 0.199, 0.038), "place": [-0.10, -0.08, 0.019]},
        "013_apple":           {"bbox": (0.076, 0.076, 0.072), "place": [0.12, 0.06, 0.035]},
    }

    # Pick target object (soup can for this demo)
    TARGET_OBJ = "005_tomato_soup_can"

    for obj_id, spec in obj_specs.items():
        ply_path = DATA_DIR / f"objects/{obj_id}/output/point_cloud/iteration_7000/point_cloud.ply"
        if not ply_path.exists():
            print(f"  {obj_id}: PLY not found, skipping")
            continue
        full = load_gaussian_cluster_from_ply(str(ply_path), max_points=999999999)
        bx, by, bz = spec["bbox"]
        margin = 1.3
        mask = (np.abs(full.positions[:, 0]) < bx / 2 * margin) & \
               (np.abs(full.positions[:, 1]) < by / 2 * margin) & \
               (full.positions[:, 2] > 0.002) & \
               (full.positions[:, 2] < bz * margin)
        obj_cluster = GaussianCluster(
            positions=full.positions[mask], quaternions=full.quaternions[mask],
            scales=full.scales[mask], opacities=full.opacities[mask], sh_dc=full.sh_dc[mask],
        )
        scene.add_object(obj_id, obj_cluster, position=np.array(spec["place"], dtype=np.float32))

    print(scene.summary())

    # ── Step 2: Detect grasps on the target object ───────────────────
    print("\nStep 2: Detecting grasps...")
    from src.pipeline.gs_to_grasp import detect_grasps

    target_cluster = scene.objects[TARGET_OBJ]
    # Transform to current world position for grasp detection
    target_world = scene.get_transformed_object(TARGET_OBJ)
    obj_pts = target_world.positions
    SH_C0 = 0.28209479177387814
    obj_colors = np.clip(SH_C0 * target_world.sh_dc + 0.5, 0, 1).astype(np.float32)

    grasps = detect_grasps(obj_pts, obj_colors, top_k=5, collision_detection=False)
    if not grasps:
        print("ERROR: No grasps detected!")
        return
    best_grasp = grasps[0]
    print(f"Best grasp: pos={best_grasp.position.round(3)}, score={best_grasp.score:.3f}")

    # ── Step 3: Generate trajectory ──────────────────────────────────
    print("\nStep 3: Generating trajectory...")
    from src.trajectory import TrajectoryGenerator, trajectory_to_actions

    place_target = np.array([0.15, -0.10, 0.025], dtype=np.float32)
    gen = TrajectoryGenerator()
    traj = gen.generate_pick_place(best_grasp, place_target)
    gen.smooth_corners(traj, radius=3)
    actions = trajectory_to_actions(traj)
    print(f"Trajectory: {len(traj)} waypoints, {len(actions)} actions")

    # Find grasp and release step indices
    grasp_step = None
    release_step = None
    for i, wp in enumerate(traj.waypoints):
        if wp.phase == "grasp" and grasp_step is None:
            grasp_step = i
        if wp.phase == "release" and release_step is None:
            release_step = i
    print(f"Grasp at step {grasp_step}, release at step {release_step}")

    # Compute grasp offset using world-space object centroid
    obj_centroid = target_world.centroid
    offset_pos, offset_quat = compute_grasp_offset(
        best_grasp.position, best_grasp.orientation, obj_centroid
    )
    print(f"Object centroid: {obj_centroid.round(3)}, offset: {offset_pos.round(4)}")

    # ── Step 4: Initialize renderers ─────────────────────────────────
    print("\nStep 4: Initializing renderers...")
    from src.pipeline.maniskill_arm_renderer import ManiSkillArmRenderer
    from src.pipeline.hybrid_renderer import HybridRenderer

    arm_renderer = ManiSkillArmRenderer(
        resolution=RESOLUTION,
        shader="default",
        camera_eye=CAMERA_EYE,
        camera_target=CAMERA_TARGET,
        camera_fovy=75.0,
    )

    hybrid = HybridRenderer(
        scene=scene,
        arm_renderer=arm_renderer,
        camera_eye=CAMERA_EYE,
        camera_target=CAMERA_TARGET,
        output_resolution=RESOLUTION,
    )

    # ── Step 5: Render all frames ────────────────────────────────────
    print(f"\nStep 5: Rendering {len(traj)} frames...")
    frames = hybrid.render_trajectory(
        trajectory=traj,
        grasp_step=grasp_step,
        release_step=release_step,
        target_object=TARGET_OBJ,
        grasp_anchor=obj_centroid,
        grasp_offset_pos=offset_pos,
        grasp_offset_quat=offset_quat,
    )

    # ── Step 6: Save frames + video ──────────────────────────────────
    print(f"\nStep 6: Saving {len(frames)} frames...")
    phase_colors = {
        "transit_pick": (51, 153, 255), "pre_grasp": (0, 204, 102),
        "approach": (255, 153, 0), "grasp": (255, 0, 0),
        "lift": (204, 0, 204), "transit_place": (51, 153, 255),
        "pre_place": (0, 204, 102), "place_descend": (255, 153, 0),
        "release": (255, 0, 0), "retreat": (128, 128, 128),
    }

    for i, (frame, wp) in enumerate(zip(frames, traj.waypoints)):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # HUD
        draw.rectangle([(0, 0), (RESOLUTION, 18)], fill=(0, 0, 0))
        grip = "OPEN" if wp.gripper > 0.5 else "CLOSED"
        color = phase_colors.get(wp.phase, (200, 200, 200))
        draw.text((4, 3), f"Step {i}/{len(traj)-1} | {wp.phase} | {grip}", fill=color)

        img.save(frames_dir / f"{i:04d}.png")

    # Create video
    import subprocess
    video_path = OUTPUT_DIR / "hybrid_pick_place.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "8",
        "-i", str(frames_dir / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(video_path),
    ], capture_output=True)

    # Save training data summary
    print(f"\n{'='*50}")
    print(f"HYBRID RENDERING COMPLETE")
    print(f"{'='*50}")
    print(f"  Frames: {len(frames)} @ {RESOLUTION}x{RESOLUTION}")
    print(f"  Actions: {actions.shape}")
    print(f"  Video: {video_path}")
    if video_path.exists():
        print(f"  Video size: {video_path.stat().st_size // 1024} KB")
    print(f"  Frames dir: {frames_dir}")
    print(f"\n  Each frame is a training sample:")
    print(f"    Image: {RESOLUTION}x{RESOLUTION} composited (3DGS scene + ManiSkill arm)")
    print(f"    Action: 7-dim delta EE (from trajectory)")
    print(f"    Language: 'pick up the object and place it at the target'")

    arm_renderer.close()


if __name__ == "__main__":
    main()

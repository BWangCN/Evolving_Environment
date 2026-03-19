"""
Calibration test: verify that 3DGS coordinates match ManiSkill coordinates.

Places objects at KNOWN positions in ManiSkill, captures multi-view images,
reconstructs with 3DGS, then checks if the reconstructed point cloud has
the objects at the same positions.

This tells us the exact transform between the two coordinate systems.
"""

import os
import sys
import numpy as np
import sapien
from pathlib import Path
from PIL import Image

os.environ.setdefault("DISPLAY", "")
sys.path.insert(0, str(Path(__file__).parent.parent))

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import get_actor_builder


DATA_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/calibration")
N_VIEWS = 60
RESOLUTION = 512

# Known object positions in ManiSkill coordinates
KNOWN_POSITIONS = {
    "005_tomato_soup_can": np.array([0.0, 0.0, 0.0]),      # table center
    "013_apple":           np.array([0.10, 0.08, 0.0]),     # front-right
    "025_mug":             np.array([-0.10, -0.08, 0.0]),   # back-left
}


def capture_calibration_scene():
    """Capture multi-view images with objects at known positions."""
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PickCube-v1", obs_mode="rgbd", num_envs=1, sim_backend="cpu",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="rt"),
        human_render_camera_configs=dict(shader_pack="rt"),
    )
    env.reset(seed=42)
    base_env = env.unwrapped
    scene = base_env.scene

    # Hide robot + default objects
    base_env.agent.robot.set_root_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, "cube"):
        base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, "goal_site"):
        base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
    for a in base_env._hidden_objects:
        a.set_pose(sapien.Pose(p=[0, 0, -10]))

    # Place objects at known positions
    actual_positions = {}
    for obj_id, target_xy in KNOWN_POSITIONS.items():
        builder = get_actor_builder(scene, id=f"ycb:{obj_id}")
        obj = builder.build(name=f"cal_{obj_id}")
        cm = obj.get_first_collision_mesh()
        z_offset = -cm.bounding_box.bounds[0, 2] if cm else 0.02
        pos = np.array([target_xy[0], target_xy[1], z_offset])
        obj.set_pose(sapien.Pose(p=pos, q=[1, 0, 0, 0]))
        actual_positions[obj_id] = pos.copy()
        print(f"  Placed {obj_id} at ManiSkill pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # Settle
    for _ in range(5):
        env.step(env.action_space.sample() * 0)

    # Create capture camera
    sub_scene = scene.sub_scenes[0]
    cam = sub_scene.add_camera(
        name="cal_cam", width=RESOLUTION, height=RESOLUTION,
        fovy=np.radians(75), near=0.01, far=10.0,
    )

    # Also get the base_camera's extrinsic for reference
    base_cam = [c for c in sub_scene.get_cameras() if "base" in c.name][0]

    # Camera orbit
    rng = np.random.default_rng(42)
    center = np.array([0.0, 0.0, 0.08])
    cam_eyes = []

    print(f"\n  Rendering {N_VIEWS} calibration views...")
    for i in range(N_VIEWS):
        azimuth = 2 * np.pi * i / N_VIEWS + rng.normal(0, 0.05)
        elevation = rng.uniform(np.radians(25), np.radians(60))
        radius = rng.uniform(0.45, 0.55)
        x = center[0] + radius * np.cos(elevation) * np.cos(azimuth)
        y = center[1] + radius * np.cos(elevation) * np.sin(azimuth)
        z = center[2] + radius * np.sin(elevation)
        eye = np.array([x, y, z])
        cam_eyes.append(eye)

        pose = sapien_utils.look_at(eye=eye, target=center)
        if hasattr(pose, "sp"):
            cam.entity.set_pose(pose.sp)
        elif hasattr(pose, "raw_pose"):
            rp = pose.raw_pose.squeeze().cpu().numpy()
            cam.entity.set_pose(sapien.Pose(p=rp[:3], q=rp[3:]))

        sub_scene.update_render()
        cam.take_picture()
        rgba = cam.get_picture("Color")
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(images_dir / f"{i:05d}.png")

    # NOW: get the SAPIEN extrinsic for each view and save them
    # We'll write TWO versions of COLMAP cameras.txt:
    # Version A: using our look_at function
    # Version B: using SAPIEN's actual extrinsic matrix (converted)
    # Then train both and compare which one gives correct 3D positions

    # Save SAPIEN extrinsics for later analysis
    sapien_exts = []
    rng2 = np.random.default_rng(42)  # same seed
    for i in range(N_VIEWS):
        azimuth = 2 * np.pi * i / N_VIEWS + rng2.normal(0, 0.05)
        elevation = rng2.uniform(np.radians(25), np.radians(60))
        radius = rng2.uniform(0.45, 0.55)
        x = center[0] + radius * np.cos(elevation) * np.cos(azimuth)
        y = center[1] + radius * np.cos(elevation) * np.sin(azimuth)
        z = center[2] + radius * np.sin(elevation)
        eye = np.array([x, y, z])

        pose = sapien_utils.look_at(eye=eye, target=center)
        if hasattr(pose, "sp"):
            cam.entity.set_pose(pose.sp)
        elif hasattr(pose, "raw_pose"):
            rp = pose.raw_pose.squeeze().cpu().numpy()
            cam.entity.set_pose(sapien.Pose(p=rp[:3], q=rp[3:]))

        sub_scene.update_render()
        cam.take_picture()
        ext = cam.get_extrinsic_matrix()  # (3, 4) SAPIEN convention
        sapien_exts.append(ext)

    np.save(DATA_DIR / "sapien_extrinsics.npy", np.array(sapien_exts))
    np.save(DATA_DIR / "camera_eyes.npy", np.array(cam_eyes))

    env.close()

    # Write COLMAP sparse model using our current method
    write_colmap_sparse(DATA_DIR, cam_eyes, center)

    # Save known positions for later comparison
    np.save(DATA_DIR / "known_positions.npy", actual_positions)
    print(f"\n  Saved calibration data to {DATA_DIR}")
    return actual_positions


def write_colmap_sparse(data_dir, cam_eyes, center):
    """Write COLMAP cameras using our current look_at method."""
    sparse_dir = data_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    W = H = RESOLUTION
    fovy = np.radians(75)
    fy = H / (2 * np.tan(fovy / 2))
    fx = fy

    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {W/2:.6f} {H/2:.6f}\n")

    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[]\n")
        for i, eye in enumerate(cam_eyes):
            # OpenCV convention: z-forward, y-down, x-right
            forward = center - eye
            forward = forward / np.linalg.norm(forward)
            up = np.array([0., 0., 1.])
            right = np.cross(forward, up)
            nr = np.linalg.norm(right)
            if nr < 1e-6:
                right = np.cross(forward, np.array([0., 1., 0.]))
                nr = np.linalg.norm(right)
            right = right / nr
            down = np.cross(forward, right)

            R_opencv = np.stack([right, down, forward], axis=0)
            # COLMAP stores R as the transpose of OpenCV R
            R_colmap = R_opencv.T
            t_colmap = -R_colmap @ eye

            # R to quaternion
            R = R_colmap
            trace = R[0, 0] + R[1, 1] + R[2, 2]
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                qx = (R[2, 1] - R[1, 2]) * s
                qy = (R[0, 2] - R[2, 0]) * s
                qz = (R[1, 0] - R[0, 1]) * s
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
            q = np.array([w, qx, qy, qz])
            q = q / np.linalg.norm(q)

            f.write(f"{i+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t_colmap[0]:.10f} {t_colmap[1]:.10f} {t_colmap[2]:.10f} 1 {i:05d}.png\n\n")

    # Seed points at known object positions + table
    rng = np.random.default_rng(42)
    all_pts = []
    # Table
    n_t = 5000
    all_pts.append(np.stack([
        rng.uniform(-0.25, 0.25, n_t),
        rng.uniform(-0.2, 0.2, n_t),
        rng.normal(0, 0.001, n_t),
    ], axis=-1))
    # Objects at known positions
    for pos in KNOWN_POSITIONS.values():
        n_o = 1000
        all_pts.append(pos.reshape(1, 3) + rng.uniform(-0.04, 0.04, (n_o, 3)))
    # Floor
    n_f = 2000
    all_pts.append(np.stack([
        rng.uniform(-0.5, 0.5, n_f),
        rng.uniform(-0.5, 0.5, n_f),
        np.full(n_f, -0.001),
    ], axis=-1))

    all_pts = np.concatenate(all_pts)
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        for pid, pt in enumerate(all_pts, 1):
            f.write(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 150 120 90 0.1\n")

    print(f"  Wrote COLMAP sparse: {len(cam_eyes)} cameras, {len(all_pts)} seed points")


def train_and_analyze():
    """Train 3DGS, then compare reconstructed object positions with ground truth."""
    import subprocess
    import torch

    # Train Gaussian Grouping
    print("\nTraining 3DGS on calibration scene...")

    # Dummy masks
    mask_dir = DATA_DIR / "object_mask"
    mask_dir.mkdir(exist_ok=True)
    for i in range(N_VIEWS):
        Image.fromarray(np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)).save(
            mask_dir / f"{i:05d}.png"
        )

    config = DATA_DIR / "config.json"
    config.write_text('{"num_classes": 2, "densify_until_iter": 5000}')

    output_dir = DATA_DIR / "output"
    subprocess.run([
        "conda", "run", "-n", "gaussian_grouping",
        "python", "train.py",
        "-s", str(DATA_DIR),
        "-m", str(output_dir),
        "--iterations", "7000",
        "--config_file", str(config),
    ], cwd="/home/bwang25/Desktop/Manipulation/gaussian-grouping",
       capture_output=True)

    ply_path = output_dir / "point_cloud/iteration_7000/point_cloud.ply"
    if not ply_path.exists():
        print("ERROR: Training failed, no PLY found")
        return

    # Load reconstructed point cloud
    from plyfile import PlyData
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    print(f"  Reconstructed: {len(xyz):,} Gaussians")
    print(f"  Bounds: X[{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}]"
          f" Y[{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}]"
          f" Z[{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")

    # Find clusters near known positions
    known = np.load(DATA_DIR / "known_positions.npy", allow_pickle=True).item()

    print(f"\n  === CALIBRATION RESULTS ===")
    print(f"  {'Object':<25} {'ManiSkill XYZ':>30} {'3DGS cluster XYZ':>30} {'Error (mm)':>12}")
    print(f"  {'-'*100}")

    for obj_id, maniskill_pos in known.items():
        # Find the cluster of Gaussians nearest to the expected position
        # Objects are ~5cm tall, so search within 10cm sphere
        dists = np.linalg.norm(xyz - maniskill_pos, axis=1)
        nearby = dists < 0.10  # within 10cm
        if nearby.sum() < 10:
            print(f"  {obj_id:<25} ({maniskill_pos[0]:+.3f},{maniskill_pos[1]:+.3f},{maniskill_pos[2]:+.3f})"
                  f"  NOT FOUND (only {nearby.sum()} pts within 10cm)")
            continue

        # Cluster centroid of nearby points (weighted by proximity)
        nearby_pts = xyz[nearby]
        nearby_dists = dists[nearby]
        weights = 1.0 / (nearby_dists + 0.001)
        cluster_center = np.average(nearby_pts, weights=weights, axis=0)

        error_mm = np.linalg.norm(cluster_center - maniskill_pos) * 1000
        print(f"  {obj_id:<25} ({maniskill_pos[0]:+.3f},{maniskill_pos[1]:+.3f},{maniskill_pos[2]:+.3f})"
              f"  ({cluster_center[0]:+.3f},{cluster_center[1]:+.3f},{cluster_center[2]:+.3f})"
              f"  {error_mm:8.1f}")

    # Also check: where is the table surface in 3DGS?
    table_mask = np.abs(xyz[:, 2]) < 0.02  # near z=0
    if table_mask.sum() > 100:
        table_z = xyz[table_mask, 2].mean()
        print(f"\n  Table surface Z in 3DGS: {table_z:.4f} (ManiSkill: 0.0000)")
        print(f"  Table Z offset: {table_z*1000:.1f} mm")


def main():
    print("=== COORDINATE CALIBRATION TEST ===\n")

    print("Step 1: Capture calibration scene...")
    positions = capture_calibration_scene()

    print("\nStep 2: Train 3DGS and analyze...")
    train_and_analyze()

    print("\n=== CALIBRATION COMPLETE ===")
    print(f"Results saved to {DATA_DIR}")


if __name__ == "__main__":
    main()

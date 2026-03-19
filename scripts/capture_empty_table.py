"""Capture multi-view images of an empty ManiSkill table (no robot, no objects)."""

import os
import numpy as np
import sapien
from pathlib import Path
from PIL import Image

os.environ.setdefault("DISPLAY", "")

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils import sapien_utils


def main():
    output_dir = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/empty_table")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_views = 90
    resolution = 512
    shader = "rt"

    print(f"Capturing empty table: {n_views} views @ {resolution}x{resolution}, shader={shader}")

    env = gym.make(
        "PickCube-v1", obs_mode="rgbd", num_envs=1, sim_backend="cpu",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
    )
    env.reset(seed=42)
    base_env = env.unwrapped
    scene = base_env.scene

    # Hide EVERYTHING: robot arm, cube, goal
    # Hide robot links
    robot = base_env.agent.robot
    for link in robot.get_links():
        # Make robot invisible by moving all links far away
        # Note: we can't easily hide individual links, so we move the whole robot
        pass
    # Move robot base far away
    robot.set_root_pose(sapien.Pose(p=[0, 0, -10]))

    # Hide cube and goal
    if hasattr(base_env, "cube"):
        base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, "goal_site"):
        base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
    for actor in base_env._hidden_objects:
        actor.set_pose(sapien.Pose(p=[0, 0, -10]))

    # Step to apply
    for _ in range(5):
        env.step(env.action_space.sample() * 0)

    # Create capture camera
    sub_scene = scene.sub_scenes[0]
    cam = sub_scene.add_camera(
        name="capture_cam", width=resolution, height=resolution,
        fovy=np.radians(75), near=0.01, far=10.0,
    )

    # Generate camera poses (same distribution as before)
    rng = np.random.default_rng(42)
    center = np.array([0.0, 0.0, 0.08])

    print(f"Rendering {n_views} views...")
    for i in range(n_views):
        azimuth = 2 * np.pi * i / n_views + rng.normal(0, 0.05)
        elevation = rng.uniform(np.radians(25), np.radians(60))
        radius = rng.uniform(0.45, 0.55)
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
        rgba = cam.get_picture("Color")
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(images_dir / f"{i:05d}.png")

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  [{i+1}/{n_views}]")

    env.close()

    # Write camera poses in COLMAP format (same as before)
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    W, H = resolution, resolution
    fovy = np.radians(75)
    fy = H / (2 * np.tan(fovy / 2))
    fx = fy

    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {W/2:.6f} {H/2:.6f}\n")

    rng2 = np.random.default_rng(42)  # same seed for same poses
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[]\n")
        for i in range(n_views):
            azimuth = 2 * np.pi * i / n_views + rng2.normal(0, 0.05)
            elevation = rng2.uniform(np.radians(25), np.radians(60))
            radius = rng2.uniform(0.45, 0.55)
            x = center[0] + radius * np.cos(elevation) * np.cos(azimuth)
            y = center[1] + radius * np.cos(elevation) * np.sin(azimuth)
            z = center[2] + radius * np.sin(elevation)
            eye = np.array([x, y, z])

            # OpenCV convention W2C
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
            R = np.stack([right, down, forward], axis=0)
            t = -R @ eye

            # R to quaternion
            trace = R[0,0]+R[1,1]+R[2,2]
            if trace > 0:
                s = 0.5/np.sqrt(trace+1.0)
                w=0.25/s; qx=(R[2,1]-R[1,2])*s; qy=(R[0,2]-R[2,0])*s; qz=(R[1,0]-R[0,1])*s
            elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
                s=2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
                w=(R[2,1]-R[1,2])/s; qx=0.25*s; qy=(R[0,1]+R[1,0])/s; qz=(R[0,2]+R[2,0])/s
            elif R[1,1]>R[2,2]:
                s=2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
                w=(R[0,2]-R[2,0])/s; qx=(R[0,1]+R[1,0])/s; qy=0.25*s; qz=(R[1,2]+R[2,1])/s
            else:
                s=2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
                w=(R[1,0]-R[0,1])/s; qx=(R[0,2]+R[2,0])/s; qy=(R[1,2]+R[2,1])/s; qz=0.25*s
            q = np.array([w,qx,qy,qz]); q = q/np.linalg.norm(q)

            f.write(f"{i+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {i:05d}.png\n\n")

    # Dense seed points (table + floor only, no objects)
    rng3 = np.random.default_rng(42)
    pts = []
    # Table surface
    n_t = 8000
    tx = rng3.uniform(-0.25, 0.25, n_t)
    ty = rng3.uniform(-0.2, 0.2, n_t)
    tz = np.zeros(n_t) + rng3.normal(0, 0.001, n_t)
    pts.append(np.stack([tx, ty, tz], axis=-1))
    # Floor
    n_f = 4000
    fx_ = rng3.uniform(-0.6, 0.6, n_f)
    fy_ = rng3.uniform(-0.6, 0.6, n_f)
    fz = np.full(n_f, -0.001) + rng3.normal(0, 0.001, n_f)
    pts.append(np.stack([fx_, fy_, fz], axis=-1))

    all_pts = np.concatenate(pts)
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        for pid, pt in enumerate(all_pts, 1):
            f.write(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 160 130 100 0.1\n")

    print(f"\nDone! {n_views} images + COLMAP sparse model at {output_dir}")
    print(f"  Seed points: {len(all_pts)}")


if __name__ == "__main__":
    main()

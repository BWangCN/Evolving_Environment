"""Capture individual YCB objects for compositional 3DGS.

Each object is placed alone at table center, captured from multiple views,
and reconstructed as a separate Gaussian cluster.

No robot arm in any capture — only the object + table (table will be subtracted later
or we use the object Gaussians above the table surface).
"""

import os
import numpy as np
import sapien
from pathlib import Path
from PIL import Image

os.environ.setdefault("DISPLAY", "")

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import get_actor_builder

# Objects to capture (complex shapes for testing reconstruction)
OBJECTS = [
    "005_tomato_soup_can",
    "025_mug",
    "024_bowl",
    "011_banana",
    "013_apple",
]

N_VIEWS = 60  # fewer views per object (smaller, simpler geometry)
RESOLUTION = 512


def capture_single_object(obj_id: str, output_dir: Path, shader: str = "rt"):
    """Capture one object alone on the table."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "PickCube-v1", obs_mode="rgbd", num_envs=1, sim_backend="cpu",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=shader),
        human_render_camera_configs=dict(shader_pack=shader),
    )
    env.reset(seed=42)
    base_env = env.unwrapped
    scene = base_env.scene

    # Hide robot
    base_env.agent.robot.set_root_pose(sapien.Pose(p=[0, 0, -10]))

    # Hide default cube + goal
    if hasattr(base_env, "cube"):
        base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, "goal_site"):
        base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
    for actor in base_env._hidden_objects:
        actor.set_pose(sapien.Pose(p=[0, 0, -10]))

    # Keep table and floor visible — object sits naturally on table
    # We'll extract the object by tight bounding box later

    # Place target object at origin, floating (no table underneath)
    builder = get_actor_builder(scene, id=f"ycb:{obj_id}")
    obj = builder.build(name=f"obj_{obj_id}")
    collision_mesh = obj.get_first_collision_mesh()
    if collision_mesh:
        bbox = collision_mesh.bounding_box.bounds  # (2, 3) min/max
        z_offset = -bbox[0, 2]  # bottom of object at z=0
        obj_height = bbox[1, 2] - bbox[0, 2]
    else:
        z_offset = 0.02
        obj_height = 0.05
    obj.set_pose(sapien.Pose(p=[0, 0, z_offset], q=[1, 0, 0, 0]))
    obj_center_z = z_offset + obj_height * 0.5

    # Settle physics (object lands on table)
    for _ in range(10):
        env.step(env.action_space.sample() * 0)

    # Camera orbit — same radius as environment for consistent SH
    sub_scene = scene.sub_scenes[0]
    cam = sub_scene.add_camera(
        name="obj_cam", width=RESOLUTION, height=RESOLUTION,
        fovy=np.radians(75), near=0.01, far=10.0,
    )

    center = np.array([0.0, 0.0, obj_center_z])
    rng = np.random.default_rng(42)

    for i in range(N_VIEWS):
        azimuth = 2 * np.pi * i / N_VIEWS + rng.normal(0, 0.05)
        elevation = rng.uniform(np.radians(20), np.radians(70))
        radius = rng.uniform(0.25, 0.40)  # close enough to fill frame, consistent SH
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

    env.close()

    # Write COLMAP sparse model
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    W, H = RESOLUTION, RESOLUTION
    fovy = np.radians(75)
    fy = H / (2 * np.tan(fovy / 2))
    fx = fy

    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list\n# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {W/2:.6f} {H/2:.6f}\n")

    rng2 = np.random.default_rng(42)
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[]\n")
        for i in range(N_VIEWS):
            azimuth = 2 * np.pi * i / N_VIEWS + rng2.normal(0, 0.05)
            elevation = rng2.uniform(np.radians(20), np.radians(70))
            radius = rng2.uniform(0.25, 0.40)
            x = center[0] + radius * np.cos(elevation) * np.cos(azimuth)
            y = center[1] + radius * np.cos(elevation) * np.sin(azimuth)
            z = center[2] + radius * np.sin(elevation)
            eye = np.array([x, y, z])

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

            trace = R[0,0]+R[1,1]+R[2,2]
            if trace > 0:
                s=0.5/np.sqrt(trace+1.0); w=0.25/s; qx=(R[2,1]-R[1,2])*s; qy=(R[0,2]-R[2,0])*s; qz=(R[1,0]-R[0,1])*s
            elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
                s=2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s; qx=0.25*s; qy=(R[0,1]+R[1,0])/s; qz=(R[0,2]+R[2,0])/s
            elif R[1,1]>R[2,2]:
                s=2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s; qx=(R[0,1]+R[1,0])/s; qy=0.25*s; qz=(R[1,2]+R[2,1])/s
            else:
                s=2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s; qx=(R[0,2]+R[2,0])/s; qy=(R[1,2]+R[2,1])/s; qz=0.25*s
            q = np.array([w,qx,qy,qz]); q = q/np.linalg.norm(q)
            f.write(f"{i+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {i:05d}.png\n\n")

    # Seed points — object region only (no table since it's hidden)
    rng3 = np.random.default_rng(42)
    n_pts = 5000
    obj_pts = np.zeros((n_pts, 3))
    obj_pts[:, 0] = rng3.uniform(-0.06, 0.06, n_pts)
    obj_pts[:, 1] = rng3.uniform(-0.06, 0.06, n_pts)
    obj_pts[:, 2] = rng3.uniform(0, obj_height * 1.2, n_pts)
    all_pts = obj_pts

    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        for pid, pt in enumerate(all_pts, 1):
            f.write(f"{pid} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 160 130 100 0.1\n")

    return len(all_pts)


def main():
    base_dir = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/objects")

    for obj_id in OBJECTS:
        print(f"\n{'='*50}")
        print(f"Capturing {obj_id}...")
        out = base_dir / obj_id
        n_pts = capture_single_object(obj_id, out, shader="rt")
        print(f"  {N_VIEWS} images + {n_pts} seed points → {out}")

        # Verify one image
        img = Image.open(out / "images" / "00000.png")
        arr = np.array(img)
        print(f"  Sample image: {arr.shape}, mean={arr.mean():.0f}")

    print(f"\n{'='*50}")
    print(f"All {len(OBJECTS)} objects captured at {base_dir}")


if __name__ == "__main__":
    main()

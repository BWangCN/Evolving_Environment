"""
Capture multi-view images of a ManiSkill tabletop scene for 3DGS reconstruction.

Renders a Franka Panda + table + YCB objects from ~90 camera viewpoints using
ManiSkill's ray-tracing renderer. Output is ready for COLMAP → Gaussian Grouping.

Objects placed on table:
  - 005_tomato_soup_can (red can, cylindrical)
  - 006_mustard_bottle (yellow bottle, complex shape)
  - 025_mug (mug with handle)
  - 024_bowl (bowl)
  - 011_banana (banana, organic shape)

Usage:
    conda activate gaussian_grouping
    python scripts/capture_maniskill_multiview.py

    # Custom objects:
    python scripts/capture_maniskill_multiview.py --objects 005_tomato_soup_can 013_apple 025_mug

    # More views:
    python scripts/capture_maniskill_multiview.py --num-views 120

    # Lower quality but faster:
    python scripts/capture_maniskill_multiview.py --shader rt-fast
"""

import argparse
import os
from pathlib import Path

import numpy as np
import sapien
import torch

# Must set before importing ManiSkill
os.environ.setdefault("DISPLAY", "")


def generate_camera_poses(
    n_views: int = 90,
    center: np.ndarray = None,
    radius_range: tuple = (0.5, 0.9),
    elevation_range: tuple = (15, 70),
    full_circle: bool = True,
):
    """Generate camera poses on a hemisphere looking at the table center.

    Args:
        n_views: Total number of viewpoints.
        center: (3,) look-at target. Default: slightly above table center.
        radius_range: (min, max) distance from center.
        elevation_range: (min, max) degrees above horizon.
        full_circle: If True, cover full 360° azimuth.

    Returns:
        List of (eye, target) pairs for look_at().
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.1])  # slightly above table

    rng = np.random.default_rng(42)
    poses = []

    for i in range(n_views):
        # Azimuth: uniform around full circle
        if full_circle:
            azimuth = 2 * np.pi * i / n_views + rng.normal(0, 0.05)
        else:
            azimuth = np.pi * i / n_views - np.pi / 2

        # Elevation: biased toward medium angles (best for 3DGS)
        el_min, el_max = np.radians(elevation_range[0]), np.radians(elevation_range[1])
        elevation = rng.uniform(el_min, el_max)

        # Radius: slight variation
        radius = rng.uniform(*radius_range)

        # Spherical to Cartesian
        x = center[0] + radius * np.cos(elevation) * np.cos(azimuth)
        y = center[1] + radius * np.cos(elevation) * np.sin(azimuth)
        z = center[2] + radius * np.sin(elevation)

        eye = np.array([x, y, z])
        poses.append((eye, center))

    return poses


def main():
    parser = argparse.ArgumentParser(
        description="Capture multi-view images of ManiSkill tabletop scene"
    )
    parser.add_argument(
        "--objects", nargs="+",
        default=[
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "025_mug",
            "024_bowl",
            "011_banana",
        ],
        help="YCB object IDs to place on table"
    )
    parser.add_argument("--num-views", type=int, default=90,
                        help="Number of viewpoints to capture")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution (square)")
    parser.add_argument("--shader", type=str, default="rt",
                        choices=["rt", "rt-med", "rt-fast", "default"],
                        help="Rendering quality")
    parser.add_argument("--output-dir", type=str,
                        default="/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/maniskill_tabletop",
                        help="Output directory for images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Objects: {args.objects}")
    print(f"Views: {args.num_views}, Resolution: {args.resolution}x{args.resolution}")
    print(f"Shader: {args.shader}")
    print(f"Output: {output_dir}")

    # ── Build scene ──────────────────────────────────────────────────
    import gymnasium as gym
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building.actors import get_actor_builder
    from PIL import Image

    # Create a simple PickCube env to get the scene infrastructure, then add objects
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        num_envs=1,
        sim_backend="cpu",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )
    env.reset(seed=args.seed)

    # Access internal scene
    base_env = env.unwrapped
    scene = base_env.scene

    # Place YCB objects on the table
    rng = np.random.default_rng(args.seed)
    object_positions = [
        [0.0, 0.0],     # center
        [0.1, 0.08],    # front-right
        [-0.1, 0.08],   # front-left
        [0.1, -0.08],   # back-right
        [-0.1, -0.08],  # back-left
        [0.0, 0.12],    # front-center
        [0.0, -0.12],   # back-center
    ]

    placed_objects = []
    for i, obj_id in enumerate(args.objects):
        try:
            builder = get_actor_builder(scene, id=f"ycb:{obj_id}")
            xy = object_positions[i % len(object_positions)]
            # Get object height from collision mesh after building
            obj = builder.build(name=f"obj_{obj_id}")

            # Place on table surface (table is at z=0 in ManiSkill)
            collision_mesh = obj.get_first_collision_mesh()
            z_offset = -collision_mesh.bounding_box.bounds[0, 2] if collision_mesh else 0.02
            obj.set_pose(sapien.Pose(
                p=[xy[0], xy[1], z_offset],
                q=[1, 0, 0, 0],
            ))
            placed_objects.append((obj_id, obj))
            print(f"  Placed {obj_id} at ({xy[0]:.2f}, {xy[1]:.2f}, {z_offset:.3f})")
        except Exception as e:
            print(f"  WARNING: Failed to place {obj_id}: {e}")

    # Step physics to settle objects
    for _ in range(10):
        env.step(env.action_space.sample() * 0)

    # ── Generate camera viewpoints ───────────────────────────────────
    # Hide PickCube default objects (red cube, green goal) so they don't clutter the scene
    for actor in base_env._hidden_objects:
        actor.set_pose(sapien.Pose(p=[0, 0, -10]))  # move far below
    if hasattr(base_env, 'cube'):
        base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, 'goal_site'):
        base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))

    camera_poses = generate_camera_poses(
        n_views=args.num_views,
        center=np.array([0.0, 0.0, 0.08]),  # slightly above table, where objects are
        radius_range=(0.45, 0.55),           # consistent distance
        elevation_range=(25, 60),
    )

    # ── Create a dedicated capture camera with wide FOV ─────────────
    print(f"\nRendering {args.num_views} views...")

    sub_scene = scene.sub_scenes[0]

    # Create a new camera entity with wider FOV for better scene coverage
    cam_entity = sub_scene.add_camera(
        name="capture_cam",
        width=args.resolution,
        height=args.resolution,
        fovy=np.radians(75),  # wider than default 57°
        near=0.01,
        far=10.0,
    )
    print(f"  Created capture camera: {args.resolution}x{args.resolution}, FOV=75°")

    for i, (eye, target) in enumerate(camera_poses):
        # Compute look-at pose using sapien utils
        pose = sapien_utils.look_at(eye=eye, target=target)
        if hasattr(pose, 'sp'):
            cam_entity.entity.set_pose(pose.sp)
        elif hasattr(pose, 'raw_pose'):
            rp = pose.raw_pose.squeeze().cpu().numpy()
            cam_entity.entity.set_pose(sapien.Pose(p=rp[:3], q=rp[3:]))
        else:
            cam_entity.entity.set_pose(pose)

        # Render
        sub_scene.update_render()
        cam_entity.take_picture()
        rgba = cam_entity.get_picture("Color")  # (H, W, 4) float32

        # Convert to uint8 RGB
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

        # Save
        img_path = images_dir / f"{i:05d}.png"
        Image.fromarray(rgb).save(img_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.num_views}] saved {img_path.name} "
                  f"(eye={eye.round(2)}, {rgb.shape})")

    env.close()

    print(f"\n{'='*50}")
    print(f"Capture complete!")
    print(f"  {args.num_views} images saved to {images_dir}")
    print(f"  Resolution: {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"  Objects: {[o[0] for o in placed_objects]}")
    print(f"\nNext steps:")
    print(f"  1. Run COLMAP: colmap automatic_reconstructor --image_path {images_dir} --workspace_path {output_dir}/colmap")
    print(f"  2. Run Gaussian Grouping on the COLMAP output")
    print(f"  3. Verify 3DGS reconstruction quality")


if __name__ == "__main__":
    main()

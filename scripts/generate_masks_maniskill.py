"""Generate segmentation masks from ManiSkill for Gaussian Grouping training."""

import numpy as np
import sapien
from pathlib import Path
from PIL import Image

import mani_skill.envs
import gymnasium as gym
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import get_actor_builder


DATA_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/maniskill_tabletop")
OBJECTS = ["005_tomato_soup_can", "006_mustard_bottle", "025_mug", "024_bowl", "011_banana"]
POSITIONS = [[0.0, 0.0], [0.1, 0.08], [-0.1, 0.08], [0.1, -0.08], [-0.1, -0.08]]
N_VIEWS = 90
SEED = 42


def main():
    mask_dir = DATA_DIR / "object_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Use default shader (supports Segmentation texture, RT does not)
    env = gym.make(
        "PickCube-v1", obs_mode="rgbd", num_envs=1, sim_backend="cpu",
        render_mode="rgb_array", sensor_configs=dict(shader_pack="default"),
    )
    env.reset(seed=SEED)
    base_env = env.unwrapped
    scene = base_env.scene

    # Place YCB objects
    for i, obj_id in enumerate(OBJECTS):
        try:
            builder = get_actor_builder(scene, id=f"ycb:{obj_id}")
            obj = builder.build(name=f"obj_{obj_id}")
            collision_mesh = obj.get_first_collision_mesh()
            z_offset = -collision_mesh.bounding_box.bounds[0, 2] if collision_mesh else 0.02
            obj.set_pose(sapien.Pose(
                p=[POSITIONS[i][0], POSITIONS[i][1], z_offset], q=[1, 0, 0, 0]
            ))
            print(f"  Placed {obj_id} at z={z_offset:.3f}")
        except Exception as e:
            print(f"  WARNING: Failed to place {obj_id}: {e}")

    # Hide PickCube default objects
    if hasattr(base_env, "cube"):
        base_env.cube.set_pose(sapien.Pose(p=[0, 0, -10]))
    if hasattr(base_env, "goal_site"):
        base_env.goal_site.set_pose(sapien.Pose(p=[0, 0, -10]))
    for actor in base_env._hidden_objects:
        actor.set_pose(sapien.Pose(p=[0, 0, -10]))

    # Settle physics
    for _ in range(10):
        env.step(env.action_space.sample() * 0)

    # Find base_camera (supports segmentation)
    sub_scene = scene.sub_scenes[0]
    cam = None
    for c in sub_scene.get_cameras():
        if "base_camera" in c.name:
            cam = c
            break
    assert cam is not None, "No base_camera found"

    # Generate same camera poses as capture script
    rng = np.random.default_rng(SEED)
    center = np.array([0.0, 0.0, 0.08])

    print(f"\nGenerating {N_VIEWS} segmentation masks...")

    for i in range(N_VIEWS):
        azimuth = 2 * np.pi * i / N_VIEWS + rng.normal(0, 0.05)
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
        seg = cam.get_picture("Segmentation")  # (H, W, 4)

        # Actor-level segmentation (channel 1) as grayscale integer IDs
        seg_ids = seg[:, :, 1].astype(np.uint8)
        unique_ids = np.unique(seg_ids)

        # Save as grayscale (mode=L) — each pixel = object ID
        # Resize to 512x512 to match images
        mask_img = Image.fromarray(seg_ids, mode="L").resize((512, 512), Image.NEAREST)
        mask_img.save(mask_dir / f"{i:05d}.png")

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  [{i+1}/{N_VIEWS}] unique IDs: {unique_ids.tolist()}, shape: {seg_ids.shape}")

    env.close()
    print(f"\nDone! {N_VIEWS} masks saved to {mask_dir}")


if __name__ == "__main__":
    main()

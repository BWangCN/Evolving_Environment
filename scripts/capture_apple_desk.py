"""
Capture an object resting ON the ManiSkill tabletop ("desk") from many upper-hemisphere views,
for the CORRECT pipeline: reconstruct whole table+object scene -> segment object cluster ->
fix the occluded-bottom hole.

Key points (lessons learned):
  - Object sits ON the table (bottom at table z=0) -> its underside is NEVER seen from the upper
    hemisphere = the real occlusion hole we later fix.
  - We write EXACT camera poses (COLMAP sparse/0), NOT COLMAP-SfM/DUSt3R.
  - We write EXACT per-view object masks (object_mask/, value 1=object, 0=rest) by hiding all
    non-object render bodies and taking the object's alpha -> perfect grouping supervision (no SAM).

Output (gaussian-grouping format): <out>/{images,object_mask}/{00000..}.png + <out>/sparse/0/*.txt
Env: gaussian_grouping
Usage:
  python scripts/capture_apple_desk.py --obj 013_apple --num-views 150 \
      --out /home/bwang25/Desktop/Manipulation/gaussian-grouping/data/apple_desk
"""
import os, argparse
import numpy as np
from pathlib import Path
os.environ.setdefault("DISPLAY", "")
import sapien
import gymnasium as gym
import mani_skill.envs  # noqa
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import get_actor_builder
from PIL import Image


def rotmat_to_qvec(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0); w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s; y = (R[0, 2] - R[2, 0]) * s; z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]); w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]); w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]); w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def w2c_from_eye(eye, center):
    """COLMAP world->cam [R|t] for a camera at eye looking at center, world up +z.
    Convention matches capture_individual_objects.py (known-good with gaussian-grouping)."""
    fwd = center - eye; fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0., 0., 1.])
    right = np.cross(fwd, up); nr = np.linalg.norm(right)
    if nr < 1e-6:
        right = np.cross(fwd, np.array([0., 1., 0.])); nr = np.linalg.norm(right)
    right /= nr
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], 0)   # world->cam
    t = -R @ eye
    return R, t


def render_bodies(entity):
    return [c for c in entity.components if "RenderBody" in type(c).__name__]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="013_apple")
    ap.add_argument("--num-views", type=int, default=150)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--shader", default="rt", choices=["rt", "rt-med", "rt-fast", "default"])
    ap.add_argument("--fovy", type=float, default=55.0)
    ap.add_argument("--radius", type=float, nargs=2, default=[0.22, 0.32])
    ap.add_argument("--elev", type=float, nargs=2, default=[15.0, 85.0])
    ap.add_argument("--out", default="/home/bwang25/Desktop/Manipulation/gaussian-grouping/data/apple_desk")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "object_mask").mkdir(parents=True, exist_ok=True)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    W = H = args.resolution

    env = gym.make("PickCube-v1", obs_mode="rgbd", num_envs=1, sim_backend="cpu",
                   render_mode="rgb_array",
                   sensor_configs=dict(shader_pack=args.shader),
                   human_render_camera_configs=dict(shader_pack=args.shader))
    env.reset(seed=args.seed)
    be = env.unwrapped
    scene = be.scene
    sub = scene.sub_scenes[0]

    # Hide robot + PickCube's cube/goal so the scene is just table + apple
    be.agent.robot.set_root_pose(sapien.Pose(p=[0, 0, -100]))
    if hasattr(be, "cube"): be.cube.set_pose(sapien.Pose(p=[0, 0, -100]))
    if hasattr(be, "goal_site"): be.goal_site.set_pose(sapien.Pose(p=[0, 0, -100]))
    for a in be._hidden_objects: a.set_pose(sapien.Pose(p=[0, 0, -100]))

    # Place the object on the table (bottom at z=0)
    builder = get_actor_builder(scene, id=f"ycb:{args.obj}")
    builder.set_scene_idxs([0])
    obj = builder.build(name=f"obj_{args.obj}")
    cm = obj.get_first_collision_mesh()
    zmin = float(cm.bounding_box.bounds[0, 2]); zmax = float(cm.bounding_box.bounds[1, 2])
    z_off = -zmin
    obj.set_pose(sapien.Pose(p=[0, 0, z_off], q=[1, 0, 0, 0]))
    for _ in range(25):
        be.scene.step()
    center = obj.pose.p[0].cpu().numpy().astype(np.float64)
    center[2] = z_off + (zmin + zmax) / 2.0   # geometric center height (origin + bbox center)
    print(f"object {args.obj}: zmin={zmin:.3f} zmax={zmax:.3f} -> center={center.round(3)}")

    obj_entity = obj._objs[0]
    # all other render bodies (table/ground/walls) to hide for the mask pass
    other_comps = []
    for e in sub.entities:
        if e is obj_entity:
            continue
        other_comps += render_bodies(e)

    cam = sub.add_camera(name="cap", width=W, height=H,
                         fovy=np.radians(args.fovy), near=0.01, far=10.0)
    fy = H / (2 * np.tan(np.radians(args.fovy) / 2)); fx = fy

    rng = np.random.default_rng(args.seed)
    n = args.num_views
    eyes = []
    for i in range(n):
        az = 2 * np.pi * i / n + rng.normal(0, 0.04)
        el = np.radians(rng.uniform(*args.elev))
        r = rng.uniform(*args.radius)
        eye = center + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        eyes.append(eye)

    n_ok = 0
    for i, eye in enumerate(eyes):
        pose = sapien_utils.look_at(eye=eye, target=center)
        psp = pose.sp if hasattr(pose, "sp") else sapien.Pose(
            p=pose.raw_pose.squeeze().cpu().numpy()[:3], q=pose.raw_pose.squeeze().cpu().numpy()[3:])
        cam.entity.set_pose(psp)

        sub.update_render(); cam.take_picture()
        rgba = cam.get_picture("Color")
        rgba = rgba.cpu().numpy() if hasattr(rgba, "cpu") else np.asarray(rgba)
        rgb = (rgba[:, :, :3] * (255.0 if rgba[..., :3].max() <= 1.0001 else 1.0)).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(out / "images" / f"{i:05d}.png")

        # mask pass: hide everything except the object, take alpha
        for c in other_comps: c.disable()
        sub.update_render(); cam.take_picture()
        a = cam.get_picture("Color")
        a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
        mask = (a[:, :, 3] > 0.5).astype(np.uint8)
        for c in other_comps: c.enable()
        Image.fromarray(mask, mode="L").save(out / "object_mask" / f"{i:05d}.png")  # 1=object, 0=rest
        n_ok += int(mask.sum() > 50)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{n}] cov={100*mask.mean():.1f}%  eye={eye.round(2)}")

    env.close()

    # ---- write COLMAP sparse/0 with EXACT poses ----
    sp = out / "sparse" / "0"
    with open(sp / "cameras.txt", "w") as f:
        f.write("# Camera list\n1 PINHOLE %d %d %.6f %.6f %.6f %.6f\n" % (W, H, fx, fy, W / 2, H / 2))
    with open(sp / "images.txt", "w") as f:
        f.write("# Image list\n")
        for i, eye in enumerate(eyes):
            R, t = w2c_from_eye(eye, center)
            q = rotmat_to_qvec(R)
            f.write(f"{i+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {i:05d}.png\n\n")
    # seed point cloud over table plane + object volume
    rng2 = np.random.default_rng(0)
    n_tab, n_obj = 12000, 4000
    tab = np.column_stack([rng2.uniform(-0.35, 0.35, n_tab), rng2.uniform(-0.35, 0.35, n_tab),
                           rng2.uniform(-0.01, 0.01, n_tab)])
    obj_pts = center + np.column_stack([rng2.uniform(-0.05, 0.05, n_obj), rng2.uniform(-0.05, 0.05, n_obj),
                                        rng2.uniform(-0.04, 0.04, n_obj)])
    pts = np.vstack([tab, obj_pts])
    cols = np.vstack([np.tile([180, 140, 100], (n_tab, 1)), np.tile([170, 40, 40], (n_obj, 1))])
    with open(sp / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        for pid, (p, c) in enumerate(zip(pts, cols), 1):
            f.write(f"{pid} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[0])} {int(c[1])} {int(c[2])} 0.5\n")
    print(f"\nDone: {n} views ({n_ok} with object), masks + exact poses -> {out}")


if __name__ == "__main__":
    main()

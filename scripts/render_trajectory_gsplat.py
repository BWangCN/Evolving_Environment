"""
Render a pick-place trajectory through a 3DGS scene using gsplat.

Produces:
  1. Wrist-camera video: camera follows EE through the trajectory
  2. Third-person video: fixed camera (matching ManiSkill VLA input)
     with the wrist-cam frustum drawn to show where the robot "looks"
  3. Combined side-by-side video

Usage:
    conda activate gaussian_grouping
    python scripts/render_trajectory_gsplat.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ───────────────────────────────────────────────────────────

PLY_PATH = "/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/maniskill_tabletop/output_v3/point_cloud/iteration_7000/point_cloud.ply"
OUTPUT_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs/trajectory_render")

# Third-person camera — use a pose within the 3DGS training distribution
# (our capture orbit was radius 0.45-0.55, elevation 25-60 degrees around table center)
THIRD_PERSON_EYE = np.array([0.35, 0.25, 0.40])  # front-right, elevated
THIRD_PERSON_TARGET = np.array([0.0, 0.0, 0.05])  # table center

RENDER_W, RENDER_H = 256, 256  # Match VLA input resolution
FPS = 10


# ── Load 3DGS model ─────────────────────────────────────────────────

def load_gsplat_model(ply_path):
    """Load 3DGS PLY into gsplat-compatible tensors."""
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    n = len(v.data)

    means = torch.tensor(
        np.stack([v["x"], v["y"], v["z"]], axis=-1), dtype=torch.float32, device="cuda"
    )

    # SH DC coefficients — gsplat v1.5 expects raw SH, shape (N, K, 3) where K=(deg+1)^2
    # For sh_degree=0, K=1
    colors_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)
    colors = torch.tensor(colors_dc, dtype=torch.float32, device="cuda").unsqueeze(1)  # (N, 1, 3)

    # Opacities
    opacities_raw = torch.tensor(v["opacity"], dtype=torch.float32, device="cuda")
    opacities = torch.sigmoid(opacities_raw)

    # Scales
    scales_raw = torch.tensor(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1),
        dtype=torch.float32, device="cuda",
    )
    scales = torch.exp(scales_raw)

    # Rotations (quaternion wxyz)
    quats = torch.tensor(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1),
        dtype=torch.float32, device="cuda",
    )
    quats = quats / quats.norm(dim=-1, keepdim=True)

    print(f"Loaded {n:,} Gaussians from {ply_path}")
    return means, colors, opacities, scales, quats


def render_gsplat(means, colors, opacities, scales, quats, viewmat, K, W, H):
    """Render one frame using gsplat."""
    from gsplat import rasterization

    # gsplat expects (C, 4, 4) viewmat, (C, 3, 3) K
    viewmat_t = torch.tensor(viewmat, dtype=torch.float32, device="cuda").unsqueeze(0)
    K_t = torch.tensor(K, dtype=torch.float32, device="cuda").unsqueeze(0)

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat_t,
        Ks=K_t,
        width=W,
        height=H,
        sh_degree=0,
    )

    rgb = renders[0].clamp(0, 1).cpu().numpy()  # (H, W, 3)
    return (rgb * 255).astype(np.uint8)


# ── Camera math ──────────────────────────────────────────────────────

def look_at_opengl(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Compute 4x4 world-to-camera matrix (OpenGL convention for gsplat)."""
    forward = eye - target  # OpenGL looks down -Z
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-6:
        # Eye-target is parallel to up — use alternative up
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        norm_right = np.linalg.norm(right)
    right = right / norm_right
    up_actual = np.cross(forward, right)

    R = np.eye(4)
    R[0, :3] = right
    R[1, :3] = up_actual
    R[2, :3] = forward
    R[:3, 3] = R[:3, :3] @ (-eye)
    return R


def intrinsic_matrix(W, H, fovy_deg=90):
    """Compute 3x3 camera intrinsic matrix."""
    fovy = np.radians(fovy_deg)
    fy = H / (2 * np.tan(fovy / 2))
    fx = fy
    cx, cy = W / 2, H / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def quat_wxyz_to_rotation(q):
    """Quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def ee_pose_to_wrist_viewmat(position, orientation_wxyz):
    """Convert EE pose to a following camera view matrix.

    Uses a simple strategy: camera hovers above and behind the EE,
    always looking at the EE position. This gives a clear view of
    the gripper and the objects regardless of EE orientation.
    """
    # Camera follows EE from above and slightly behind
    cam_offset = np.array([0.0, -0.15, 0.20])  # behind (-Y) and above (+Z)
    cam_pos = position + cam_offset
    cam_target = position

    return look_at_opengl(cam_pos, cam_target)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wrist_dir = OUTPUT_DIR / "wrist"
    third_dir = OUTPUT_DIR / "third_person"
    combined_dir = OUTPUT_DIR / "combined"
    wrist_dir.mkdir(exist_ok=True)
    third_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)

    # Load 3DGS
    means, colors, opacities, scales, quats = load_gsplat_model(PLY_PATH)

    # Generate trajectory (same as pipeline test)
    from src.pipeline.gs_to_grasp import extract_scene_pointcloud, detect_grasps
    from src.trajectory import TrajectoryGenerator, trajectory_to_actions, compute_camera_poses

    pts, cols = extract_scene_pointcloud(PLY_PATH, max_points=100000)
    mask = (pts[:, 2] > 0.005) & (pts[:, 2] < 0.2) & \
           (pts[:, 0] > -0.2) & (pts[:, 0] < 0.2) & \
           (pts[:, 1] > -0.15) & (pts[:, 1] < 0.15)

    grasps = detect_grasps(pts[mask], cols[mask], top_k=5, collision_detection=False)
    best = grasps[0]
    place_target = np.array([0.15, -0.10, 0.025], dtype=np.float32)

    gen = TrajectoryGenerator()
    traj = gen.generate_pick_place(best, place_target)
    gen.smooth_corners(traj, radius=3)
    actions = trajectory_to_actions(traj)

    print(f"\nTrajectory: {len(traj)} waypoints")
    print(f"Rendering {len(traj)} frames at {RENDER_W}x{RENDER_H}...")

    # Camera intrinsics
    K_wrist = intrinsic_matrix(RENDER_W, RENDER_H, fovy_deg=90)
    K_third = intrinsic_matrix(RENDER_W, RENDER_H, fovy_deg=75)

    # Third-person camera (fixed)
    viewmat_third = look_at_opengl(THIRD_PERSON_EYE, THIRD_PERSON_TARGET)

    # Phase labels and colors for overlay
    phase_colors = {
        "transit_pick": (51, 153, 255),
        "pre_grasp": (0, 204, 102),
        "approach": (255, 153, 0),
        "grasp": (255, 0, 0),
        "lift": (204, 0, 204),
        "transit_place": (51, 153, 255),
        "pre_place": (0, 204, 102),
        "place_descend": (255, 153, 0),
        "release": (255, 0, 0),
        "retreat": (128, 128, 128),
    }

    for i, wp in enumerate(traj.waypoints):
        # Wrist camera render
        viewmat_wrist = ee_pose_to_wrist_viewmat(wp.position, wp.orientation)
        wrist_img = render_gsplat(means, colors, opacities, scales, quats,
                                  viewmat_wrist, K_wrist, RENDER_W, RENDER_H)

        # Third-person render (same every frame since scene is static)
        if i == 0:
            third_img = render_gsplat(means, colors, opacities, scales, quats,
                                      viewmat_third, K_third, RENDER_W, RENDER_H)
            third_base = third_img.copy()
        else:
            third_img = third_base.copy()

        # Add phase label + gripper state overlay
        wrist_pil = Image.fromarray(wrist_img)
        draw_w = ImageDraw.Draw(wrist_pil)
        phase = wp.phase
        grip_text = "OPEN" if wp.gripper > 0.5 else "CLOSED"
        pc = phase_colors.get(phase, (128, 128, 128))
        draw_w.rectangle([(0, 0), (RENDER_W, 18)], fill=(0, 0, 0, 180))
        draw_w.text((4, 2), f"{phase} | grip:{grip_text}", fill=pc)
        draw_w.text((4, RENDER_H - 14), f"step {i}/{len(traj)-1}", fill=(200, 200, 200))

        third_pil = Image.fromarray(third_img)
        draw_t = ImageDraw.Draw(third_pil)
        draw_t.rectangle([(0, 0), (RENDER_W, 18)], fill=(0, 0, 0, 180))
        draw_t.text((4, 2), f"Third-person (fixed)", fill=(200, 200, 200))

        # Save individual frames
        wrist_pil.save(wrist_dir / f"{i:04d}.png")
        third_pil.save(third_dir / f"{i:04d}.png")

        # Combined side-by-side
        combined = Image.new("RGB", (RENDER_W * 2 + 4, RENDER_H), (30, 30, 30))
        combined.paste(wrist_pil, (0, 0))
        combined.paste(third_pil, (RENDER_W + 4, 0))
        combined.save(combined_dir / f"{i:04d}.png")

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(traj)}] phase={phase}, pos=({wp.position[0]:.3f},{wp.position[1]:.3f},{wp.position[2]:.3f})")

    # Create videos
    print("\nCreating videos...")
    import subprocess

    for name, src_dir in [("wrist_camera", wrist_dir), ("third_person", third_dir), ("combined", combined_dir)]:
        out_path = OUTPUT_DIR / f"{name}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(src_dir / "%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out_path),
        ], capture_output=True)
        print(f"  {out_path} ({out_path.stat().st_size // 1024} KB)")

    # Also save a single comparison image (first, middle, last frames)
    key_frames = [0, len(traj) // 4, len(traj) // 2, 3 * len(traj) // 4, len(traj) - 1]
    fig_w = RENDER_W * len(key_frames)
    fig_h = RENDER_H * 2 + 20
    summary = Image.new("RGB", (fig_w, fig_h), (30, 30, 30))
    for col, idx in enumerate(key_frames):
        w_img = Image.open(wrist_dir / f"{idx:04d}.png")
        t_img = Image.open(third_dir / f"{idx:04d}.png")
        summary.paste(w_img, (col * RENDER_W, 0))
        summary.paste(t_img, (col * RENDER_W, RENDER_H + 20))
    summary.save(OUTPUT_DIR / "trajectory_summary.png")
    print(f"  Summary: {OUTPUT_DIR / 'trajectory_summary.png'}")

    print(f"\nDone! Outputs at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

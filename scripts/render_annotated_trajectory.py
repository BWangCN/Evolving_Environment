"""Render 3DGS scene from a fixed viewpoint with trajectory overlay."""

import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

PLY = "/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/maniskill_tabletop/output_v3/point_cloud/iteration_7000/point_cloud.ply"
OUT = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs/trajectory_render")

W, H = 512, 512


def load_model(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    means = torch.tensor(np.stack([v["x"], v["y"], v["z"]], axis=-1), dtype=torch.float32, device="cuda")
    colors = torch.tensor(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1), dtype=torch.float32, device="cuda").unsqueeze(1)
    opacities = torch.sigmoid(torch.tensor(np.array(v["opacity"]), dtype=torch.float32, device="cuda"))
    scales = torch.exp(torch.tensor(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1), dtype=torch.float32, device="cuda"))
    quats = torch.tensor(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1), dtype=torch.float32, device="cuda")
    quats = quats / quats.norm(dim=-1, keepdim=True)
    print(f"Loaded {len(v.data):,} Gaussians")
    return means, colors, opacities, scales, quats


def look_at(eye, target, up=np.array([0., 0., 1.])):
    """OpenCV convention: x-right, y-down, z-forward (matches GG/COLMAP/gsplat)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    nr = np.linalg.norm(right)
    if nr < 1e-6:
        right = np.cross(forward, np.array([0., 1., 0.]))
        nr = np.linalg.norm(right)
    right = right / nr
    down = np.cross(forward, right)
    R = np.eye(4)
    R[0, :3] = right
    R[1, :3] = down
    R[2, :3] = forward
    R[:3, 3] = R[:3, :3] @ (-eye)
    return R


def render(means, colors, opacities, scales, quats, viewmat, fovy_deg=75):
    from gsplat import rasterization
    fovy = np.radians(fovy_deg)
    fy = H / (2 * np.tan(fovy / 2))
    fx = fy
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
    vm = torch.tensor(viewmat, dtype=torch.float32, device="cuda").unsqueeze(0)
    Kt = torch.tensor(K, dtype=torch.float32, device="cuda").unsqueeze(0)
    renders, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=vm, Ks=Kt, width=W, height=H, sh_degree=0,
    )
    return (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def project(pt3d, viewmat, fovy_deg=75):
    fovy = np.radians(fovy_deg)
    fy = H / (2 * np.tan(fovy / 2))
    fx = fy
    p = viewmat[:3, :3] @ pt3d + viewmat[:3, 3]
    if p[2] <= 0:
        return None
    u = int(fx * p[0] / p[2] + W / 2)
    v = int(fy * p[1] / p[2] + H / 2)
    if 0 <= u < W and 0 <= v < H:
        return (u, v)
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames_dir = OUT / "annotated"
    frames_dir.mkdir(exist_ok=True)

    means, colors, opacities, scales, quats = load_model(PLY)

    # Use a viewpoint that we verified produces good renders
    # (camera 15 from training set: front-right elevated view)
    eye = np.array([0.183, 0.326, 0.378])
    target = np.array([0.0, 0.0, 0.08])
    viewmat = look_at(eye, target)

    # Render base scene
    print("Rendering base scene...")
    base_img = render(means, colors, opacities, scales, quats, viewmat)
    Image.fromarray(base_img).save(OUT / "scene_base.png")
    print(f"  Base: {base_img.shape}, range [{base_img.min()}, {base_img.max()}]")

    # Generate trajectory
    from src.pipeline.gs_to_grasp import extract_scene_pointcloud, detect_grasps
    from src.trajectory import TrajectoryGenerator, trajectory_to_actions

    pts, cols = extract_scene_pointcloud(PLY, max_points=100000)
    mask = (pts[:, 2] > 0.005) & (pts[:, 2] < 0.2) & \
           (pts[:, 0] > -0.2) & (pts[:, 0] < 0.2) & \
           (pts[:, 1] > -0.15) & (pts[:, 1] < 0.15)
    grasps = detect_grasps(pts[mask], cols[mask], top_k=5, collision_detection=False)
    best = grasps[0]
    place_target = np.array([0.15, -0.10, 0.025], dtype=np.float32)

    gen = TrajectoryGenerator()
    traj = gen.generate_pick_place(best, place_target)
    gen.smooth_corners(traj, radius=3)
    print(f"Trajectory: {len(traj)} waypoints, grasp={best.position.round(3)}, place={place_target}")

    # Phase colors
    pc = {
        "transit_pick": (51, 153, 255), "pre_grasp": (0, 204, 102),
        "approach": (255, 153, 0), "grasp": (255, 0, 0),
        "lift": (204, 0, 204), "transit_place": (51, 153, 255),
        "pre_place": (0, 204, 102), "place_descend": (255, 153, 0),
        "release": (255, 0, 0), "retreat": (128, 128, 128),
    }

    # Render frames with trajectory overlay
    print(f"Rendering {len(traj)} frames with trajectory overlay...")
    for i, wp in enumerate(traj.waypoints):
        img = Image.fromarray(base_img.copy())
        draw = ImageDraw.Draw(img)

        # Draw full trajectory path (dimmed)
        for j in range(1, len(traj)):
            p1 = project(traj.waypoints[j - 1].position, viewmat)
            p2 = project(traj.waypoints[j].position, viewmat)
            if p1 and p2:
                phase = traj.waypoints[j].phase
                c = pc.get(phase, (128, 128, 128))
                # Dim future path
                if j > i:
                    c = tuple(v // 3 for v in c)
                draw.line([p1, p2], fill=c, width=2)

        # Grasp target marker (red square)
        g = project(best.position, viewmat)
        if g:
            draw.rectangle([g[0] - 4, g[1] - 4, g[0] + 4, g[1] + 4], outline=(255, 50, 50), width=2)

        # Place target marker (yellow square)
        p = project(place_target, viewmat)
        if p:
            draw.rectangle([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(255, 255, 0), width=2)

        # Current EE position (large circle)
        cur = project(wp.position, viewmat)
        if cur:
            r = 7
            grip_color = (0, 255, 0) if wp.gripper > 0.5 else (255, 0, 0)
            draw.ellipse([cur[0] - r, cur[1] - r, cur[0] + r, cur[1] + r],
                         fill=grip_color, outline=(255, 255, 255), width=2)

        # HUD bar
        draw.rectangle([(0, 0), (W, 22)], fill=(0, 0, 0))
        grip_text = "OPEN" if wp.gripper > 0.5 else "CLOSED"
        color = pc.get(wp.phase, (200, 200, 200))
        draw.text((4, 4), f"Step {i}/{len(traj)-1} | {wp.phase} | gripper: {grip_text}", fill=color)

        img.save(frames_dir / f"{i:04d}.png")

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(traj)}] {wp.phase}")

    # Create video
    import subprocess
    video_path = OUT / "pick_place_trajectory.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "8",
        "-i", str(frames_dir / "%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(video_path),
    ], capture_output=True)
    print(f"\nVideo: {video_path} ({video_path.stat().st_size // 1024} KB)")
    print(f"Frames: {frames_dir}/")
    print(f"Base scene: {OUT / 'scene_base.png'}")


if __name__ == "__main__":
    main()

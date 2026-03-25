"""
Full 3DGS → SuGaR Refined Textured Mesh → ManiSkill Sim Pipeline for all 5 objects.

Steps per object:
  1. 3DGS training (Gaussian Grouping, white bg, 7K iter)
  2. SuGaR full pipeline (coarse → refine → textured mesh extraction)
  3. Crop mesh to object region, fix MTL, generate collision hull
Then:
  4. Load all into ManiSkill
  5. AnyGrasp → trajectory → execute → visualize

Usage:
    CUDA_HOME=/usr/local/cuda conda run -n gaussian_grouping python scripts/run_full_3dgs_pipeline.py
"""

import sys, os, subprocess, shutil, numpy as np, torch
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data/objects")
GG_DIR = Path("/home/bwang25/Desktop/Manipulation/gaussian-grouping")
SUGAR_DIR = Path("/home/bwang25/Desktop/Manipulation/SuGaR")
OUTPUT_DIR = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs/full_3dgs_pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTS = {
    "005_tomato_soup_can": {"bbox": (0.069, 0.069, 0.103), "place": [0.00, 0.00, 0.001], "density": 800,
                            "crop": ([-0.05, -0.05, -0.005], [0.05, 0.05, 0.12])},
    "025_mug":             {"bbox": (0.107, 0.095, 0.081), "place": [-0.10, 0.08, 0.001], "density": 500,
                            "crop": ([-0.07, -0.07, -0.005], [0.07, 0.07, 0.10])},
    "024_bowl":            {"bbox": (0.162, 0.162, 0.055), "place": [0.10, -0.08, 0.001], "density": 400,
                            "crop": ([-0.10, -0.10, -0.005], [0.10, 0.10, 0.07])},
    "011_banana":          {"bbox": (0.070, 0.199, 0.038), "place": [-0.08, -0.10, 0.001], "density": 300,
                            "crop": ([-0.12, -0.05, -0.005], [0.12, 0.05, 0.05])},
    "013_apple":           {"bbox": (0.076, 0.076, 0.072), "place": [0.10, 0.08, 0.001], "density": 600,
                            "crop": ([-0.05, -0.05, -0.005], [0.05, 0.05, 0.08])},
}
TARGET_OBJ = "005_tomato_soup_can"


def step1_train_3dgs(obj_id):
    """Train 3DGS with Gaussian Grouping."""
    print(f"\n  [3DGS] Training {obj_id}...")
    scene_dir = DATA_DIR / obj_id
    output_dir = scene_dir / "output"
    config_file = scene_dir / "config.json"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    config_file.write_text('{"num_classes": 2, "densify_until_iter": 5000}')

    env = os.environ.copy()
    env["PYTHONPATH"] = str(GG_DIR / "submodules/diff-gaussian-rasterization") + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run([
        sys.executable, "train.py",
        "-s", str(scene_dir), "-m", str(output_dir) + "/",
        "--config_file", str(config_file),
        "--iterations", "7000", "--num_classes", "2", "--white_background",
    ], cwd=str(GG_DIR), env=env, capture_output=True, text=True)

    ply = output_dir / "point_cloud/iteration_7000/point_cloud.ply"
    if ply.exists():
        print(f"  [3DGS] OK: {ply.stat().st_size // 1024} KB")
        return True
    print(f"  [3DGS] FAILED")
    return False


def step2_sugar_full(obj_id):
    """Run full SuGaR: coarse → refine → textured mesh."""
    print(f"  [SuGaR] Full pipeline for {obj_id}...")
    scene = str(DATA_DIR / obj_id)
    ckpt = str(DATA_DIR / obj_id / "output") + "/"

    # Clean old SuGaR output
    for d in ["coarse", "coarse_mesh", "refined", "refined_mesh"]:
        p = SUGAR_DIR / "output" / d / obj_id
        if p.exists():
            shutil.rmtree(p)

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(p for p in env.get("PYTHONPATH", "").split(":") if "gaussian-grouping" not in p)

    result = subprocess.run([
        sys.executable, "train.py",
        "-s", scene, "-c", ckpt, "-i", "7000", "-r", "density",
        "--high_poly", "True",
        "--export_uv_textured_mesh", "True", "--export_ply", "True",
        "--eval", "True", "--gpu", "0", "--white_background", "True",
    ], cwd=str(SUGAR_DIR), env=env, capture_output=True, text=True)

    # Find textured mesh
    mesh_dir = SUGAR_DIR / "output/refined_mesh" / obj_id
    obj_files = list(mesh_dir.glob("*.obj")) if mesh_dir.exists() else []
    if obj_files:
        print(f"  [SuGaR] Textured mesh: {obj_files[0].name} ({obj_files[0].stat().st_size // (1024*1024)} MB)")
        return obj_files[0]
    print(f"  [SuGaR] FAILED")
    if result.stderr:
        print(f"  Last error: {result.stderr[-200:]}")
    return None


def step3_crop_and_prepare(obj_id, sugar_obj_path):
    """Crop textured mesh, fix MTL, generate collision hull."""
    import open3d as o3d

    print(f"  [Crop] Processing {obj_id}...")
    spec = OBJECTS[obj_id]
    mesh_dir = DATA_DIR / obj_id / "mesh"
    mesh_dir.mkdir(exist_ok=True)

    # Copy texture PNG
    png_files = list(sugar_obj_path.parent.glob("*.png"))
    if png_files:
        texture_src = png_files[0]
        texture_dst = mesh_dir / f"{obj_id}_texture.png"
        shutil.copy2(str(texture_src), str(texture_dst))

    # Load and crop
    m = o3d.io.read_triangle_mesh(str(sugar_obj_path), enable_post_processing=True)
    cmin, cmax = spec["crop"]
    bbox = o3d.geometry.AxisAlignedBoundingBox(cmin, cmax)
    cropped = m.crop(bbox)

    # Keep largest component
    if len(cropped.triangles) > 0:
        tc, ts, _ = cropped.cluster_connected_triangles()
        tc = np.asarray(tc); ts = np.asarray(ts)
        if len(ts) > 1:
            cropped.remove_triangles_by_mask(tc != np.argmax(ts))
            cropped.remove_unreferenced_vertices()

    cropped.compute_vertex_normals()
    v = np.asarray(cropped.vertices)

    if len(v) == 0:
        print(f"  [Crop] FAILED: empty after crop")
        return False

    # Shift bottom to z=0
    v[:, 2] -= v[:, 2].min()
    cropped.vertices = o3d.utility.Vector3dVector(v)

    print(f"  [Crop] {len(v):,}v, size: {((v.max(0)-v.min(0))*1000).round(0)}mm, UV: {cropped.has_triangle_uvs()}")

    # Save visual mesh
    visual_path = mesh_dir / f"{obj_id}_visual.obj"
    o3d.io.write_triangle_mesh(str(visual_path), cropped)

    # Fix MTL to reference texture
    mtl_path = mesh_dir / f"{obj_id}_visual.mtl"
    if mtl_path.exists():
        mtl_content = mtl_path.read_text()
        if "map_Kd" not in mtl_content:
            # Find the material name
            lines = mtl_content.strip().split("\n")
            with open(str(mtl_path), "w") as f:
                for line in lines:
                    f.write(line + "\n")
                f.write(f"map_Kd {obj_id}_texture.png\n")

    # Collision hull
    hull, _ = cropped.compute_convex_hull()
    hull.compute_vertex_normals()
    collision_path = mesh_dir / f"{obj_id}_collision.obj"
    o3d.io.write_triangle_mesh(str(collision_path), hull)
    print(f"  [Crop] Collision: {len(hull.vertices)}v, watertight={hull.is_watertight()}")

    return True


def step4_sim_pipeline():
    """Load all objects, run AnyGrasp, execute trajectories."""
    import mani_skill.envs, gymnasium as gym, sapien, sapien.physx as physx
    from mani_skill.utils import sapien_utils
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("SIM PIPELINE")
    print("=" * 60)

    env = gym.make("PickCube-v1", obs_mode="rgbd", num_envs=1, sim_backend="cpu",
                    render_mode="rgb_array", control_mode="pd_joint_pos",
                    sensor_configs=dict(shader_pack="default"))
    env.reset(seed=42)
    base = env.unwrapped
    if hasattr(base, "cube"): base.cube.set_pose(sapien.Pose(p=[0,0,-10]))
    if hasattr(base, "goal_site"): base.goal_site.set_pose(sapien.Pose(p=[0,0,-10]))
    for a in base._hidden_objects: a.set_pose(sapien.Pose(p=[0,0,-10]))

    # Load objects
    actors = {}
    for obj_id, spec in OBJECTS.items():
        mesh_dir = DATA_DIR / obj_id / "mesh"
        visual = mesh_dir / f"{obj_id}_visual.obj"
        collision = mesh_dir / f"{obj_id}_collision.obj"
        if not visual.exists() or not collision.exists():
            print(f"  {obj_id}: SKIP (missing mesh)")
            continue

        builder = base.scene.create_actor_builder()
        builder.add_visual_from_file(filename=str(visual))
        phys = physx.PhysxMaterial(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        builder.add_convex_collision_from_file(filename=str(collision), material=phys, density=spec["density"])
        builder.initial_pose = sapien.Pose(p=spec["place"], q=[1, 0, 0, 0])
        builder.set_scene_idxs([0])
        actors[obj_id] = builder.build(name=obj_id)
        print(f"  {obj_id}: loaded")

    for _ in range(300):
        base.scene.step()
    for oid, a in actors.items():
        p = a.pose.p[0].cpu().numpy()
        print(f"  {oid} settled: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]")

    # Save scene
    frame = env.render()
    if isinstance(frame, torch.Tensor): frame = frame.cpu().numpy()
    if frame.ndim == 4: frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
    Image.fromarray(frame).save(OUTPUT_DIR / "initial_scene.png")
    print(f"\n  Scene: {OUTPUT_DIR / 'initial_scene.png'}")

    # Motion planner
    planner = PandaArmMotionPlanningSolver(env, debug=False, vis=False, base_pose=base.agent.robot.pose, print_env_info=False)

    # AnyGrasp
    import open3d as o3d
    target_pos = actors[TARGET_OBJ].pose.p[0].cpu().numpy()
    mesh_o3d = o3d.io.read_triangle_mesh(str(DATA_DIR / TARGET_OBJ / "mesh" / f"{TARGET_OBJ}_visual.obj"))
    mesh_o3d.compute_vertex_normals()
    pcd = mesh_o3d.sample_points_uniformly(10000)
    pts = np.asarray(pcd.points, dtype=np.float32) + target_pos.astype(np.float32)
    colors = np.full((len(pts), 3), 0.5, dtype=np.float32)

    from src.pipeline.gs_to_grasp import detect_grasps
    grasps = detect_grasps(pts, colors, top_k=10, collision_detection=False)
    if not grasps:
        print("  No grasps!"); env.close(); return
    for i, g in enumerate(grasps[:3]):
        print(f"  Grasp {i}: pos={g.position.round(3)}, score={g.score:.3f}")

    # Execute trajectories
    ee_ori = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    place_targets = [
        np.array([0.15, -0.12, 0.02], dtype=np.float32),
        np.array([-0.12, 0.12, 0.02], dtype=np.float32),
        np.array([0.05, -0.15, 0.02], dtype=np.float32),
    ]

    n_traj = min(3, len(grasps))
    fig, axes = plt.subplots(n_traj, 6, figsize=(24, 4 * n_traj))
    if n_traj == 1: axes = axes[np.newaxis, :]

    for t_idx in range(n_traj):
        grasp = grasps[t_idx]
        place = place_targets[t_idx % len(place_targets)]
        print(f"\n  Traj {t_idx+1}: grasp={grasp.position.round(3)}")

        env.reset(seed=42 + t_idx)
        if hasattr(base, "cube"): base.cube.set_pose(sapien.Pose(p=[0,0,-10]))
        if hasattr(base, "goal_site"): base.goal_site.set_pose(sapien.Pose(p=[0,0,-10]))
        for a in base._hidden_objects: a.set_pose(sapien.Pose(p=[0,0,-10]))
        for oid, spec in OBJECTS.items():
            if oid in actors:
                actors[oid].set_pose(sapien.Pose(p=spec["place"], q=[1,0,0,0]))
        for _ in range(100): base.scene.step()

        gp = grasp.position.copy()
        steps = [
            ("start", None, True), ("transit", np.array([gp[0],gp[1],0.30]), True),
            ("pre_grasp", np.array([gp[0],gp[1],gp[2]+0.10]), True),
            ("approach", gp, True), ("grasp", gp, False),
            ("lift", np.array([gp[0],gp[1],gp[2]+0.15]), False),
            ("transit_place", np.array([place[0],place[1],0.30]), False),
            ("pre_place", np.array([place[0],place[1],place[2]+0.10]), False),
            ("place", place.copy(), False), ("release", place.copy(), True),
            ("retreat", np.array([place[0],place[1],place[2]+0.15]), True),
        ]

        frames, labels = [], []
        for name, pos, grip_open in steps:
            if pos is not None:
                tp = np.concatenate([pos.astype(np.float64), ee_ori])
                qpos = base.agent.robot.get_qpos().cpu().numpy()[0]
                res = planner.planner.plan_screw(tp, qpos, time_step=0.1)
                if res["status"] != "Success":
                    res = planner.planner.plan_qpos_to_pose(tp, qpos, time_step=0.1, wrt_world=True)
                fq = res["position"][-1] if res["status"] == "Success" else qpos[:7]
                action = np.zeros(8, dtype=np.float32)
                action[:7] = fq[:7]
                action[7] = 1.0 if grip_open else -1.0
                for _ in range(20): env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))

            f = env.render()
            if isinstance(f, torch.Tensor): f = f.cpu().numpy()
            if f.ndim == 4: f = f[0]
            frames.append(f.copy()); labels.append(name)

        # Save video
        fdir = OUTPUT_DIR / f"traj_{t_idx}_frames"; fdir.mkdir(exist_ok=True)
        pc = {"start":(200,200,200),"transit":(51,153,255),"pre_grasp":(0,204,102),
              "approach":(255,153,0),"grasp":(255,0,0),"lift":(204,0,204),
              "transit_place":(51,153,255),"pre_place":(0,204,102),"place":(255,153,0),
              "release":(255,0,0),"retreat":(128,128,128)}
        for fi, (frame, label) in enumerate(zip(frames, labels)):
            if frame.dtype != np.uint8:
                frame = (np.clip(frame,0,1)*255).astype(np.uint8) if frame.max()<=1.0 else frame.astype(np.uint8)
            img = Image.fromarray(frame); draw = ImageDraw.Draw(img)
            draw.rectangle([(0,0),(img.width,24)], fill=(0,0,0))
            draw.text((4,4), f"Step {fi}/{len(frames)-1} | {label}", fill=pc.get(label,(200,200,200)))
            img.save(fdir / f"{fi:04d}.png")
        subprocess.run(["ffmpeg","-y","-framerate","2","-i",str(fdir/"%04d.png"),
                        "-c:v","libx264","-pix_fmt","yuv420p",str(OUTPUT_DIR/f"traj_{t_idx}.mp4")], capture_output=True)

        key_idx = [0,2,4,5,8,10] if len(frames)>=11 else list(range(min(6,len(frames))))
        for col, fi in enumerate(key_idx[:6]):
            frame = frames[fi]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame,0,1)*255).astype(np.uint8) if frame.max()<=1.0 else frame.astype(np.uint8)
            axes[t_idx,col].imshow(frame); axes[t_idx,col].set_title(labels[fi],fontsize=10); axes[t_idx,col].axis("off")
        axes[t_idx,0].set_ylabel(f"Traj {t_idx+1}\nscore={grasp.score:.2f}", fontsize=11)

    plt.suptitle(f"3DGS → SuGaR Textured Mesh Pipeline: {TARGET_OBJ}", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR/"trajectory_overview.png"), dpi=120); plt.close()
    print(f"\n  Overview: {OUTPUT_DIR/'trajectory_overview.png'}")
    env.close()


def main():
    print("=" * 60)
    print("FULL 3DGS → TEXTURED MESH → SIM PIPELINE")
    print("=" * 60)

    # Steps 1-3 for each object
    for obj_id in OBJECTS:
        print(f"\n{'='*50}")
        print(f"  {obj_id}")
        print(f"{'='*50}")

        # Check if 3DGS already trained
        ply = DATA_DIR / obj_id / "output/point_cloud/iteration_7000/point_cloud.ply"
        if not ply.exists():
            ok = step1_train_3dgs(obj_id)
            if not ok: continue
        else:
            print(f"  [3DGS] Already trained")

        # Check if SuGaR textured mesh exists
        mesh_dir = SUGAR_DIR / "output/refined_mesh" / obj_id
        obj_files = list(mesh_dir.glob("*.obj")) if mesh_dir.exists() else []
        if not obj_files:
            sugar_path = step2_sugar_full(obj_id)
            if sugar_path is None: continue
        else:
            sugar_path = obj_files[0]
            print(f"  [SuGaR] Already done: {sugar_path.name}")

        # Crop and prepare
        step3_crop_and_prepare(obj_id, sugar_path)

    # Step 4: Sim pipeline
    step4_sim_pipeline()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Results: {OUTPUT_DIR}/")
    print(f"  initial_scene.png, trajectory_overview.png")
    print(f"  traj_0.mp4, traj_1.mp4, traj_2.mp4")


if __name__ == "__main__":
    main()

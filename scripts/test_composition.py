"""Test compositional scene: empty table + objects extracted by tight bbox."""
import sys, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline.compositional_scene import CompositionalScene, load_gaussian_cluster_from_ply, GaussianCluster

DATA = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/data")
OUT = Path("/home/bwang25/Desktop/Manipulation/Evolving_Environment/logs/compositional_test")
OUT.mkdir(parents=True, exist_ok=True)

# Load empty table
env_cluster = load_gaussian_cluster_from_ply(
    str(DATA / "empty_table/output/point_cloud/iteration_7000/point_cloud.ply"),
    max_points=999999999,
)
print(f"Environment: {env_cluster.n:,} Gaussians")
scene = CompositionalScene(env_cluster)

# Object bboxes and placements
obj_specs = {
    "005_tomato_soup_can": {"bbox": (0.069, 0.069, 0.103), "place": [0.0, 0.0, 0.05]},
    "025_mug":             {"bbox": (0.107, 0.095, 0.081), "place": [-0.10, 0.08, 0.04]},
    "024_bowl":            {"bbox": (0.162, 0.162, 0.055), "place": [0.10, -0.08, 0.028]},
    "011_banana":          {"bbox": (0.070, 0.199, 0.038), "place": [-0.10, -0.08, 0.019]},
    "013_apple":           {"bbox": (0.076, 0.076, 0.072), "place": [0.12, 0.06, 0.035]},
}

for obj_id, spec in obj_specs.items():
    ply_path = DATA / f"objects/{obj_id}/output/point_cloud/iteration_7000/point_cloud.ply"
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
        positions=full.positions[mask],
        quaternions=full.quaternions[mask],
        scales=full.scales[mask],
        opacities=full.opacities[mask],
        sh_dc=full.sh_dc[mask],
    )
    print(f"  {obj_id}: {full.n:,} → {obj_cluster.n:,} (bbox filter)")
    scene.add_object(obj_id, obj_cluster, position=np.array(spec["place"], dtype=np.float32))

print(f"\n{scene.summary()}")

# Render
from gsplat import rasterization

means, colors, opacities, scales, quats = scene.get_render_tensors("cuda")

eye = np.array([0.183, 0.326, 0.378])
target = np.array([0.0, 0.0, 0.08])
forward = target - eye
forward /= np.linalg.norm(forward)
up = np.array([0., 0., 1.])
right = np.cross(forward, up)
right /= np.linalg.norm(right)
down = np.cross(forward, right)
R = np.eye(4)
R[0, :3] = right
R[1, :3] = down
R[2, :3] = forward
R[:3, 3] = R[:3, :3] @ (-eye)

W = H = 512
fovy = np.radians(75)
fy = H / (2 * np.tan(fovy / 2))
fx = fy
K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])

vm = torch.tensor(R, dtype=torch.float32, device="cuda").unsqueeze(0)
Kt = torch.tensor(K, dtype=torch.float32, device="cuda").unsqueeze(0)
renders, _, _ = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities,
    colors=colors, viewmats=vm, Ks=Kt, width=W, height=H, sh_degree=0,
)
img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(img).save(OUT / "composed_bbox_v2.png")
print(f"\nSaved: {OUT / 'composed_bbox_v2.png'}")
print(f"Image: mean={img.mean():.0f}, range=[{img.min()},{img.max()}]")

"""Reproduce the IMG_0365 desk plane DETERMINISTICALLY (np.random.default_rng(0), same as
deskleak_cross_0365.py) and PERSIST explicit n + off (and the bi inlier set) to a permanent path.
Same seed -> identical inliers/plane as the earlier Case-B analysis, NOT a fresh independent fit."""
import os, numpy as np, torch
from plyfile import PlyData

MODEL="output/mouse_apple_0365/point_cloud/iteration_30000"
OUT="output/mouse_apple_0365/desk_plane.npy"

d=PlyData.read(f"{MODEL}/point_cloud.ply")["vertex"].data; N=len(d)
xyz=np.stack([d["x"],d["y"],d["z"]],1).astype(np.float64)
obj=np.stack([d[f"obj_dc_{i}"] for i in range(16)],1).astype(np.float32)
clf=torch.load(f"{MODEL}/classifier.pth",map_location="cpu")
W=clf["weight"].squeeze(-1).squeeze(-1).numpy(); b=clf["bias"].numpy()
ids=(obj@W.T+b).argmax(1)

# --- identical RANSAC to deskleak_cross_0365.py: rng(0), 2000 iters, tol=0.005*diag ---
rng=np.random.default_rng(0); diag=np.linalg.norm(xyz.max(0)-xyz.min(0)); tol=0.005*diag; bi=None
for _ in range(2000):
    p=xyz[rng.choice(N,3,replace=False)]; n=np.cross(p[1]-p[0],p[2]-p[0]); nn=np.linalg.norm(n)
    if nn<1e-9: continue
    n/=nn; off=-n@p[0]; inl=np.abs(xyz@n+off)<tol
    if bi is None or inl.sum()>bi.sum(): bi=inl
# refit plane on inliers (SVD) -> the n/off actually used downstream
Pin=xyz[bi]; c0=Pin.mean(0); _,_,Vt=np.linalg.svd(Pin-c0,full_matrices=False); n=Vt[-1]; off=-n@c0
# orient n toward objects (above desk), matching the convention used in analysis
objmask=ids>0
if np.median(xyz[objmask]@n+off) < np.median(xyz[bi]@n+off): n,off=-n,-off
n=n/np.linalg.norm(n)

# --- sanity checks ---
print("=== regenerated desk plane (deterministic rng(0)) ===")
print(f"  inlier count = {int(bi.sum())} ({100*bi.sum()/N:.1f}% of {N})   [earlier Case-B analysis: 417,872]")
print(f"  n   = {np.round(n,6).tolist()}")
print(f"  off = {off:.6f}")
# n should be roughly vertical (its dominant component is the table 'up'); print angle to each world axis
for ax,name in zip(np.eye(3),["X","Y","Z"]):
    print(f"  angle(n, world {name}) = {np.degrees(np.arccos(np.clip(abs(n@ax),0,1))):.1f} deg")
obj_h=np.median(xyz[objmask]@n+off); desk_h=np.median(xyz[bi]@n+off)
print(f"  objects median height above plane = {obj_h-desk_h:.4f} (should be >0: objects sit above desk) -> {'OK' if obj_h>desk_h else 'BAD'}")

assert int(bi.sum())==417872, f"inlier count {int(bi.sum())} != earlier 417,872 -> regenerated fit DIFFERS, STOP"
print("  SANITY: inlier count matches the earlier analysis exactly (417,872) -> same deterministic plane")

np.save(OUT, dict(n=n, off=float(off), bi=bi, inlier_count=int(bi.sum()), seed=0, tol_frac=0.005), allow_pickle=True)
print(f"\nPERSISTED -> {OUT}  (explicit n + off + bi inlier set)")

"""STEP 1: diagnose mouse_0323_segmented.ply before cleaning. Read-only. Print percentiles for
flatness (sharp flakes), opacity + scale (soft blobs), off-core distance (both junk types), color
(desk leakage). Decide separability of each junk type vs the dense mouse body."""
import numpy as np
from plyfile import PlyData

SRC = "output/mouse_gg_0323/mouse_0323_segmented.ply"
d = PlyData.read(SRC)["vertex"].data
N = len(d)
xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
scales = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1).astype(np.float64))
opacity = 1.0/(1.0+np.exp(-d["opacity"].astype(np.float64)))
C0 = 0.28209479177387814
rgb = np.clip(C0*np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1) + 0.5, 0, 1)

ssort = np.sort(scales, 1)
smin, smax = ssort[:, 0], ssort[:, 2]
flat = smin/np.maximum(smax, 1e-12)              # sharp flake -> near 0

# dense mouse core: robust center via median, core = densest cluster by kNN distance
# (chunked kNN to keep memory sane for ~82k pts)
K = 16
M = len(xyz)
d_k = np.empty(M)
B = 2048
for i in range(0, M, B):
    chunk = xyz[i:i+B]
    dd = np.sqrt(np.maximum(((chunk[:, None, :] - xyz[None, :, :])**2).sum(2), 0.0))
    part = np.partition(dd, K, axis=1)[:, 1:K+1]
    d_k[i:i+B] = part.mean(1)
core = d_k <= np.percentile(d_k, 60)             # densest 60% = mouse body
core_center = np.median(xyz[core], 0)
# off-core distance = distance to nearest core gaussian (0 if in a dense region)
off = np.empty(M)
core_xyz = xyz[core]
for i in range(0, M, B):
    chunk = xyz[i:i+B]
    dd = np.sqrt(np.maximum(((chunk[:, None, :] - core_xyz[None, :, :])**2).sum(2), 0.0))
    off[i:i+B] = dd.min(1)

warm = rgb[:, 0] - rgb[:, 2]
gold = (rgb[:, 0] > 0.40) & (warm > 0.12) & (rgb[:, 1] > rgb[:, 2])

def pct(name, a, fmt="{:.4f}"):
    print(f"  {name:20s} " + "  ".join(f"p{p}=" + fmt.format(np.percentile(a, p)) for p in [50, 90, 95, 99, 100]))

print(f"SRC (read-only): {SRC}\ntotal Gaussians: {N}\n")
print("FLATNESS smin/smax (sharp flake -> ~0):"); pct("flat", flat)
print("OPACITY (soft haze -> low):"); pct("opacity", opacity)
print("OFF-CORE distance (junk -> high; body ~0):"); pct("off_core", off)
print("SCALE max-axis (soft blobs -> large):"); pct("max_scale", smax)
print(f"COLOR: gold/desk-tinted = {int(gold.sum())} ({100*gold.sum()/N:.1f}%)")
print("\nseparability cross-tab (off-core is required for removal):")
offhi = off > np.percentile(off, 90)
print(f"  off-core(>p90 ={np.percentile(off,90):.4f}): {int(offhi.sum())}")
fl = flat < np.percentile(flat, 50)
print(f"  PASS-A cand  flat<{np.percentile(flat,50):.4f} & off-core>p90 : {int((fl&offhi).sum())}")
lo = opacity < np.percentile(opacity, 25)
lg = smax > np.percentile(smax, 90)
print(f"  PASS-B cand  (opacity<{np.percentile(opacity,25):.3f} | scale>{np.percentile(smax,90):.3f}) & off-core>p90 : {int(((lo|lg)&offhi).sum())}")
print(f"  gold & off-core>p90 : {int((gold&offhi).sum())}")
for lab, m in [("CORE body", core), ("off-core>p90", offhi)]:
    print(f"  [{lab:13s}] n={int(m.sum()):6d}  flat_med={np.median(flat[m]):.3f}  op_med={np.median(opacity[m]):.3f}  "
          f"scale_med={np.median(smax[m]):.3f}  off_med={np.median(off[m]):.4f}")
np.save("/tmp/diag0323.npy", dict(flat=flat, opacity=opacity, off=off, smax=smax, gold=gold, core=core), allow_pickle=True)
print("\nsaved arrays -> /tmp/diag0323.npy")

"""STEP 2: two-pass clean of mouse_0323_segmented.ply. Reads original READ-ONLY, writes only
mouse_0323_clean.ply. OFF-CORE is required for any removal -> the dense mouse body is never touched.
PASS A = sharp flakes (flat & off-core). PASS B = soft blobs (low-opacity|large-scale, & off-core),
+ optional desk-colored off-core. Thresholds from diag_0323_clean.py percentiles."""
import numpy as np
from plyfile import PlyData, PlyElement

SRC = "output/mouse_gg_0323/mouse_0323_segmented.ply"
DST = "output/mouse_gg_0323/mouse_0323_clean.ply"
a = np.load("/tmp/diag0323.npy", allow_pickle=True).item()
flat, opacity, off, smax, gold, core = a["flat"], a["opacity"], a["off"], a["smax"], a["gold"], a["core"]

d = PlyData.read(SRC)["vertex"].data
N = len(d)

# thresholds (from printed distributions)
T_off   = np.percentile(off, 90)     # 0.472  off-core gate (REQUIRED for any removal)
T_flat  = 0.05                       # flake platelet (body median 0.024 protected by off-gate anyway)
T_op    = 0.01                       # near-invisible haze (p25 ~0.006)
T_scale = 0.10                       # clearly large blob (p95=0.090, p99=0.186)

offcore = off > T_off
passA = offcore & (flat < T_flat)                          # sharp flakes
passB = offcore & ((opacity < T_op) | (smax > T_scale))    # soft haze blobs
passC = offcore & gold                                     # desk-colored off-core (optional)
remove = passA | passB | passC
keep = ~remove

print(f"SRC (read-only): {SRC}\ntotal: {N}")
print(f"thresholds: off-core>{T_off:.3f} (REQUIRED)  flat<{T_flat}  opacity<{T_op}  scale>{T_scale}\n")
print(f"  PASS A sharp flakes (flat & off-core)            : {int(passA.sum()):5d} ({100*passA.sum()/N:.2f}%)")
print(f"  PASS B soft blobs (lowop|largescale & off-core)  : {int(passB.sum()):5d} ({100*passB.sum()/N:.2f}%)")
print(f"  PASS C desk-color off-core (optional)            : {int(passC.sum()):5d} ({100*passC.sum()/N:.2f}%)")
print(f"  A&B overlap                                      : {int((passA&passB).sum()):5d}")
print(f"  TOTAL removed                                    : {int(remove.sum()):5d} ({100*remove.sum()/N:.2f}%)")
print(f"  remaining                                        : {int(keep.sum())}")

# SAFETY: how much of the dense CORE body did we remove? (must be ~0)
core_removed = int((remove & core).sum())
print(f"\nSAFETY: dense-core Gaussians removed = {core_removed} ({100*core_removed/int(core.sum()):.3f}% of core)")
if core_removed/max(int(core.sum()),1) > 0.01:
    print("  ** WARNING: removing >1% of the dense core — STOP and reconsider **")
else:
    print("  OK: body protected (off-core gate held).")

# write clean copy (contiguous rebuild to avoid plyfile view quirk)
kept = d[keep]
new = np.empty(len(kept), dtype=[(n, d.dtype[n]) for n in d.dtype.names])
for n in d.dtype.names: new[n] = kept[n]
PlyData([PlyElement.describe(new, "vertex")], text=False).write(DST)
# viewer ply (standard 3DGS fields only)
std = [n for n in d.dtype.names if n and not n.startswith("obj_dc")]
nv = np.empty(len(kept), dtype=[(n, d.dtype[n]) for n in std])
for n in std: nv[n] = kept[n]
PlyData([PlyElement.describe(nv, "vertex")], text=False).write(DST.replace(".ply", "_viewer.ply"))
print(f"\nsaved -> {DST}  (+ _viewer.ply)   original untouched")

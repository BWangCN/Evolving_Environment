"""Inpaint the VACATED SPOT (hole + contact-shadow RING) of a moved object. Corrected version:
 - MEASURES the contact-shadow extent (radial desk-luminance profile) instead of guessing a radius;
 - removes ALL desk(label0) Gaussians within that measured region (footprint + shadow ring);
 - fills with a DENSE synthesized patch of clean-desk Gaussians (copy attributes of nearby clean desk
   onto a grid on the plane) so it blends to the white table with no gaps/faint patch.
Reusable mouse|apple. Persisted plane (STOP if missing). Inputs untouched."""
import os, argparse, numpy as np, torch
from plyfile import PlyData, PlyElement

GG="/home/ravi/Desktop/Ravi/gaussian-grouping"; MODEL=f"{GG}/output/mouse_apple_0365/point_cloud/iteration_30000"
ap=argparse.ArgumentParser()
ap.add_argument("--object",choices=["mouse","apple"],required=True)
ap.add_argument("--scene",required=True,help="moved scene ply to inpaint")
ap.add_argument("--seg",default=None); ap.add_argument("--classifier",default=f"{MODEL}/classifier.pth")
ap.add_argument("--plane",default=f"{GG}/output/mouse_apple_0365/desk_plane.npy")
ap.add_argument("--margin",type=float,default=0.15,help="extra fraction of R added beyond measured shadow")
ap.add_argument("--out",required=True)
args=ap.parse_args()
LABEL=1 if args.object=="mouse" else 2
SEG=args.seg or f"{GG}/output/mouse_apple_0365/{args.object}_0365_segmented.ply"
if not os.path.exists(args.plane): raise SystemExit(f"STOP: persisted plane missing {args.plane}; run persist_desk_plane_0365.py")
pl=np.load(args.plane,allow_pickle=True).item(); n=np.asarray(pl["n"],float); n/=np.linalg.norm(n); off=float(pl["off"])
a0=np.array([1.,0,0]) if abs(n[0])<0.9 else np.array([0,1.,0]); u=np.cross(n,a0); u/=np.linalg.norm(u); v=np.cross(n,u)

d=PlyData.read(args.scene)["vertex"].data; N=len(d); names=list(d.dtype.names)
xyz=np.stack([d["x"],d["y"],d["z"]],1).astype(np.float64)
C0=0.28209479177387814; rgb=np.clip(C0*np.stack([d["f_dc_0"],d["f_dc_1"],d["f_dc_2"]],1)+0.5,0,1)
lum=0.299*rgb[:,0]+0.587*rgb[:,1]+0.114*rgb[:,2]
obj=np.stack([d[f"obj_dc_{i}"] for i in range(16)],1).astype(np.float32)
clf=torch.load(args.classifier,map_location="cpu"); W=clf["weight"].squeeze(-1).squeeze(-1).numpy(); b=clf["bias"].numpy()
ids=(obj@W.T+b).argmax(1)
U,V,h=xyz@u,xyz@v,xyz@n+off
desk=ids==0; desk_h=np.median(h[desk]); deskb=desk&(np.abs(h-desk_h)<0.5)

seg=PlyData.read(SEG)["vertex"].data; sp=np.stack([seg["x"],seg["y"],seg["z"]],1).astype(np.float64)
sU,sV=sp@u,sp@v; cU,cV=np.median(sU),np.median(sV); R=np.percentile(np.hypot(sU-cU,sV-cV),85)
rad=np.hypot(U-cU,V-cV)

# --- MEASURE shadow extent: radial luminance recovery to clean baseline ---
baseline=np.median(lum[deskb&(rad>3*R)&(rad<6*R)])
shadow_to=R
for frac in np.arange(0.6,3.01,0.2):
    lo,hi=frac*R,(frac+0.2)*R; m=deskb&(rad>=lo)&(rad<hi)
    if m.sum()>20 and np.median(lum[m])<baseline-0.05: shadow_to=hi
R_region=shadow_to+args.margin*R
print(f"object={args.object} footprint R={R:.4f} clean baseline lum={baseline:.3f}")
print(f"MEASURED shadow extends to {shadow_to:.4f} ({shadow_to/R:.2f}R) -> inpaint region radius={R_region:.4f} ({R_region/R:.2f}R)")

# --- SHAPE-AWARE removal: circular region OR any darkened desk (catches elongated shadow crescent) ---
shadow_dark=deskb&(rad<3.2*R)&(lum<baseline-0.035)           # catch fainter/elongated penumbra
remove=deskb&((rad<R_region)|shadow_dark)
Rb=float(np.percentile(rad[remove],99)) if remove.any() else R_region   # robust boundary (ignore stray far)
remove=remove&(rad<=Rb+1e-6)                                 # keep removal within the fillable disc
print(f"removing {int(remove.sum())} desk gaussians (circular {R_region:.3f} + darkened crescent), boundary Rb={Rb:.3f}")

# --- LOCAL-BRIGHTNESS-MATCHED fill: reflect the IMMEDIATE surrounding desk inward across the boundary.
# Source = clean desk in annulus [Rb, 2Rb]; reflect r_out->(2Rb-r_out) so r_out=Rb->edge, r_out=2Rb->center.
# This carries the LOCAL brightness ramp + view-dependent SH continuously into the hole (no bright step).
src=deskb&(rad>=Rb)&(rad<2*Rb)&(lum>baseline-0.06)
si=np.where(src)[0]
# density match: annulus[Rb,2Rb] area=3x disc -> subsample to ~1/3 so opacity doesn't over-accumulate
rng=np.random.default_rng(0); keepfrac=(Rb**2)/((2*Rb)**2-Rb**2)   # disc / annulus = 1/3
si=si[rng.random(len(si))<keepfrac]
M=len(si); assert M>40,f"too few local source desk gaussians ({M})"
th=np.arctan2(V[si]-cV,U[si]-cU); r_out=rad[si]; r_in=2*Rb-r_out      # reflect inward
newU=cU+r_in*np.cos(th); newV=cV+r_in*np.sin(th)
dU=newU-U[si]; dV=newV-V[si]
fill=np.array(d[si])
fill["x"]=fill["x"]+dU*u[0]+dV*v[0]; fill["y"]=fill["y"]+dU*u[1]+dV*v[1]; fill["z"]=fill["z"]+dU*u[2]+dV*v[2]
print(f"fill: reflected {M} LOCAL surrounding-desk gaussians inward (keep SH; local brightness ramp matched)")

kept=d[~remove]; merged=np.concatenate([kept,fill])
print(f"scene {N} - removed {int(remove.sum())} + filled {M} = {len(merged)}")
new=np.empty(len(merged),dtype=[(k,merged.dtype[k]) for k in names])
for k in names: new[k]=merged[k]
PlyData([PlyElement.describe(new,"vertex")],text=False).write(args.out)
std=[k for k in names if k and not k.startswith("obj_dc")]
mv=np.empty(len(merged),dtype=[(k,merged.dtype[k]) for k in std])
for k in std: mv[k]=merged[k]
PlyData([PlyElement.describe(mv,"vertex")],text=False).write(args.out.replace(".ply","_viewer.ply"))
print(f"saved -> {args.out} (+ _viewer.ply)")

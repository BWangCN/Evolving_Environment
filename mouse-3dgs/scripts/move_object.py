"""RoboSplat-style rigid in-plane move of ONE object (mouse|apple) in the IMG_0365 scene, kept ON the
table, merged back into the full scene. Reusable via CLI. Does NOT overwrite originals.

Plane: REUSES the saved desk-plane inlier set from the earlier IMG_0365 RANSAC (/tmp/leak0365.npy 'bi'),
deriving n/off by SVD on those exact inliers (no fresh RANSAC).

Transform correctness:
 - positions: full rigid transform (yaw about table normal through the object centroid + in-plane offset
   + vertical re-seat onto the plane).
 - rotations (quaternions): composed with the yaw rotation (identity if yaw=0).
 - SH (f_rest): rotated by the SAME rotation via a sampling-based per-band real-SH rotation that matches
   3DGS's SH basis exactly (identity if yaw=0). f_dc (DC band) is rotation-invariant -> unchanged.
   A self-test (round-trip) is printed. scale/opacity/obj_dc unchanged.
"""
import os, argparse, numpy as np
from plyfile import PlyData, PlyElement
import torch

GG="/home/ravi/Desktop/Ravi/gaussian-grouping"
MODEL=f"{GG}/output/mouse_apple_0365/point_cloud/iteration_30000"

ap=argparse.ArgumentParser()
ap.add_argument("--object",choices=["mouse","apple"],required=True)
ap.add_argument("--dx",type=float,default=None,help="in-plane offset along u (scene units); default=auto visible")
ap.add_argument("--dy",type=float,default=0.0,help="in-plane offset along v")
ap.add_argument("--yaw",type=float,default=0.0,help="yaw degrees about table normal")
ap.add_argument("--reduce_sh_to_dc",dest="reduce_sh_to_dc",action="store_true",default=True,
                help="zero SH deg1-3 of the MOVED object only (default ON: translated splats can't keep valid higher-SH near cameras)")
ap.add_argument("--keep_full_sh",dest="reduce_sh_to_dc",action="store_false",
                help="keep full SH on the moved object (renders wrong color at off-trained view dirs)")
ap.add_argument("--full",default=f"{MODEL}/point_cloud.ply")
ap.add_argument("--classifier",default=f"{MODEL}/classifier.pth")
ap.add_argument("--seg",default=None,help="object segmented ply (default derived from object)")
ap.add_argument("--plane",default=f"{GG}/output/mouse_apple_0365/desk_plane.npy")
ap.add_argument("--cameras",default=f"{GG}/output/mouse_apple_0365/cameras.json")
ap.add_argument("--out",default=None)
args=ap.parse_args()
LABEL=1 if args.object=="mouse" else 2
SEG=args.seg or f"{GG}/output/mouse_apple_0365/{args.object}_0365_segmented.ply"
OUT=args.out or f"{GG}/output/mouse_apple_0365/scene_0365_moved_{args.object}.ply"

# ---------- desk plane: LOAD persisted explicit n/off (NO re-fit; STOP if missing) ----------
if not os.path.exists(args.plane):
    raise SystemExit(f"STOP: persisted desk plane not found at {args.plane}.\n"
                     f"  Re-run: python persist_desk_plane_0365.py  (deterministic rng(0), reproduces the same plane).\n"
                     f"  NOT re-fitting a fresh plane (would shift the table frame).")
pl=np.load(args.plane,allow_pickle=True).item()
assert "n" in pl and "off" in pl, f"{args.plane} missing explicit n/off"
n=np.asarray(pl["n"],float); n=n/np.linalg.norm(n); off=float(pl["off"]); bi=pl.get("bi",None)
full=PlyData.read(args.full)["vertex"].data; Nfull=len(full)
xyz_full=np.stack([full["x"],full["y"],full["z"]],1).astype(np.float64)
clf=torch.load(args.classifier,map_location="cpu")
W=clf["weight"].squeeze(-1).squeeze(-1).numpy(); b=clf["bias"].numpy()
obj_full=np.stack([full[f"obj_dc_{i}"] for i in range(16)],1).astype(np.float32)
ids_full=(obj_full@W.T+b).argmax(1)
a0=np.array([1.,0,0]) if abs(n[0])<0.9 else np.array([0,1.,0]); u=np.cross(n,a0); u/=np.linalg.norm(u); v=np.cross(n,u)
print(f"LOADED persisted desk plane: n={np.round(n,5).tolist()} off={off:.5f} inliers={pl.get('inlier_count','?')}")

# ---------- load object ----------
seg=PlyData.read(SEG)["vertex"].data; Nobj=len(seg)
names=list(seg.dtype.names)
nlabel=int((ids_full==LABEL).sum())
print(f"object={args.object} (label {LABEL})  seg gaussians={Nobj}  full-scene label-{LABEL} gaussians={nlabel}  match={Nobj==nlabel}")
pos=np.stack([seg["x"],seg["y"],seg["z"]],1).astype(np.float64)

# default offset: CAMERA-AWARE. Translating flat-platelet Gaussians changes the view-angle onto them;
# a large move near close cameras tips them edge-on -> semi-transparent/glassy. So cap the implied
# angular change (~7 deg) -> dx ~ 0.12 * median camera distance to the object. Stays opaque + on-table.
objc=np.median(xyz_full[ids_full==LABEL],0)
import json
_cams=json.load(open(args.cameras))
_camdist=np.median([np.linalg.norm(np.array(c["position"])-objc) for c in _cams])
if args.dx is None:
    args.dx=float(0.12*_camdist)
    print(f"auto dx = 0.12 * median cam-dist({_camdist:.2f}) = {args.dx:.4f}  (keeps view-angle change small -> object stays opaque)")
ang=np.degrees(np.arctan2(abs(args.dx)+abs(args.dy),_camdist))
print(f"implied view-angle change from move: ~{ang:.1f} deg (>~15 deg risks the flat-gaussian glassy artifact)")
theta=np.radians(args.yaw)

# ---------- rotation about n (Rodrigues) ----------
def rot_about_axis(axis,th):
    a=axis/np.linalg.norm(axis); K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K)
R=rot_about_axis(n,theta)
centroid=pos.mean(0)

# ---------- positions: yaw about centroid -> in-plane offset -> reseat ----------
p1=(R@(pos-centroid).T).T+centroid
p2=p1+args.dx*u+args.dy*v
base_before=np.percentile(pos@n+off,1)
base_after =np.percentile(p2@n+off,1)
# GROUND to the LOCAL desk height at the DESTINATION (the table is not perfectly planar; preserving
# the object's own base height makes it sink/float where the local surface differs). Fall back to
# own-base if no local desk found.
Uf,Vf,hf=xyz_full@u,xyz_full@v,xyz_full@n+off
destU,destV=np.median(p2@u),np.median(p2@v)
Rfoot=np.percentile(np.hypot((pos@u)-np.median(pos@u),(pos@v)-np.median(pos@v)),85)
desk=ids_full==0; desk_h_all=np.median(hf[desk])
near=desk&(np.hypot(Uf-destU,Vf-destV)<1.5*Rfoot)&(np.abs(hf-desk_h_all)<0.5)
if int(near.sum())>20:
    local_h=float(np.median(hf[near])); dz=local_h-base_after; mode=f"local desk @dest (h={local_h:.4f}, {int(near.sum())} desk pts)"
else:
    dz=base_before-base_after; mode="own base (no local desk at dest)"
p3=p2+dz*n
base_final=np.percentile(p3@n+off,1)
print(f"\nRE-SEAT: grounded to {mode}")
print(f"  base_before(own)={base_before:.4f}  base_after_move={base_after:.4f}  dz={dz:+.4f}  base_final={base_final:.4f}")

# ---------- rotations (quaternions, w,x,y,z) ----------
def quat_mul(q1,q2):  # Hamilton, (w,x,y,z)
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],1)
quats=np.stack([seg["rot_0"],seg["rot_1"],seg["rot_2"],seg["rot_3"]],1).astype(np.float64)
if abs(theta)>1e-9:
    qy=np.array([np.cos(theta/2),*(np.sin(theta/2)*n)])  # rotation quaternion about n
    new_q=quat_mul(qy,quats); new_q/=np.linalg.norm(new_q,axis=1,keepdims=True)
    print("quaternions: rotated by yaw (composed) and renormalized")
else:
    new_q=quats; print("quaternions: yaw=0 -> unchanged (exact)")

# ---------- SH rotation (sampling-based, matches 3DGS basis); identity if yaw=0 ----------
C1=0.4886025119029199
C2=[1.0925484305920792,-1.0925484305920792,0.31539156525252005,-1.0925484305920792,0.5462742152960396]
C3=[-0.5900435899266435,2.890611442640554,-0.4570457994644658,0.3731763325901154,-0.4570457994644658,1.445305721320277,-0.5900435899266435]
def sh_basis_band(dirs,l):  # (M,3) unit dirs -> (M,2l+1) 3DGS real-SH basis values
    x,y,z=dirs[:,0],dirs[:,1],dirs[:,2]; xx,yy,zz=x*x,y*y,z*z
    if l==1: return np.stack([C1*(-y),C1*(z),C1*(-x)],1)
    if l==2: return np.stack([C2[0]*x*y,C2[1]*y*z,C2[2]*(2*zz-xx-yy),C2[3]*x*z,C2[4]*(xx-yy)],1)
    if l==3: return np.stack([C3[0]*y*(3*xx-yy),C3[1]*x*y*z,C3[2]*y*(4*zz-xx-yy),C3[3]*z*(2*zz-3*xx-3*yy),
                              C3[4]*x*(4*zz-xx-yy),C3[5]*z*(xx-yy),C3[6]*x*(xx-3*yy)],1)
def band_rot_matrix(R,l):   # D such that c' = D @ c  (rotate function by R)
    rng=np.random.default_rng(0); s=rng.standard_normal((2*l+1,3)); s/=np.linalg.norm(s,axis=1,keepdims=True)
    Pm=sh_basis_band(s,l); Qm=sh_basis_band((R.T@s.T).T,l)  # Y_m(R^{-1} s): R^{-1}=R.T
    return np.linalg.solve(Pm,Qm).T   # c' = (P^{-1}Q) c
# f_rest layout: channel-major 15 each: ch*15 + coeff; coeff blocks l1[0:3],l2[3:8],l3[8:15]
rest_names=[f"f_rest_{i}" for i in range(45)]
have_rest=all(rn in names for rn in rest_names)
frest=np.stack([seg[rn] for rn in rest_names],1).astype(np.float64) if have_rest else None
if abs(theta)>1e-9 and have_rest:
    D1,D2,D3=band_rot_matrix(R,1),band_rot_matrix(R,2),band_rot_matrix(R,3)
    # self-test: round-trip R then R^{-1} ~ identity
    rt=band_rot_matrix(R.T,1)@D1
    print(f"SH rotation: sampling-based, EXACT to 3DGS basis. self-test (l=1 round-trip max|.-I|)={np.abs(rt-np.eye(3)).max():.2e}")
    new_rest=frest.copy()
    for ch in range(3):
        blk=frest[:,ch*15:(ch+1)*15]
        blk[:,0:3]=blk[:,0:3]@D1.T; blk[:,3:8]=blk[:,3:8]@D2.T; blk[:,8:15]=blk[:,8:15]@D3.T
        new_rest[:,ch*15:(ch+1)*15]=blk
else:
    new_rest=frest
    print("SH (f_rest): yaw=0 -> unchanged (exact); f_dc (DC) rotation-invariant -> unchanged")

# ---------- write moved object record (preserve ALL fields) ----------
moved=np.array(seg)  # copy
moved["x"],moved["y"],moved["z"]=p3[:,0],p3[:,1],p3[:,2]
for i,k in enumerate(["rot_0","rot_1","rot_2","rot_3"]): moved[k]=new_q[:,i].astype(moved[k].dtype)
if have_rest and new_rest is not None:
    for i,rn in enumerate(rest_names): moved[rn]=new_rest[:,i].astype(moved[rn].dtype)
# rotate placeholder normals if present/nonzero
for trio in [("nx","ny","nz")]:
    if all(t in names for t in trio):
        nrm=np.stack([seg[trio[0]],seg[trio[1]],seg[trio[2]]],1).astype(np.float64)
        if np.abs(nrm).sum()>0:
            nrm=(R@nrm.T).T
            for i,t in enumerate(trio): moved[t]=nrm[:,i].astype(moved[t].dtype)

# ---------- reduce moved object SH to DC (deg1-3 -> 0); MOVED object ONLY ----------
if args.reduce_sh_to_dc and have_rest:
    for rn in rest_names: moved[rn]=0.0
    print(f"SH reduction: zeroed deg1-3 (f_rest) on the MOVED object only ({Nobj} gaussians); DC (f_dc) kept. "
          f"Scene + stationary object keep FULL SH.")
else:
    print("SH reduction: OFF (moved object keeps full SH)")

# ---------- merge: full scene minus label-L, plus moved object ----------
keep_scene=ids_full!=LABEL
scene_kept=full[keep_scene]
# concatenate (same dtype)
merged=np.concatenate([scene_kept, moved])
print(f"\nMERGE: full={Nfull}  removed(label {LABEL})={int((~keep_scene).sum())}  added(moved)={len(moved)}  -> merged={len(merged)}")
print(f"  count check: {Nfull} - {int((~keep_scene).sum())} + {len(moved)} = {Nfull-int((~keep_scene).sum())+len(moved)} == merged {len(merged)}: {Nfull-int((~keep_scene).sum())+len(moved)==len(merged)}")
PlyData([PlyElement.describe(merged,"vertex")],text=False).write(OUT)
# viewer copy (standard fields)
std=[k for k in names if k and not k.startswith("obj_dc")]
mv=np.empty(len(merged),dtype=[(k,merged.dtype[k]) for k in std])
for k in std: mv[k]=merged[k]
PlyData([PlyElement.describe(mv,"vertex")],text=False).write(OUT.replace(".ply","_viewer.ply"))
print(f"saved -> {OUT} (+ _viewer.ply)   originals untouched")
print(f"in-plane offset applied: dx={args.dx:.4f} (u) dy={args.dy:.4f} (v)  yaw={args.yaw} deg")

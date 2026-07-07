"""Undistort IMG_0365 integer-label masks to match COLMAP's undistorted images, using NEAREST-NEIGHBOR
(labels must stay {0,1,2}). Build the map from COLMAP's distorted OPENCV cam + undistorted PINHOLE cam,
VERIFY it reproduces COLMAP's own image undistortion (bilinear), then remap masks with NEAREST."""
import os, glob, argparse, numpy as np, cv2
from PIL import Image
from scene.colmap_loader import read_intrinsics_binary

_ap=argparse.ArgumentParser(); _ap.add_argument("--base",default="/home/ravi/Desktop/Ravi/3dgs/data/IMG_0365")
BASE=_ap.parse_args().base
RAW_IMG=f"{BASE}/input"; COLMAP_IMG=f"{BASE}/images"; RAW_MASK=f"{BASE}/object_mask_dinosam2"
OUT_MASK=f"{BASE}/object_mask_dinosam2_undist"
os.makedirs(OUT_MASK, exist_ok=True)

dc=list(read_intrinsics_binary(f"{BASE}/distorted/sparse/0/cameras.bin").values())[0]   # OPENCV
uc=list(read_intrinsics_binary(f"{BASE}/sparse/0/cameras.bin").values())[0]              # PINHOLE
fx,fy,cx,cy,k1,k2,p1,p2 = dc.params
Kd=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float64); dist=np.array([k1,k2,p1,p2],np.float64)
ufx,ufy,ucx,ucy=uc.params[:4]; Ku=np.array([[ufx,0,ucx],[0,ufy,ucy],[0,0,1]],np.float64)
Wd,Hd=dc.width,dc.height; Wu,Hu=uc.width,uc.height
print(f"distorted OPENCV {Wd}x{Hd} k=({k1:.4f},{k2:.4f},{p1:.5f},{p2:.5f})")
print(f"undistorted PINHOLE {Wu}x{Hu}")
m1,m2=cv2.initUndistortRectifyMap(Kd,dist,np.eye(3),Ku,(Wu,Hu),cv2.CV_32FC1)

# --- VERIFY map reproduces COLMAP's undistorted images (bilinear) ---
print("\nVERIFY: my cv2-undistort (bilinear) vs COLMAP images/ :")
diffs=[]
_chk=[os.path.basename(f).split(".")[0] for f in sorted(glob.glob(f"{COLMAP_IMG}/*.jpg"))]
for stem in [_chk[0], _chk[len(_chk)//2], _chk[-1]]:
    raw=cv2.imread(f"{RAW_IMG}/{stem}.jpg"); mine=cv2.remap(raw,m1,m2,cv2.INTER_LINEAR)
    col=cv2.imread(f"{COLMAP_IMG}/{stem}.jpg")
    if mine.shape!=col.shape: print(f"  {stem}: SHAPE MISMATCH mine{mine.shape} colmap{col.shape}"); continue
    mad=float(np.abs(mine.astype(int)-col.astype(int)).mean())
    diffs.append(mad); print(f"  {stem}: mean abs pixel diff = {mad:.2f} (0-255)  shape={mine.shape[1]}x{mine.shape[0]}")
print(f"  -> {'ALIGNED (low diff = same undistortion)' if max(diffs)<6 else 'WARNING: high diff, map may not match COLMAP'}")

# --- undistort masks with NEAREST-NEIGHBOR ---
masks=sorted(glob.glob(f"{RAW_MASK}/*.png")); allvals=set()
mk={1:[],2:[]}
for fp in masks:
    stem=os.path.basename(fp).split(".")[0]
    m=np.array(Image.open(fp))                       # {0,1,2}, 1080x1920
    mu=cv2.remap(m,m1,m2,cv2.INTER_NEAREST)          # NEAREST -> labels preserved
    allvals|=set(np.unique(mu).tolist())
    Image.fromarray(mu.astype(np.uint8),"L").save(f"{OUT_MASK}/{stem}.png")
    mk[1].append(int((mu==1).sum())); mk[2].append(int((mu==2).sum()))
print(f"\nundistorted {len(masks)} masks -> {OUT_MASK}")
print(f"HARD CHECK unique values across ALL undistorted masks: {sorted(allvals)}  (must be subset of [0,1,2])")
assert set(allvals).issubset({0,1,2}), "LABEL CORRUPTION: non-{0,1,2} values present!"
HW=Hu*Wu
print(f"  mouse keep_frac median={100*np.median(mk[1])/HW:.3f}%  apple keep_frac median={100*np.median(mk[2])/HW:.3f}%")
print(f"  no-overlap preserved (integer mask): single label per pixel by construction")

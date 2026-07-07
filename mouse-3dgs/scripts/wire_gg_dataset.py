"""STAGE 2: wire fresh MULTI-CLASS GG dataset for IMG_0365 (mouse+apple). Undistorted images + COLMAP
poses + undistorted integer-label masks {0,1,2} as multi-class identity supervision. Verifies gates.
Overwrites nothing."""
import os, glob, shutil, argparse, numpy as np
from PIL import Image
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from scene.dataset_readers import storePly

_ap=argparse.ArgumentParser()
_ap.add_argument("--src",default="/home/ravi/Desktop/Ravi/3dgs/data/IMG_0365")
_ap.add_argument("--dst",default="/home/ravi/Desktop/Ravi/gaussian-grouping/data/mouse_apple_0365")
_a=_ap.parse_args(); SRC=_a.src; DST=_a.dst
assert not os.path.exists(DST), f"{DST} exists - refuse to overwrite"
for s in ["images","object_mask","sparse/0","check"]: os.makedirs(f"{DST}/{s}",exist_ok=True)

for fp in sorted(glob.glob(f"{SRC}/images/*.jpg")): shutil.copy(fp,f"{DST}/images/{os.path.basename(fp)}")
for fp in sorted(glob.glob(f"{SRC}/object_mask_dinosam2_undist/*.png")): shutil.copy(fp,f"{DST}/object_mask/{os.path.basename(fp)}")
for f in ["cameras.bin","images.bin","points3D.bin"]: shutil.copy(f"{SRC}/sparse/0/{f}",f"{DST}/sparse/0/{f}")
xyz,rgb,_=read_points3D_binary(f"{DST}/sparse/0/points3D.bin"); storePly(f"{DST}/sparse/0/points3D.ply",xyz,rgb)

# GATES
imgs=sorted(glob.glob(f"{DST}/images/*.jpg")); masks=sorted(glob.glob(f"{DST}/object_mask/*.png"))
poses=read_extrinsics_binary(f"{DST}/sparse/0/images.bin"); cam=list(read_intrinsics_binary(f"{DST}/sparse/0/cameras.bin").values())[0]
istems={os.path.basename(f).split('.')[0] for f in imgs}; mstems={os.path.basename(f).split('.')[0] for f in masks}
pstems={os.path.splitext(p.name)[0] for p in poses.values()}
print("=== STAGE 2 GATES ===")
print(f"images={len(imgs)} masks={len(masks)} poses={len(poses)} points3D={len(xyz)}")
print(f"GATE counts equal 130: {len(imgs)==len(masks)==len(poses)==130}")
print(f"GATE names align: {istems==mstems==pstems}")
ir=Image.open(imgs[0]).size[::-1]; mr=Image.open(masks[0]).size[::-1]
print(f"GATE res image {ir} == mask {mr} == camera {(cam.height,cam.width)}: {ir==mr==(cam.height,cam.width)}")
mm=all(Image.open(f"{DST}/images/{s}.jpg").size==Image.open(f"{DST}/object_mask/{s}.png").size for s in istems)
print(f"GATE every image matches its mask res: {mm}")
vals=set(); perobj={1:0,2:0}
for p in masks:
    m=np.array(Image.open(p)); vals|=set(np.unique(m).tolist())
    perobj[1]+=int((m==1).any()); perobj[2]+=int((m==2).any())
print(f"GATE mask values across all 130: {sorted(vals)} (MULTI-CLASS: must be subset of {{0,1,2}} and CONTAIN 1 AND 2)")
print(f"GATE both objects present: mouse(1) in {perobj[1]} frames, apple(2) in {perobj[2]} frames (multi-class, not collapsed)")
print(f"GATE sparse/0: {sorted(os.listdir(f'{DST}/sparse/0'))}")
print(f"\nassembled -> {DST}")

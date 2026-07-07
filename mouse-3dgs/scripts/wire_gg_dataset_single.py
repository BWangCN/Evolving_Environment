"""Wire a fresh GG dataset identical to mouse_gg_0323 EXCEPT the masks (DINO+SAM2 instead of adaptive).
Reuses the same undistorted frames + same COLMAP poses (no re-run). New masks go in object_mask/ so the
train command differs from mouse_gg_0323 only by dataset path. Verifies gates. Overwrites nothing."""
import os, glob, shutil
import numpy as np
from PIL import Image
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

REF  = "/home/ravi/Desktop/Ravi/gaussian-grouping/data/mouse_gg_0323"          # same frames + poses
NEWM = "/home/ravi/Desktop/Ravi/3dgs/data/IMG_0323/object_mask_dinosam2"        # NEW masks
DST  = "/home/ravi/Desktop/Ravi/gaussian-grouping/data/mouse_gg_0323_dinosam2"
assert not os.path.exists(DST), f"{DST} exists - refusing to overwrite"
for sub in ["images","object_mask","sparse/0","check"]: os.makedirs(f"{DST}/{sub}", exist_ok=True)

# SAME frames + SAME poses (copied from the working run, unchanged)
for fp in sorted(glob.glob(f"{REF}/images/*.jpg")): shutil.copy(fp, f"{DST}/images/{os.path.basename(fp)}")
for f in ["cameras.bin","images.bin","points3D.bin","points3D.ply"]:
    shutil.copy(f"{REF}/sparse/0/{f}", f"{DST}/sparse/0/{f}")
# NEW masks as identity supervision (named object_mask -> object_path stays 'object_mask')
for fp in sorted(glob.glob(f"{NEWM}/*.png")): shutil.copy(fp, f"{DST}/object_mask/{os.path.basename(fp)}")

# ---- GATES ----
imgs = sorted(glob.glob(f"{DST}/images/*.jpg")); masks = sorted(glob.glob(f"{DST}/object_mask/*.png"))
poses = read_extrinsics_binary(f"{DST}/sparse/0/images.bin")
cam = list(read_intrinsics_binary(f"{DST}/sparse/0/cameras.bin").values())[0]
istems = {os.path.basename(f).split('.')[0] for f in imgs}
mstems = {os.path.basename(f).split('.')[0] for f in masks}
pstems = {os.path.splitext(p.name)[0] for p in poses.values()}
print("=== GATES ===")
print(f"images={len(imgs)} object_masks={len(masks)} poses={len(poses)}")
print(f"GATE counts equal 130: {len(imgs)==len(masks)==len(poses)==130}")
print(f"GATE names align (img==mask==pose): {istems==mstems==pstems}")
ir = Image.open(imgs[0]).size[::-1]; mr = Image.open(masks[0]).size[::-1]
print(f"GATE res image {ir} == mask {mr} == camera {(cam.height,cam.width)}: {ir==mr==(cam.height,cam.width)}")
mm = all(Image.open(f"{DST}/images/{s}.jpg").size == Image.open(f"{DST}/object_mask/{s}.png").size for s in istems)
print(f"GATE every image matches its mask resolution: {mm}")
ids = set()
for p in masks[:15]: ids |= set(np.unique(np.array(Image.open(p))).tolist())
print(f"GATE mask ID values: {sorted(ids)} (0=bg/desk, 1=mouse)")
# confirm masks are the NEW dinosam2 ones (tighter): median keep vs old adaptive
kf = np.median([(np.array(Image.open(p))>0).mean()*100 for p in masks])
okf = np.median([(np.array(Image.open(p))>0).mean()*100 for p in sorted(glob.glob(f"{REF}/object_mask/*.png"))])
print(f"GATE masks are NEW (DINO+SAM2): median keep={kf:.2f}%  (old adaptive run was {okf:.2f}%)")
print(f"GATE sparse/0: {sorted(os.listdir(f'{DST}/sparse/0'))}")
print(f"\nassembled -> {DST}")

# Gaussian Segmentation + Hole Reconstruction (object-on-desk pipeline)

Produce a **clean, complete standalone object Gaussian** from a simulated scene: capture an
object resting on a desk, reconstruct the whole table+object scene in 3DGS, **segment the
object's Gaussian cluster** out of the desk, and **reconstruct the holes** — both the desk
disocclusion where the object sat *and* the object's own occluded bottom (it never sees its
underside while resting on the table).

This directory contains **only our additions**. The segmentation + inpainting engine is
[Gaussian Grouping (lkeab)](https://github.com/lkeab/gaussian-grouping) — **download it
yourself** (see Setup); we do not redistribute it.

## Why this pipeline (lessons learned)
- The object **sits on the table** (bottom at table `z=0`), so its underside is never seen from
  the upper hemisphere — that is the real **occlusion hole** the inpainting step fixes.
- We write **exact camera poses** (sim ground truth) and **exact per-view object masks**
  (by hiding all non-object render bodies and taking the object's alpha) → perfect grouping
  supervision. **No COLMAP SfM, no DUSt3R, no SAM.** (DUSt3R/COLMAP-free on isolated objects gave
  catastrophic poses; for synthetic data with known poses, always use the GT poses.)
- Capturing each object mesh *floating in mid-air* is **not** this pipeline — it sidesteps the
  segmentation + occluded-bottom problem that the coverage/editing method is actually about.

## What's here
```
scripts/capture_apple_desk.py                 # (in ../scripts) capture object-on-desk -> gaussian-grouping format
gaussian_segmentation_hole_recon/
  configs/object_removal_apple_desk.json      # gaussian-grouping object-removal config (segment object out + inpaint desk hole)
  configs/object_inpaint_apple_desk.json      # gaussian-grouping object-inpaint config (fill the occluded-bottom hole, LaMa + finetune)
  patches/gaussian_grouping_desk.patch        # our small edits to gaussian-grouping's edit_object_{removal,inpaint}.py
```

## External dependencies (download these yourself)
- **Gaussian Grouping** — https://github.com/lkeab/gaussian-grouping (their license applies).
  Follow their install (3DGS rasterizer, `simple-knn`, etc.). On Blackwell/5090 build the CUDA
  extensions with `TORCH_CUDA_ARCH_LIST="12.0"` and add `#include <cfloat>` to `simple-knn`.
- **LaMa** inpainting weights, as required by gaussian-grouping's inpaint step (`big-lama`).
- ManiSkill (SAPIEN) + a YCB asset pack for the capture (env `gaussian_grouping`).

## Pipeline
**1. Capture the object on the desk (this repo).** Writes `images/`, `object_mask/` (1=object),
and `sparse/0/` with exact poses, in gaussian-grouping's expected layout:
```bash
python scripts/capture_apple_desk.py --obj 013_apple --num-views 150 \
    --out <gaussian-grouping>/data/apple_desk
```

**2. Set up gaussian-grouping.** Clone it, install, then apply our additions:
```bash
cd <gaussian-grouping>
git apply <this-repo>/gaussian_segmentation_hole_recon/patches/gaussian_grouping_desk.patch
mkdir -p config/object_removal config/object_inpaint
cp <this-repo>/gaussian_segmentation_hole_recon/configs/object_removal_apple_desk.json config/object_removal/apple_desk.json
cp <this-repo>/gaussian_segmentation_hole_recon/configs/object_inpaint_apple_desk.json config/object_inpaint/apple_desk.json
```

**3. Reconstruct + learn the grouping field.** Run gaussian-grouping's training on
`data/apple_desk` (the per-view `object_mask/` supplies exact grouping supervision — no SAM/DEVA
needed). See the upstream README for the exact `train.py` invocation.

**4. Segment + reconstruct holes.** `select_obj_id: [1]` selects the object cluster.
- `edit_object_removal.py` with `config/object_removal/apple_desk.json` → removes the object and
  inpaints the **desk disocclusion** left behind.
- `edit_object_inpaint.py` with `config/object_inpaint/apple_desk.json` → LaMa-inpaints + finetunes
  (`finetune_iteration: 10000`, `lambda_dlpips: 0.5`) to **fill the object's occluded-bottom hole**.

Result: a scene-segmented object Gaussian with both holes reconstructed, ready for editable
insert/move synthesis.

## Patch contents
`patches/gaussian_grouping_desk.patch` is a small diff against upstream
`edit_object_removal.py` / `edit_object_inpaint.py` (guards the video-writer when a render split
has zero views, so the desk pipeline's empty splits don't crash). Apply with `git apply`.

## Attribution
Segmentation + inpainting: **Gaussian Grouping**, Ye et al. (ECCV 2024) — see their repo/license.
This directory adds only the simulation capture, configs, and the small patch above.

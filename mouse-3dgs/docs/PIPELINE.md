# End-to-End Pipeline — Video → Digital Copy of an Object

This is the **master runbook**: how a raw phone video of an object (or two) on a desk becomes a
clean, isolated, movable 3D Gaussian model. It ties together the scripts in `scripts/` with the
two upstream repos you re-clone and the model weights you download.

For **why** each stage exists, read [`HISTORY.md`](HISTORY.md). For the **environment build**
(torch, CUDA extensions, the `sm_89`/cfloat notes), read the top-level [`../README.md`](../README.md).
For **moving this repo + data to a new machine**, read [`MIGRATION.md`](MIGRATION.md).

> **Nothing here is fully turnkey.** Masking is semi-interactive (a GroundingDINO text prompt and a
> seed frame per object), COLMAP needs a capture with genuine top-down coverage to register (the
> "solidity gate" — see HISTORY Phase 5), and several scripts still carry hard-coded
> `IMG_0365`-style paths (the later `--base/--src/--dst` flags parametrize *some* of them). Treat
> the commands below as the proven recipe to adapt per capture, not a single button.

---

## Components you assemble (each in its own conda env — isolation is non-negotiable)

| Component | What it does here | Re-clone / install | Env |
|---|---|---|---|
| **SeeDo** (`github.com/ai4ce/SeeDo`) | video→frames + DINO+SAM2 masking. Your masking drivers (`scripts/mask_*_dinosam2*.py`) run **inside** this checkout. | `git clone https://github.com/ai4ce/SeeDo.git` (has GroundingDINO + segment-anything-2); `pip install -r SeeDo/requirements.txt` | `seedo` |
| **COLMAP** | camera poses + undistortion | `conda create -n colmap -c conda-forge colmap` (or system/apt) | `colmap` |
| **gaussian-grouping** (`github.com/lkeab/gaussian-grouping`) | GG training + everything that reads the trained `.ply`. Your wiring/select/edit scripts run **inside** this checkout. | see [`../README.md`](../README.md) §1–§5 (`--recursive`, build the two CUDA exts) | `gaussian_grouping` |

**Weights to download (kept OUT of git — see MIGRATION.md):**
- `sam_vit_h_4b8939.pth` (SAM ViT-H) — for the masking step.
- GroundingDINO config + weights (`groundingdino_swint_ogc.pth`) — pulled per GroundingDINO's README.
- `big-lama` — for the final inpaint (stage 11).

**How to use the `scripts/` here:** they are version-controlled *glue*. On the new machine, run each
from inside the checkout named in its "run-from" column (copy it in, or add the checkout to
`PYTHONPATH`) under that stage's env — they import from SeeDo's / GG's local packages.

---

## The 11 stages

Assume a capture `data/IMG_XXXX/` with the raw video and (once extracted) an `input/` frame folder.

### Stage 1 — Video → frames  ·  env `seedo` · run-from SeeDo
```bash
conda activate seedo
python SeeDo/convert_video.py --input IMG_XXXX.MOV --output IMG_XXXX_30fps.mp4   # upstream SeeDo
# then sample frames into data/IMG_XXXX/input/  (e.g. ffmpeg -i IMG_XXXX_30fps.mp4 data/IMG_XXXX/input/%04d.jpg)
```

### Stage 2 — COLMAP poses + undistortion  ·  env `colmap` · run-from GG
```bash
conda activate colmap
cd gaussian-grouping
python convert.py -s /abs/path/data/IMG_XXXX          # feature-extract → match → mapper → undistort
# yields data/IMG_XXXX/{images,sparse/0,distorted/…}
```
> This is the registration gate: if top-down frames don't register, the object's top surface will
> come out see-through later. Fix at capture time, not with a filter.

### Stage 3 — Object masks (GroundingDINO + SAM2)  ·  env `seedo` · run-from SeeDo
```bash
conda activate seedo
# multi-object (bg=0, obj1=1, obj2=2, …):
python scripts/mask_objects_dinosam2.py --base /abs/path/data/IMG_XXXX
# single-object variant:
python scripts/mask_object_dinosam2_single.py --base /abs/path/data/IMG_XXXX
# → data/IMG_XXXX/object_mask_dinosam2/<frame>.png  (integer-label masks)
```
> The GroundingDINO text prompts and seed frame are set inside the script (e.g. `"mouse"`,
> `"apple"`); edit them for a new object. Masks are the **object-only identity masks** (mostly
> black), not a keep-mask (HISTORY Phase 2).

### Stage 4 — Undistort masks to match COLMAP, then verify  ·  env `gaussian_grouping` · run-from GG
```bash
conda activate gaussian_grouping
python scripts/undistort_masks.py --base /abs/path/data/IMG_XXXX   # NEAREST remap; labels stay {0,1,2}
python scripts/verify_masks.py --img data/IMG_XXXX/input --msk data/IMG_XXXX/object_mask_dinosam2 \
       --out data/IMG_XXXX                                          # no-overlap / drift / overlays gate
```

### Stage 5 — Wire the GG dataset (frames + poses + masks)  ·  env `gaussian_grouping` · run-from GG
```bash
python scripts/wire_gg_dataset.py --src /abs/path/data/IMG_XXXX --dst output_data/mouse_apple_XXXX
# single-object same-poses variant: scripts/wire_gg_dataset_single.py
```

### Stage 6 — Train Gaussian Grouping  ·  env `gaussian_grouping` · run-from GG
```bash
python train.py -s output_data/mouse_apple_XXXX -m output/mouse_apple_XXXX \
       --config config/gaussian_dataset/train.json     # to 30k iters; identity supervision on full scene
```

### Stage 7 — Select the object by identity  ·  env `gaussian_grouping` · run-from GG
```bash
python scripts/select_by_identity.py     # writes <object>_segmented.ply per label (fresh names)
```
Inspect top-down + oblique: **solid = success** (HISTORY Phase 5). This is your **digital copy**.

### Stage 8 — Trim flakes (optional cleanup)  ·  env `gaussian_grouping` · run-from GG
```bash
python scripts/diag_0323_clean.py        # computes thresholds → /tmp/diag0323.npy
python scripts/trim_flakes.py            # flat-AND-off-core removal → *_clean.ply (body untouched)
```

### Stage 9 — Persist the desk plane (once per scene; required before any move)  ·  env `gaussian_grouping` · run-from GG
```bash
python scripts/persist_desk_plane.py     # deterministic RANSAC → output/.../desk_plane.npy
```
> **Reuse this file for every move — never silently re-fit.** It is DATA (gitignored), moved
> out-of-band with the rest of `output/`.

### Stage 10 — Move an object in-scene  ·  env `gaussian_grouping` · run-from GG
```bash
python scripts/move_object.py --object mouse --dx <units> --yaw <deg> --reduce_sh_to_dc \
       --out output/mouse_apple_XXXX/scene_moved_mouse.ply
```
> `--reduce_sh_to_dc` (default ON) fixes color after translation; it does **not** fix geometry —
> there is a finite max move distance (~13° view-change was the mouse's usable limit). STOP and
> report rather than produce a transparent object (HISTORY Phase 7).

### Stage 11 — Inpaint the vacated spot  ·  env `gaussian_grouping` · run-from GG
```bash
python scripts/inpaint_vacated.py --object mouse --scene output/mouse_apple_XXXX/scene_moved_mouse.ply \
       --out output/mouse_apple_XXXX/scene_moved_mouse_inpainted.ply
```
> Removes the darkened desk Gaussians over the **measured** shadow ring (footprint + margin) and
> fills with local-brightness-matched desk. STOPs if the persisted plane is missing. Needs
> `big-lama` weights for the LaMa-backed variant.

---

## Condensed flow

```
video ─▶ (seedo) convert_video + frames
      ─▶ (colmap) convert.py  ── poses/undistort ──┐
      ─▶ (seedo) mask_*_dinosam2 ── object masks ──┤
      ─▶ (gg) undistort_masks → verify_masks → wire_gg_dataset
      ─▶ (gg) train.py ── trained scene + identity field
      ─▶ (gg) select_by_identity ── DIGITAL COPY (.ply)
      ─▶ (gg) trim_flakes ── clean copy
      ─▶ (gg) persist_desk_plane → move_object → inpaint_vacated ── in-scene manipulation
```

# Mouse 3DGS — Gaussian Grouping Extraction & In-Scene Manipulation

Extract a clean, solid 3D model of an object (a computer mouse) from a full-scene capture
using **3D Gaussian Splatting + Gaussian Grouping**, then reposition it in-scene
convincingly. This README is the **from-scratch install** for the *proven* pipeline — the
one that produced a solid, correctly-colored, movable mouse.

This README is the environment build for the **Gaussian Grouping half** (training + everything
that reads the trained `.ply`). For the **complete video → digital-copy runbook** — including the
front-half stages (video→frames + DINO+SAM2 masking in the separate `seedo` env, and COLMAP
poses) — see [`docs/PIPELINE.md`](docs/PIPELINE.md), which ties all 11 stages and their envs
together.

For the full story of how the pipeline evolved and **why** each choice was made, read
[`docs/HISTORY.md`](docs/HISTORY.md). To move this repo/data between machines, see
[`docs/MIGRATION.md`](docs/MIGRATION.md).

> **Scope note:** this repo is the mouse/object 3DGS pipeline. The masking front-half runs in its
> own `seedo` env (SeeDo = GroundingDINO + SAM2) and COLMAP in a `colmap` env — both documented in
> `docs/PIPELINE.md` and re-cloned/installed separately, never bundled here. The RoboSplat /
> robot-arm work is intentionally out of scope — it isn't proven yet and is documented separately
> once it works on the target machine.

---

## Why this stack (the short version)

Every choice below exists because a simpler path failed. The details are in `HISTORY.md`;
the headlines:

- **Per-project conda env, never a shared base.** The original pain was environment
  impermanence (Colab wiping state). Isolation is enforced so a new tool can never break a
  working pipeline. *Verify every install lands in the right env with
  `pip show <pkg> | grep Location`.*
- **torch matched to the machine's CUDA, installed before everything else.** Mismatched
  CUDA is the #1 cause of the `diff-gaussian-rasterization` / `simple-knn` build failures
  that plagued early setup. Get torch right first; the rest follows.
- **The two CUDA extensions are compiled locally, not pip-installed blindly.** They build
  against *your* torch + GPU architecture. On Ada (RTX 40-series, `sm_89`) this needs
  `TORCH_CUDA_ARCH_LIST="8.9"` and a small **cfloat fix** for `simple-knn` — see the build
  step. Skipping these = a failed compile.
- **Gaussian Grouping over mask-then-reconstruct.** GG keeps everything in one coherent
  reconstruction (consistent geometry + lighting), which is what makes in-scene
  repositioning work. The tradeoff (boundary-tightness transparency) was solved by *capture
  coverage*, not by switching methods — see below.

---

## 0. Prerequisites

- Linux (Ubuntu). A recent NVIDIA driver.
- Miniconda / Anaconda.
- `git`, `build-essential`, and a CUDA toolkit **matching your driver** (needed to compile
  the extensions; if there's no system `nvcc`, install the toolkit into the conda env).
- COLMAP (for camera poses) — `sudo apt install colmap` or build from source.

Confirm the GPU and CUDA situation first — this determines the torch install line:

```bash
nvidia-smi                                          # GPU model + max supported CUDA
nvcc --version 2>/dev/null || echo "no system nvcc" # is there a system toolkit?
```

---

## 1. Clone the pipeline repos

```bash
# This repo (scripts + docs):
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Gaussian Grouping (the segmentation engine) — includes 3DGS as submodules:
git clone https://github.com/lkeab/gaussian-grouping.git --recursive
# (If you cloned without --recursive:  git submodule update --init --recursive )
```

---

## 2. Create the isolated environment

```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping
# Sanity: confirm you're in the right env before ANY install.
which python        # must point inside .../envs/gaussian_grouping/
```

---

## 3. ⚠️ INSTALL TORCH — GPU-SPECIFIC

This is the one block that depends on the machine's GPU. The command below is the **verified
working build** on the current workstation — **RTX 4070 Ti SUPER (Ada, `sm_89`), driver 575.57.08
(max CUDA 12.9), no system `nvcc`** — confirmed by the running `gaussian_grouping` env:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

> The card supports CUDA 12.x, but **cu118 is the proven stack** on `sm_89` — the CUDA extensions,
> spconv/xformers siblings, and every trained result were built against it. Don't jump to a cu12
> wheel unless you're prepared to rebuild and re-verify the extensions.
>
> On a **different** GPU, re-derive this line: run `nvidia-smi` (max supported CUDA) and
> `nvcc --version`, then pick the matching `--index-url` — do NOT reuse this pin blindly. A wrong
> CUDA build is exactly the failure this README exists to prevent.

After install, verify the GPU is visible:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expect: 2.1.2+cu118  11.8  True
```

---

## 4. Install the Python deps

```bash
pip install -r requirements.txt
```

---

## 5. Compile the CUDA extensions (the step that used to fail)

These live under the Gaussian Grouping submodules. They compile against the torch you just
installed and your GPU's compute capability.

**Set the architecture flag for your GPU first.** On this workstation the card is an
**RTX 4070 Ti SUPER — Ada, compute capability 8.9** (verified via
`nvidia-smi --query-gpu=compute_cap`), so:

```bash
# RTX 4070 Ti SUPER (Ada, sm_89). On a different card, set its compute capability
# instead (Ampere / RTX 30-series / A6000 = 8.6; check NVIDIA's list).
export TORCH_CUDA_ARCH_LIST="8.9"

cd gaussian-grouping
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

> **The `simple-knn` cfloat fix:** if `simple-knn` fails to compile with an error about
> `FLT_MAX` / `<cfloat>`, add `#include <cfloat>` at the top of
> `submodules/simple-knn/simple_knn.cu`, then re-run the install. This was required on the
> workstation's toolchain and is the known fix.

Verify both imported cleanly:

```bash
python -c "import diff_gaussian_rasterization, simple_knn; print('extensions OK')"
```

---

## 6. Get SAM (mask generation) + inpainting weights

```bash
# SAM — for generating the object-only identity masks:
pip install git+https://github.com/facebookresearch/segment-anything.git
# Download a SAM checkpoint (e.g. sam_vit_h) per the SAM README into a weights/ dir
# that is gitignored.

# LaMa / big-lama — for the final desk/underside inpaint (Phase 6/7). Pull the
# big-lama weights per the LaMa instructions; keep them outside git (gitignored).
```

---

## 7. Run the pipeline

The stages (see `HISTORY.md` for the reasoning behind each):

```
capture → COLMAP poses → SAM object-only masks → GG training (identity supervision
on the full unmasked scene) → select object by identity → move/inpaint
```

**7a. COLMAP poses** (the registration gate — if top-down frames don't register, the top
surface won't be solid):

```bash
# Standard COLMAP feature-extract → match → mapper → image_undistorter,
# then arrange into GG's expected sparse/0/ layout. See gaussian-grouping's data docs.
```

**7b. SAM masks** — generate the **object-only identity mask** (object = 1, everything else
= 0; mostly black). *Not* the object+table "keep-mask" — that belonged only to an abandoned
detour and is not used here.

**7c. Train Gaussian Grouping** on the FULL unmasked scene, using the masks as **identity
supervision** (labels which Gaussians are the object via multi-view consistency), not as an
image crop:

```bash
cd gaussian-grouping
python train.py -s PATH_TO_SCENE -m output/mouse_gg --config configs/gaussian_dataset/train.json
# Train to 30k iterations. Watch it in the terminal (launch manually) rather than blind.
```

**7d. Select the object by identity** and check solidity (the top-down view is the whole
point — it must read solid, not see-through):

```bash
python scripts/select_by_identity.py --model output/mouse_gg --label OBJECT_LABEL \
    --out output/mouse_gg/mouse_selected.ply
# Inspect top-down + oblique. Solid = success.
```

**7e. Move an object in-scene** (the mouse+apple frontier). Uses the persisted desk plane
and reduces the moved object's SH to DC so it renders correct color after translation:

```bash
python scripts/move_object.py \
    --scene output/mouse_apple_0365 \
    --plane output/mouse_apple_0365/desk_plane.npy \
    --label 1 \
    --reduce_sh_to_dc \
    --out output/mouse_apple_0365/scene_moved.ply
```

See `scripts/` for the move/inpaint utilities and their flags.

---

## Known ceilings & guards (don't relearn these the hard way)

- **Solidity comes from capture coverage.** If the object looks see-through from a direction,
  the fix is almost always *more capture angles there* (re-capture), not a filter. The
  `IMG_0323` recapture with top-down coverage is what made the mouse solid.
- **DC reduction fixes color, not geometry.** After a translation, higher-SH bands are
  out-of-domain garbage → reduce to DC. But DC can't fix that flat Gaussians go transparent
  when viewed edge-on. There is a **finite max move distance** (~13° view-change was the
  mouse's usable limit; ~19°+ is the risk zone). Guard moves against it; STOP and report
  rather than produce a transparent object.
- **The contact-shadow ghost extends beyond the footprint.** When inpainting a vacated spot,
  treat the *measured* shadow extent (footprint + margin), and match fill brightness to the
  local penumbra — a global fill brightness leaves a bright patch + dark crescent at steep
  angles.
- **Never overwrite originals; reuse the persisted plane, never silently re-fit it.**

---

## Repo layout

```
.
├── README.md              # this file — from-scratch install for the GG mouse pipeline
├── requirements.txt       # Python deps (torch installed separately, GPU-specific)
├── .gitignore             # keeps .ply/captures/weights/outputs OUT of git
├── scripts/               # ALL pipeline glue: masking, wiring, select, trim, plane, move, inpaint
└── docs/
    ├── PIPELINE.md        # master runbook — the full video→object flow across all 3 envs
    ├── HISTORY.md         # dated experiment log + why each pivot happened
    └── MIGRATION.md       # push-to-GitHub + move-data-to-new-machine runbook
```

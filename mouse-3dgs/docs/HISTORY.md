# Project History — Mouse 3DGS Extraction & Manipulation

A dated, decision-by-decision record of the project: what each experiment was, what
happened, and **why** it caused the next change of direction. This is the "why we did it
this way" companion to the README. It begins at the point work moved onto the **Robotixx
lab workstation** (earlier Colab attempts are deliberately omitted — they were an
environment dead-end, not part of the working pipeline).

> Dates are drawn from the working-session record. Where a single session covered several
> steps, the steps are listed in the order they happened within that session.

---

## The goal (unchanged throughout)

Extract a **clean, solid, photorealistic 3D model of a computer mouse** from a full-scene
capture (mouse on a desk), using **3D Gaussian Splatting** + **Gaussian Grouping** for
segmentation — so the mouse can stand alone as a reusable asset, and later be
**repositioned in-scene** convincingly. The later frontier extended this to a mouse+apple
scene and, eventually, toward robot-arm manipulation demos (RoboSplat-style).

---

## Phase 0 — Environment foundation (mid-June 2026)

**What:** Moved off Colab onto the Robotixx lab Linux workstation
(`ravi@robotixx-System-Product-Name`), an RTX-class GPU box. Stood up the base
`gaussian-splatting` conda environment (graphdeco-inria repo) and trained the first
**local** full-quality reconstruction of the `IMG_0280` capture to the full 30,000
iterations (PSNR ≈ 39.23, ~15.5 it/s — far better than the 7k-iter Colab run at 34.20).

**Why it mattered:** The recurring pain of the whole project up to this point was
**environment impermanence** — Colab wiped the environment on every reset, forcing brutal
rebuilds. The lesson, applied from here forward: keep the *base* environment stable, give
**each project its own conda env**, and keep **code and data separate** from the
environment. This principle is why later work never contaminated a working env, and why the
RoboSplat setup (Phase 7) was deliberately isolated.

**Environment discipline established here:**
- Per-project conda envs (`gaussian_splatting`, later `gaussian_grouping`, `seedo`, etc.),
  never a shared mutable base.
- Code/data separation — large `.ply`s, captures, and weights live outside git.
- On this GPU (Ada, `sm_89`): compile with `TORCH_CUDA_ARCH_LIST="8.9"`, plus the
  **cfloat fix** for `simple-knn` to build.

---

## Phase 1 — First pipeline end-to-end (IMG_0280) & the core problem

**What:** Ran the full extraction pipeline on `IMG_0280`:
`capture → COLMAP poses (130 frames @ 0.71px reprojection, undistorted to 1065×1884) →
SAM mouse-only masks → Gaussian Grouping training (identity supervision on the full
unmasked scene) → select the mouse by identity`. This produced the first clean, isolated
mouse — the **"tight" selection**.

**What happened — the defining flaw:** The tight mouse **reads as see-through from the top
down.** Diagnosis: the `IMG_0280` capture never gave the optimizer top-surface coverage, so
the top of the mouse had no multi-view support. The mouse-only identity mask also draws the
identity boundary tightly around the mouse's visible outline, discarding **~94% of the
contact layer** (the shadow/floor where the mouse meets the desk). This is
**boundary-tightness behavior, not a wiring bug.**

**Why it drove the next steps:** Everything in Phases 2–4 is a response to this transparency
problem — first patching it spatially, then studying an alternative approach, then finally
going *around* it with a better capture.

---

## Phase 2 — Contact-layer patches (opt1 / opt2) on IMG_0280

**What:** Two spatial approaches to add the missing contact layer back, run one at a time
for comparison:
- **Option 1 — footprint-clipped interface slab** (`mouse_opt1_base.ply`): mouse-identity
  Gaussians UNION a thin interface slab clipped tightly to the mouse footprint, using the
  RANSAC desk-plane normal as "up." Surgical; minimal desk.
- **Option 2 — snug bounding box** (`mouse_opt2_bbox.ply`): keep ALL Gaussians inside a
  snug 3D box around the mouse, identity ignored, box bottom just below the desk surface.
  Simpler; includes a thin desk disc and possible floaters.

This set up a **three-way comparison**: *tight* (cleanest, but transparent) vs *opt1*
(solid, minimal desk skirt) vs *opt2* (solid, but desk disc / floaters).

**The mask scare (resolved):** A worry surfaced that the wrong mask had been wired in. It
resolved cleanly: the pipeline used the correct **mouse-only identity mask** (mostly black,
~1.6% white). The **mouse+table "keep-mask"** (mostly white) only ever belonged to an
**abandoned masked-3DGS detour** (`output/IMG_0280_masked`) — the two were never crossed.
That detour's only value had been killing background floater haze by baking the keep-mask
into the RGBA alpha so photometric loss saw only mouse+table; it was never the result.

**Why we didn't just pick a patch:** The patches worked but were heuristics bolted onto a
capture that lacked the underlying data. That pointed two ways at once — study whether a
*different segmentation method* avoids the problem (Phase 3), and consider whether a
*better capture* removes the need for patching at all (Phase 4). Both were pursued.

---

## Phase 3 — RoboSplat study (the alternative segmentation approach)

**What:** Studied the RoboSplat paper (novel-demonstration generation by editing 3DGS
scenes). The key finding was their **segmentation choice**, which is the *opposite* of ours:

- **Our route (Gaussian Grouping):** train one full-scene reconstruction, attach a learned
  per-Gaussian **identity field** supervised by 2D masks, then select Gaussians by identity.
- **RoboSplat route (mask-then-reconstruct):** mask the 2D images first and
  **re-reconstruct the object from scratch** from only the masked object pixels. The robot
  arm, separately, is decomposed by **URDF** — each Gaussian is assigned to a robot link by
  spatial proximity to that link's known point cloud; anything near no link/object is
  background.

**Why it mattered:** RoboSplat's object reconstruction sidesteps the exact transparency
problem — object Gaussians are optimized purely to render the object well, with no
"carve a subset out of a joint scene" step that orphans contact-shadow Gaussians. The
authors *knew* about Gaussian Grouping (cited it) and still chose mask-then-reconstruct — a
meaningful signal. **Crucial caveat:** RoboSplat never needs a watertight standalone object;
their moved objects get re-rendered into many synthetic views where small contact-region
imperfections wash out. **Our goal (a clean reusable asset) is stricter**, which is why the
contact-layer problem bit us harder than it bit them.

---

## Phase 4 — RoboSplat-style mouse-only reconstruction (the informative failure)

**What:** Actually tried the full RoboSplat object path on the mouse — mask first, fresh
reconstruct, depth priors. Gave a recognizable mouse but with two noise problems across
iterations: first long cross-frame **spikes**, then sharp **boundary flakes**.

**What we learned (the valuable part):** Worked through removal with scripted
scale/opacity/outlier filters, then a planned flat-AND-off-surface geometric filter. Spun up
several approaches:
- **ap1** — manual SuperSplat cleanup.
- **ap2** — eroded-mask retrain. **Failed *informatively*: eroding the mask changed
  nothing** — which *proved the flakes were not mask-boundary artifacts* but persistent
  geometry (born of the seed cloud + densification).
- **ap3** — eroded-mask + depth.
- **ap2b** — a geometry-based attack (flat-AND-off-surface filter), the logical
  consequence of ap2's finding.

**Why we set it aside:** ap2 reframed the problem from "bad masks" to "persistent seeded
geometry," but the cleaner fix was becoming obvious from the other direction — the root
cause was *capture coverage*, not segmentation method. So the RoboSplat thread was parked at
ap2b in favor of the recapture.

---

## Phase 5 — The recapture (IMG_0323): the breakthrough

**What:** Shot a **better capture with more angles, including top-down**, specifically to
fix the transparency at its root. Ran the full pipeline from raw video, GG identity approach
(explicitly *not* RoboSplat), trained to 30k iterations → `output/mouse_gg_0323`.

**What happened — it worked:**
- Identity field clean: **99.4–99.9% coverage** including top-down views.
- Mouse selected at **82,224 Gaussians**.
- Solidity check: **SOLID from above** (≤0.4% see-through).

The top-down coverage gave the optimizer exactly the multi-view support `IMG_0280` lacked.

**Why this retired an entire branch:** This solved the problem by going *around* it instead
of patching it. The whole transparent-path branch — opt1/opt2 contact patches, the
masked-3DGS detour — was **retired**. Confirmed principle: *the COLMAP registration gate
and the solidity check are the same question asked at both ends of the pipeline* — if the
top-down frames register, the top surface gets support and comes out solid.

---

## Phase 6 — Cleanup saga (the three-layer onion)

With a solid mouse in hand, the remaining work was trimming leftover junk. It came off in
layers:

1. **First look:** soft haze + sharp flakes around the solid mouse → planned a two-filter
   cleanup.
2. **The segmented file leaked the desk:** a solid slab of table had been pulled into the
   selection. Removing that slab was necessary before any flake filter — with the slab
   present it would pollute the "distance from mouse core" measurement the flake filter
   relies on. Reselected → `mouse_0323_reselected.ply`.
3. **With the slab gone,** only scattered flakes remained, sitting cleanly in the off-core
   tail. Options: SuperSplat hand-pass (box-select, ~2 min, no script) vs the scripted ap2b
   flat-AND-off-surface filter (repeatable, distribution-based) vs scripted-then-SuperSplat.

**Why the ordering mattered:** Each layer had to be removed in sequence because each one
*masked the measurement* the next step depended on. The planned finish after flake-trim was
always **hole inpainting** (desk where the mouse sat + the mouse's unseen underside) via
GG's `edit_object_inpaint` with LaMa / `big-lama` weights.

---

## Phase 7 — Mouse+apple scene (IMG_0365): moving objects

**What:** Moved to a two-object scene (mouse + apple, `output/mouse_apple_0365`) with the
goal: **reposition an object and re-render the combined scene to look real.** Objects were
label-separated (mouse = label 1, apple = label 2, desk = label 0), so a rigid move is
clean: transform positions by R+t, rotate quaternions by R, rotate SH by R, then merge the
`.ply`s back (viewers render the concatenation of all Gaussians).

**What happened, in order:**

1. **First move worked**, but the moved object rendered with **white/glassy SH garbage**.
   Diagnosis: higher-SH coefficients (deg 1–3) encode color as a function of view
   direction, trained *only* for directions the cameras originally saw the mouse from. A
   translation of ~1.63 units with cameras only ~4.5 units away asks those coefficients to
   predict color 21–40° outside their trained domain — deg-3 extrapolation diverges into
   exactly that glassy garbage. **The higher-SH bands are stale/out-of-domain after a large
   translation, not information.**
2. **Fix — `--reduce_sh_to_dc`:** reduce the moved object's SH to DC (degree 0) so it
   renders correct flat base color from all angles (at the original frame, DC-only and
   full-SH are identical — confirming no real signal lost). Added as a flag to
   `move_object_0365.py`, default ON for moved objects. **This fixed the color.**
3. **The persisted desk plane:** RANSAC desk plane fit once and saved to
   `desk_plane.npy` (n=[-0.025,-0.803,-0.595], off=2.767, 417,872 inliers); all subsequent
   moves **reuse** it (STOP if missing — never silently re-fit), so every move sits on the
   same table and the on-table base-height re-seat check is consistent.
4. **The vacated-spot ghost (the real remaining work):** moving the object exposes where it
   sat — a faint **contact-shadow ghost** on the desk. First inpaint attempts left a
   residual. Diagnosis: the baked-in contact shadow is a *ring of darkened desk Gaussians
   extending BEYOND the footprint*; an inpaint covering only the footprint leaves the shadow
   ring surviving = the ghost. Fix: expand the treated region to the **measured** shadow
   extent (footprint + margin, ~1.35R), remove the darkened desk Gaussians, and fill.
5. **Steep-angle fill refinement:** a global clean-desk fill brightness (0.88) was too
   bright against a darker penumbra ring (0.72–0.76), leaving a bright patch + thin dark
   crescent at steep view angles. Fix: **local-brightness-matched fill** — sample donor
   brightness from the immediately-surrounding desk with a ramp matching the penumbra
   gradient, so there is no brightness step.

**The hard ceiling discovered here (carry forward):** DC reduction fixes *color* at
distance, but does nothing for **geometry**. The object's Gaussians were only optimized to
look solid from the *original* angles; move far enough and you view the splats edge-on and
thin, and the object goes transparent again — no SH trick rescues that. The mouse hit
~13° view-change at its usable limit; risk zone ~19°+. So every move carries a
**distance guard**: if reaching a target position needs a view-angle change large enough to
risk flat-Gaussian transparency, STOP and report the max safe distance rather than produce a
transparent object. (The apple, being rounder with more legitimate specular, is flattened
*more* by DC reduction — reported per-move.)

**Where Phase 7 stands:** move + DC-color-fix working; vacated-spot inpaint tuned through
the local-brightness-matched fill; the goal from the last session was a clearly bigger
on-table move plus making the inpaint genuinely ghost-free at the larger exposed area,
reusing the persisted plane.

---

## Phase 8 — Toward robot-arm manipulation (RoboSplat migration) — IN PROGRESS

**What:** Decided the next frontier is rendering a **robot arm interacting with the objects**
for RoboSplat-style manipulation demos. Key realization from the study: for manipulation
demos the arm essentially *must* be **URDF-driven** (a demo is a trajectory of joint poses;
a static Gaussian blob can't articulate). Chose the **Franka Panda** to match RoboSplat's
preprocessed data (path of least resistance). Plan: stand up RoboSplat's own demo generation
first (their arm, their scene, their data) to de-risk, *then* tackle the real integration —
getting their arm into our mouse scene, which needs the ICP + differentiable-rendering
**frame alignment** their preprocessing already did for their data and which our captures
never needed.

**Also flagged (RoboArmGS, Nov 2025):** naively binding static Gaussians to URDF links
produces motion artifacts because idealized URDF motion ≠ real arm motion. RoboSplat sidesteps
this because it poses arms into *synthetic* demo configurations (imperfections wash out),
which is our situation too — so the naive URDF route is acceptable *for demos*, not for
matching real arm video.

**The blocker that triggered this migration:** On the Robotixx workstation, the RoboSplat
environment build hit a **disk wall** — 9.3 GB free / 100% used, with the disk-hungry steps
(requirements, in-env CUDA 11.8 toolkit, PyTorch3D + diff-gaussian-rasterization builds,
`data.zip`) still ahead. Rather than push into a guaranteed mid-build disk-full failure, work
is **migrating to a second machine with adequate space** — which is the reason this repo and
these docs exist.

**Repo boundary note:** the RoboSplat setup is intentionally NOT part of the from-scratch
README — that README documents only the *proven* GG mouse pipeline (Phases 0–7). RoboSplat
lives here in the history as the direction of travel; its clean install will be documented
separately once it's actually proven on the new machine.

---

## Cross-cutting lessons (the "why we work this way" summary)

- **Environment isolation is non-negotiable.** Born from Colab wipes; enforced as
  per-project conda envs with code/data separation. Every new tool (e.g. RoboSplat) gets its
  *own* env and must never install into a working one — verified with
  `pip show <pkg> | grep Location` after each install.
- **Capture coverage beats post-hoc patching.** The single biggest win (Phase 5) came from
  re-capturing with top-down angles, not from any clever filter. The registration gate and
  the solidity check are the same question at both ends.
- **Diagnose before scaling.** Repeatedly, the right move was to understand *why* a residual
  existed (ap2's mask-erosion null result; the SH out-of-domain diagnosis; the contact-shadow
  ring) before applying a bigger/repeated fix. Scaling an unproven step just makes a bigger
  mess.
- **Know the hard ceilings.** DC reduction fixes color but not geometry; there is a
  finite max move distance set by how flat the Gaussians are. Guard against it explicitly.
- **Never overwrite originals; fresh output names per experiment.** Every move/selection
  writes a new `.ply`; the persisted plane is reused, never silently re-fit.

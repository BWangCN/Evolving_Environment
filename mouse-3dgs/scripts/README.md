# scripts/

All the **authored glue** for the video → digital-copy pipeline, ordered by stage. The full
runbook with per-stage commands is [`../docs/PIPELINE.md`](../docs/PIPELINE.md); the reasoning
behind each stage is [`../docs/HISTORY.md`](../docs/HISTORY.md).

**How to run these:** each script executes *inside* a specific upstream checkout (it imports that
repo's local packages) and under a specific conda env — see the run-from / env columns. They are
the real working versions from the workstation: several still carry hard-coded `IMG_0365`-style
default paths (the later ones take `--base/--src/--dst`); adapt per capture.

| Stage | script | env | run-from | workstation original |
|---|---|---|---|---|
| 3 — masks, multi-object | `mask_objects_dinosam2.py` | `seedo` | SeeDo | `SeeDo/mask_dinosam2_multi.py` |
| 3 — masks, single-object | `mask_object_dinosam2_single.py` | `seedo` | SeeDo | `SeeDo/mask_dinosam2.py` |
| 4 — undistort masks | `undistort_masks.py` | `gaussian_grouping` | GG¹ | `3dgs/undistort_masks_365.py` |
| 4 — verify masks | `verify_masks.py` | `gaussian_grouping` | anywhere | `3dgs/verify_multi365.py` |
| 5 — wire GG dataset, multi | `wire_gg_dataset.py` | `gaussian_grouping` | GG | `gaussian-grouping/wire_multi365.py` |
| 5 — wire GG dataset, single | `wire_gg_dataset_single.py` | `gaussian_grouping` | GG | `gaussian-grouping/wire_dinosam2.py` |
| 7 — select by identity | `select_by_identity.py` | `gaussian_grouping` | GG | `gaussian-grouping/extract_objects_0365.py` |
| 8 — flake-trim thresholds | `diag_0323_clean.py` | `gaussian_grouping` | GG | `gaussian-grouping/diag_0323_clean.py` |
| 8 — flake-trim filter | `trim_flakes.py` | `gaussian_grouping` | GG | `gaussian-grouping/filter_0323_clean.py` |
| 9 — persist desk plane | `persist_desk_plane.py` | `gaussian_grouping` | GG | `gaussian-grouping/persist_desk_plane_0365.py` |
| 10 — move object | `move_object.py` | `gaussian_grouping` | GG | `gaussian-grouping/move_object_0365.py` |
| 11 — inpaint vacated spot | `inpaint_vacated.py` | `gaussian_grouping` | GG | `gaussian-grouping/inpaint_vacated_0365.py` |

¹ "GG" = inside the `gaussian-grouping` checkout (these import its `scene.colmap_loader` etc.).
Stages 1 (video→frames), 2 (COLMAP), and 6 (GG training) use the upstream repos' own scripts
(`SeeDo/convert_video.py`, `gaussian-grouping/convert.py`, `gaussian-grouping/train.py`) — not
duplicated here.

## What the key scripts do

- **`mask_objects_dinosam2.py`** — GroundingDINO text-prompt boxes on a seed frame → SAM ViT-H
  precise masks → SAM2 video propagation with one `obj_id` per object, in one pass. Output:
  integer-label mask per frame (bg=0, obj1=1, obj2=2), overlaps resolved by SAM2 logit. Prompts /
  seed frame are set in-script — edit for a new object.
- **`undistort_masks.py`** — remap masks to COLMAP's undistorted geometry with NEAREST (labels must
  stay exact integers), after verifying the remap reproduces COLMAP's own image undistortion.
- **`verify_masks.py`** — the mask gate: per-object stability/drift, NO-OVERLAP, swap detection,
  overlay renders. Run before wiring; don't train on unverified masks.
- **`wire_gg_dataset.py`** — assemble undistorted frames + COLMAP poses + integer-label masks into
  GG's expected `images/ + object_mask/ + sparse/0/` layout. Refuses to overwrite an existing dst.
- **`select_by_identity.py`** — keep only Gaussians whose learned GG identity label is the object
  (classifier over `obj_dc` features); writes one segmented `.ply` per label, fresh names. This is
  the digital copy. (Phase 5/7.)
- **`diag_0323_clean.py` → `trim_flakes.py`** — two-pass cleanup: diag computes data-driven
  percentile thresholds; the filter removes sharp flakes (flat AND off-core) and soft blobs.
  Off-core is required for any removal, so the dense body is never touched. (Phase 6.)
- **`persist_desk_plane.py`** — deterministic (seeded) RANSAC desk-plane fit, persisted to
  `desk_plane.npy`. Fit **once per scene**; every later move REUSES it — never silently re-fit.
- **`move_object.py`** — rigid move of one object's Gaussians (positions R+t, quaternions by R, SH
  by the same R via exact per-band real-SH rotation), re-seat on the persisted plane, merge back.
  `--reduce_sh_to_dc` default ON (translated splats can't keep valid higher-SH); STOPs if the
  plane file is missing. (Phase 7.)
- **`inpaint_vacated.py`** — measures the contact-shadow extent from the radial desk-luminance
  profile (footprint + ring, not a guessed radius), removes the darkened desk Gaussians there, and
  fills with a dense local-brightness-matched clean-desk patch. STOPs if the plane is missing.

**Do NOT put `.ply`s, weights, `desk_plane.npy`, or captures here** — those are data, moved
separately per [`../docs/MIGRATION.md`](../docs/MIGRATION.md) and excluded by `.gitignore`.

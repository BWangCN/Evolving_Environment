# 3DGS-EvoHome Progress Log

> **Synced with:** [3DGS_Timeline.md](3DGS_Timeline.md)
> **Rule:** When a timeline task is completed, mark it `[x]` in both this file AND `3DGS_Timeline.md`.

---

## Week 1 — Mar 16–22 | Launch + Benchmark Framing

### Mar 16 (Sun) — Decision Day + A1/A2/A3 Verified

#### Part 1: Research & Decisions (done on Windows 4080 machine)

- Read through full technical pipeline document (`3DGS_EvoHome_Technical_Pipeline.md`)
- Created project progress tracking (`progress.md`)
- **VLA backbone decision: π0.5** (non-RLFT SOTA)
  - Compared π0 / π0.5 / π0.6 in detail (architecture, open-source status, GPU requirements)
  - Deep-dived into π0.5's knowledge insulation mechanism (block-wise causal attention mask, separate expert params, K/V weak gradient path)
  - Decided: **insulation (not isolation) + single-stage training with flow matching loss only**
  - Per-env LoRA + per-env TFA structurally eliminates cross-environment forgetting
  - Insulation vs isolation vs no-isolation will be an ablation experiment
- **3DGS tool analysis:** Compared gsplat / nerfstudio / original 3DGS (INRIA)
  - Recommendation: **nerfstudio (Feature Splatting)** for reconstruction+segmentation, **gsplat** for editing+batch rendering
  - Original 3DGS ruled out (non-commercial license, less scriptable)
- **Literature review completed:** 35+ papers on Gaussian segmentation, editing, inpainting, robot manipulation
  - Saved to `3DGS_Literature_Review.md`
  - Key finding: RoboSplat (RSS 2025, closest competitor) avoids inpainting problem — our opportunity
  - Best inpainting approach: Gaussian Grouping (clone + LaMa + fine-tune)
  - No existing work combines 3DGS data synthesis with continual learning — our unique contribution
- **3DGS tool decision locked: 方案 D** — Gaussian Grouping (core) + gsplat (batch rendering)
- **Windows compatibility:** Gaussian Grouping 需要 WSL2 运行，原生 Windows 太多 patch
- **Development environment decision:** 本地 4080 用 WSL2, 5090 做大规模重建, Hopper 做 VLA 训练
- **All decisions documented:** `decisions.md` (4 locked + 8 pending)
- **Pipeline gap analysis saved:** `pipeline_gap_analysis.md`
- **Simulation environment decision:** BEHAVIOR-1K / OmniGibson
- **EvoHome-Bench draft designed:** `EvoHome_Bench_Design.md`
  - 5 environments (Base → Add → Replace → Layout → Full Change)
  - 15 tasks per env (10 specific + 5 shared for forgetting measurement)
  - 3 language variants per task → 45 evaluations per env
  - Performance matrix protocol, FTS/FR/ZIC/CE metrics
  - Multi-view capture protocol for 3DGS (~90 images per env)
- **Git repo initialized:** github.com/BWangCN/Evolving_Environment (private)
- **Core pipeline code implemented (50 tests passing):** see Code Reference below
- **Next steps (5090 Linux):** Gaussian Grouping setup + π0.5 inference verification

#### Part 2: Linux 5090 Environment Setup + A1/A2/A3 End-to-End Verification

- **Cloned Gaussian Grouping** to `/home/bwang25/Desktop/Manipulation/gaussian-grouping/`
- **Created conda environment `gaussian_grouping`:**
  - Python 3.10, PyTorch 2.10.0+cu128, CUDA toolkit 12.8 (via conda)
  - RTX 5090 is Blackwell architecture (SM 120, compute capability 12.0)
  - **cu126 PyTorch does NOT work** — shows "sm_120 is not compatible" warnings. **Must use cu128+**
  - System CUDA is 13.1 (driver 590.48.01) — forward compatible with CUDA 12.8 runtime

- **Compiled CUDA extensions for Blackwell (SM 120):**
  - `diff-gaussian-rasterization` — compiled cleanly with `TORCH_CUDA_ARCH_LIST="12.0"`
  - `simple-knn` — **required a source fix:** added `#include <cfloat>` for `FLT_MAX` undefined error on newer CUDA
  - Build recipe: `PYTHONNOUSERSITE=1 CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0" pip install --no-build-isolation <package>`
  - Both extensions import and work correctly

- **Installed COLMAP 3.13.0** via conda-forge (CUDA-enabled)

- **Installed DEVA + SAM for mask preparation:**
  - DEVA package installed (`pip install -e .`)
  - Downloaded all pretrained models to `Tracking-Anything-with-DEVA/saves/`:
    - `sam_vit_h_4b8939.pth` (2.4 GB) — SAM ViT-H
    - `DEVA-propagation.pth` (277 MB)
    - `groundingdino_swint_ogc.pth` (694 MB)
    - `mobile_sam.pt` (41 MB)
    - `GroundingDINO_SwinT_OGC.py` (config)
  - `segment_anything` installed
  - **Known issue:** GroundingDINO CUDA extension compilation fails due to conda GCC 14.3 header conflicts (`_Float32` undefined errors in nvcc). **Not blocking** — only needed for custom dataset mask preparation; pre-converted datasets work fine. Fix: force system GCC (`CC=/usr/bin/gcc CXX=/usr/bin/g++`) or remove conda compiler packages.

- **A1/A2/A3 End-to-End Test — PASSED:**
  - **Dataset:** bear scene from HuggingFace (`mqye/Gaussian-Grouping`), 96 images
  - **A1 (COLMAP):** Pre-converted sparse reconstruction loaded successfully (96 cameras, 63,659 initial points)
  - **A2 (3DGS Training):**
    - 30,000 iterations completed in ~17 minutes
    - Training speed: ~30 it/s on RTX 5090 (30 GB VRAM used)
    - Loss: 1.257 (iter 0) → ~0.15 (iter 30,000)
    - Checkpoints saved at iterations 1,000 / 7,000 / 30,000
    - Final model: `output/bear/point_cloud/iteration_30000/point_cloud.ply` (935 MB) + `classifier.pth` (19 KB)
  - **A3 (SAM Segmentation Rendering):**
    - Rendered all 96 views (~1 min, 1.6 it/s)
    - Output types generated:
      - `renders/` — 96 photorealistic RGB novel-view images
      - `objects_pred/` — 96 predicted segmentation masks (per-object color coding)
      - `gt/` — 96 ground truth RGB images
      - `gt_objects_color/` — 96 ground truth segmentation masks
      - `objects_feature16/` — 96 identity encoding feature visualizations (16-dim)
      - `concat/` — 96 side-by-side comparison images (GT | Render | GT Seg | Pred Seg | Features)
  - **Segmentation quality:** Excellent
    - Bear statue (body + base) segmented as one coherent object (brown) across all viewpoints
    - Stone pedestal segmented separately (green)
    - Individual tree trunks segmented (purple, orange, light blue)
    - Foliage, ground, and background all separated
    - Segmentation is **3D consistent** — same object retains same color/ID across different viewpoints
  - **Conclusion:** Gaussian Grouping works end-to-end on RTX 5090 for our pipeline. Reconstruction is photorealistic, segmentation is clean and 3D-consistent. Ready to proceed to custom datasets and Stage B.

- **Output location:** `gaussian-grouping/output/bear/train/ours_30000/`
  - Visual results can be found at:
    - Rendered RGB: `renders/00000.png` through `renders/00095.png`
    - Segmentation: `objects_pred/00000.png` through `objects_pred/00095.png`
    - Comparisons: `concat/00000.png` through `concat/00095.png`
    - Feature maps: `objects_feature16/00000.png` through `objects_feature16/00095.png`

---

### Code Reference (src/)

#### `src/scene/` — Scene Object System

| File | What it does | Key usage |
|------|-------------|-----------|
| `object.py` | `SceneObject` dataclass — holds geometry (point cloud, bbox, centroid) + semantics (category, description) + affordances (graspable, is_container, is_surface) for one object | `obj = SceneObject("bowl_01", "bowl", "white bowl", point_cloud=pts)` |
| `affordance.py` | `infer_affordances(obj)` — given category name, auto-fills graspable/container/surface flags and valid task types. Extend `AFFORDANCE_DB` dict for new categories | `infer_affordances(obj)  # modifies obj in-place` |
| `registry.py` | `SceneObjectRegistry` — per-environment container for all objects. Query by category, affordance, task type. Get combined point cloud with exclusions (for collision checking) | `reg.add_object("mug_01", "mug", "red mug", point_cloud=pts)` / `reg.get_containers()` / `reg.get_scene_point_cloud(exclude=["bowl_01"])` |

#### `src/task/` — Task Planning & Language

| File | What it does | Key usage |
|------|-------------|-----------|
| `planner.py` | `TaskPlanner` — takes a registry, enumerates all valid `ManipulationTask(task_type, obj, target)` triples. Handles size filtering (e.g., object must fit in container opening) | `planner = TaskPlanner(registry)` / `tasks = planner.enumerate_tasks()` / `tasks = planner.enumerate_tasks([TaskType.PLACE_IN])` |
| `language.py` | `LanguageGenerator` — generates diverse natural language instructions per task. Currently rule-based templates; `backend="llm"` reserved for future LLM integration | `gen = LanguageGenerator()` / `gen.generate(task, n_variants=3)` → `["place the spoon in the bowl", ...]` |

#### `src/trajectory/` — Trajectory Generation Pipeline

| File | What it does | Key usage |
|------|-------------|-----------|
| `generator.py` | `TrajectoryGenerator` — given a `GraspPose` (mock or from AnyGrasp) + place target, generates phased waypoint trajectory: home→transit→pre_grasp→approach→grasp→lift→transit_place→pre_place→place→release→retreat. Also validates against `CollisionChecker` | `gen = TrajectoryGenerator()` / `traj = gen.generate_pick_place(grasp, place_target)` / `gen.validate_trajectory(traj, checker)` |
| `collision.py` | `CollisionChecker` — inflated safety volume collision detection using KDTree. Phase-aware margins: larger during transport, smaller during placement. Supports grasped-object-in-hand collision checking | `checker = CollisionChecker(scene_cloud, gripper_margin=0.03)` / `checker.check_trajectory(waypoints)` |
| `place_target.py` | `compute_place_target()` — computes 3D place position from target object's geometry + task type (place_on → top surface, place_in → inside container, place_next_to → offset, stack_on → directly above) | `target_pos = compute_place_target(TaskType.PLACE_IN, bowl_obj)` |
| `action_format.py` | `trajectory_to_actions()` — converts waypoint sequence to π0.5 delta actions `[Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper]`, normalized by `ACTION_POS_SCALE` and `ACTION_ROT_SCALE` | `actions = trajectory_to_actions(traj)  # shape (T-1, 7)` |
| `camera.py` | `compute_camera_poses()` — transforms EE poses to wrist camera poses via fixed EE-to-camera offset. Output used for 3DGS/gsplat rendering | `cam_poses = compute_camera_poses(traj)  # list of (pos, quat)` |
| `interpolation.py` | Low-level math: linear position interpolation (`lerp`), spherical orientation interpolation (`slerp`), quaternion↔euler conversions. All quaternions use wxyz convention | `slerp(q1, q2, t)` / `quat_to_euler(q)` / `euler_to_quat(rpy)` |

#### `src/config/franka.py` — Robot & Action Constants

Franka Panda specs (home pose, gripper limits, workspace bounds), wrist camera intrinsics, trajectory phase heights, collision margins, π0.5 action normalization scales. **Edit this file to tune parameters.**

#### End-to-end usage example

```python
from src.scene import SceneObjectRegistry
from src.scene.object import TaskType
from src.task import TaskPlanner, LanguageGenerator
from src.trajectory import (
    TrajectoryGenerator, GraspPose, CollisionChecker,
    compute_place_target, trajectory_to_actions, compute_camera_poses,
)

# 1. Build scene
reg = SceneObjectRegistry("E1")
reg.add_object("bowl_01", "bowl", "white bowl", point_cloud=bowl_pts)
reg.add_object("spoon_01", "spoon", "metal spoon", point_cloud=spoon_pts)

# 2. Plan tasks
tasks = TaskPlanner(reg).enumerate_tasks([TaskType.PLACE_IN])
lang = LanguageGenerator().generate(tasks[0], n_variants=3)

# 3. Mock grasp (replace with AnyGrasp on Linux)
grasp = GraspPose(position=spoon_pos, orientation=down_quat)

# 4. Compute place target from bowl geometry
place = compute_place_target(TaskType.PLACE_IN, reg.get("bowl_01"))

# 5. Generate trajectory
traj = TrajectoryGenerator().generate_pick_place(grasp, place)

# 6. Collision check (exclude bowl = place target)
checker = CollisionChecker(reg.get_scene_point_cloud(exclude=["bowl_01"]))
assert TrajectoryGenerator().validate_trajectory(traj, checker)

# 7. Convert to training data format
actions = trajectory_to_actions(traj)        # (T-1, 7) π0.5 actions
cam_poses = compute_camera_poses(traj)       # for 3DGS rendering
# → Render images at each cam_pose → pair with actions + lang → (I, a, l) training data
```

### Mar 16 (Sun) — continued, late night
- **Evaluation module implemented** (`src/evaluation/`):
  - `EvoHomeBenchMetrics`: input performance matrix → FTS, FR, ZIC, CE, EvoHome Score, method comparison table
  - `CARSScheduler`: competence-aware adaptive replay scheduling with decoupled VLM/decoder support
- **Batch stress test** (`tests/test_batch_generation.py`):
  - 165 pick-place trajectories across all task types, 90.3% acceptance rate
  - Action statistics: position deltas within [-3.4, 3.2] normalized, orientation deltas within [-3.1, 3.2]
  - Camera-EE distance perfectly consistent (std=0.000000m)
- **Total: 74 tests passing** (15 scene + 13 task + 22 trajectory + 4 batch + 20 evaluation)

### Mar 17 (Mon)
- **Critical finding: OmniGibson (Isaac Sim) does NOT support A100/H100** — requires RT Cores
  - Confirmed via NVIDIA docs + GitHub Issue #1872 (segfault on A100 headless)
  - OmniGibson role reduced to **evaluation only** (not in batch data synthesis pipeline)
- **Pipeline bottleneck analysis:**
  - Batch rendering (gsplat) is the bottleneck — 10K-50K images per environment
  - gsplat is pure CUDA, works on A100 — **Hopper is the main compute platform**
  - OmniGibson NOT in the data synthesis loop (Stage B is entirely 3DGS-based)
- **Evaluation platform alternatives investigated:**
  - ManiSkill 3 (SAPIEN/Vulkan): works on A100, GPU-parallel (30K+ FPS), but fewer household assets
  - Decision pending: OmniGibson on 5090 (richer assets, serial) vs ManiSkill on Hopper (faster, parallel)
- **Hopper storage plan designed:**
  - HOME (60GB, backed up): code repos, conda base, scripts
  - SCRATCH (unlimited, 90-day purge): conda envs, model weights, training outputs
  - PROJECTS (1TB): persistent checkpoints, final results
- **Key distinction articulated: EvoHome vs LIBERO**
  - LIBERO: continual learning unit = task (new task with new demos)
  - EvoHome: continual learning unit = environment (same tasks, changing world, zero new demos)
- **Development strategy decided:** 5090 先跑通最小 E2E pipeline → 再迁移 Hopper 做 scale

**Week 1 Tasks — Decisions (Windows):**
- [x] VLA backbone → π0.5
- [x] 3DGS tool → Gaussian Grouping + gsplat
- [x] Simulation env → BEHAVIOR-1K / OmniGibson
- [x] Grasp model → AnyGrasp + motion planner
- [x] Robot → Franka Panda
- [x] EvoHome-Bench draft
- [x] Target objects + environment sequence
- [x] Literature review (35+ papers)
- [x] Pipeline gap analysis

**Week 1 Tasks — Code Modules (Windows):** ✅ ALL DONE

- [x] `src/scene/`: SceneObject + Registry + AffordanceInference + tests
- [x] `src/task/`: TaskPlanner + LanguageGenerator + tests
- [x] `src/trajectory/`: TrajectoryGenerator, CollisionChecker, PlaceTargetComputer, ActionFormatter, CameraPoseComputer + tests
- [x] `src/config/franka.py`: Franka specs, action format, collision margins
- [x] `src/evaluation/`: EvoHomeBenchMetrics + CARSScheduler + tests
- [x] 74 tests passing, batch stress test 165 trajectories @ 90.3% acceptance

**Week 1 Tasks — Setup (Linux 5090):** ✅ ALL DONE (Mar 17)
- [x] Gaussian Grouping environment setup + A1/A2/A3 end-to-end verified (bear scene)
- [x] Gaussian Grouping scene editing: object removal + inpainting verified
- [x] gsplat 1.5.3 installed + PLY bridge verified (renders from arbitrary camera pose)
- [x] AnyGrasp SDK installed (MinkowskiEngine + pointnet2 compiled for Blackwell)
- [x] π0.5 (openpi) inference verified: (15, 8) action output from 224x224 RGB + prompt
- [ ] OmniGibson installation (evaluation only, deferred — not needed for data synthesis)

---

### Mar 17 (Mon) — continued, afternoon — Minimal E2E Pipeline Setup

#### Gaussian Grouping Scene Editing (Stage B8/B9 Verified)

- **Object Removal — PASSED:**
  - Removed bear statue (object ID 34, threshold 0.3) from trained 3DGS scene
  - Removal PLY saved: `output/bear/point_cloud_object_removal/iteration_30000/point_cloud.ply` (764 MB, down from 892 MB)
  - All 96 training views rendered to `output/bear/train/ours_object_removal/iteration_30000/`
  - Bear cleanly removed, leaving ghostly artifact hole on stone pedestal (expected — needs inpainting)
  - **Bug fixed:** `edit_object_removal.py` line 128 — empty test camera set caused `UnboundLocalError`. Added early return guard.

- **3D Inpainting — PASSED:**
  - Used pre-computed LaMa 2D pseudo labels from HuggingFace (`data/bear/images_inpaint_unseen/`)
  - Fine-tuned 10,000 iterations with L1 + LPIPS loss (λ_dlpips=0.5)
  - Training speed: ~8 it/s on RTX 5090 (27 GB VRAM)
  - First attempt OOM at iter 4800 during densification → retried with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → completed successfully
  - Inpaint PLY saved: `output/bear/point_cloud_object_inpaint/iteration_9999/point_cloud.ply` (2.8 GB — larger due to densified Gaussians filling the hole)
  - **Result quality: Excellent** — stone pedestal surface cleanly filled, no visible artifacts from any viewpoint
  - **Bug fixed:** `edit_object_inpaint.py` same empty test set bug as removal script.

- **Object Relocation — NOT IN OFFICIAL CODE:**
  - Gaussian Grouping has no built-in relocation (SE(3) transform on Gaussian groups)
  - Would require custom code to manipulate `_xyz` (positions) and `_rotation` (quaternions) of selected Gaussians
  - **Deferred to custom implementation** when working with tabletop scenes

#### gsplat Batch Rendering (Stage B13 Verified)

- **gsplat 1.5.3** installed into `gaussian_grouping` conda env via `pip install gsplat`
- **PLY bridge verified:** Loaded inpainted scene PLY (9,561,153 Gaussians) → rendered 640x480 RGB from arbitrary camera pose
- Render output: `(1, 480, 640, 3)` float tensor, range [0.007, 0.974]
- **CUDA JIT compilation note:** Must set `CUDA_HOME=/usr/local/cuda` (system CUDA 13.1 supports SM 120). Conda CUDA toolkit is only 12.4 and does NOT support `compute_120`.
- First JIT compile takes ~38 seconds; subsequent runs use cached kernels
- Test render saved: `Evolving_Environment/gsplat_test_render.png`

#### AnyGrasp Grasp Estimation (Environment Ready, License Pending)

- **Cloned** `anygrasp_sdk` to `/home/bwang25/Desktop/Manipulation/anygrasp_sdk/`
- **MinkowskiEngine v0.5.4** compiled successfully after two fixes:
  1. **NVTX headers:** Added CPATH to nvidia nvtx include dir
  2. **CUDA 12.8 `__to_address` fix:** Created local copy of `/usr/include/c++/11/bits/shared_ptr_base.h` with `std::__to_address` qualification (no sudo needed — used CPATH override)
  - Build recipe:
    ```
    CPATH="$FIX_INCLUDE:$NVTX_INCLUDE:/usr/include/x86_64-linux-gnu:$CPATH"
    CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0"
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas_library_dirs=${CONDA_PREFIX}/lib --blas=openblas
    ```
- **pointnet2** compiled and installed (SM 120)
- **Core dependencies** installed: numpy, Pillow, scipy, tqdm, open3d
  - `graspnetAPI` failed to install (distutils.msvccompiler removed in modern setuptools) — not needed for grasp detection, only for GraspNet-1Billion evaluation
- **OpenSSL 1.1** installed via conda for license_checker binary
- **Feature ID obtained:** `8535717028844380837`
- **⚠️ BLOCKING:** License registration required at https://forms.gle/XVV3Eip8njTYJEBo6 (~2 business days)

#### π0.5 (openpi) VLA Inference (Verified)

- **Cloned** `openpi` to `/home/bwang25/Desktop/Manipulation/openpi/` (with submodules: aloha, LIBERO)
- **Environment:** uv-managed `.venv` (Python 3.11.13, JAX 0.5.3 + CUDA 12, PyTorch 2.7.1)
  - Separate from `gaussian_grouping` env (different Python version + JAX)
  - Created via `GIT_LFS_SKIP_SMUDGE=1 uv sync`
- **Checkpoint downloaded:** `gs://openpi-assets/checkpoints/pi05_droid` → cached at `~/.cache/openpi/`
- **Inference test — PASSED:**
  - Config: `pi05_droid` (action_dim=32, action_horizon=15)
  - Input: 224x224 uint8 RGB images + text prompt + joint state
  - Output: `actions.shape = (15, 8)` — 15-step horizon, 8-dim (7 joint deltas + gripper)
  - Actions range: [-0.58, 0.97] (reasonable normalized range)
  - First inference: ~26s (includes JAX JIT compilation); subsequent calls much faster
  - **GPU memory:** ~12 GB for inference (cannot run simultaneously with GG inpainting)

#### CUDA Extension Rebuild Notes

- PyTorch is installed in **user-site** (`~/.local/lib/python3.10/site-packages/`), NOT in the conda env
- Current version: PyTorch 2.9.1+cu128 (was 2.10.0 before gsplat install)
- `simple-knn` and `diff-gaussian-rasterization` had to be recompiled after PyTorch version change
- Build recipe for all CUDA extensions on Blackwell:
  ```bash
  CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0" \
  CFLAGS="-I/usr/include -I/usr/include/x86_64-linux-gnu" \
  pip install --no-build-isolation --force-reinstall .
  ```

---

## Week 2 — Mar 23–29 | AnyGrasp + Scene Editing + First Data

**Week 2 Tasks:**
- [ ] AnyGrasp: receive license → test on 3DGS point clouds (5090)
- [x] ~~Gaussian Grouping: object segmentation + removal (5090)~~ — done Mar 17
- [x] ~~gsplat: PLY bridge + rendering from arbitrary camera pose (5090)~~ — done Mar 17
- [x] ~~π0.5: inference verification (5090)~~ — done Mar 17
- [ ] Gaussian Grouping: object relocation via SE(3) transform (custom code needed)
- [ ] Connect AnyGrasp → TrajectoryGenerator → gsplat end-to-end
- [ ] First rendered (I, a, l) triplets
- [ ] Evaluation platform decision: OmniGibson (5090) vs ManiSkill (Hopper)

### Mar 17 (Mon) — evening — ManiSkill 3 + π0.5 Integration

#### Simulation Platform Decision: ManiSkill 3

- **OmniGibson blocked:** Isaac Sim 4.5 crashes on RTX 5090 (Blackwell). Isaac Sim 5.0 migration has no ETA.
- **ManiSkill 3** selected: Vulkan rasterization + RT rendering mode, Franka Panda default, works on 5090 now
- **ManiSkill 3.0.0b22** installed into `gaussian_grouping` conda env
- **PickCube-v1** verified with `rt-fast` shader — Franka + red cube + green goal rendered at 512x512
- Missing deps fixed during install: `lxml`, `decorator` (user-site packaging issues)
- Build fix: `toppra` required `CC=/usr/bin/gcc` (conda GCC header conflict)

#### π0.5 → ManiSkill Integration (End-to-End Verified)

- **Architecture:** openpi policy server (port 8000, `.venv` Python 3.11) ↔ openpi-client (ManiSkill env, `gaussian_grouping` Python 3.10)
- **Action mapping:** π0.5 DROID output `(15, 8)` → take first step `[:7]` (drop terminate flag) → clip to `[-1, 1]` → ManiSkill `pd_ee_delta_pose`
- **Image mapping:** ManiSkill render (512x512) → resize to 224x224 uint8 → both `exterior_image_1_left` and `wrist_image_left` (ManiSkill has no default wrist cam)
- **State mapping:** ManiSkill `qpos[:7]` → `joint_position`, `qpos[7]` → `gripper_position`
- **Result:** Robot moves coherently (not random) but doesn't complete pick task. Expected — π0.5 pretrained on real DROID data, not ManiSkill sim. Action scale and visual domain differ.
- **Key insight for paper:** This confirms π0.5 can receive images from any source and produce actions. Fine-tuning on our 3DGS synthetic data will adapt it to our visual domain.
- **Output:** `pi05_maniskill_demo.mp4` (51 frames), start/end frame PNGs

#### Adapter v2: Correct Action Format (LIBERO, not DROID)

- **Critical discovery:** `pi05_droid` outputs **joint velocities** (8-dim), NOT delta EE pose. `pi05_libero` outputs **delta EE pose** (7-dim) which matches ManiSkill's `pd_ee_delta_pose` 1:1.
- **LIBERO action format:** `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]` (7-dim, meters/radians, quantile denormalized)
- **LIBERO state format:** `[ee_pos(3), ee_axis_angle(3), gripper_qpos(2)]` (8-dim)
- **Downloaded `pi05_libero` checkpoint** to `~/.cache/openpi/` (~6 GB)
- **Adapter written:** `src/adapters/pi05_maniskill.py`
  - `maniskill_obs_to_pi05()`: ManiSkill obs → LIBERO input (tcp_pose → axis-angle, qpos → gripper, resize 224x224)
  - `pi05_action_to_maniskill()`: LIBERO actions → ManiSkill (take step from chunk, clip [-1,1], scale factor)
  - `run_episode()`: Full loop with replanning every N steps
- **Test result:** Robot moves purposefully (approaches cube, opens gripper) but doesn't complete pick (visual domain gap). Action format confirmed correct.
- **Normalization details:** π0.5 uses quantile denormalization. Raw model output in [-1,1] → `(x+1)/2 * (q99-q01) + q01` → absolute units (meters, radians)
- **Action scale tuning needed:** ManiSkill controller gains differ from LIBERO's robosuite. The `action_scale` parameter in the adapter controls this.

**Week 2 Checkpoint:** Real grasp poses + scene editing + first rendered training images

---

## Week 3 — Mar 30–Apr 5 | Synthetic Data at Scale + VLA Baseline

**Week 3 Tasks:**
- [ ] Full E2E pipeline on E1 (hundreds of trajectories, 5090)
- [ ] Data augmentation: camera perturbation, pose randomization, lighting
- [ ] π0.5 zero-shot baseline → first LoRA finetune on E1 (5090 → Hopper)
- [ ] Sanity check: finetuned > zero-shot on E1?

**Week 3 Checkpoint:** VLA finetunes on synthetic data + zero-shot vs finetuned comparison

---

## Week 4 — Apr 6–12 | Continual Learning Framework (Hopper A100)

**Week 4 Tasks:**
- [ ] Per-env LoRA + TFA implementation on openpi (JAX)
- [ ] Two-env sequential adaptation: E1 → E2
- [ ] Baselines: naive FT, EWC, experience replay, LoRA-only
- [ ] First leaderboard numbers

**Week 4 Checkpoint:** Our method + ≥3 baselines have numbers on EvoHome-Bench

---

## Week 5 — Apr 13–19 | Scale to 5 Envs + Anti-Forgetting (Hopper A100)

**Week 5 Tasks:**
- [ ] Synthetic data for E2–E5
- [ ] Sequential adaptation: E1 → E2 → ... → E5, full 5×5 performance matrix
- [ ] 3DGS Environment Bank + CARS replay
- [ ] Compare: no replay vs uniform replay vs CARS

**Week 5 Checkpoint:** Full 5-env results + replay strategy comparison

---

## Week 6 — Apr 20–26 | Ablations + Analysis

**Week 6 Tasks:**
- [ ] Ablations: insulation vs isolation, clean vs noisy trajectories, LoRA rank, TFA size
- [ ] Per-task-type and per-change-type forgetting analysis
- [ ] EvoHome-Bench materials: overview figure, failure analysis

**Week 6 Checkpoint:** All ablations done + analysis complete

---

---

### Mar 17 (Mon) — late evening — Strategic Decisions (TBD)

#### Paper Framing: Data Source for VLA Fine-Tuning — TBD

Three options identified for how to generate the training data that fine-tunes the VLA:

**Framing A: Sim demos only (simplest)**
- Use simulator's built-in motion planner to generate expert demonstrations
- Focus paper contribution on CL method (per-env LoRA + TFA + CARS)
- 3DGS pipeline shown as proof-of-concept for real-world, not used in main experiments
- Risk: reviewers say "what's novel about fine-tuning on sim demos?"

**Framing B: 3DGS pipeline only (purist)**
- Use 3DGS pipeline even in simulation (render sim → reconstruct → edit → gsplat)
- Technically impressive but hard to justify when sim has better alternatives
- Risk: reviewers say "why not just use the sim's own trajectory generator?"

**Framing C: Compare both (strongest, recommended if time permits)**
- Baseline 1: sim motion planner demos → fine-tune VLA
- Baseline 2 (ours): 3DGS synthetic data → fine-tune VLA
- Show 3DGS achieves comparable performance to sim demos
- Argument: "In the real world, sim demos don't exist — 3DGS is the only option"
- CL story works with either data source (orthogonal contribution)
- Risk: most work, tight on 10-week timeline

**Decision: TBD.** Start with Framing A (fastest to get results), build toward C if time permits. The CL contribution (per-env LoRA + TFA + evolving environments) stands regardless of data source.

#### Evaluation Platform: Dual-Track — TBD

Running both in parallel, will decide which to use for the paper:

**Track 1: LIBERO on 5090 (fast path)**
- π0 already works at ~90%+ success (existing checkpoint)
- Modify environments via BDDL (add/remove objects, change layout)
- Show degradation → CL method → recovery
- Limitation: MuJoCo rendering, less photorealistic

**Track 2: ManiSkill 3 + fine-tune π0 on Hopper (slow path)**
- π0 currently 0% on ManiSkill (needs fine-tuning on ManiSkill demos)
- Generate demos via ManiSkill's motion planner (automatic)
- Convert to openpi format → LoRA fine-tune π0 on Hopper
- Better rendering (RT mode), GPU-parallel evaluation on A100
- Limitation: ~1 week setup before experiments can start

**Decision: TBD.** Start LIBERO now (immediate results). Submit ManiSkill fine-tuning to Hopper in parallel. Decide primary benchmark by Week 3 based on which produces better results.

#### VLA Backbone Choice — TBD

- **π0** as pretrained backbone (DROID/LIBERO checkpoints available)
- Incorporate **π0.5's knowledge insulation** into our CL framework
- Add **per-env LoRA** (VLM) + **TFA** (action decoder) for continual adaptation
- Paper framing: "We build on π0 architecture, add insulation [cite π0.5], and propose per-env LoRA + TFA for environment-level CL"
- Not attacked because: (1) π0.5 is not fine-tuned on any sim benchmark, (2) we clearly cite the architectural choices, (3) our contribution is the CL method + data pipeline, not the base VLA

Available simulation checkpoints:
- `pi0_libero` — π0 on LIBERO (~90%+)
- `pi05_libero` — π0.5 on LIBERO (96.85%) ← strongest
- `pi0_aloha_sim` — π0 on ALOHA sim
- Neither has ManiSkill checkpoint → ManiSkill requires fine-tuning either way

**Decision: TBD.** If LIBERO → use π0.5 directly (96.85%, no training needed). If ManiSkill → fine-tune π0 or π0.5 on ManiSkill demos (both need same work).

#### ManiSkill VLA Landscape — No Public π0/π0.5 Checkpoint Exists

| Model | ManiSkill checkpoint public? | Method | Best result |
|-------|------------------------------|--------|-------------|
| π0 | **No** | piRL did RL fine-tuning (not released) | 85.7% |
| π0.5 | **No** | piRL did RL fine-tuning (not released) | 84.8% |
| OpenVLA | Yes (HuggingFace) | SFT + PPO | 97.66% |
| Octo | Yes (HuggingFace) | SFT | 9-90% |

**Potential bonus contribution:** If we fine-tune π0.5 on ManiSkill and release the checkpoint, this would be the **first publicly available π0.5-ManiSkill model**. Low effort, high community impact.

#### ManiSkill Fine-Tuning Pipeline (if we go this route)

```
ManiSkill motion planning demos (.h5, no images)
    ↓  replay_trajectory (add RGB + pd_ee_delta_pose)
ManiSkill demos with rendered images (.h5)
    ↓  ManiSkill-to-LeRobot converter (official, PR #1174)
LeRobot v2.1 dataset (Parquet + MP4)
    ↓  openpi compute_norm_stats.py
    ↓  openpi train.py (LoRA fine-tune π0.5, config modeled on pi05_libero)
π0.5-ManiSkill checkpoint
```

Key gotchas: openpi needs LeRobot v2.1 (not v3.0), action format must be 7-dim delta EE pose, state must be 8-dim (ee_pos + axis_angle + gripper_qpos), no wrist camera by default in ManiSkill (duplicate base view or add one).

---

### Mar 18 (Tue) — ManiSkill Dataset Generation

#### Overnight Replay Approach — FAILED

- **Attempt 1 (motionplanning demos):** Replay with `--target-control-mode pd_ee_delta_pose` → 0.6-30% success. Control mode conversion (joint pos → delta EE) fails for most trajectories via IK solver.
- **Attempt 2 (RL demos + `--use-env-states`):** RL demos already in `pd_ee_delta_pose`, but replay failed because demos were recorded with `physx_cuda` (GPU sim) and replay ran on CPU sim. States diverge due to floating-point differences → 3-4 episodes saved per task regardless of tier.
- **Attempt 3 (RL demos + `--use-env-states --sim-backend gpu`):** Fixed GPU mismatch → 100% replay for PickCube (10/10). But other tasks still only 3-4 episodes — same PhysX non-determinism issue.
- **Conclusion:** Replay-based approach is fundamentally unreliable for GPU-sim recorded demos. Abandoned replay entirely.

#### Fresh Rollout Approach — SUCCESS

- **New strategy:** Load pre-trained RL policy checkpoint, run live in ManiSkill with RGB rendering, record trajectories directly. No replay = no determinism issues.
- **Script:** `scripts/generate_fresh_demos.py`
- **Critical bug found:** ManiSkill's PPO baseline uses **Tanh** activation, not ELU. Using ELU → 0% success. Using Tanh → 99% success. Activation function must match exactly.
- **RL policy success rates:** PickCube 98.6%, StackCube 93.5%, PullCube 100%, LiftPegUpright 98.9%

#### Episode Length Investigation

- **RL policy demos: ~15 steps** per episode (policy solves tasks aggressively fast)
- **ManiSkill standard: 50 steps** (`max_episode_steps=50` in official env registration)
- **LIBERO: 200-400 steps** at 20Hz
- **DROID: ~250 steps** at 15Hz
- **RPD (VLA on ManiSkill):** Explicitly filtered to ≤50 steps for VLA fine-tuning
- **Conclusion:** 50 steps at 20Hz is the ManiSkill standard and validated by RPD. Our episodes are short (~15 steps actual manipulation + idle after success) but this matches what RPD used.

#### Action Scaling Experiment

Tested scaling down RL policy actions for smoother trajectories:
- scale=1.0: 100% success, 15 steps (jerky)
- scale=0.5: 95% success, 24 steps (smoother)
- scale=0.3: 40% success, 38 steps (too slow for policy)
- scale=0.1: 0% success (policy can't finish)
- **Decision:** Keep scale=1.0 (matching RPD's approach). VLA action horizon handles the step count.

#### VLA Training Data Format

π0.5 fine-tuning requires `action_horizon` prediction. From 50-step episodes with `action_horizon=10`:
- Sliding window produces ~40 training samples per episode
- 2000 episodes × 40 = ~80,000 samples per task
- 4 tasks = ~320,000 total training samples

#### Final Dataset Generation (Running, Started Mar 18 ~1:00 PM)

- **Script:** `scripts/generate_fresh_demos.py --task all --num-demos 2000`
- **Log:** `logs/fresh_demos_all.log` (PID 287180)
- **Tasks:** PickCube-v1, StackCube-v1, PullCube-v1, LiftPegUpright-v1
- **Per episode:**
  - `rgb`: (T+1, 256, 256, 3) uint8 — RT-fast rendered, gzip compressed
  - `actions`: (T, 7) float32 — delta EE pose [dx,dy,dz,drx,dry,drz,gripper]
  - `state`: (T+1, 8) float32 — LIBERO format [ee_pos(3), axis_angle(3), gripper_qpos(2)]
  - `tcp_pose`: (T+1, 7) float32, `qpos`: (T+1, 9) float32
  - `instruction`: per-episode language string (e.g., "pick up the red cube and move it to the green target")
- **Settings:** max_episode_steps=50, control_freq=20Hz, 256×256 RGB, gzip compression
- **Estimated:** ~8 GB total, ~36 min generation time
- **Visualizations saved to:** `/home/bwang25/Desktop/Manipulation/visualization/`
  - Per-task: start/mid/end PNGs, slow-motion MP4s, 4-episode grid MP4s

#### Overnight Dataset Generation (Started Mar 17 ~10:35 PM)

- **Script:** `scripts/generate_maniskill_dataset.sh` (PID 173951)
- **Log:** `logs/dataset_generation.log`
- **Tasks:** PickCube-v1, StackCube-v1, PegInsertionSide-v1, LiftPegUpright-v1, PullCube-v1
- **Tiers:** 10, 50, 200, 500, 1000 demos per task (5 tiers × 5 tasks = 25 replay jobs)
- **Output:** `~/.maniskill/demos/<task>/tier_<N>/trajectory.rgb.pd_ee_delta_pose.*.h5`
- **Note:** motionplanning→pd_ee_delta_pose conversion has ~30-50% success rate (expected for control mode conversion). Larger tiers will have enough valid episodes.
- **Estimated time:** 4-8 hours total
- **Next step (morning):** Check log, verify rendered h5 files, then convert to LeRobot format

---

### Mar 17 (Mon) — late evening — Competitor Analysis: Continual Learning for VLAs

#### CRL-VLA (arXiv 2602.03445, Feb 2026)
**"CRL-VLA: Continual Vision-Language-Action Learning"** — Westlake/ZJU + HKU

- **Problem:** Task-level CL for VLAs via on-policy RL (PPO). New tasks arrive sequentially; model must learn without forgetting old tasks.
- **VLA backbone:** OpenVLA-OFT (not π0.5)
- **CL method:** RL-algorithmic — dual-critic architecture with frozen Goal-Conditioned Value critic (anchors old task values) + trainable MC critic (drives new task learning). Asymmetric advantage regulation bounds forgetting theoretically.
- **No LoRA, no EWC, no parameter isolation** — purely critic/advantage-side approach.
- **Benchmark:** LIBERO (task-stream CL). Single-task CL: 0.98 FAR with 0.03 forgetting (near-oracle).
- **No synthetic data, no 3DGS.**
- **Key limitation for us:** Task-level CL only — physical environment never changes. Only the language instruction changes between tasks.

#### LifeLong-RFT (arXiv 2602.10503, Feb 2026)
**"Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning"** — UCAS/CASIA

- **Problem:** Catastrophic forgetting during sequential SFT of VLAs. Proposes GRPO (Group Relative Policy Optimization, from DeepSeek-Math) as a drop-in replacement for SFT.
- **VLA backbone:** NORA-Long (3B, discrete actions via FAST+ tokenizer) — NOT π0.5 or OpenVLA
- **CL method:** Offline RL via GRPO + Multi-Dimensional Process Reward (token accuracy + trajectory alignment + format compliance). Experience replay with 5 demos per old task.
- **Full-parameter finetuning** — no LoRA, no parameter isolation.
- **Benchmark:** LIBERO (task-stream CL) + SimplerEnv + real-world Franka.
- **Key results:** +22% AUC improvement over SFT on LIBERO CL. LIBERO-Goal: AUC 54.4 → 90.3.
- **Real-world CL:** Beats π0 (SFT) and OpenVLA (SFT) on sequential Franka tasks.
- **No synthetic data, no 3DGS.** Still needs 10 demos per new task + 5 replay demos per old task.
- **Key limitation for us:** Task-level CL only. No environment/visual domain change.

#### Simple Recipe Works (arXiv 2603.11653, Mar 2026)
**"Simple Recipe Works: VLAs are Natural Continual Learners with RL"** — UT Austin

- **Key finding:** Simple sequential LoRA fine-tuning with RL **already achieves near-zero forgetting** across OpenVLA-OFT, π0, and OpenVLA on 5 benchmarks (LIBERO suites + RoboCasa + ManiSkill).
- **Explanation:** Large pretrained models + LoRA's low-rank constraint + on-policy RL = implicit anti-forgetting.
- **Implication for us:** Supports our use of LoRA as a strong CL primitive. But they only test task-level CL, not environment-level.

#### Positioning of EvoHome vs All Three

| Dimension | CRL-VLA | LifeLong-RFT | Simple Recipe | **EvoHome (Ours)** |
|-----------|---------|-------------|---------------|-------------------|
| **CL unit** | Task | Task | Task | **Environment** |
| **What changes** | Language goal | Language goal | Language goal | **Physical world** (objects, layout) |
| **Demos needed** | Online RL rollouts | 10 per new task | Online RL rollouts | **Zero** (3DGS scan only) |
| **Synthetic data** | None | None | None | **3DGS pipeline** (core contribution) |
| **CL mechanism** | Dual-critic RL | GRPO + replay | LoRA + RL | Per-env LoRA + TFA + CARS |
| **VLA backbone** | OpenVLA-OFT | NORA-Long (3B) | OpenVLA/π0 | **π0.5** |
| **Benchmark** | LIBERO | LIBERO | LIBERO+RoboCasa+ManiSkill | **EvoHome-Bench** (novel) |

**Our key differentiators (none of the three have these):**
1. **Environment-level CL** — the physical world changes, not just the task. This is a harder, more realistic problem.
2. **Zero-demonstration adaptation** — no new human demos or RL interaction needed. A phone scan of the new environment → 3DGS → unlimited synthetic data.
3. **3DGS data synthesis pipeline** — entirely novel in the VLA continual learning space.
4. **Generative replay via 3DGS Environment Bank** — replay from neural scene representations, not stored demos.

**Potential concern:** "Simple Recipe Works" shows LoRA + RL already handles task-level CL well. Our response: task-level CL is solved (or nearly so); **environment-level CL is the open problem**, and it requires a data pipeline (3DGS) that task-level methods don't address.

### Mar 18 (Tue) — continued, afternoon — Pipeline Visualization + Trajectory Smoothing

#### Pipeline Visualization Script (`scripts/visualize_pipeline.py`)

- **Created comprehensive Open3D + matplotlib visualization** with 5 modes:
  - `scene`: 3DGS point cloud with object segmentation highlighted (tested on bear scene)
  - `grasps`: AnyGrasp detections with gripper frames and coordinate axes (scale-adaptive sizing)
  - `trajectory`: 3D pick-place trajectory with phase-colored segments, gripper state indicators
  - `plot2d`: Matplotlib 2D analysis — XZ/XY views, action delta profiles, gripper state timeline
  - `all`: Run all above in sequence
- **Mock data mode** (`--mock`): Synthetic tabletop scene at robot scale for instant testing without GPU/3DGS
- **Real data mode**: Tested on bear scene (3M Gaussians, 338K object Gaussians, 256 object classes)
- **AnyGrasp on real 3DGS — VERIFIED:** Extracted bear object point cloud from segmented 3DGS → AnyGrasp detected 10 grasps (scores 0.05-0.07, low because bear scene is COLMAP-scale, not metric tabletop)
- **Outputs saved:** `logs/trajectory_viz_v2.png`, `logs/grasps_real_bear_v2.png`, `logs/scene_viz_v2.png`, `logs/trajectory_plot2d.png`

#### ManiSkill RL vs Motion-Planned Trajectory Comparison

- **Key finding: Both ManiSkill motion planner AND our TrajectoryGenerator produce piecewise-linear trajectories**
  - ManiSkill `plan_screw()` (mplib): straight-line screw motion in SE(3), time-optimal via TOPP-RA. Joint-space diffs std ~0.01, 50-86 steps, 5 plateau steps per trajectory.
  - Our `TrajectoryGenerator`: lerp+slerp interpolation between keyframes. 83 waypoints, constant actions within phases.
  - **RL policy trajectories are fundamentally different:** curved paths, continuously varying actions (diff std ~0.3), 12-17 steps, no plateaus, bang-bang behavior.
- **Decision: Keep linear trajectories** — they provide clearer action patterns for VLA learning. The visual signal (3DGS images) is the dominant adaptation signal, not trajectory shape.
- **Comparison plots saved:** `logs/maniskill_rl_trajectories_xz.png`, `logs/maniskill_rl_actions.png`, `logs/mp_vs_rl_actions.png`

#### Trajectory Corner Smoothing

- **Problem identified:** Sharp direction changes at phase boundaries within the same logical move (e.g., transit_pick → pre_grasp = 36.7°, transit_place → pre_place = 119.9°)
- **Solution:** Added `smooth_corners()` post-processing method to `TrajectoryGenerator`
  - Gaussian-weighted local blending around within-move phase boundaries only
  - `SMOOTH_TRANSITIONS` set defines which junctions to smooth: `{(transit_pick, pre_grasp), (transit_place, pre_place)}`
  - Junctions between separate moves (e.g., lift → transit_place) intentionally left sharp
  - Grasp and release contact poses pinned (zero drift)
  - Configurable `radius` parameter (default 3, tested up to 5)
- **Results (radius=3):**
  - transit_pick → pre_grasp: 36.7° → 18.5°
  - transit_place → pre_place: 119.9° → 83.6°
  - lift → transit_place: 60.1° → 60.1° (unchanged, separate moves)
  - Max position deviation: 24mm, mean: 1.2mm
- **All 22 trajectory tests still passing**
- **Comparison plots saved:** `logs/corner_smoothing_comparison.png`, `logs/selective_smoothing.png`

### Mar 18 (Tue) — evening — ManiSkill → 3DGS Reconstruction Pipeline

#### ManiSkill Multi-View Capture (`scripts/capture_maniskill_multiview.py`)

- **Script captures multi-view images** of ManiSkill tabletop scene for 3DGS reconstruction
- **YCB objects on table:** tomato soup can, mustard bottle (fell off), mug, bowl, banana + Franka Panda arm
- **90 views** at 512x512 with full RT ray tracing (32 spp, OptiX denoiser)
- Camera orbits hemisphere: radius 0.45-0.55m, elevation 25-60°, FOV 75°
- Custom camera entity with wider FOV than ManiSkill defaults
- PickCube default objects hidden during capture
- **Output:** `data/maniskill_tabletop/images/` (90 PNGs)

#### COLMAP Challenges with Synthetic Renders

- **COLMAP registered all 90/90 images** but only triangulated **3 sparse points**
- Root cause: synthetic renders have too-perfect surfaces → SIFT features match but triangulation fails due to lack of texture variance
- **Solution: Bypass COLMAP triangulation entirely** — wrote known camera poses directly in COLMAP text format
- Camera intrinsics known exactly (PINHOLE, fx=fy=333.6, 512x512)
- Generated **12,500 dense seed points** covering table, floor, objects, robot region
- **Key insight:** For sim-to-3DGS pipeline, COLMAP SfM is unnecessary — camera poses are known from the simulator

#### Segmentation Masks (`scripts/generate_masks_maniskill.py`)

- ManiSkill's `default` shader supports Segmentation texture; RT shader does not
- Generated **grayscale integer ID masks** (mode=L) from actor-level segmentation
- 90 masks at 512x512, unique IDs: 0-27 (table, floor, robot parts, 4 objects)
- Required matching camera poses between RT capture and default-shader mask generation

#### Gaussian Grouping Training — SUCCESS

- **Config:** 7000 iterations, 30 object classes, 12,500 initial points
- **Result:** 336,112 Gaussians (101 MB PLY), trained in ~5 min on RTX 5090
- **Reconstruction quality:** Excellent — wood table texture, object shapes, robot arm, shadows all faithfully reconstructed
- **Segmentation quality:** Objects correctly separated from table and floor, each with unique ID
- **Output:** `data/maniskill_tabletop/output_v3/`
  - `point_cloud/iteration_7000/point_cloud.ply` + `classifier.pth`
  - `train/ours_7000/renders/` (90 3DGS renders)
  - `train/ours_7000/gt/` (90 GT images for comparison)
  - `train/ours_7000/concat/` (side-by-side GT|Render|Seg comparisons)

#### LeRobot Dataset Conversion — VERIFIED

- **Script:** `scripts/convert_maniskill_to_lerobot.py`
- **Verification script:** `scripts/verify_lerobot_dataset.py`
- **Result:** 8,000 episodes, 94,865 frames, 12.8 GB at `~/.cache/huggingface/lerobot/bwang25/maniskill_pi05`
- All checks passed: episode counts match, frame lengths match, images valid, actions in [-1,1], instructions present
- Ready for π0.5 LoRA fine-tuning

#### Pipeline Validated: ManiSkill RT → Known Poses → Gaussian Grouping → 3DGS + Segmentation

This is the first end-to-end test of the sim-to-3DGS pipeline. The metric-scale reconstructed scene can now be used for: AnyGrasp grasp detection → trajectory generation → gsplat rendering → VLA training data.

#### Full Pipeline End-to-End: 3DGS → AnyGrasp → Trajectory → gsplat Render

- **AnyGrasp on metric-scale 3DGS:** Extracted tabletop region from reconstructed scene → 5 grasps detected (scores 0.10-0.22, widths 0.058-0.079m within Franka Panda range)
- **Trajectory from real grasps:** Best grasp → pick-place trajectory (83 waypoints, smoothed), 82 actions in π0.5 delta format
- **gsplat rendering verified:** Identified camera convention mismatch (gsplat uses OpenCV W2C = [R | -R@pos], not OpenGL). Fixed and verified — renders match Gaussian Grouping quality.
- **Trajectory visualization video:** `logs/trajectory_render/pick_place_trajectory.mp4` — 3DGS rendered scene with animated trajectory overlay (phase colors, gripper state, grasp/place markers)

#### Hybrid Rendering Architecture — Design Decisions (Mar 18)

Resolved 6 design bottlenecks for moving Gaussian clusters along robot trajectories:

**Bottleneck 1 — Grasp anchor point: RESOLVED**
- Use grasp contact position (from AnyGrasp) as the SE(3) anchor
- Physically correct: object pivots around contact point, not centroid
- `T_offset = inv(T_gripper_at_grasp) @ T_object`

**Bottleneck 2 — Gaussian rotation: RESOLVED**
- Full SE(3) transform on both positions AND quaternions
- `pos_new = R_delta @ (pos_old - anchor) + anchor_new`
- `quat_new = quat_delta * quat_old`
- No shortcuts — maintains physical correctness even with wrist rotation

**Bottleneck 3 — Robot arm frozen in 3DGS: RESOLVED**
- All major VLAs (π0, π0.5, OpenVLA) use third-person camera where arm is visible and critical
- π0.5 takes 3 images: `base_0_rgb` (third-person) + `left_wrist_0_rgb` + `right_wrist_0_rgb`
- **Solution: Hybrid depth compositing**
  - ManiSkill renders robot arm at each trajectory step (RGB + depth + segmentation mask, robot = actor IDs 2-14)
  - gsplat renders 3DGS scene + moved object Gaussians (RGB+D mode)
  - Per-pixel depth comparison: `final[px] = closer_surface_rgb`
  - Handles occlusion both directions (arm in front of object AND object in front of arm)

**Bottleneck 4 — Object removal / inpainting: RESOLVED**
- **Compositional 3DGS approach**: reconstruct empty environment and individual objects separately
- Empty table has full surface (no occlusion from objects)
- Each object is a separate Gaussian cluster stored in SceneObjectRegistry
- At render time: concatenate environment Gaussians + object Gaussians at desired SE(3) poses
- No inpainting ever needed — environment evolution = swap objects in/out of composition

**Bottleneck 5 — Shadows baked into 3DGS: RESOLVED**
- Not a concern. VLA images are 224×224 — shadows are negligible pixels
- Geometry and object identity are what the VLA learns from
- Simulator shadows are already low quality

**Bottleneck 6 — Rendering speed: RESOLVED — it's a strength**
- ~10s per trajectory (AnyGrasp + traj gen + dual render + compositing)
- 500 trajectories per environment in ~1.5 hours on single 5090
- 5 environments × 500 trajectories = 2,500 trajectories in ~10 hours
- Compare: 2,500 real-world demonstrations would take weeks of human effort

#### Final Rendering Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│ For each trajectory step:                                │
│                                                          │
│  ManiSkill                        gsplat                 │
│  ┌──────────────┐                ┌──────────────────┐   │
│  │ Set robot    │                │ Environment 3DGS  │   │
│  │ joint pos    │                │ (empty table)     │   │
│  │ via IK       │                │        +          │   │
│  │              │                │ Object Gaussians  │   │
│  │ Render:      │                │ (SE(3) moved to   │   │
│  │  RGB + Depth │                │  current EE pos)  │   │
│  │  + Seg Mask  │                │                   │   │
│  │ (robot only) │                │ Render: RGB + D   │   │
│  └──────┬───────┘                └────────┬──────────┘   │
│         │                                 │              │
│         └──────────┐   ┌──────────────────┘              │
│                    ▼   ▼                                 │
│              ┌─────────────────┐                         │
│              │ Depth Composite │                         │
│              │ Per-pixel:      │                         │
│              │  closer wins    │                         │
│              └────────┬────────┘                         │
│                       ▼                                  │
│              ┌─────────────────┐                         │
│              │ (Image, Action, │                         │
│              │  Language)      │                         │
│              │ Training Triplet│                         │
│              └─────────────────┘                         │
└─────────────────────────────────────────────────────────┘
```

---

## Mar 19 — SuGaR Mesh Extraction + Sim Pipeline

### SuGaR Mesh Extraction — WORKING (with caveats)

- **SuGaR installed** at `/home/bwang25/Desktop/Manipulation/SuGaR/`
- pytorch3d 0.7.9 built for Blackwell SM 120
- Rasterizer conflict resolved: GG needs `sh_objs` support, SuGaR doesn't. GG's rasterizer built in-place at `gaussian-grouping/submodules/diff-gaussian-rasterization/`. Use `PYTHONPATH` override when running GG.
- `np.byte` → `np.uint8` fix in `SuGaR/sugar_scene/cameras.py:107`

### Object Capture (ManiSkill → 3DGS)

- **Script:** `scripts/capture_individual_objects.py`
- 120 views per object, Fibonacci sphere coverage (-80° to +80° elevation)
- Objects: `005_tomato_soup_can`, `025_mug`, `024_bowl`, `011_banana`, `013_apple`
- **Key fix:** Remove `RenderBodyComponent` from `scene-0_ground` and `scene-0_table-workspace` entities — moving them to z=-100 is NOT enough, they still render
- White background via alpha compositing (SAPIEN RGBA → alpha channel → white bg)
- Object masks from alpha channel saved to `object_mask/`

### 3DGS → Mesh Pipeline Issues & Solutions

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Thin shell meshes | Only top-hemisphere views (20°-70° elev) | Full sphere capture (Fibonacci spiral) |
| Green noise in mesh | 3DGS reconstructs green background | White background + `--white_background` flag |
| Gray ground in images | SAPIEN ground plane renders with alpha=1 | Remove `RenderBodyComponent` from ground entity |
| SuGaR `np.byte` crash | NumPy version incompatibility | Change to `np.uint8` |
| GG `sh_objs` error | Wrong rasterizer loaded (SuGaR's vs GG's) | Build GG's rasterizer, use PYTHONPATH override |
| GG `config.json` not found | CWD-relative path | Pass `--config_file` explicitly |

### Sim Pipeline — WORKING (IK-based)

- **Script:** `scripts/run_sim_pipeline.py`
- Loads SuGaR OBJ meshes into ManiSkill as dynamic actors
- AnyGrasp detects grasps on mesh point clouds
- IK motion planning via mplib (all waypoints reachable)
- 11-step pick-place: start → transit → pre_grasp → approach → grasp → lift → transit_place → pre_place → place_descend → release → retreat
- Control: `pd_joint_pos` with motion planner (NOT `pd_ee_delta_pose` — action scaling mismatch)

### Visualization Paths

| What | Path |
|------|------|
| **Object input images** | `data/objects/<id>/images/` (120 PNG each) |
| **Object masks** | `data/objects/<id>/object_mask/` (120 PNG each) |
| **3DGS checkpoints** | `data/objects/<id>/output/point_cloud/iteration_7000/point_cloud.ply` |
| **3DGS renders** | `data/objects/<id>/output/train/ours_7000/renders/` |
| **3DGS GT vs Render** | `data/objects/<id>/output/train/ours_7000/concat/` |
| **SuGaR coarse models** | `SuGaR/output/coarse/<id>/sugarcoarse_.../15000.pt` |
| **SuGaR raw meshes** | `SuGaR/output/coarse_mesh/<id>/sugarmesh_...ply` |
| **Cropped OBJ meshes** | `data/objects/<id>/mesh/<id>.obj` |
| **Sim initial scene** | `logs/sim_pipeline/initial_scene.png` |
| **Sim trajectory video** | `logs/sim_pipeline/traj_0.mp4`, `traj_1.mp4`, `traj_2.mp4` |
| **Sim trajectory frames** | `logs/sim_pipeline/traj_0_frames/0000-0010.png` |
| **Sim trajectory overview** | `logs/sim_pipeline/trajectory_overview.png` |
| **3DGS quality check** | `/tmp/3dgs_quality_check.png` (input vs 3DGS render comparison) |
| **Mesh comparison plot** | `/tmp/sugar_all_meshes.png` or `/tmp/sugar_filtered_meshes.png` |

### VGGT Pipeline — WORKING (Mar 19, replaces 3DGS+SuGaR)

**Why pivot:** SuGaR mesh extraction from 3DGS failed repeatedly — background Gaussians produced noisy mesh fragments. VGGT (feed-forward transformer, CVPR 2025 Best Paper) reconstructs directly from images in seconds.

**Pipeline:** ManiSkill RGB captures → VGGT point cloud → mask filter → scale calibrate → Poisson mesh → ManiSkill sim

**Script:** `scripts/run_vggt_pipeline.py`

**Results (all 5 objects):**

| Object | Verts | Tris | Size (mm) | Expected (mm) |
|--------|-------|------|-----------|---------------|
| Tomato Can | 321,309 | 638,942 | 60x75x101 | 69x69x103 |
| Mug | 520,715 | 1,032,662 | 85x98x101 | 107x95x81 |
| Bowl | 351,687 | 700,231 | 162x159x105 | 162x162x55 |
| Banana | 92,012 | 182,988 | 198x81x63 | 70x199x38 |
| Apple | 349,445 | 695,588 | 63x71x75 | 76x76x72 |

**Sim pipeline:** All 3 trajectories executed successfully, all IK solutions OK.

**VGGT Visualization Paths:**

| What | Path |
|------|------|
| **All meshes overview** | `logs/vggt_pipeline/all_meshes_vggt.png` |
| **Sim initial scene** | `logs/vggt_pipeline/initial_scene.png` |
| **Trajectory overview** | `logs/vggt_pipeline/trajectory_overview.png` |
| **Trajectory videos** | `logs/vggt_pipeline/traj_0.mp4`, `traj_1.mp4`, `traj_2.mp4` |
| **Trajectory frames** | `logs/vggt_pipeline/traj_*_frames/` |
| **Object meshes** | `data/objects/<id>/mesh/<id>.obj` |

**Key improvements over 3DGS+SuGaR:**
- Objects are recognizable solid shapes (can, mug, bowl, banana, apple) — not splattered fragments
- Reconstruction takes seconds per object (vs 20+ min for 3DGS+SuGaR)
- No background noise problem (mask-filtered point cloud + Poisson mesh)
- Scale calibrated to real-world dimensions using known YCB bbox sizes

**Remaining issues to address:**
- Mesh visualization in matplotlib is sparse (dense meshes render poorly); actual OBJ files are high quality
- Bowl height is 2x expected (105mm vs 55mm) — VGGT scale estimation less accurate for flat objects
- Object colors in ManiSkill use material override (not VGGT vertex colors) — could improve with texture mapping
- Poisson mesh may have extra surface where point cloud has gaps — could improve with alpha shape or ball pivoting

### 3DGS → SuGaR Refined Textured Mesh — WORKING (Mar 20)

**Full SuGaR pipeline verified on tomato_soup_can:**
1. 3DGS training (white bg, 120 views, 7K iter) → 116K Gaussians ✓
2. SuGaR coarse (15K iter) → coarse mesh ✓
3. SuGaR extract mesh → mesh with correct size (67×66×104mm) ✓
4. SuGaR refine (15K iter) → refined Gaussians bound to mesh ✓
5. SuGaR extract textured mesh → .obj + .mtl + .png texture atlas ✓
6. Crop to object region → 50K verts, UV preserved ✓
7. Separate collision hull (watertight convex, 1815v) ✓
8. Import to ManiSkill → sits on table correctly, texture renders ✓

**Key fixes that made it work:**
- White background images + `--white_background` flag (no background Gaussians)
- Remove ground plane RenderBodyComponent (not just move to z=-100)
- Separate visual mesh (detailed, textured) and collision mesh (watertight convex hull)
- Crop refined mesh to object bbox before import (removes background geometry)
- Fix MTL file to reference texture PNG (`map_Kd`)

**Visualization paths:**

| What | Path |
|------|------|
| Input images | `logs/debug_3dgs/step1_input_images.png` |
| Object masks | `logs/debug_3dgs/step1_masks.png` |
| GT vs 3DGS renders | `logs/debug_3dgs/step3_gt_vs_render.png` |
| GT\|Render\|Seg | `logs/debug_3dgs/step3_concat.png` |
| Gaussian cloud | `logs/debug_3dgs/step4_gaussian_cloud.png` |
| All-angle renders | `logs/debug_3dgs/step4_all_angle_renders.png` |
| Mesh method comparison | `logs/debug_3dgs/mesh_method_comparison.png` |
| Flat-color mesh in sim | `logs/debug_3dgs/mesh_in_sim_test.png` |
| Textured mesh in sim | `logs/debug_3dgs/textured_mesh_cropped_in_sim.png` |
| Textured mesh files | `data/objects/005_tomato_soup_can/mesh/005_tomato_soup_can_textured_cropped.obj` + `.mtl` + `.png` |
| Collision mesh | `data/objects/005_tomato_soup_can/mesh/005_tomato_soup_can_collision.obj` |

**TBD:**
- [ ] Texture appears dark in ManiSkill default shader — try `rt` shader or adjust ambient light
- [x] Run full SuGaR refined pipeline on all 5 objects
- [ ] Integrate textured meshes into full sim pick-place pipeline

### Mar 20 (Thu) — Debugging Bowl/Mug + Functional Semantics Insight

#### Debug: Bowl & Mug Reconstruction Quality

Investigated why 024_bowl and 025_mug reconstructions are "very bad."

**Step 1: Input images are clean.** White backgrounds, good 120-view coverage, no green screen artifacts.

**Step 2: 3DGS reconstruction is fine.** GT vs render comparison shows near-perfect quality for all objects including bowl and mug. The Gaussian clouds have correct shapes (bowl concavity, mug handle visible). 3DGS handles concave objects perfectly because it's volumetric (splatting, not meshing).

**Step 3: SuGaR mesh extraction is the bottleneck.** Marching cubes level-set extraction fails on concave geometry:
- SuGaR meshes the entire scene (700K+ faces) including background Gaussians spread over 600-900mm
- Crop to object region captures only 2-3% of faces
- For concave objects, the cropped faces are **fragmented into many disconnected components**
- "Keep largest component" throws away most of the object: bowl keeps 13K/100K faces, mug 13K/100K
- Tomato can (convex) works because the cropped faces form one clean connected surface

| Object | SuGaR raw faces | In crop | After largest component | Quality |
|--------|----------------|---------|------------------------|---------|
| Tomato can | 326K | 2,888 | 101K | Good |
| Bowl | 736K | 21,362 | 13,586 | Bad (fragmented) |
| Mug | 766K | 17,651 | 13,246 | Bad (fragmented) |

**Decision: Not worth fixing.** SuGaR's concavity limitation is a known engineering issue, not a novel research problem. Not a paper contribution.

#### Gripper Width Analysis

Panda gripper max opening = 80mm. Object graspability:
- Tomato can (66mm) — fits ✓
- Banana (38mm min) — easy ✓
- Apple (73mm) — borderline (3.5mm clearance/side)
- Bowl (162mm) — impossible
- Mug (82mm + handle) — impossible

Checked ManiSkill alternatives: only XArm6 + Robotiq 2F-85 (85mm) available — only 5mm wider, not worth switching (requires new motion planner + full pipeline rewrite).

**Decision: Object selection problem, not gripper problem.** Select convex objects < 70mm for grasping. Use concave objects as semantic targets (containers).

#### Key Insight: Functional Semantic Digital Twins

**Thesis:** Digital twins need geometry + appearance + **functional semantics**.

Prior work (SyncTwist etc.) generates pick-and-place trajectories that ignore object identity — every object is just a rigid body to relocate. No semantic understanding.

**Our approach:** Use object functional roles (container, food, tool, stackable) to generate semantically meaningful long-horizon task sequences:
- "Pick up the apple and place it in the bowl"
- "Stack the cans"
- "Clear the table: fruits into bowl first, then move bowl"

**Implementation plan (proof of concept):**
1. Hard-coded template system for task planning (maps object roles + layout → task sequence)
2. Hard-coded trajectory generator for semantic primitives (pick, place-in, stack, etc.)
3. Demonstrates that functional semantics enable rich trajectory synthesis beyond trivial pick-place
4. Concave objects (bowl, plate) become *destinations*, not grasp targets — sidesteps both gripper and SuGaR issues

**Recommended object set:**
- Graspable (convex, <70mm): 005_tomato_soup_can, 011_banana, 004_sugar_box, 006_mustard_bottle, 010_potted_meat_can, 014_lemon, 016_pear
- Semantic targets (containers, use GT mesh): 024_bowl, 029_plate
- Enables combinatorial task generation from object relationships

---

### Mar 22 (Sat) — Hopper Setup + π0.5 LoRA Finetuning Pipeline Verified + Training Submitted

#### Hopper Cluster Setup (A100)

- [x] Connected to GMU Hopper via VPN (`vpn.gmu.edu`, GENERAL group) + SSH
- [x] Set up passwordless SSH (`ssh hopper` alias in `~/.ssh/config`)
- [x] Installed uv + cloned openpi to `~/openpi/`
- [x] Resolved `rerun-sdk` glibc incompatibility (`uv sync --no-install-package rerun-sdk`)
- [x] Verified JAX 0.5.3 sees GPU: `CudaDevice(id=0)` on A100 MIG 3g.40gb slice
- [x] Resolved SSL certificate issue for GCS downloads (`SSL_CERT_FILE` env var with certifi)
- [x] Downloaded pi0.5 base checkpoint (11.6GB) to `/scratch/bwang25/openpi_cache/`
- [x] Transferred tokenizer from 5090 to Hopper

#### Dataset Transfer

- [x] Transferred ManiSkill LeRobot dataset (13GB, 8000 eps) from 5090 to `/scratch/bwang25/maniskill_pi05/`
- [x] Created symlink from `$HF_HOME/lerobot/bwang25/maniskill_pi05` → scratch dataset
- [x] Verified dataset loads correctly in openpi data pipeline

#### Custom Training Config

- [x] Created `LeRobotManiSkillDataConfig` class in `config.py` — handles flat-key dataset format (image, wrist_image, state, actions → observation/image, observation/wrist_image, observation/state, actions, prompt)
- [x] Created `pi05_maniskill` config (full finetuning, needs A100 80GB)
- [x] Created `pi05_maniskill_lora` config (LoRA finetuning, fits 40GB MIG)
  - Model: `pi05=True, action_horizon=10, gemma_2b_lora + gemma_300m_lora`
  - LoRA ranks: 16 (VLM), 32 (action expert) — openpi defaults
  - Batch size: 32 (test) / 64 (production A100 80GB)
  - LR: 5e-5 cosine decay, 1K warmup
  - Steps: 20,000
  - Freeze filter: all base weights frozen, only LoRA params trained
  - EMA disabled (standard for LoRA)

#### Pipeline Verification

- [x] Computed normalization statistics (mean/std for states and actions, ~1 hour)
- [x] Full finetuning test on 40GB MIG: **OOM at 53.6GB** — confirms A100 80GB needed for full finetuning
- [x] LoRA finetuning test on 40GB MIG: **SUCCESS** — model loaded, data flowing, training steps executing
  - Checkpoint loaded in 5 seconds (12.5GB, 2.5 GiB/s)
  - Data pipeline verified: 32×224×224×3 images, 32×32 states, 32×200 tokenized prompts
  - LoRA parameters visible: `lora_a` and `lora_b` in attention (q, k, v, out) and MLP layers

#### Production Training Submitted

- [x] Created SLURM script `slurm_train_pi05_lora.sh` (A100 80GB, 2-day limit, batch_size=64)
- [x] Submitted job to Hopper `gpuq` partition: `sbatch slurm_train_pi05_lora.sh`
- [x] Job queued (SLURM job ID 6589595), pending A100 allocation
- [x] wandb tracking: `beichenwang2000-george-mason-university/openpi`
- [x] Checkpoints saved to `/scratch/bwang25/checkpoints/pi05_maniskill_lora/pi05_lora_v1/`
- [x] Estimated training time: ~30 hours

**Key files on Hopper:**
- Config: `~/openpi/src/openpi/training/config.py` (LeRobotManiSkillDataConfig + pi05_maniskill_lora)
- SLURM: `~/openpi/slurm_train_pi05_lora.sh`
- Norm stats: `~/openpi/assets/pi05_maniskill_lora/`
- Logs: `/scratch/bwang25/logs/pi05-lora-*.out`

### Mar 23 (Sun) — Evaluation + Training Data Quality Analysis

#### Finetuned pi0.5 LoRA Evaluation on ManiSkill (5090)

- [x] Training completed on Hopper A100 40GB (~24 hours, 20K steps, batch_size=32)
- [x] Downloaded checkpoint (step 19999) + norm stats to 5090
- [x] Policy server launched, inference loop running on ManiSkill PickCube
- [x] **Result: 0/5 success** — model outputs saturated actions, no coherent manipulation

#### Root Cause Analysis

- [x] **Camera view mismatch ruled out** — training data and inference both use `env.render()` (same third-person view, verified by visual comparison)
- [x] **Action space confirmed matching** — both use `pd_ee_delta_pose` [-1, 1] format
- [x] **Root cause identified: RL demo quality** — RL policy actions are jerky, saturating at ±1.0, with only 17 steps per episode
  - RL actions: range [-1.0, +1.0], step delta mean abs = 0.1505, episodes = 17 steps avg
  - Motion planning actions (converted): range [-0.25, +0.40], step delta mean abs = 0.005, episodes = 77 steps avg
  - RL is **30x less smooth** than motion planning
  - pi0.5 action chunks (horizon=10) get only 1-2 replanning cycles per 17-step RL episode
  - LIBERO actions (reference): rotation q01~[-0.12, -0.19], q99~[0.14, 0.31] — much smaller than RL's ±1.0
  - Quantile normalization on RL data maps [-1, 1] → [-1, 1] (no useful normalization)

#### Decision: Switch to Motion Planning Demos

- [x] **Checked MP demo availability:** PickCube-v1 (1000 eps, 74 steps avg), StackCube-v1 (1000 eps, 107 steps avg). PullCube-v1 and LiftPegUpright-v1 have no MP demos.
- [x] **ManiSkill replay tool verified:** converts joint position actions → pd_ee_delta_pose (100% success on test batch)
- [x] Converted actions are smooth: pos deltas ~0.01-0.15m, rot deltas ~0.01-0.12rad (matches LIBERO distribution)
- [x] Each 77-step MP episode → 68 valid training samples (vs 8 from RL) = **8.5x more data**
- [x] Step 1 DONE: ManiSkill replay tool converted MP trajectories → pd_ee_delta_pose for PickCube (1000 eps, 78K frames) and StackCube (1000 eps, 108K frames). 100% success.
- [x] Converted action quality verified: pos deltas ~[-0.2, +0.35], rot deltas ~[-0.08, +0.08] — matches LIBERO distribution
- [ ] **TBD: Step 2** — Re-render converted MP trajectories with RGB (`generate_mp_rgb.py`). Script needs fix: `env_states` is stored as HDF5 group, not array. Need to handle nested state dict format for `set_state_dict()`.
- [ ] **TBD: Step 3** — Convert to LeRobot format (`convert_maniskill_to_lerobot.py --demo-dir ~/.maniskill/demos_mp`)
- [ ] **TBD: Step 4** — Upload to Hopper, compute new norm stats, retrain LoRA
- [x] Scripts created: `generate_mp_rgb.py`, `generate_mp_dataset.sh`, updated `convert_maniskill_to_lerobot.py` (added `--demo-dir` flag)

### Mar 24 (Mon) — NaN Debugging + Normalization Fix + Pipeline Video + Project Direction

#### Training NaN Debugging

- [x] v2 training (MP data) produced NaN at step 0 on Hopper
- [x] Confirmed data is clean (no NaN/Inf in dataset or norm stats)
- [x] Confirmed `use_quantile_norm=False` still produces NaN → pi0.5 requires quantile normalization
- [x] Confirmed widening q01/q99 to min range 0.2 still produces NaN
- [x] **Root cause**: MP motion planner keeps gripper orientation fixed → rotation deltas ~±0.04 → quantile norm with MP's own ranges amplifies gradients → NaN
- [x] Confirmed RL dataset still trains successfully with current config → issue is MP-specific
- [x] **Fix (D15)**: Use RL data's normalization stats (q01/q99 ranges [-1, 1]) for MP data. Small MP values map to small normalized values, matching pi0.5's expected input distribution
- [x] Verified fix works: Step 0 produces real loss on interactive GPU node
- [x] v3 training submitted on Hopper A100 40GB (batch_size=32, 20K steps, RL norm stats + MP data)

#### Pipeline Visualization

- [x] Created `record_pipeline_video.py` — smooth 20fps video of full pipeline
- [x] Recorded video: SuGaR meshes → AnyGrasp (10 grasps) → motion-planned pick-and-place
- [x] Tested textured meshes — SuGaR textures are too dark (baked lighting). Solid colors used instead.
- [x] Video: `logs/pipeline_video/full_pipeline.mp4` (9.1s, 181 frames)

#### Texture Quality (Parallel Investigation — Non-Blocking)

- [ ] SuGaR texture extraction bakes lighting into albedo → dark textures in simulation
- [ ] Options to investigate: relighting, albedo extraction, different reconstruction, or accept solid colors
- [ ] **Not blocking**: solid colors work for benchmark. Texture improvement is a quality enhancement.

#### Project Direction (Prof Xiao Guidance)

- Professor advised: the **pipeline + benchmark** is the core contribution, not a novel CL algorithm
- Standard CL methods (LoRA + replay) are sufficient for experiments
- Priority: make EvoHome-Bench solid, run thorough experiments, show VLA degradation + mitigation
- Texture quality is parallel/optional — solid colors are acceptable for controlled benchmarks

---

## Weeks 7–10 — Apr 27–May 28 | Writing + Polish + Submission

- **Week 7 (Apr 27–May 3):** Supplementary experiments + Method/Bench writing skeleton
- **Week 8 (May 4–10):** All figures/tables final + Experiment/Bench draft + supplementary video
- **Week 9 (May 11–17):** Intro, Related Work, Conclusion, Abstract. Draft to Professor by **May 14**
- **Week 10 (May 18–28):** Revisions, polish. **May 25: SUBMIT ABSTRACT. May 28: SUBMIT PAPER.**

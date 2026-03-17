# 3DGS-EvoHome Progress Log

> **Synced with:** [3DGS_Timeline.md](3DGS_Timeline.md)
> **Rule:** When a timeline task is completed, mark it `[x]` in both this file AND `3DGS_Timeline.md`.

---

## Week 1 — Mar 16–22 | Launch + Benchmark Framing

### Mar 16 (Sun)
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

**Week 1 Tasks — Setup (Linux 5090):** DEFERRED (machine down Mar 16, back online Mar 17)
- [ ] Gaussian Grouping + first scene reconstruction
- [ ] π0.5 inference on LIBERO
- [ ] OmniGibson installation (evaluation only, not needed for data synthesis)

---

## Week 2 — Mar 23–29 | AnyGrasp + Scene Editing + First Data

**Week 2 Tasks:**
- [ ] AnyGrasp: install + license + test on 3DGS point clouds (5090)
- [ ] Gaussian Grouping: object segmentation + removal + relocation (5090)
- [ ] gsplat: first rendered (I, a, l) triplets (5090)
- [ ] π0.5: inference verification (5090)
- [ ] Evaluation platform decision: OmniGibson (5090) vs ManiSkill (Hopper)

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

## Weeks 7–10 — Apr 27–May 28 | Writing + Polish + Submission

- **Week 7 (Apr 27–May 3):** Supplementary experiments + Method/Bench writing skeleton
- **Week 8 (May 4–10):** All figures/tables final + Experiment/Bench draft + supplementary video
- **Week 9 (May 11–17):** Intro, Related Work, Conclusion, Abstract. Draft to Professor by **May 14**
- **Week 10 (May 18–28):** Revisions, polish. **May 25: SUBMIT ABSTRACT. May 28: SUBMIT PAPER.**

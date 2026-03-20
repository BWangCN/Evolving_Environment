# 3DGS-EvoHome Technical Decisions

> **Last Updated:** March 19, 2026 (MAJOR: drop 3DGS as core tech → per-object scanning + VGGT, method-agnostic framework)
> **Synced with:** [3DGS_Timeline.md](3DGS_Timeline.md) | [progress.md](progress.md) | [pipeline_gap_analysis.md](pipeline_gap_analysis.md)

---

## Locked Decisions

### D1. VLA Backbone — π0.5
- **Choice:** π0.5 (Physical Intelligence, non-RLFT SOTA)
- **Why not π0:** No knowledge insulation — VLM backbone gets polluted by action loss gradients
- **Why not π0.6:** Not officially open-sourced; RECAP RL mechanism adds unnecessary complexity
- **Codebase:** [openpi](https://github.com/Physical-Intelligence/openpi) (JAX), model weights on HuggingFace (`lerobot/pi05_base`)
- **Date:** Mar 16, 2026

### D2. Training Strategy — Insulation + Single-Stage
- **Choice:** Keep π0.5's native knowledge insulation (block-wise causal attention mask), single-stage training with flow matching loss only
- **Why not full isolation (stop_gradient):**
  - Would require a separate VLM loss (undefined, potentially misaligned)
  - Weak K/V gradient to LoRA is beneficial — aligns VLM representations toward action generation
  - Per-env LoRA + per-env TFA already structurally eliminates cross-environment forgetting
- **Why not no insulation (π0-style):** Full gradient backflow to VLM risks representation distortion on small synthetic datasets
- **Ablation plan:** Compare insulation vs isolation vs no-isolation as ablation experiment
- **Date:** Mar 16, 2026

### D3. 3D Reconstruction — Per-Object Scanning, Method-Agnostic (REVISED Mar 19, twice)

> **MAJOR PIVOT:** 3DGS is no longer a core technology. The framework is reconstruction-method-agnostic. 3DGS is one option among several (VGGT, Hunyuan3D, etc.) and may appear as an ablation, not as the headline contribution.

- **Choice:** Per-object scanning (not scene-level). Each new object is photographed individually (e.g., on green screen or turntable) → reconstructed to mesh → imported into sim.
- **Why per-object instead of scene-level:**
  - Eliminates the need for segmentation entirely (no Gaussian Grouping, no SAM)
  - Avoids green screen / table bleeding artifacts during mesh extraction
  - More realistic user interaction — nobody re-scans their whole kitchen for one new mug
  - Scene layout captured separately (one overhead photo + object detection, or RGB-D frame)
- **Recommended reconstruction method: VGGT** (Facebook Research, arXiv 2503.11651)
  - Feed-forward transformer: 5-10 images → point cloud + depth + camera params in <1 second
  - Point cloud → Poisson/Ball Pivoting → mesh (~1-2 minutes total per object)
  - Open source, commercial license available, 2-21 GB VRAM
  - Used by SyncTwin (arXiv 2601.09920) for robotic digital twin construction
- **Alternative reconstruction methods (for ablation):**
  - 3DGS (Gaussian Grouping or vanilla): higher fidelity but 30-60 images, 15-30 min
  - Hunyuan3D 2.0: single image, but scale ambiguity (needs RGB-D + ICP)
  - 2DGS: best mesh quality, but requires retraining
- **What's removed from pipeline:**
  - ~~Gaussian Grouping~~ → not needed (no scene-level segmentation)
  - ~~SAM / DEVA~~ → not needed
  - ~~gsplat~~ → not needed (sim handles rendering)
  - ~~SuGaR~~ → not needed (VGGT outputs point cloud directly)
  - ~~COLMAP~~ → VGGT replaces it (or use known camera poses from turntable)
  - ~~Hybrid depth compositing~~ → sim handles everything
  - ~~3DGS inpainting (LaMa)~~ → not needed (sim composes from individual objects)
- **What remains from 3DGS era:**
  - Object Library concept (store mesh + appearance per object, reuse across environments)
  - The "scan → reconstruct → sim" paradigm (but with lighter tools)
- **Impact on paper:**
  - Title must change: remove "3D Gaussian Splatting", focus on "digital twin" / "scene digitization" / "evolving environments"
  - Framework is method-agnostic: VGGT vs 3DGS vs Hunyuan3D as ablation
  - Stronger paper: not tied to one 3D reconstruction method
- **Competitor context:**
  - SyncTwin (arXiv 2601.09920): VGGT → mesh → Isaac Sim, 71-93% success, planning-only, no CL
  - Li et al. (arXiv 2603.13825): Hunyuan3D → mesh → Isaac Sim, 12-62% success, planning-only, no CL
  - Both validate "digital twin → sim" approach but neither does learning or CL
- **Date:** Mar 16 (3DGS core), Mar 17 (3DGS verified), Mar 19 AM (3DGS as digitizer), **Mar 19 PM (MAJOR: drop 3DGS as core, per-object VGGT, method-agnostic)**

### D4. Development Environment (UPDATED Mar 17)
- **Windows 4080 (local):** Documentation, code modules (no GPU deps), quick iteration
- **Linux 5090 (dev machine):** **Primary dev — 3DGS, AnyGrasp, π0.5 inference, OmniGibson eval** (has RT Cores)
- **GMU Hopper (A100 cluster):** VLA finetuning (LoRA + TFA), gsplat batch rendering at scale
- **Strategy:** 5090 先跑通最小 E2E pipeline → 迁移 Hopper 做 scale
- **Hopper storage:** HOME (code, 60GB) / SCRATCH (envs + weights, unlimited, 90d purge) / PROJECTS (results, 1TB)
- **5090 Environment Details (verified Mar 17):**
  - RTX 5090: Blackwell architecture, SM 120 (compute capability 12.0), 32 GB VRAM
  - System CUDA: 13.1 (driver 590.48.01) — **required for gsplat JIT** (conda CUDA 12.4 doesn't support SM 120)
  - Conda env `gaussian_grouping`: Python 3.10, PyTorch 2.9.1+cu128 (user-site), CUDA toolkit 12.8
    - Contains: Gaussian Grouping, gsplat 1.5.3, MinkowskiEngine 0.5.4, AnyGrasp deps, pointnet2
  - openpi `.venv`: Python 3.11, JAX 0.5.3, PyTorch 2.7.1 (separate env, uv-managed)
    - Checkpoint cached at `~/.cache/openpi/`
  - **Cannot run GG inpainting + π0.5 inference simultaneously** (each uses 12-27 GB VRAM)
- **Date:** Mar 16–17, 2026

---

### D5. Simulation Environment — ManiSkill 3 — DECIDED Mar 17
- **Choice:** ManiSkill 3 (SAPIEN/Vulkan) for evaluation + visualization
- **Why not OmniGibson:** Isaac Sim 4.5 crashes on RTX 5090 (Blackwell). OmniGibson's Isaac Sim 5.0 migration has no ETA.
- **ManiSkill advantages:** Works on 5090 now, has RT rendering mode (`rt-fast`), Franka Panda default, 2K+ objects (YCB, PartNet), GPU-parallel evaluation on A100
- **ManiSkill limitations:** Fewer household assets than BEHAVIOR-1K (2K vs 9K), RT mode is single-env only
- **Installation (Mar 17):** ManiSkill 3.0.0b22 installed in `gaussian_grouping` conda env. PickCube-v1 verified with rt-fast shader on 5090.
- **Data synthesis pipeline (REVISED Mar 19):** 3DGS → export mesh → ManiSkill (trajectory execution + rendering). ManiSkill IS now in the data synthesis loop (handles all rendering + physics). gsplat no longer used for training image rendering.
- **Future option:** Migrate to OmniGibson when Isaac Sim 5.0 port lands (richer assets, better photorealism)
- **Date:** Mar 16–17, 2026

### D6. Grasp Model — AnyGrasp — INSTALLED Mar 17
- **Choice:** AnyGrasp (T-RO 2023)
- **Why not GraspGen:** Too new (2025), less community adoption, harder to justify to reviewers
- **Why not GaussianGrasper:** Its language grounding is redundant (we already have object segmentation); grasp quality itself isn't better than AnyGrasp
- **Pipeline:** 3DGS → segment object → extract point cloud from Gaussians → AnyGrasp → all valid grasp poses used as diverse training data (no selection heuristic)
- **Installation status (Mar 17):** SDK cloned, MinkowskiEngine 0.5.4 + pointnet2 compiled for Blackwell. License registration submitted (feature ID: `8535717028844380837`). Awaiting approval (~2 business days).
- **Fallback:** Contact-GraspNet (fully open-source) if license is denied
- **Date:** Mar 16, 2026 (installed Mar 17)

### D7. Trajectory Generation — Motion Planner (NOT Reference Trajectory Library)
- **Choice:** Use classical motion planner to generate trajectories from grasp poses. **Eliminates** the reference trajectory library entirely.
- **Pipeline:** AnyGrasp generates N candidates → collision check filters to M valid → motion planner generates trajectory per grasp (home → pre-grasp → approach → grasp → lift) → render in 3DGS
- **Why this is better:** True zero-demonstration (not "amortized demonstration"); no dependency on human demos; diverse grasp poses = diverse training data; motion planning is classical and reliable
- **Impact:** B5 (trajectory matching) and B6 (trajectory retargeting) from original pipeline are eliminated
- **Date:** Mar 16, 2026

### D8. Robot Initial Position — Fixed Home Pose + Minor Randomization
- **Choice:** Fixed home pose (arm above workspace ~40cm) with small perturbations (±2-3cm position, ±5° orientation)
- **No large randomization:** VLA learns visuo-motor policy; large initial position changes make same-object observations too different
- **Diversity comes from:** Environment composition changes (evolving objects/layouts), NOT robot starting position
- **Date:** Mar 16, 2026

### D9. Robot Platform — Franka Panda
- **Choice:** Franka Panda (7-DOF, parallel-jaw gripper, max opening 8cm)
- **Why:** BEHAVIOR-1K official standard, π0/π0.5 evaluation standard, parallel-jaw compatible with AnyGrasp
- **Home pose:** Joint angles [0, -π/4, 0, -3π/4, 0, π/2, π/4], EE ~40cm above workspace
- **Date:** Mar 16, 2026

### D10. Pipeline Architecture — Per-Object Digital Twin + Sim (REVISED Mar 19 PM)
- **Choice:** Per-object 3D scanning → mesh → Object Library → compose in ManiSkill → sim handles ALL rendering, physics, trajectory execution
- **Full pipeline:**
  ```
  Offline (each new object, once):
      Object on green screen/turntable → 5-10 photos → VGGT → point cloud → mesh
      → store in Object Library (mesh + metadata)

  Online (when environment changes):
      Overhead photo of table → detect which objects, where →
      retrieve meshes from Object Library →
      place in ManiSkill at detected poses →
      AnyGrasp → motion plan → sim execute → record (I, a, l) →
      π0.5 LoRA fine-tune (per-env adapter)

  Continual (across environments):
      Object Library persists across E1→E2→...→En
      Replay: reload old environment's objects + layout in sim → regenerate data
  ```
- **Key advantages over 3DGS-centric pipeline:**
  - No segmentation needed (per-object scan = no table/background contamination)
  - No COLMAP, no Gaussian Grouping, no gsplat, no SuGaR, no SAM
  - 2 minutes per object (VGGT) vs 45 minutes (3DGS pipeline)
  - Reconstruction method is pluggable (VGGT / 3DGS / Hunyuan3D)
- **Key advantages over Li et al. / SyncTwin:**
  - They do planning (zero-shot, 12-93% success); we do learning (VLA adapts over time)
  - They need RGB-D sensor for pose alignment; our turntable setup needs only RGB
  - We address continual adaptation; they don't
- **Object Library replaces 3DGS Environment Bank:**
  - Store: per-object mesh + texture + metadata (size, category, affordances)
  - Store: per-environment layout snapshot (object IDs + poses)
  - Replay: reload any historical environment in sim from stored objects + layout
  - Simpler than storing 3DGS scenes, more flexible (remix objects across environments)
- **Paper contributions (updated):**
  1. Problem definition: evolving environment CL for VLAs (+ EvoHome-Bench)
  2. Digital twin framework: scan → reconstruct → sim → synthesize data (method-agnostic)
  3. Decoupled CL: per-env LoRA + TFA + CARS + Object Library replay
  4. Ablation: VGGT vs 3DGS vs Hunyuan3D as reconstruction backends
- **Date:** Mar 19, 2026

---

## Pending Decisions

### ~~P2. Inpainting Strategy~~ — REMOVED (Mar 19)
- **Reason:** Per-object scanning + sim composition eliminates all inpainting needs. No holes to fill.

### P3. Language Instruction Generation
- **Candidates:** LLM template expansion, manual template library
- **Target:** Week 2

### P5. LoRA Rank
- **Candidates:** r = 8, 16, 32
- **Target:** Week 3-4 (empirical tuning)

### P6. TFA Residual Architecture
- **Open questions:** MLP vs shallow transformer? How many parameters?
- **Target:** Week 3-4

### P7. Number of EvoHome-Bench Environments
- **Current plan:** 5 environments
- **Open:** Could go to 7-8 if time permits
- **Target:** Week 2

### P8. Real Robot Experiments
- **Open:** In scope or simulation only? CoRL strongly prefers real robot.
- **Target:** Week 4-5 (decide based on progress)

### ~~P9. Gaussian → Mesh Method~~ — SUPERSEDED (Mar 19)
- **Reason:** With per-object VGGT pipeline, Gaussian → Mesh conversion is no longer needed. VGGT outputs point cloud directly → Poisson mesh. 3DGS only appears as ablation comparison.

### P10. Scene Layout Detection (NEW, Mar 19)
- **Problem:** Per-object meshes are in Object Library, but how to detect their current poses in the real environment?
- **Candidates:**
  - Overhead RGB photo → VLM/object detection → estimate positions
  - RGB-D frame → point cloud segmentation + colored-ICP (SyncTwin approach)
  - Manual specification (simplest, for sim experiments)
- **Target:** Week 3-4

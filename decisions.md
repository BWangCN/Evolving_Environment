# 3DGS-EvoHome Technical Decisions

> **Last Updated:** March 17, 2026 (PM — pipeline components verified)
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

### D3. 3DGS Tool — Gaussian Grouping + gsplat (方案 D) — VERIFIED Mar 17
- **Choice:** Gaussian Grouping for core pipeline + gsplat for batch rendering
- **Gaussian Grouping handles:** COLMAP, 3DGS reconstruction, SAM segmentation, object editing, inpainting (LaMa + clone + fine-tune)
- **gsplat handles:** Large-scale batch rendering (native batch API + multi-GPU DDP), data augmentation rendering
- **Bridge:** PLY format export/import between the two — **VERIFIED**: gsplat successfully loads GG's PLY output (9.5M Gaussians) and renders from arbitrary camera poses
- **Gaussian Grouping limitation found (Mar 17):** Object relocation (SE(3) transform on Gaussian groups) is NOT implemented in official code — needs custom implementation
- **Why not nerfstudio + gsplat (方案 A):** Feature Splatting has no inpainting; nerfstudio CLI-oriented design fights programmatic use
- **Why not Gaussian Grouping only (方案 B):** Cannot handle batch rendering of thousands of training images efficiently
- **Why not gsplat only (方案 C):** Too much from-scratch engineering for 10-week timeline
- **Date:** Mar 16, 2026 (verified Mar 17)

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
- **Data synthesis pipeline (Stage B) is entirely 3DGS-based:** AnyGrasp → motion plan → Gaussian edit → gsplat render. Pure CUDA, runs on A100. ManiSkill NOT in data synthesis loop.
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

---

## Pending Decisions

### P2. Inpainting Strategy Details — BASELINE VERIFIED Mar 17
- **Baseline:** Gaussian Grouping's LaMa approach — **verified on bear scene**. 2D LaMa pseudo labels + 10K fine-tune → clean inpainting, no visible artifacts. Loss: 0.27 → 0.10.
- **Note:** Densification during fine-tuning is memory-intensive (27 GB VRAM). Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Alternatives:** GScream (depth-guided), InFusion (diffusion prior), 3DGS-CD (duplicate + optimize)
- **Target:** Week 2-3 (baseline locked, alternatives only if quality insufficient on tabletop scenes)

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

# 3DGS-EvoHome Technical Decisions

> **Last Updated:** March 16, 2026
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

### D3. 3DGS Tool — Gaussian Grouping + gsplat (方案 D)
- **Choice:** Gaussian Grouping for core pipeline + gsplat for batch rendering
- **Gaussian Grouping handles:** COLMAP, 3DGS reconstruction, SAM segmentation, object editing, inpainting (LaMa + clone + fine-tune)
- **gsplat handles:** Large-scale batch rendering (native batch API + multi-GPU DDP), data augmentation rendering
- **Bridge:** PLY format export/import between the two (one-time engineering)
- **Why not nerfstudio + gsplat (方案 A):** Feature Splatting has no inpainting; nerfstudio CLI-oriented design fights programmatic use
- **Why not Gaussian Grouping only (方案 B):** Cannot handle batch rendering of thousands of training images efficiently
- **Why not gsplat only (方案 C):** Too much from-scratch engineering for 10-week timeline
- **Date:** Mar 16, 2026

### D4. Development Environment
- **Windows 4080 (local):** 3DGS validation, documentation, quick iteration — **via WSL2 (Ubuntu 22.04)**
- **Linux 5090 (dev machine):** Large-scale 3DGS reconstruction, VLA inference validation
- **GMU Hopper (A100 cluster):** VLA finetuning (LoRA + TFA), EvoHome-Bench full evaluation, batch rendering at scale
- **Reason for WSL2:** Gaussian Grouping native Windows requires 4-5 file patches + DEVA has Ubuntu-only dependencies; WSL2 uses Windows GPU driver directly with zero extra effort
- **Date:** Mar 16, 2026

---

### D5. Simulation Environment — BEHAVIOR-1K (OmniGibson)
- **Choice:** BEHAVIOR-1K / OmniGibson as the simulation environment for benchmark evaluation
- **Date:** Mar 16, 2026

### D6. Grasp Model — AnyGrasp
- **Choice:** AnyGrasp (T-RO 2023)
- **Why not GraspGen:** Too new (2025), less community adoption, harder to justify to reviewers
- **Why not GaussianGrasper:** Its language grounding is redundant (we already have object segmentation); grasp quality itself isn't better than AnyGrasp
- **Pipeline:** 3DGS → segment object → extract point cloud from Gaussians → AnyGrasp → all valid grasp poses used as diverse training data (no selection heuristic)
- **Date:** Mar 16, 2026

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

### P2. Inpainting Strategy Details
- **Baseline:** Gaussian Grouping's LaMa approach
- **Alternatives:** GScream (depth-guided), InFusion (diffusion prior), 3DGS-CD (duplicate + optimize)
- **Target:** Week 2-3

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

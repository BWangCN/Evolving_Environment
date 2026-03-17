# 3DGS Evolving Home — CoRL 2026 Timeline

**Paper:** Zero-Interaction Continual Adaptation of VLA in Evolving Home Environments via 3D Gaussian Splatting
**Target:** CoRL 2026 — Abstract May 25, Paper May 28
**Duration:** 10 Weeks (Mar 16 — May 28, 2026)
**Strategy:** New Problem Definition + EvoHome-Bench/Challenge + Decoupled CL Method

> **Major pipeline update (Mar 16):** Replaced reference trajectory library with AnyGrasp + motion planner.
> True zero-demonstration pipeline. See `decisions.md` D6/D7.

---

## Week 1 — Mar 16–22 | Launch + Pipeline Foundation

### Decisions (Windows) ✅
- [x] Decide VLA backbone → **π0.5** (insulation + single-stage training)
- [x] Decide 3DGS tool → **Gaussian Grouping + gsplat** (方案 D)
- [x] Decide simulation env → **BEHAVIOR-1K / OmniGibson**
- [x] Decide grasp model → **AnyGrasp** + motion planner (eliminates reference trajectory library)
- [x] Decide robot → **Franka Panda** (parallel-jaw gripper, BEHAVIOR-1K standard)
- [x] EvoHome-Bench draft: environment sequence, metrics, evaluation protocol
- [x] Define target objects and 5-environment progression
- [x] Literature review (35+ papers on 3DGS editing/segmentation/inpainting)
- [x] Pipeline gap analysis completed

### Trajectory Generation Module (Windows) 🔧
- [ ] Define Franka Panda specs: workspace limits, gripper max opening (8cm), home pose, EE-to-camera transform
- [ ] Define action format: π0.5-compatible (Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper) + normalization range
- [ ] Implement `TrajectoryGenerator`: grasp_pose + place_target → waypoint sequence (home → pre-grasp → approach → grasp → lift → transport → pre-place → release)
- [ ] Implement Cartesian interpolation (linear + slerp for orientation)
- [ ] Implement `CollisionChecker`: trajectory waypoints × scene point cloud → pass/reject
- [ ] Implement `ActionFormatter`: EE pose sequence → π0.5 delta actions + gripper state
- [ ] Implement `CameraPoseComputer`: EE poses → wrist camera poses (for 3DGS rendering)
- [ ] Unit tests with mock grasp poses + matplotlib 3D visualization
- [ ] Package as standalone module: `src/trajectory/`

### 3DGS + VLA Setup (Linux 5090) — parallel
- [ ] Gaussian Grouping environment setup + first scene reconstruction
- [ ] π0.5 (openpi) inference verification on LIBERO
- [ ] BEHAVIOR-1K / OmniGibson installation + first scene loading

**✅ Checkpoint: Trajectory module tested with mock data + 3DGS pipeline runs on one scene**

---

## Week 2 — Mar 23–29 | Grasp Pipeline + Scene Editing

### AnyGrasp Integration (Linux 5090)
- [ ] Install AnyGrasp (register license)
- [ ] Test: OmniGibson scene → depth → point cloud → AnyGrasp → grasp poses
- [ ] Test: 3DGS scene → extract Gaussian positions → AnyGrasp → grasp poses
- [ ] Connect AnyGrasp output to TrajectoryGenerator → verify end-to-end

### Scene Editing (Linux 5090)
- [ ] Gaussian Grouping: object segmentation on tabletop scene
- [ ] Object removal + inpainting (LaMa) test
- [ ] Object relocation (move Gaussian group to new pose)

### Task Planning (Windows)
- [ ] Implement `LanguageGenerator`: object list + task templates → diverse language instructions
- [ ] Implement `TaskPlanner`: given objects in scene → enumerate valid (task, language) pairs
- [ ] Define place targets per task type (pick-place, rearrange, stack)

### Metrics Module (Windows)
- [ ] Implement `EvoHomeBenchMetrics`: performance matrix → FTS, FR, ZIC, CE, EvoHome Score
- [ ] Unit tests with synthetic performance matrices

**✅ Checkpoint: Full grasp→trajectory→action chain works + scene editing verified**

---

## Week 3 — Mar 30–Apr 5 | First Synthetic Data + VLA Baseline

### Synthetic Data Generation (Linux 5090 / A100)
- [ ] Full pipeline end-to-end: scene → segment → grasp → plan → edit scene → render → (I, a, l)
- [ ] First batch of synthetic data for E1 (base kitchen, 5 objects)
- [ ] Data augmentation: camera perturbation, object pose randomization
- [ ] gsplat batch rendering integration for scale

### VLA Baseline (Hopper A100)
- [ ] π0.5 zero-shot evaluation on BEHAVIOR-1K tasks → baseline numbers
- [ ] First LoRA finetune attempt on E1 synthetic data
- [ ] Sanity check: does finetuned model improve over zero-shot on E1?

### EvoHome-Bench Evaluation Pipeline (Linux)
- [ ] Automated evaluation script: load model → run episodes → compute metrics
- [ ] E1 performance baseline established

**✅ Checkpoint: VLA finetunes on synthetic data + zero-shot vs finetuned comparison**

---

## Week 4 — Apr 6–12 | Continual Learning Framework

### Decoupled CL Implementation (Hopper A100)
- [ ] Per-environment LoRA on VLM backbone (openpi JAX)
- [ ] Trajectory Flow Anchoring (TFA) residual on action decoder
- [ ] Two-environment sequential adaptation: E1 → E2
- [ ] Measure forgetting: performance on E1 after adapting to E2

### Baselines (Hopper A100)
- [ ] Naive finetuning baseline
- [ ] EWC baseline
- [ ] Experience replay baseline (raw image replay)
- [ ] LoRA-only baseline (no TFA)
- [ ] First leaderboard numbers

**✅ Checkpoint: Our method + ≥3 baselines have numbers on EvoHome-Bench**

---

## Week 5 — Apr 13–19 | Scale Up + Anti-Forgetting

### Scale to 5 Environments (All machines)
- [ ] Generate synthetic data for E2–E5
- [ ] Sequential adaptation: E1 → E2 → E3 → E4 → E5
- [ ] Full 5×5 performance matrix

### Anti-Forgetting (Hopper A100)
- [ ] 3DGS Environment Bank implementation
- [ ] CARS (Competence-Aware Adaptive Replay Scheduling)
- [ ] Compare: no replay vs uniform replay vs CARS

**✅ Checkpoint: Full 5-env results + replay strategy comparison**

---

## Week 6 — Apr 20–26 | Ablations + Analysis

- [ ] Ablation: insulation vs isolation vs no-isolation (gradient flow)
- [ ] Ablation: clean planner trajectories vs noisy human-style trajectories
- [ ] Ablation: LoRA rank (8 vs 16 vs 32)
- [ ] Ablation: TFA residual size
- [ ] Ablation: per-env LoRA + TFA vs per-env LoRA only vs TFA only
- [ ] Per-task-type forgetting analysis
- [ ] Per-change-type forgetting analysis (object add vs replace vs layout)

**✅ Checkpoint: All ablations done + analysis complete**

---

## Week 7 — Apr 27–May 3 | Experiments 80% + Writing Skeleton 20%

- [ ] Supplementary experiments, edge cases
- [ ] **Writing:** Method + EvoHome-Bench section skeleton

**✅ Checkpoint: Supplementary experiments done + Method/Bench skeleton**

---

## Week 8 — May 4–10 | Experiments Wrap-up 60% + Writing 40%

- [ ] All figure and table data confirmed FINAL
- [ ] EvoHome-Bench overview figure created
- [ ] Supplementary video recorded
- [ ] **Writing:** Experiment + EvoHome-Bench section draft

**✅ Checkpoint: All experiment numbers final + Experiment/Bench draft**

---

## Week 9 — May 11–17 | Last Experiments 30% + Writing 70%

- [ ] Final supplementary experiments (if Professor Xiao requests after review)
- [ ] **Writing:** Introduction, Related Work, Conclusion, Abstract
- [ ] Full draft to Professor Xiao for review (by **Wed May 14** at latest)

**✅ Checkpoint: Complete draft to Professor + All experiments 100% done**

---

## Week 10 — May 18–25 | 🚨 Full-Time Writing + Submission

- [ ] May 18–20: Major revisions based on Professor feedback
- [ ] May 21–23: Polish writing, figures, verify number consistency across all tables
- [ ] May 24: Supplementary material finalized, proofreading complete
- [ ] **May 25 (Mon): SUBMIT ABSTRACT**
- [ ] May 26–27: Final paper polish
- [ ] **May 28 (Thu): SUBMIT PAPER**

---

## Risk Flags

| Risk | When | Mitigation |
|------|------|------------|
| 3DGS synthetic data quality too low → VLA degrades | Week 3 | Sanity check: finetune small batch, check trend |
| AnyGrasp license or compatibility issues | Week 2 | Fallback: Contact-GraspNet (fully open source) |
| Flow matching decoder doesn't respond to TFA | Week 4 | Fallback: standard LoRA on decoder |
| Planner trajectories too different from π0.5 pretraining distribution | Week 3 | This is expected to be a feature, not bug; ablation in Week 6 |
| Not enough sequential environments for convincing CL story | Week 5 | Prioritize diverse object types over sheer environment count |
| OmniGibson setup issues on 5090 | Week 1-2 | Can use Hopper as backup; OmniGibson has Docker images |
| Not enough writing time | Week 10 | Start skeletons Week 7; all figures/tables final by Week 8 |

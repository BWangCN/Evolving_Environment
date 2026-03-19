# 3DGS Evolving Home — CoRL 2026 Timeline

**Paper:** Zero-Interaction Continual Adaptation of VLA in Evolving Home Environments via 3D Gaussian Splatting
**Target:** CoRL 2026 — Abstract May 25, Paper May 28
**Duration:** 10 Weeks (Mar 16 — May 28, 2026)
**Strategy:** New Problem Definition + EvoHome-Bench/Challenge + Decoupled CL Method

> **Major pipeline update (Mar 16):** Replaced reference trajectory library with AnyGrasp + motion planner.
> True zero-demonstration pipeline. See `decisions.md` D6/D7.

---

## Week 1 — Mar 16–22 | Launch + Pipeline Foundation

### Decisions ✅ DONE
- [x] VLA backbone → **π0.5** (insulation + single-stage training)
- [x] 3DGS tool → **Gaussian Grouping + gsplat** (方案 D)
- [x] Simulation env → **BEHAVIOR-1K / OmniGibson**
- [x] Grasp model → **AnyGrasp** + motion planner (eliminates reference trajectory library)
- [x] Robot → **Franka Panda** (parallel-jaw gripper, BEHAVIOR-1K standard)
- [x] Pipeline gap analysis completed
- [x] EvoHome-Bench draft: environment sequence, metrics, evaluation protocol
- [x] Define target objects and 5-environment progression
- [x] 3DGS literature review (35+ papers)

### Code — Scene + Task + Trajectory ✅ DONE (originally Week 2)
- [x] `src/scene/`: SceneObject + Registry + AffordanceInference
- [x] `src/task/`: TaskPlanner + LanguageGenerator (template, LLM extension point)
- [x] `src/config/franka.py`: Franka specs, action format, collision margins
- [x] `src/trajectory/`: TrajectoryGenerator, CollisionChecker (inflated volumes), PlaceTargetComputer, ActionFormatter (π0.5 deltas), CameraPoseComputer
- [x] `src/trajectory/generator.py`: chain() for multi-step sequences

### Code — Evaluation ✅ DONE (originally Week 5)
- [x] `src/evaluation/metrics.py`: EvoHomeBenchMetrics (FTS, FR, ZIC, CE, EvoHome Score, method comparison)
- [x] `src/evaluation/cars.py`: CARS scheduler (decoupled VLM/decoder replay)

### Testing ✅ DONE
- [x] 74 tests passing (scene 15 + task 13 + trajectory 22 + evaluation 20 + batch 4)
- [x] Batch stress test: 165 trajectories, 90.3% acceptance rate
- [x] Action distribution validated, camera consistency verified

### Writing — Related Works ✅ DONE
- [x] Comprehensive related works survey (`related_works.md`)
- [x] BibTeX references generated (`references.bib`)

### Infrastructure Analysis (Mar 17) ✅ DONE
- [x] Hopper storage plan: HOME (code) / SCRATCH (envs, weights) / PROJECTS (results)
- [x] OmniGibson/Isaac Sim confirmed **incompatible with A100** (no RT Cores) — evaluation only
- [x] Pipeline bottleneck = gsplat batch rendering (pure CUDA, runs on A100)
- [x] OmniGibson NOT in data synthesis loop → reduced to evaluation platform
- [x] ManiSkill 3 identified as GPU-parallel A100-compatible evaluation alternative (pending decision)
- [x] Development strategy: **5090 first** (debug E2E) → Hopper (scale up)

### 3DGS + VLA Setup (Linux 5090) ✅ DONE (Mar 17)
- [x] Gaussian Grouping environment setup + first scene reconstruction (bear, 96 images, 30K iters)
- [x] Gaussian Grouping scene editing: object removal + 3D inpainting (LaMa + fine-tune) verified
- [x] gsplat 1.5.3 installed, PLY bridge verified (9.5M Gaussians → 640x480 render)
- [x] AnyGrasp SDK installed (MinkowskiEngine 0.5.4 + pointnet2 compiled for Blackwell SM 120)
- [x] π0.5 (openpi) inference verified: `pi05_droid` config → (15, 8) action output
- [x] AnyGrasp license received + installed (Mar 18). 3,097 grasps detected on bear 3DGS point cloud.
- [x] Minimal E2E pipeline — **VERIFIED Mar 18:**
  - [x] Step 1: Extract object point cloud from segmented 3DGS (338K Gaussians for bear, obj_dc features + classifier)
  - [x] Step 2: AnyGrasp grasp detection (5 grasps on bear object, scores 0.08-0.10)
  - [x] Step 3: Convert AnyGrasp GraspGroup → GraspPose (rotation matrix → wxyz quaternion)
  - [x] Step 4: TrajectoryGenerator → 83-waypoint pick-place trajectory, 82 actions (7-dim delta EE)
  - [ ] Step 5: gsplat render at camera poses → (I, a, l) triplets (code ready, needs tabletop scene for proper scale)

**✅ Checkpoint: All code modules built + 74 tests passing + infrastructure plan locked + all 4 pipeline components installed & individually verified**

---

## Week 2 — Mar 23–29 | AnyGrasp Integration + Scene Editing + First Data

### AnyGrasp Integration (Linux 5090)
- [x] Install AnyGrasp SDK + MinkowskiEngine + pointnet2 (all compiled for Blackwell SM 120)
- [x] License received + installed (Mar 18), checkpoint loaded, 3,097 grasps detected on bear 3DGS scene
- [x] Test: 3DGS scene → extract Gaussian positions → point cloud → AnyGrasp → grasp poses (Mar 18, 10 grasps on bear object)
- [x] Connect AnyGrasp output to TrajectoryGenerator → verify end-to-end (Mar 18, `scripts/visualize_pipeline.py`)
- [x] Replace mock grasp poses with real AnyGrasp output in batch pipeline (Mar 18, `scripts/visualize_pipeline.py` grasps mode)

### Scene Editing (Linux 5090)
- [x] Gaussian Grouping: object removal on bear scene (object ID 34, 764 MB PLY)
- [x] 3D Inpainting: LaMa pseudo labels + 10K iter fine-tune (2.8 GB PLY, clean result)
- [x] Object relocation design complete (Mar 18): SE(3) transform on Gaussian positions + quaternions, grasp contact as anchor, hybrid depth compositing with ManiSkill for robot arm
- [ ] Object relocation implementation: compositional 3DGS (empty env + object clusters) + hybrid rendering code
- [ ] Gaussian Grouping: object segmentation on **custom tabletop scene** (bear was demo dataset)

### First Synthetic Data (Linux 5090)
- [x] gsplat batch rendering: loaded edited PLY (9.5M Gaussians) → rendered from arbitrary camera pose
- [x] ManiSkill RT → 3DGS reconstruction pipeline verified (Mar 18, 90 views, 336K Gaussians, metric scale)
- [ ] Render trajectories in 3DGS scene → first (I, a, l) triplets
- [ ] Visual quality check on rendered training images

### Evaluation Platform ✅ DECIDED (Mar 17)
- [x] Decide: ~~OmniGibson~~ → **ManiSkill 3** (OmniGibson blocked on RTX 5090, Isaac Sim 5.0 migration pending)
- [x] ManiSkill 3.0.0b22 installed, PickCube-v1 verified with RT rendering on 5090
- [x] π0.5 → ManiSkill integration verified (policy server + client, action format mapping)
- [x] ManiSkill dataset generation: 4 tasks × 2000 demos, fresh RL policy rollouts + RT rendering + language instructions
- [x] Convert ManiSkill demos to LeRobot v2.1 format (Mar 18, 8000 eps, 94865 frames, verified)
- [ ] Submit π0.5 LoRA fine-tuning on ManiSkill to Hopper
- [ ] Prototype custom EvoHome task definitions in ManiSkill (multi-step, LIBERO-Long-style)

### LIBERO Evolving Environment Benchmark — **Assigned to Sal Yang** (NEW, Mar 18)
- [ ] Set up LIBERO environment on 5090 (Python 3.8 venv + robosuite)
- [ ] Verify π0.5 baseline on LIBERO-Long (expected ~92.4% success)
- [ ] Construct evolving environment benchmark: modify LIBERO environments via BDDL (add/remove/reposition objects, change layouts)
- [ ] Define environment evolution sequence (E1 → E2 → E3 → E4 → E5) within LIBERO
- [ ] Test baseline VLAs (π0.5, π0) on evolved environments → measure performance degradation
- [ ] Document benchmark protocol: tasks per environment, evaluation episodes, success criteria

**✅ Checkpoint: Real grasp poses + scene editing + first rendered training images + ManiSkill dataset generated + LIBERO benchmark construction started**

---

## Week 3 — Mar 30–Apr 5 | Synthetic Data at Scale + VLA Baseline

### Synthetic Data Generation (Linux 5090 / A100)
- [ ] Full pipeline end-to-end: scene → segment → grasp → plan → edit → render → (I, a, l)
- [ ] Generate full E1 dataset (hundreds of trajectories)
- [ ] Data augmentation: camera perturbation, object pose randomization, lighting

### VLA Baseline (5090 → Hopper A100)
- [ ] π0.5 zero-shot inference sanity check (5090 first, then migrate to Hopper)
- [ ] First LoRA finetune attempt on E1 synthetic data (Hopper A100)
- [ ] Sanity check: finetuned model improves over zero-shot on E1?

### EvoHome-Bench Evaluation Pipeline
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
- [ ] CARS integration with real competence scores
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

## Week 10 — May 18–25 | Full-Time Writing + Submission

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
| Planner trajectories too different from π0.5 pretraining distribution | Week 3 | Expected to be a feature, not bug; ablation in Week 6 |
| Not enough sequential environments for convincing CL story | Week 5 | Prioritize diverse object types over sheer environment count |
| ~~OmniGibson setup issues on 5090~~ | ~~Week 1-2~~ | **RESOLVED**: OmniGibson only for eval; ManiSkill as A100-compatible alternative |
| OmniGibson incompatible with A100 (no RT Cores) | Week 2 | Option A: OmniGibson on 5090 (serial eval); Option B: ManiSkill on Hopper (GPU-parallel) |
| Not enough writing time | Week 10 | Start skeletons Week 7; all figures/tables final by Week 8 |

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
- **Next steps (5090 Linux):** Gaussian Grouping setup + π0.5 inference verification

### Mar 17 (Mon)
- *(pending)*

### Mar 18 (Tue)
- *(pending)*

### Mar 19 (Wed)
- *(pending)*

### Mar 20 (Thu)
- *(pending)*

### Mar 21 (Fri)
- *(pending)*

### Mar 22 (Sat)
- *(pending)*

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

**Week 1 Tasks — Code Modules (Windows):**

*Phase 1: Scene Object System (`src/scene/`)*
- [ ] `SceneObject` dataclass (geometry + semantics + affordances)
- [ ] `SceneObjectRegistry` (管理场景中所有物体)
- [ ] `AffordanceInference` (category → affordances/valid_tasks)
- [ ] Unit tests with mock objects

*Phase 2: Task Planning (`src/task/`)*
- [ ] `TaskPlanner` (registry → 枚举合法 task-object pairs)
- [ ] `LanguageGenerator` (task + objects → 多样化 language instructions)
- [ ] Unit tests

*Phase 3: Trajectory Generation (`src/trajectory/`)*
- [ ] Franka specs + π0.5 action format definition
- [ ] `PlaceTargetComputer` (place_on/in/next_to/stack from object geometry)
- [ ] `TrajectoryGenerator` (grasp + place_target → phased waypoints)
- [ ] Cartesian interpolation + slerp orientation
- [ ] `CollisionChecker` (inflated volumes, phase-aware margin)
- [ ] `ActionFormatter` (EE poses → π0.5 delta actions + normalization)
- [ ] `CameraPoseComputer` (EE pose → wrist camera pose)
- [ ] Unit tests + matplotlib 3D visualization

**Week 1 Tasks — Setup (Linux 5090):**
- [ ] Gaussian Grouping + first scene reconstruction
- [ ] π0.5 inference on LIBERO
- [ ] BEHAVIOR-1K / OmniGibson installation

**Week 1 Checkpoint:** Trajectory module tested with mock data + 3DGS pipeline runs on one scene

---

## Week 2 — Mar 23–29 | Synthetic Pipeline + Metrics Lock-in

**Week 2 Tasks (from Timeline):**
- [ ] Implement object-level Gaussian editing (move, add, remove objects)
- [ ] Build synthetic trajectory generation pipeline
- [ ] Finalize EvoHome-Bench metrics: FTS, FR, ZIC, CE
- [ ] Define standardized evaluation protocol

**Week 2 Checkpoint:** Synthetic pipeline built + EvoHome-Bench metrics & protocol finalized

---

## Week 3 — Mar 30–Apr 5 | First Synthetic Data + Eval Pipeline

**Week 3 Tasks (from Timeline):**
- [ ] First batch of synthetic data generated
- [ ] VLA finetuning experiments launched
- [ ] EvoHome-Bench evaluation pipeline built
- [ ] Quick sanity check: finetune with small batch, verify positive trend

**Week 3 Checkpoint:** VLA can finetune on synthetic data + EvoHome-Bench pipeline runs automatically

---

## Week 4 — Apr 6–12 | Core CL Experiments + Baselines

**Week 4 Tasks (from Timeline):**
- [ ] Sequential environment CL experiments (2–3 environments)
- [ ] Full implementation of decoupled CL framework (per-env LoRA + TFA)
- [ ] Run CL baselines on EvoHome-Bench
- [ ] Your method vs baselines — first numbers on leaderboard

**Week 4 Checkpoint:** Your method + ≥3 baselines have numbers on EvoHome-Bench

---

## Week 5 — Apr 13–19 | Scale Up + Benchmark Refinement

**Week 5 Tasks (from Timeline):**
- [ ] Expand to 4–5 sequential environments
- [ ] 3DGS Environment Bank + CARS experiments
- [ ] EvoHome-Bench difficulty levels finalized
- [ ] Leaderboard table populated with all methods

**Week 5 Checkpoint:** EvoHome-Bench leaderboard has ≥4 methods + Difficulty levels defined

---

## Week 6 — Apr 20–26 | Ablations + Benchmark Materials

**Week 6 Tasks (from Timeline):**
- [ ] All CL ablations completed
- [ ] EvoHome-Bench paper materials: overview figure, baseline failure analysis
- [ ] Edge case testing on challenging environment transitions
- [ ] Challenge concept drafted

**Week 6 Checkpoint:** All ablations done + EvoHome-Bench materials ready

---

## Week 7 — Apr 27–May 3 | Experiments 80% + Writing Skeleton 20%

**Week 7 Tasks (from Timeline):**
- [ ] Supplementary experiments, edge cases
- [ ] Writing: Method + EvoHome-Bench section skeleton

**Week 7 Checkpoint:** Supplementary experiments done + Method/Bench skeleton

---

## Week 8 — May 4–10 | Experiments Wrap-up 60% + Writing 40%

**Week 8 Tasks (from Timeline):**
- [ ] All figure and table data confirmed FINAL
- [ ] EvoHome-Bench overview figure created
- [ ] Supplementary video recorded
- [ ] Writing: Experiment + EvoHome-Bench section draft

**Week 8 Checkpoint:** All experiment numbers final + Experiment/Bench draft

---

## Week 9 — May 11–17 | Last Experiments 30% + Writing 70%

**Week 9 Tasks (from Timeline):**
- [ ] Final supplementary experiments
- [ ] Writing: Introduction, Related Work, Conclusion, Abstract
- [ ] Full draft to Professor Xiao for review (by Wed May 14)

**Week 9 Checkpoint:** Complete draft to Professor + All experiments 100% done

---

## Week 10 — May 18–25 | Full-Time Writing + Submission

**Week 10 Tasks (from Timeline):**
- [ ] May 18–20: Major revisions based on Professor feedback
- [ ] May 21–23: Polish writing, figures, verify number consistency
- [ ] May 24: Supplementary material finalized, proofreading complete
- [ ] **May 25 (Mon): SUBMIT ABSTRACT**
- [ ] May 26–27: Final paper polish
- [ ] **May 28 (Thu): SUBMIT PAPER**

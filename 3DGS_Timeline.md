# 3DGS Evolving Home — CoRL 2026 Timeline

**Paper:** Zero-Interaction Continual Adaptation of VLA in Evolving Home Environments via 3D Gaussian Splatting  
**Target:** CoRL 2026 — Abstract May 25, Paper May 28  
**Duration:** 10 Weeks (Mar 16 — May 28, 2026)  
**Strategy:** New Problem Definition + EvoHome-Bench/Challenge + Decoupled CL Method

---

## Week 1 — Mar 16–22 | Launch + Benchmark Framing

- [ ] Decide VLA backbone (π0 / OpenVLA / other) and 3DGS reconstruction tool
- [ ] Run through first scene reconstruction end-to-end
- [ ] EvoHome-Bench draft: sequential environment protocol, initial metrics list
- [ ] Define first batch of target objects and environment sequence

**✅ Checkpoint: 3DGS pipeline runs on one scene + EvoHome-Bench dimension draft**

---

## Week 2 — Mar 23–29 | Synthetic Pipeline + Metrics Lock-in

- [ ] Implement object-level Gaussian editing (move, add, remove objects)
- [ ] Build synthetic trajectory generation pipeline
- [ ] Finalize EvoHome-Bench metrics: Forward Transfer Score, Forgetting Rate, Zero-Interaction Compliance, Compute Efficiency
- [ ] Define standardized evaluation protocol (same base VLA, same env sequence, same eval objects)

**✅ Checkpoint: Synthetic pipeline built + EvoHome-Bench metrics & protocol finalized**

---

## Week 3 — Mar 30–Apr 5 | First Synthetic Data + Eval Pipeline

- [ ] First batch of synthetic data generated
- [ ] VLA finetuning experiments launched
- [ ] EvoHome-Bench evaluation pipeline built (adaptation method in → forward/backward/cost metrics out)
- [ ] Quick sanity check: finetune with small batch, verify performance trend is positive

**✅ Checkpoint: VLA can finetune on synthetic data + EvoHome-Bench pipeline runs automatically**

---

## Week 4 — Apr 6–12 | Core CL Experiments + Baselines

- [ ] Sequential environment CL experiments (2–3 environments)
- [ ] Full implementation of decoupled CL framework (per-env LoRA + Trajectory Flow Anchoring)
- [ ] Run CL baselines on EvoHome-Bench: naive finetune, EWC, experience replay, LoRA-only
- [ ] Your method vs baselines — first numbers on leaderboard

**✅ Checkpoint: Your method + ≥3 baselines have numbers on EvoHome-Bench**

---

## Week 5 — Apr 13–19 | Scale Up + Benchmark Refinement

- [ ] Expand to 4–5 sequential environments
- [ ] 3DGS Environment Bank + Competence-Aware Replay Scheduling experiments
- [ ] EvoHome-Bench refinement: difficulty levels finalized (object-level / layout / full scene change)
- [ ] Leaderboard table populated with all methods

**✅ Checkpoint: EvoHome-Bench leaderboard has ≥4 methods + Difficulty levels defined**

---

## Week 6 — Apr 20–26 | Ablations + Benchmark Materials

- [ ] All CL ablations completed (per-env LoRA vs shared, TFA vs naive LoRA, replay scheduling variants)
- [ ] EvoHome-Bench paper materials: overview figure, baseline failure analysis
- [ ] Edge case testing on challenging environment transitions
- [ ] Challenge concept drafted: standardized submission format description

**✅ Checkpoint: All ablations done + EvoHome-Bench materials ready**

---

## Week 7 — Apr 27–May 3 | Experiments 80% + Writing Skeleton 20%

- [ ] Supplementary experiments, edge cases
- [ ] **Writing:** Method + EvoHome-Bench section skeleton (structure + content points, no polish)

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
| 3DGS synthetic data quality too low → VLA degrades | Week 3 | Quick sanity check: finetune small batch, check trend |
| Flow matching decoder doesn't respond to TFA | Week 4 | Fallback: standard LoRA on decoder as backup |
| Not enough sequential environments for convincing CL story | Week 5 | Prioritize diverse object types over sheer environment count |
| Not enough writing time | Week 10 | Start skeletons Week 7; all figures/tables final by Week 8 |

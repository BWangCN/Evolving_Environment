# Future Paper: Plug-and-Play Object Manipulation via 3D-Grounded Manipulation Embeddings

> **Target venue:** ICLR 2027 (deadline ~Oct 2026)
> **Prerequisite:** CoRL 2026 paper (EvoHome — 3DGS digital twin + CL framework)
> **Status:** Idea stage. Seeds planted in CoRL paper's Future Work section.

---

## Core Idea

**Objects carry their own "manipulation driver."**

When a new object enters the environment, its 3D scan (3DGS/mesh) is encoded into a compact manipulation embedding. This embedding is injected into the VLA as conditioning — the VLA instantly knows how to manipulate the object, with zero gradient-based training.

```
Current paradigm:  new object → scan → generate data → train LoRA (hours) → can manipulate
Our paradigm:      new object → scan → encode (1 second) → can manipulate
```

Analogy: word embeddings carry semantic knowledge so LLMs can reason about new words. Manipulation embeddings carry actionable 3D knowledge so VLAs can manipulate new objects.

---

## Why ICLR (not just CoRL/RSS)

This is not just a robotics contribution. It's a **representation learning** question:

> "Can we learn a mapping from 3D geometric structure to actionable manipulation knowledge, such that the representation transfers zero-shot to novel objects?"

This connects to:
- **In-context learning:** the embedding serves as "context" that tells the VLA how to act, analogous to few-shot prompting in LLMs
- **Hypernetworks:** encoder generates task-specific weights from input description
- **Meta-learning:** amortized adaptation — replace per-task optimization with learned initialization/generation
- **3D foundation models:** emerging area (Point-E, Shap-E, OpenShape) — we add the "actionable" dimension
- **Object-centric representation learning:** each object has a disentangled, interpretable representation

The framing should be: **"3D Structure as Manipulation Prior"** — geometric structure predicts manipulation strategy, and this mapping can be learned.

---

## Technical Framework

### Manipulation Encoder

```
Input:  Object 3D representation (point cloud / mesh / 3DGS features)
Output: Manipulation embedding z ∈ R^d (e.g., d=256)

z encodes:
  - Grasp affordances (where to grasp, what grip type)
  - Physical properties (approximate weight, center of mass, friction)
  - Functional properties (container? stackable? tool?)
  - Manipulation dynamics (how it moves when grasped, stable orientations)
```

### Injection into VLA

Three possible mechanisms (increasing novelty):

**M1: Concatenation** — z appended to VLM input tokens as additional context
  - Simplest, minimal architecture change
  - VLM learns to attend to z via cross-attention

**M2: FiLM conditioning** — z modulates intermediate features
  - z → affine transform parameters → applied to VLM/decoder hidden states
  - More expressive, object-specific feature modulation

**M3: LoRA generation** — z → hypernetwork → LoRA weights
  - Most powerful: each object gets its own adapter, generated in one forward pass
  - True "plug-and-play" — object literally carries its own weights

### Training the Encoder

**Data source:** Our CoRL pipeline generates exactly the right data.

```
For each object o_i in diverse training set (1000+ objects):
    1. 3DGS scan → 3D representation R_i
    2. Import mesh into sim
    3. AnyGrasp → diverse grasps → sim execution → successful trajectories T_i
    4. Collect (R_i, T_i) pairs

Meta-training objective:
    θ* = argmin_θ Σ_i L(VLA(image, lang, f_θ(R_i)), T_i)

where f_θ is the manipulation encoder parameterized by θ.
```

For M3 (hypernetwork), the objective becomes:
```
    θ* = argmin_θ Σ_i L(VLA(image, lang; LoRA=g_θ(R_i)), T_i)
```

### Evaluation

**Key experiments:**
1. **Zero-shot novel object manipulation:** Train encoder on objects A, test on held-out objects B. No fine-tuning on B.
2. **Comparison with fine-tuning:** Show encode-to-adapt matches or approaches train-to-adapt quality with 1000x less compute at deployment.
3. **Embedding analysis:** What does z capture? Visualize embedding space — do functionally similar objects cluster? (mugs together, bowls together, tools together)
4. **Cross-embodiment transfer:** Does z transfer across different robot arms? (If z captures object properties, not robot-specific skills, it should.)
5. **Compositional generalization:** Novel combinations of known object types in new arrangements.

---

## Relationship to CoRL Paper

```
CoRL 2026 (Paper 1):                    ICLR 2027 (Paper 2):
├── Problem: Evolving env CL            ├── Problem: Zero-shot object adaptation
├── 3DGS → sim pipeline (builds infra)  ├── Uses pipeline from Paper 1
├── Per-env LoRA + TFA + CARS           ├── Replaces LoRA training with encoding
├── EvoHome-Bench                       ├── Extends benchmark with zero-shot eval
└── Contribution: CL framework          └── Contribution: New adaptation paradigm
```

Paper 1 generates the training data and infrastructure.
Paper 2 uses that infrastructure to train the manipulation encoder.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Embedding doesn't generalize to truly novel geometries | Train on very diverse objects (1000+); use shape primitives as augmentation |
| M3 hypernetwork too hard to train | Start with M1/M2; M3 as ablation or follow-up |
| Sim-to-real gap in encoder training | Train in sim, but 3DGS scans are from real objects — bridges the gap |
| Not enough novelty for ICLR | Frame as representation learning, not just robotics; add theoretical analysis of why geometry predicts manipulation |
| Concurrent work appears | The 3D-to-manipulation-embedding idea is sufficiently specific that exact replication is unlikely |

---

## Potential Title Ideas

- "Object Embeddings for Instant Manipulation: Learning 3D-Grounded Manipulation Priors"
- "Plug-and-Play Manipulation: Zero-Shot Adaptation via 3D Manipulation Encoders"
- "3D Structure as Manipulation Prior: From Geometry to Grasping in One Forward Pass"
- "Manipulation Drivers: Objects That Carry Their Own Control Knowledge"

---

## Timeline Sketch

- **May 2026:** CoRL submitted. Pipeline and EvoHome-Bench ready.
- **Jun 2026:** Design encoder architecture. Generate large-scale (3D, trajectory) dataset in sim.
- **Jul 2026:** Train manipulation encoder (M1 → M2). First zero-shot results.
- **Aug 2026:** Ablations, embedding analysis, cross-embodiment experiments.
- **Sep 2026:** Writing. Target ICLR 2027 deadline (~Oct 1).

---

## Seeds in CoRL Paper (Future Work Section)

> "While our current framework adapts via gradient-based LoRA fine-tuning on synthesized data, an exciting direction is to learn a manipulation encoder that maps an object's 3D representation directly to adapter weights — enabling instantaneous, training-free adaptation when new objects enter the environment. Our 3DGS-to-sim pipeline naturally generates the large-scale (3D representation, manipulation trajectory) pairs needed to train such an encoder. We leave this encode-to-adapt paradigm as future work."

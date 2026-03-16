# Zero-Interaction Continual Adaptation of VLA in Evolving Home Environments via 3D Gaussian Splatting

## Technical Pipeline Documentation

**Project Codename:** 3DGS-EvoHome  
**Target Venue:** CoRL 2026  
**Last Updated:** March 16, 2026

---

## 1. Problem Definition

### 1.1 Core Problem Statement

Pre-trained Vision-Language-Action (VLA) models achieve impressive zero-shot generalization across a wide range of manipulation tasks. However, when deployed into a specific home environment, they inevitably encounter objects, layouts, and configurations that fall outside their pretraining distribution. A home environment is not static — residents continuously introduce new objects (a new coffee mug, a child's toy, a different brand of cereal box), rearrange furniture, and change the scene layout. The VLA must adapt to these changes without requiring per-object human demonstrations every time the environment evolves.

We formalize this as the **Evolving Home Environment Continual Adaptation** problem:

Given a pre-trained VLA $\pi_0$ and a sequence of home environments $\{E_1, E_2, ..., E_n\}$ where each $E_i$ introduces novel objects or layout changes relative to $E_{i-1}$, the goal is to produce an adapted policy $\pi_n$ that:

1. **Succeeds on all tasks across all environments** $E_1$ through $E_n$ (no catastrophic forgetting)
2. **Requires zero per-object human demonstration** for adaptation — the only human effort is a one-time reference trajectory library collected during system initialization
3. **Adapts via 3DGS-synthesized data only** — no additional real-world robot interaction needed per new object

### 1.2 Why This Problem Matters

Current VLA research focuses almost exclusively on pretraining generalization. But even the strongest VLAs (e.g., π0 achieving ~26% success rate across 50 household tasks in the BEHAVIOR Challenge 2025) cannot handle the combinatorial diversity of personalized home environments. Our framework addresses the **last mile of personalization** — this contribution is complementary to, not competing with, VLA scaling.

### 1.3 Key Distinction from Existing Work

Unlike RoboSplat (RSS 2025) which requires **per-task human demonstrations** for each new object and does not address continual learning, our method requires only a **one-time reference trajectory library** collected during system initialization. The accurate claim is: **"amortized demonstration cost with zero per-object human effort"**, NOT "zero demonstration."

---

## 2. System Overview

The full pipeline consists of five major stages:

```
┌─────────────────────────────────────────────────────────┐
│                    INITIALIZATION (One-Time)             │
│                                                         │
│  [Real Robot] → Collect Reference Trajectory Library    │
│       (K tasks × M demonstrations per task)             │
│                                                         │
│  [Pre-trained VLA π₀] → Base model checkpoint           │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              PER-ENVIRONMENT ADAPTATION LOOP             │
│                                                         │
│  For each new environment Eᵢ:                           │
│                                                         │
│  Stage A: 3DGS Scene Reconstruction                     │
│      ↓                                                  │
│  Stage B: Synthetic Manipulation Data Generation        │
│      ↓                                                  │
│  Stage C: Decoupled Continual Learning                  │
│      ↓                                                  │
│  Stage D: Evaluation on EvoHome-Bench                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Stage A: 3DGS Scene Reconstruction

### 3.1 Purpose

Reconstruct the current home environment as an editable, photorealistic 3D Gaussian Splatting representation that supports object-level manipulation for downstream synthetic data generation.

### 3.2 Input

Multi-view RGB images of the tabletop/workspace from a calibrated camera (either handheld or mounted on the robot arm). Approximately 30–60 images per scene, covering diverse viewpoints with sufficient overlap. Include close-up shots of target objects for high-quality reconstruction.

### 3.3 Process

1. **Image Capture Protocol:** Capture images from multiple elevations (low, eye-level, overhead) and azimuthal angles around the workspace. Ensure >60% overlap between adjacent views. Record camera intrinsics and extrinsics via a calibration target or SfM pipeline (e.g., COLMAP).

2. **3DGS Training:** Train a 3DGS model on the captured images. The output is a set of 3D Gaussians $\{(\mu_k, \Sigma_k, \alpha_k, c_k)\}_{k=1}^{N}$ where $\mu_k$ is position, $\Sigma_k$ is covariance, $\alpha_k$ is opacity, and $c_k$ is spherical harmonic color coefficients.

3. **Object Segmentation:** Use a foundation model (e.g., SAM, Grounded-SAM) to segment individual objects within the 3DGS scene. Each object becomes a separable subset of Gaussians that can be independently manipulated (translated, rotated, removed).

4. **3DGS Environment Bank Registration:** Store the reconstructed scene in the 3DGS Environment Bank (see Section 6.2) indexed by environment ID $E_i$.

### 3.4 Output

An editable 3DGS scene where individual objects can be moved, added, or removed, and photorealistic novel-view RGB images can be rendered from any camera pose.

### 3.5 Tools & Dependencies

Candidate tools: gsplat, nerfstudio, 3DGS original codebase. Object segmentation: Grounded-SAM, LERF, or Gaussian Grouping.

### 3.6 Key Technical Risks

- **Reconstruction quality on small/thin/reflective objects.** Mitigation: increase capture density around challenging objects, use close-up supplementary views.
- **Object segmentation errors.** Mitigation: manual correction for the first few environments; in production, leverage iterative SAM refinement.

---

## 4. Stage B: Synthetic Manipulation Data Generation

### 4.1 Purpose

Generate geometrically and visually consistent training data (image-action pairs) for the VLA by composing the 3DGS scene with reference trajectories, without any additional human demonstration.

### 4.2 Input

- Editable 3DGS scene from Stage A
- Reference Trajectory Library $\mathcal{T} = \{(\tau_1, l_1), (\tau_2, l_2), ..., (\tau_K, l_K)\}$ where each $\tau_j$ is a demonstrated end-effector trajectory and $l_j$ is the corresponding language instruction (e.g., "pick up the red mug", "place the bowl on the shelf")
- Target object poses in the current environment $E_i$

### 4.3 Process

#### 4.3.1 Trajectory Retargeting

For each target object in $E_i$, select the most appropriate reference trajectory from $\mathcal{T}$ based on:

- **Grasp type compatibility:** match the object's graspable geometry to trajectories with similar grasp configurations
- **Task type:** match the language instruction to the intended task (pick, place, pour, etc.)

Apply rigid-body transformation to the reference trajectory to align it with the target object's pose in $E_i$:

$$\tau'_j(t) = R_{align} \cdot \tau_j(t) + t_{align}$$

where $R_{align}$ and $t_{align}$ are computed from the relative pose between the reference object position and the target object position.

#### 4.3.2 Gaussian Scene Editing

To synthesize intermediate states of manipulation:

1. **Pre-grasp:** render the scene as-is from the robot's camera viewpoint at each timestep of $\tau'_j$
2. **During grasp & transport:** detach the target object's Gaussians from the scene, attach them to the end-effector frame, and re-render. This produces geometrically consistent images where the object moves with the gripper.
3. **Post-place:** move the object's Gaussians to the placement location, re-render.

At each timestep $t$, render an RGB image $I_t$ from the robot's camera pose corresponding to $\tau'_j(t)$, producing paired data $(I_t, a_t, l_j)$ where $a_t$ is the action at timestep $t$.

#### 4.3.3 Data Augmentation

- **Viewpoint perturbation:** small random perturbations to camera pose during rendering to increase visual diversity
- **Lighting variation:** modify spherical harmonic coefficients to simulate different lighting conditions
- **Object pose randomization:** vary target object poses within a plausible range on the workspace surface
- **Background variation:** swap backgrounds or add/remove distractor objects to improve robustness

### 4.4 Output

A dataset $\mathcal{D}_i = \{(I_t, a_t, l)\}$ for environment $E_i$, containing image-action-language triplets suitable for VLA finetuning.

### 4.5 Key Technical Risks

- **Trajectory retargeting failures** when object geometry is very different from reference. Mitigation: maintain a diverse reference library covering multiple grasp types.
- **Visual artifacts in Gaussian editing** during object transport (floating Gaussians, shadow inconsistencies). Mitigation: implement shadow-aware rendering; accept minor artifacts as data augmentation in itself.
- **Physical plausibility** — synthesized trajectories may violate collision constraints. Mitigation: add collision checking against the remaining scene Gaussians and discard infeasible trajectories.

---

## 5. Stage C: Decoupled Continual Learning

### 5.1 Purpose

Adapt the VLA to environment $E_i$ using synthesized data $\mathcal{D}_i$ while preserving performance on all previously learned environments $E_1, ..., E_{i-1}$.

### 5.2 Why Decoupled?

Modern VLAs (e.g., π0, Octo) typically have a **dual-architecture** design:

1. **VLM Backbone** (vision-language understanding): a large pretrained vision-language model that encodes images and language instructions into a shared representation. Architecture: transformer-based, typically frozen or LoRA-finetuned.
2. **Action Decoder** (action generation): a module that takes the VLM representation and outputs actions. Architecture varies — π0 uses a **flow matching** decoder, others use diffusion policy or MLP heads.

These two components have **fundamentally different learning dynamics and forgetting patterns:**

- The VLM backbone needs to learn **what the new objects look like and what the language instructions mean in this context**. This is primarily a perceptual/semantic task. Forgetting manifests as confusing visually similar objects across environments.
- The action decoder needs to learn **how to execute the motor actions for new object geometries**. This is primarily a motor control task. Forgetting manifests as degraded action precision on previously learned objects.

Applying the same CL strategy (e.g., LoRA everywhere, or EWC everywhere) to both is suboptimal because their forgetting mechanisms differ. Hence, **decoupled continual learning** — a tailored strategy for each component.

### 5.3 VLM Backbone: Per-Environment LoRA

#### 5.3.1 Design

For each environment $E_i$, train a low-rank adapter $\Delta W_i = B_i A_i$ (LoRA) on top of the frozen VLM backbone. The base VLM weights $W_0$ are never modified.

At inference in environment $E_i$:

$$W_{\text{active}} = W_0 + \Delta W_i$$

#### 5.3.2 Rationale

- **No forgetting by design:** since $W_0$ is frozen and each environment has its own adapter, adapting to $E_n$ cannot affect the representation learned for $E_1$....$E_{n-1}$.
- **Storage efficient:** each LoRA adapter is ~1-5% of the base model parameters.
- **Environment identification:** at deployment, the system needs to know which environment is active. This can be achieved via a lightweight environment classifier, or simply by user selection.

#### 5.3.3 Training Details

- LoRA rank: $r = 16$ (to be tuned)
- Applied to: attention Q, V projection matrices in the VLM
- Learning rate: 1e-4 with cosine schedule
- Trained on: $\mathcal{D}_i$ (synthesized data for environment $E_i$) + replay data (see Section 6)

### 5.4 Action Decoder: Trajectory Flow Anchoring (TFA)

#### 5.4.1 Why Not LoRA on the Decoder?

The action decoder (flow matching) operates in a fundamentally different space than the VLM. Flow matching learns a **velocity field** $v_\theta(x_t, t, c)$ that transforms noise $x_0 \sim \mathcal{N}(0,I)$ into action sequences through an ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t, c)$$

where $c$ is the conditioning from the VLM. Applying LoRA to the flow matching network directly modifies the global velocity field, which can cause previously learned flow paths to drift — even small perturbations in the velocity field accumulate through ODE integration, potentially catastrophically distorting action outputs for old tasks.

#### 5.4.2 TFA Design

Instead of modifying the base velocity field, TFA learns a **residual flow field** per task cluster:

$$v_{\text{adapted}}(x_t, t, c) = v_{\theta_0}(x_t, t, c) + \Delta v_{\phi_i}(x_t, t, c)$$

where:

- $v_{\theta_0}$ is the frozen base flow matching decoder from the pretrained VLA
- $\Delta v_{\phi_i}$ is a lightweight residual network specific to environment/task cluster $i$

The residual field is initialized to near-zero, meaning the adapted model starts from the base model's behavior and makes minimal corrections.

#### 5.4.3 Training

- $\Delta v_{\phi_i}$ is parameterized as a small MLP or shallow transformer with skip connections
- Trained with flow matching loss on $\mathcal{D}_i$:

$$\mathcal{L}_{\text{TFA}} = \mathbb{E}_{t, x_0, x_1} \| (v_{\theta_0}(x_t, t, c) + \Delta v_{\phi_i}(x_t, t, c)) - u_t(x_t | x_0, x_1) \|^2$$

where $u_t$ is the conditional vector field target.

- Regularization: $\|\Delta v_{\phi_i}\|^2$ penalty to keep residuals small

#### 5.4.4 Rationale

- **Anchoring:** the base velocity field provides a strong prior; the residual only needs to learn the delta for new objects/tasks
- **No forgetting on the base field:** $\theta_0$ is frozen, so all previously learned flow paths are preserved
- **Task-specific corrections:** each environment gets its own residual, handling environment-specific motor adaptations

### 5.5 Combined Architecture at Inference

For environment $E_i$:

```
Image + Language
      │
      ▼
[Frozen VLM Backbone + LoRA_i]  →  conditioning c
      │
      ▼
[Frozen Flow Matching Decoder + TFA Residual_i]  →  action
```

All frozen components are shared; only $\Delta W_i$ (LoRA) and $\Delta v_{\phi_i}$ (TFA residual) are environment-specific.

---

## 6. Anti-Forgetting Mechanisms

### 6.1 Overview

Even though the architecture is decoupled and per-environment, replay is still necessary for two reasons:

1. During training on $E_i$, the LoRA adapter and TFA residual could overfit to $\mathcal{D}_i$ if trained in isolation. Replay from previous environments provides regularization.
2. The **shared conditioning interface** between VLM and decoder means that even with per-environment adapters, there can be subtle distribution shift in the conditioning space. Replay mitigates this.

### 6.2 3DGS Environment Bank as Generative Replay Buffer

#### 6.2.1 Concept

Traditional replay buffers store raw data $(I, a, l)$ from previous environments. This is storage-intensive and provides no ability to generate novel viewpoints or augmented data.

The 3DGS Environment Bank stores the **3DGS reconstructions** of all previous environments $\{E_1, ..., E_{i-1}\}$. When replay data is needed:

1. Select an environment $E_j$ from the bank
2. Render novel-view images from the stored 3DGS scene
3. Combine with the corresponding reference trajectories to produce fresh $(I, a, l)$ triplets

This is a **generative replay buffer** — it can produce an unlimited number of visually diverse replay samples from a compact 3DGS representation, rather than replaying the exact same stored images.

#### 6.2.2 Advantages

- **Storage efficiency:** a 3DGS scene is typically 50–200 MB, far smaller than storing thousands of training images
- **Data diversity:** each replay sample can have a different camera viewpoint, lighting, or object perturbation
- **No distribution collapse:** generative replay avoids the overfitting-to-replay-buffer problem common in experience replay

### 6.3 Competence-Aware Adaptive Replay Scheduling (CARS)

#### 6.3.1 Motivation

Not all previous environments need equal replay. An environment where the VLA is already highly competent needs less replay than one where performance is borderline. Uniform replay wastes compute on environments that don't need it.

#### 6.3.2 Design

Maintain a competence score $s_j$ for each environment $E_j$, estimated by periodically evaluating the current model on a small held-out validation set from each environment's 3DGS-generated data.

Replay probability for environment $E_j$ when training on $E_i$:

$$p(E_j) \propto \max(0, \; s_{\text{threshold}} - s_j)$$

Environments with competence above the threshold receive no replay. Environments with degraded competence receive proportionally more replay.

#### 6.3.3 Asymmetric Treatment

Because the VLM and decoder have different forgetting patterns, CARS maintains **separate competence scores** for perception (VLM) and action (decoder):

- **Perception competence:** measured by whether the VLM correctly identifies the target object and task in its internal representations (e.g., via a probe classifier on the VLM's last layer)
- **Action competence:** measured by end-effector trajectory similarity between the adapted model's output and the reference trajectory

Replay scheduling decisions are made independently for the VLM and decoder components.

---

## 7. EvoHome-Bench: Benchmark Design

### 7.1 Purpose

EvoHome-Bench is the first standardized benchmark for evaluating continual VLA adaptation in evolving home environments. It defines the problem, the evaluation protocol, and the metrics that the community should use.

### 7.2 Benchmark Protocol

#### 7.2.1 Environment Sequence

A standardized sequence of $N$ environments $\{E_1, E_2, ..., E_N\}$, where each environment represents a different home configuration:

- **$E_1$ (Base):** a tabletop with 5 common household objects (e.g., mug, bowl, bottle, plate, spoon)
- **$E_2$ (Object Addition):** $E_1$ + 3 new objects (e.g., a toy, a vitamin bottle, a tape dispenser)
- **$E_3$ (Object Replacement):** some $E_1$ objects replaced with visually similar but different instances (e.g., a different mug, a different bowl)
- **$E_4$ (Layout Change):** same objects as $E_3$, but rearranged on the table
- **$E_5$ (Full Scene Change):** new background, new objects, new layout — maximum distribution shift

#### 7.2.2 Task Set

For each environment, a set of language-conditioned manipulation tasks:

- Pick-and-place: "pick up the [object] and place it on the [location]"
- Object rearrangement: "move the [object] to the left of the [reference]"
- Tool use: "use the [tool] to push the [object]"

Each environment has 10–20 tasks, with shared tasks across environments to measure backward stability.

#### 7.2.3 Evaluation Protocol

1. **Sequential Adaptation:** adapt to $E_1, E_2, ..., E_N$ in order, using only the method's designated adaptation mechanism
2. **After each adaptation step $E_i$:** evaluate on ALL environments $E_1, ..., E_i$
3. **Report the full performance matrix** $P \in \mathbb{R}^{N \times N}$ where $P_{ij}$ = success rate on $E_j$ after adapting through $E_i$

### 7.3 Core Metrics

#### 7.3.1 Forward Transfer Score (FTS)

How well does the method adapt to a new environment?

$$FTS_i = \text{SuccessRate}(E_i) \text{ after adapting to } E_i$$

Higher is better. Measures adaptation capability.

#### 7.3.2 Forgetting Rate (FR)

How much does performance on old environments degrade?

$$FR_{i,j} = \text{SuccessRate}(E_j \text{ after } E_j) - \text{SuccessRate}(E_j \text{ after } E_i), \quad i > j$$

$$FR_{\text{avg}} = \frac{1}{N(N-1)/2} \sum_{i>j} FR_{i,j}$$

Lower is better. Measures backward stability.

#### 7.3.3 Zero-Interaction Compliance (ZIC)

Does the method require per-object human demonstrations for each new environment?

$$ZIC = \begin{cases} 1 & \text{if no per-object demos needed} \\ 0 & \text{otherwise} \end{cases}$$

Binary metric. Our method scores 1; methods like RoboSplat score 0.

#### 7.3.4 Compute Efficiency (CE)

$$CE_i = \frac{\text{GPU-hours for adapting to } E_i}{\text{FTS}_i}$$

Lower is better. Measures the compute cost per unit of adaptation quality.

#### 7.3.5 Overall Score

$$\text{EvoHome Score} = \frac{1}{N} \sum_{i=1}^{N} FTS_i \times (1 - FR_{\text{avg}}) \times ZIC$$

A single scalar summarizing the overall quality of continual adaptation.

### 7.4 Difficulty Levels

| Level | Description | Environment Change |
|-------|-------------|-------------------|
| Easy | Single object added to known scene | Object addition only |
| Medium | Multiple objects added + minor rearrangement | Object addition + layout shift |
| Hard | Object replacement + major rearrangement | Different instances of same categories |
| Extreme | Full scene change (new objects, new layout, new background) | Maximum distribution shift |

### 7.5 Baseline Methods for Leaderboard

1. **Naive Finetuning:** finetune the full VLA on $\mathcal{D}_i$ without any replay or architectural protection
2. **EWC (Elastic Weight Consolidation):** compute Fisher Information Matrix on each environment's data, penalize changes to important weights
3. **Experience Replay:** store raw training images from previous environments, mix into current training
4. **LoRA-Only:** per-environment LoRA on both VLM and decoder (no TFA, no CARS)
5. **Ours (Full):** per-env LoRA + TFA + 3DGS Environment Bank + CARS

---

## 8. Experimental Plan

### 8.1 Hardware & Compute

- **Robot platform:** [to be specified — Franka Emika / UR5 / other]
- **Camera:** wrist-mounted RGB camera for VLA observation + external camera for 3DGS capture
- **GPU:** NVIDIA RTX 3080 (primary development) — note: may need A100/H100 for full VLA finetuning; explore efficient adaptation techniques if compute-limited
- **3DGS training:** ~15-30 min per scene on 3080
- **VLA finetuning:** dependent on backbone size; target < 4 GPU-hours per environment adaptation

### 8.2 VLA Backbone Selection

Candidates (to be decided in Week 1):

| Model | Architecture | Action Decoder | Open-Source | Feasibility on 3080 |
|-------|-------------|---------------|-------------|---------------------|
| π0 | PaliGemma VLM + Flow Matching | Flow Matching | Partial | Likely need larger GPU |
| OpenVLA | Prismatic VLM + MLP | Deterministic | Yes | Feasible with LoRA |
| Octo | Transformer + Diffusion | Diffusion Policy | Yes | Feasible |
| RoboVLM | Vision-Language + Action | Various | Yes | Feasible |

**Decision criteria:** open-source availability, flow matching or diffusion decoder (needed for TFA), feasibility on available compute.

**Note:** If using a non-flow-matching decoder (e.g., diffusion policy), TFA design needs to be adapted — the core idea (residual correction on frozen base decoder) still applies, but the parameterization changes from residual velocity field to residual score function.

### 8.3 Object & Scene Selection

#### 8.3.1 Target Objects

Select 15–25 household objects across categories:

- Kitchen: mugs, bowls, plates, utensils, bottles, food containers
- Office: tape, stapler, pens, notebooks
- Personal: toys, cosmetics, keys, phone
- Challenging: transparent objects, reflective objects, deformable objects (cloth, plush toys)

#### 8.3.2 Environment Sequence Design

Design 5 environments following the EvoHome-Bench protocol (Section 7.2.1). Ensure each environment introduces genuinely novel visual and geometric challenges.

### 8.4 Ablation Plan

| Experiment | VLM Strategy | Decoder Strategy | Replay |
|-----------|-------------|-----------------|--------|
| Naive FT | Full finetune | Full finetune | None |
| EWC | EWC | EWC | None |
| Exp. Replay | Full finetune | Full finetune | Raw image replay |
| LoRA-Only | Per-env LoRA | Per-env LoRA | 3DGS replay |
| Ours w/o CARS | Per-env LoRA | TFA | Uniform 3DGS replay |
| Ours w/o TFA | Per-env LoRA | Per-env LoRA | 3DGS + CARS |
| **Ours (Full)** | **Per-env LoRA** | **TFA** | **3DGS + CARS** |

### 8.5 Expected Results & Hypotheses

1. **Naive FT will show severe catastrophic forgetting** — FTS may be acceptable, but FR will be high
2. **EWC will underperform** because Fisher-based regularization is not well-suited for flow matching decoders (the loss landscape is highly non-diagonal)
3. **Experience Replay will be decent but storage-heavy and limited in data diversity** compared to 3DGS generative replay
4. **LoRA-Only will show that per-env LoRA on the decoder is less stable than TFA** because LoRA directly modifies the velocity field
5. **CARS will provide measurable improvement over uniform replay** especially as the number of environments grows
6. **Full system will achieve the best FTS × (1 - FR) product**

---

## 9. Paper Structure (Planned)

1. **Introduction** (1 page): Frame evolving home environment as an unsolved, formally undefined problem. Establish the need for a benchmark.
2. **Related Work** (0.75 page): 3DGS for robotics (RoboSplat, RoboGSim, GigaWorld-0, RoboPaint), Continual Learning for robotics, VLA adaptation methods (TwinRL-VLA).
3. **Problem Formulation** (0.5 page): Formal definition of the evolving home environment continual adaptation problem.
4. **EvoHome-Bench** (1.25 pages): Full benchmark specification — protocol, metrics, difficulty levels. This is a **first-class contribution**, not a subsection of experiments.
5. **Method** (2 pages): 3DGS synthetic data pipeline, decoupled CL framework (per-env LoRA + TFA), 3DGS Environment Bank, CARS.
6. **Experiments** (2 pages): Ablations, baseline comparisons, leaderboard results, qualitative analysis.
7. **Conclusion** (0.5 page): Contribution summary + EvoHome Challenge announcement.

---

## 10. Competitive Landscape & Differentiation

### 10.1 Key Competitors

| Work | What They Do | What We Do Differently |
|------|-------------|----------------------|
| **RoboSplat** (RSS 2025) | 3DGS for manipulation data synthesis | They require per-task demos; we require zero per-object demos. They don't address CL. |
| **RoboGSim** | 3DGS for simulating manipulation | Simulation pipeline focus; no CL, no evolving environment |
| **TwinRL-VLA** | RL-based VLA adaptation | RL requires reward engineering; we use amortized human demos |
| **RoboPaint** | Inpainting for data augmentation | 2D inpainting lacks 3D geometric consistency |
| **GigaWorld-0** | Large-scale 3DGS world generation | Scene generation focus; no manipulation, no CL |

### 10.2 Our Unique Contributions

1. **First formal definition** of the evolving home environment continual adaptation problem
2. **First benchmark** (EvoHome-Bench) for this problem
3. **First method** that achieves zero per-object human effort through 3DGS-based data synthesis + decoupled CL
4. **First analysis** of forgetting patterns in dual-architecture VLAs (VLM vs. action decoder)

---

## 11. Open Questions & Decision Points

These decisions must be finalized during **Week 1–2** of the timeline:

1. **VLA backbone choice** — which model to use as $\pi_0$? Constrained by compute (3080), open-source availability, and decoder architecture (need flow matching or diffusion for TFA).
2. **LoRA rank** — $r = 8, 16, 32$? Higher rank = more capacity but more parameters per environment.
3. **TFA residual architecture** — how large should $\Delta v_{\phi_i}$ be? Too small = insufficient adaptation; too large = risk of overfitting.
4. **Reference trajectory library size** — how many tasks × how many demos? Minimum viable size for the paper.
5. **Number of environments in EvoHome-Bench** — 5 is the current plan; is this sufficient to demonstrate meaningful CL? Could go to 7–8 if time permits.
6. **Real robot experiments** — are they in scope for this paper, or is simulation sufficient? CoRL strongly prefers real robot results.

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\pi_0$ | Pre-trained VLA base model |
| $\pi_i$ | Adapted model after environment $E_i$ |
| $E_i$ | The $i$-th environment configuration |
| $\mathcal{D}_i$ | Synthesized training dataset for $E_i$ |
| $\mathcal{T}$ | Reference trajectory library |
| $W_0$ | Frozen VLM backbone weights |
| $\Delta W_i$ | LoRA adapter for environment $E_i$ |
| $v_{\theta_0}$ | Frozen base flow matching velocity field |
| $\Delta v_{\phi_i}$ | TFA residual velocity field for environment $E_i$ |
| $s_j$ | Competence score for environment $E_j$ |
| $P_{ij}$ | Success rate on $E_j$ after adapting through $E_i$ |

# EvoHome-Bench: Detailed Design Document

> **Status:** DRAFT — Week 1 design phase
> **Simulation Backend:** BEHAVIOR-1K / OmniGibson
> **Synced with:** [3DGS_EvoHome_Technical_Pipeline.md](3DGS_EvoHome_Technical_Pipeline.md) Section 7

---

## 1. Design Principles

1. **Progressive difficulty** — each environment introduces a specific, isolated type of distribution shift
2. **Shared tasks for forgetting measurement** — some tasks recur across environments to enable backward stability analysis
3. **Grounded in BEHAVIOR-1K assets** — all objects and scenes use existing OmniGibson assets for reproducibility
4. **Automated evaluation** — success determined by BDDL goal predicates, no human judgment needed
5. **3DGS-compatible** — each environment can be captured as multi-view images for 3DGS reconstruction

---

## 2. Environment Sequence

### 2.1 Base Scene Selection

**Scene:** `house_single_floor` kitchen area (or equivalent BEHAVIOR-1K kitchen scene)
**Robot:** Franka Panda (7-DOF, wrist-mounted RGB camera)
**Workspace:** Kitchen countertop / table

### 2.2 Five-Environment Progression

| Env | Name | Change Type | Difficulty | What Changes |
|-----|------|-------------|------------|--------------|
| **E1** | Base Kitchen | — | Baseline | 5 common objects on countertop |
| **E2** | Object Addition | +Objects | Easy | E1 + 3 new objects added |
| **E3** | Instance Replacement | Swap | Medium | 2 objects in E2 replaced with visually similar but different instances |
| **E4** | Layout Rearrangement | Spatial | Hard | Same objects as E3, significantly different spatial arrangement |
| **E5** | Full Scene Change | Everything | Extreme | Different room, new objects, new layout, new background |

### 2.3 Concrete Object Assignments

#### E1 — Base Kitchen (5 objects)
| Object | OmniGibson Synset | Role |
|--------|-------------------|------|
| Red mug | `mug.n.04` | Graspable, containable |
| White bowl | `bowl.n.01` | Graspable, open container |
| Water bottle | `bottle.n.01` | Graspable, tall |
| Plate | `plate.n.04` | Flat surface, placement target |
| Spoon | `spoon.n.01` | Tool, thin/small |

#### E2 — Object Addition (E1 + 3 new objects)
| Added Object | OmniGibson Synset | Why This Object |
|--------------|-------------------|-----------------|
| Toy car | `toy.n.03` | Novel category, small, irregular shape |
| Vitamin bottle | `pill_bottle.n.01` | Cylindrical, visually different from water bottle |
| Tape dispenser | `dispenser.n.01` | Irregular geometry, challenging grasp |

#### E3 — Instance Replacement (swap 2 objects)
| Original | Replaced With | Challenge |
|----------|---------------|-----------|
| Red mug → | Blue mug (different shape/texture, same category) | Visual similarity but different grasp affordance |
| White bowl → | Wooden bowl (different material/color) | Same geometry class, different appearance |

#### E4 — Layout Rearrangement
- Same 8 objects as E3
- Objects moved to opposite ends of countertop
- Clustered differently (e.g., kitchen tools together vs. spread out)
- Tests spatial generalization without visual novelty

#### E5 — Full Scene Change
| Aspect | Change |
|--------|--------|
| Room | Different BEHAVIOR-1K scene (e.g., `restaurant_brunch` or living room area) |
| Objects | 5 entirely new objects (e.g., wine glass, notebook, stapler, candle, plush toy) |
| Layout | New spatial arrangement |
| Background | Different lighting, textures, furniture |
| Shared tasks | At least 3 task types shared with E1 for forgetting measurement (pick-place on new objects) |

---

## 3. Task Design

### 3.1 Task Categories

| Category | Template | Complexity |
|----------|----------|------------|
| **Pick-and-Place** | "Pick up the {object} and place it on the {target}" | Single-step |
| **Relative Placement** | "Move the {object} to the left/right of the {reference}" | Spatial reasoning |
| **Object Retrieval** | "Pick up the {object}" | Simple grasp |
| **Stacking** | "Stack the {object_A} on top of the {object_B}" | Precision placement |
| **Tool Use** | "Use the {tool} to push the {object}" | Multi-object interaction |

### 3.2 Tasks Per Environment

Each environment has **15 tasks**: 10 environment-specific + 5 shared across environments.

#### Shared Tasks (appear in ALL environments, adapted to available objects)
| ID | Task Template | Purpose |
|----|---------------|---------|
| S1 | "Pick up the {mug-like object}" | Backward stability — grasping |
| S2 | "Place the {object} on the plate" | Backward stability — placement |
| S3 | "Move the {object_A} to the left of {object_B}" | Backward stability — spatial |
| S4 | "Pick up the {small object}" | Backward stability — fine grasp |
| S5 | "Stack {object_A} on {object_B}" | Backward stability — precision |

In each environment, these templates are instantiated with the environment's specific objects. This enables direct comparison of the same task type across environments.

#### E1-Specific Tasks (10 tasks)
| ID | Task | BDDL Goal |
|----|------|-----------|
| E1-T1 | "Pick up the red mug" | `(not (ontop mug.n.04_1 countertop.n.01_1)) and (grasping robot mug.n.04_1)` |
| E1-T2 | "Place the red mug on the plate" | `(ontop mug.n.04_1 plate.n.04_1)` |
| E1-T3 | "Pick up the spoon" | `(grasping robot spoon.n.01_1)` |
| E1-T4 | "Move the bowl to the left of the mug" | `(nextto bowl.n.01_1 mug.n.04_1)` |
| E1-T5 | "Pick up the water bottle" | `(grasping robot bottle.n.01_1)` |
| E1-T6 | "Place the spoon in the bowl" | `(inside spoon.n.01_1 bowl.n.01_1)` |
| E1-T7 | "Place the water bottle on the plate" | `(ontop bottle.n.01_1 plate.n.04_1)` |
| E1-T8 | "Stack the bowl on the plate" | `(ontop bowl.n.01_1 plate.n.04_1)` |
| E1-T9 | "Move the plate to the right of the bowl" | `(nextto plate.n.04_1 bowl.n.01_1)` |
| E1-T10 | "Pick up the bowl and place it next to the bottle" | `(nextto bowl.n.01_1 bottle.n.01_1)` |

*(E2-E5 tasks follow similar patterns with their respective objects)*

### 3.3 Language Diversity

For each task, generate **3 language variants** to test language robustness:
- Imperative: "Pick up the red mug"
- Descriptive: "Grab the mug with the red color"
- Contextual: "Get the mug from the counter"

Total: 15 tasks × 3 variants = **45 language-conditioned evaluations per environment**

---

## 4. Evaluation Protocol

### 4.1 Sequential Adaptation Protocol

```
For i = 1 to 5:
    1. Present environment E_i
    2. Method adapts to E_i (using its designated mechanism)
    3. Evaluate on ALL environments E_1 through E_i
    4. Record success rates → fill row i of performance matrix P
```

### 4.2 Performance Matrix

```
        E1    E2    E3    E4    E5
After E1 [ P11   -     -     -     -  ]
After E2 [ P21  P22    -     -     -  ]
After E3 [ P31  P32   P33    -     -  ]
After E4 [ P41  P42   P43   P44    -  ]
After E5 [ P51  P52   P53   P54   P55 ]
```

P_ij = success rate on E_j's tasks after adapting through E_i.

**Diagonal** = Forward Transfer Score (how well did we adapt?)
**Below diagonal** = Forgetting measurement (did old performance degrade?)

### 4.3 Evaluation Episodes Per Cell

Each P_ij cell = **3 trials per task × 15 tasks = 45 episodes**
- Vary initial object poses slightly between trials (within a plausible range)
- Report mean success rate + standard error

Total evaluation episodes per full matrix: 45 × 15 cells = **675 episodes**

### 4.4 Success Criteria

Per episode:
- **Success (1):** All BDDL goal predicates satisfied within time limit
- **Failure (0):** Time limit exceeded or goal not met
- **Time limit:** 150 steps (configurable)

---

## 5. Metrics

### 5.1 Primary Metrics

| Metric | Formula | Measures |
|--------|---------|----------|
| **FTS_i** (Forward Transfer) | P_ii | Adaptation quality |
| **FR_avg** (Forgetting Rate) | Mean of (P_jj - P_ij) for all i > j | Backward stability |
| **ZIC** (Zero-Interaction) | 1 if no per-object demos, else 0 | Demo efficiency |
| **CE_i** (Compute Efficiency) | GPU-hours for E_i / FTS_i | Cost-effectiveness |

### 5.2 Composite Score

$$\text{EvoHome Score} = \frac{1}{N} \sum_{i=1}^{N} FTS_i \times (1 - FR_{avg}) \times ZIC$$

### 5.3 Diagnostic Metrics (Per-Environment Breakdown)

| Metric | Purpose |
|--------|---------|
| **Per-task-type forgetting** | Does forgetting differ by task type (pick vs place vs spatial)? |
| **Per-change-type forgetting** | Does adding objects cause more forgetting than layout changes? |
| **VLM vs Decoder forgetting** | Separate perception errors from motor errors |
| **Adaptation speed** | How many synthetic samples needed to reach X% success? |

---

## 6. Baselines

| Method | Description | Expected Behavior |
|--------|-------------|-------------------|
| **Zero-shot π0.5** | No adaptation, just run base model | Low FTS on novel objects, no forgetting (nothing learned) |
| **Naive Finetuning** | Full finetune on D_i, no replay | Good FTS, catastrophic forgetting |
| **EWC** | Fisher regularization | Moderate FTS, moderate forgetting |
| **Experience Replay** | Store raw images, mix into training | Decent FTS + forgetting control, storage-heavy |
| **LoRA-Only** | Per-env LoRA on both VLM and decoder | Good FTS, moderate forgetting on decoder |
| **Ours (Full)** | Per-env LoRA + TFA + 3DGS Bank + CARS | Best FTS × (1-FR) product |

---

## 7. Multi-View Capture Protocol (for 3DGS)

For each environment E_i, capture images for 3DGS reconstruction:

### 7.1 Camera Trajectory
- **~50 viewpoints** around the workspace
- 3 elevation levels: low (15°), eye-level (45°), overhead (75°)
- Full 360° azimuthal coverage
- Additional close-up shots of each target object (5 per object)
- Total: ~50 global + ~40 close-up = **~90 images per environment**

### 7.2 Capture Parameters
- Resolution: 640×480 (match VLA input resolution)
- Modalities: RGB + depth + camera intrinsics + extrinsics
- Save format: PNG (RGB) + NPY (depth) + JSON (camera params)

### 7.3 OmniGibson API Usage
```python
# Pseudocode for multi-view capture
sensor = VisionSensor(name="capture_cam", ...)
for pose in camera_trajectory:
    sensor.set_position_orientation(position=pose[:3], orientation=pose[3:])
    og.sim.step()  # render
    rgb = sensor.get_obs()["rgb"]
    depth = sensor.get_obs()["depth"]
    params = sensor.camera_parameters  # intrinsics + extrinsics
    save(rgb, depth, params, pose_id)
```

---

## 8. Open Design Questions

- [ ] Exact OmniGibson synsets for each object (need to verify availability in asset library)
- [ ] Time limit per episode: 150 steps sufficient?
- [ ] Should E5 share ANY objects with E1-E4, or be completely novel?
- [ ] Multi-step tasks: include in v1 or defer to v2?
- [ ] Real robot evaluation: if included, which subset of tasks?

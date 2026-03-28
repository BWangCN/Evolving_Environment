# Stage 1 — Structured Selection & Phase Design

> **Research framing:** Probe model lifelong learning ability from an *environment-evolution perspective* —
> specifically how a policy handles incremental object addition and periodic removal in a kitchen
> manipulation domain.

---

## 1. Object Universe Inventory

All objects come from LIBERO-Plus scenes.  We distinguish three tiers:

### Tier A — Core task objects (appear as primary manipulation targets in libero_90)

| Object | Tasks | Shape class | Approx. size | Grasp difficulty |
|--------|-------|------------|--------------|-----------------|
| `akita_black_bowl_1_main` | 34 | Hemisphere | 15 cm Ø | Easy — wide rim, stable |
| `plate_1_main` | 28 | Flat disk | 20 cm Ø | Hard — low profile, slippery |
| `ketchup_1_main` | 19 | Tapered bottle | 20 cm tall | Medium — narrow neck |
| `red_coffee_mug_1_main` | 17 | Cylinder+handle | 10 cm | Medium — handle ambiguity |
| `black_book_1_main` | 17 | Rectangular slab | 25×18×3 cm | Medium — flat, thin |
| `butter_1_main` | 16 | Rectangular box | 12×6×4 cm | Easy — blocky |
| `chocolate_pudding_1_main` | 15 | Short cylinder | 7 cm Ø | Easy — compact |
| `porcelain_mug_1_main` | 15 | Cylinder+handle | 10 cm | Medium |
| `alphabet_soup_1_main` | 14 | Cylinder (can) | 10 cm Ø | Easy |
| `cream_cheese_1_main` | 14 | Short box | 9×7×4 cm | Easy |
| `tomato_sauce_1_main` | 14 | Cylinder (can/bottle) | 14 cm | Easy |
| `chefmate_8_frypan_1_main` | 10 | Pan+handle | 30 cm | Hard — long lever, off-CoM |
| `wine_bottle_1_main` | 6 | Tall tapered bottle | 30 cm | Hard — tall, round, slippery |
| `moka_pot_1_main` | 6 | Stepped cylinder | 15 cm | Medium — octagonal grip |

### Tier B — Add-pool distractor objects (appear only in `_add_N` variants of libero_10)

These objects are *already rigged* in LIBERO-Plus as scene distractors, making them ideal
controlled objects-to-add in your phase transitions:

| Category | Objects |
|----------|---------|
| Bags | `bag_of_yeast_1/2_main` |
| Bottles | `bottle_of_alfredo_sauce_1/2`, `bottle_of_antihistamines_1/2`, `bottle_of_aspirin_1/2`, `bottle_of_baby_oil_1/2`, `bottle_of_barbecue_sauce__2_1/2/3` |
| Boxes | `box_of_shampoo__1_1/2`, `box_of_vegetable_juice__1_1/2/3`, `box_of_yogurt__1_1/2/3`, `boxed_ink_cartridge__3_1/2/3` |
| Cans | `can_of_sardines_1/2`, `can_of_soda__1_1/2`, `canned_food__1_1/2` |

These are semantically coherent (kitchen/pantry context) and geometrically diverse —
useful as distractor additions that should *not* be grasped.

---

## 2. Phase 1 Object Set Selection

**Criteria applied (in priority order):**

1. **Demo availability** — Object must be the primary target in ≥10 libero_90 tasks
   so bootstrapping from existing demonstrations is practical.
2. **Geometric diversity** — Select across shape classes: hemisphere, cylinder, flat, box, bottle.
3. **Grasp difficulty spread** — At least 2 easy, 2 medium, 2 hard to give a meaningful
   performance range for measuring plasticity later.
4. **Semantic coherence** — All objects must plausibly co-exist in a kitchen/dining context
   (discard study/office objects for Phase 1).

### Recommended Phase 1 Core Set (6 objects)

| # | Object | Shape | Difficulty | Rationale |
|---|--------|-------|-----------|-----------|
| 1 | `akita_black_bowl_1_main` | Hemisphere | **Easy** | Highest demo count (34); canonical pick-place anchor |
| 2 | `alphabet_soup_1_main` | Cylinder/can | **Easy** | Symmetric, no orientation ambiguity; 14 tasks |
| 3 | `butter_1_main` | Rectangular box | **Easy** | Blocky, stable; 16 tasks |
| 4 | `red_coffee_mug_1_main` | Cylinder+handle | **Medium** | Handle-grasp tests pose estimation; 17 tasks |
| 5 | `moka_pot_1_main` | Octagonal cylinder | **Medium** | Non-circular cross-section; 6 tasks (acceptable) |
| 6 | `plate_1_main` | Flat disk | **Hard** | Low-profile challenge; 28 tasks for ample demos |

**Rejected candidates and reasons:**
- `chefmate_8_frypan_1_main` — Hard, long handle, off-CoM grasping; too extreme for Phase 1 baseline
  (better as a Phase 2 addition to spike difficulty).
- `wine_bottle_1_main` — Hard; few tasks (6) limits demo diversity.
- `black_book_1_main` — Study domain; breaks kitchen coherence.
- `ketchup_1_main` — Redundant with bottle category already covered by moka_pot.

### Phase 1 Task Coverage

With this 6-object set, the following libero_90 task types are directly exercisable:
- Bowl placement (on plate, in basket, stacking)
- Can pick-place (to basket, to tray)
- Mug pick-place (to plate, positional variants)
- Plate arrangement
- Butter pick-place (to basket, to drawer)
- Moka pot (on stove)

Estimated demo count available from libero_90 clean tasks: **~150–200 trajectories** covering
these 6 objects (exact count depends on initstate variants selected).

---

## 3. Phase Transition Design

### 3.1 Addition Policy

**m = 1 object added per transition** (recommended over m=2+)

*Rationale:* Adding 1 object at a time gives clean attribution — you can isolate which object
caused any degradation. Adding 2+ conflates two signals. Only increase to m=2 in later phases
once you have calibrated the single-object response.

**Selection criterion for the added object:**

Use a *staged novelty* strategy rather than random:

```
Phase 2 add: 1 Tier-B distractor (semantically neutral, same kitchen domain)
             → tests spatial distraction without task relevance
Phase 3 add: 1 Tier-A object (new task-relevant object, e.g. ketchup_1_main)
             → tests task-relevant interference
Phase 4 add: 1 hard Tier-A object (chefmate_8_frypan_1_main)
             → tests high-difficulty object impact on existing skills
Phase 5+:    cycle between distractor and task-relevant additions
```

This gives you a controlled novelty ramp that makes phase effects interpretable.

**Concrete Phase Schedule:**

| Phase | Added object | Category | Expected difficulty impact |
|-------|-------------|----------|---------------------------|
| 1 | *(baseline — 6 objects)* | — | Calibration |
| 2 | `can_of_soda__1_1_main` | Tier-B distractor | Minimal (spatially interfering only) |
| 3 | `ketchup_1_main` | Tier-A task-relevant | Medium (new grasp target, related shape) |
| 4 | `chefmate_8_frypan_1_main` | Tier-A hard | High (demands new policy region) |
| 5 | `bottle_of_alfredo_sauce_1_main` | Tier-B distractor | Minimal |
| 6 | `cream_cheese_1_main` | Tier-A task-relevant | Low-medium (similar to butter) |

### 3.2 Removal Policy

**n = 1 object removed per transition, with reintroduction after 2 phases**

*Rationale:* Permanent removal makes backward-transfer measurement impossible — you can never
test if the model retained the skill. Cyclic removal (remove for 2 phases, reintroduce) lets you
measure:
- **Forgetting rate** during the absence window
- **Reacquisition speed** when the object returns
- Whether the model exhibits *catastrophic forgetting* vs. *graceful degradation*

**Removal selection:** Remove the most recently *mastered* object (highest rolling success rate)
rather than the most recently added. This maximizes the signal-to-noise on forgetting because
you are measuring decay from a peak.

**Reintroduction schedule:** Object returns exactly 2 phases after removal. This creates
a predictable cycle visible in learning curves.

**Concrete example trajectory:**
```
Phase 1:  {bowl, soup, butter, mug, moka, plate}           — 6 objects
Phase 2:  {bowl, soup, butter, mug, moka, plate, soda}     — add soda
Phase 3:  {soup, butter, mug, moka, plate, soda, ketchup}  — add ketchup, remove bowl
Phase 4:  {soup, butter, mug, moka, plate, soda, ketchup, frypan} — add frypan, remove soup?
Phase 5:  {..., bowl}                                        — bowl reintroduced (2 phases later)
```

### 3.3 Phase Transition Trigger

**Recommended: Fixed episode count (primary) + success rate gate (secondary)**

```
Transition fires when BOTH:
  (a) ≥ N_min = 500 episodes completed in current phase, AND
  (b) Rolling success rate over last 100 episodes ≥ θ = 0.60
```

*Why both:*
- Fixed minimum (a) ensures every phase gets enough data for fair comparison across phases.
- Success gate (b) prevents transitions from firing while the model is still in rapid learning —
  you want to measure the *plateau* performance before introducing new objects, not the
  mid-learning performance.
- If (b) is never reached within N_max = 2000 episodes, transition fires anyway (avoids infinite
  phases for tasks the model never masters).

**Alternative trigger (for ablation):** Pure fixed count at N=1000. Simpler, fully reproducible,
no hyperparameter sensitivity — use this as your control condition baseline.

---

## 4. Metrics Suite

To fully characterize lifelong learning, instrument these per-phase and per-object metrics:

| Metric | Definition | What it measures |
|--------|-----------|-----------------|
| **BWT** (Backward Transfer) | Avg success Δ on held-out old-object tasks after phase change | Forgetting |
| **FWT** (Forward Transfer) | Success on new object at end of phase vs. from-scratch | Plasticity |
| **Reacquisition rate** | Episodes to reach θ=0.60 on reintroduced object | Long-term retention |
| **Stability score** | Std dev of success on Phase-1 objects across all phases | Robustness of anchor skills |
| **Peak performance** | Max rolling success per object | Upper bound per difficulty class |

---

## 5. Open Questions to Resolve Before Implementation

1. **Task set per object:** Each phase operates on *which tasks*? Full libero_90 subset touching
   those objects, or a fixed canonical task per object? Fixed tasks = cleaner curves;
   full subset = higher ecological validity.

2. **Distractor placement:** When a Tier-B object is added, does it spawn *on the workspace*
   (forcing the policy to ignore it) or just in the scene graph? The `_add_N` BDDL variants
   already handle this — directly reuse those spawn regions.

3. **Demo augmentation:** When a new object is added, do you provide (a) zero new demos
   (pure online adaptation), (b) a few-shot demo bundle (e.g., 5 trajectories), or
   (c) full demo set for that object? Each tests a different learning regime.
   Recommendation: use (b) 5-shot to stay within realistic lifelong learning assumptions.

4. **Evaluation frequency:** Eval after every K=50 episodes or only at phase boundaries?
   More frequent = better curve resolution but more compute.

5. **Model choice:** Which base VLA? OpenVLA, π0, or a simpler BC baseline? The phase design
   above is model-agnostic, but the threshold θ=0.60 may need calibration per model.

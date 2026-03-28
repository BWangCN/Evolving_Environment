# Preliminary Curriculum Design Plan

This plan formalizes the object selection criteria based on empirical physics constraints and maps the exact curriculum transitions for hybrid VLA / $\pi_0$ evaluation.

---

## 1. Structured Selection & Phase 1 Design

To bootstrap our initial phase efficiently while capturing the highest quality data, we select an exact subset from our recent script findings that maximizes existing support while bridging into more complex generalization.

### The Phase 1 Core Set (10 Objects)
*We rely exactly on the 10 objects inherently supported by `prepare_groceries` and `set_table` configurations.*

| Object | Geo Class | Size Constraint | Grasp Difficulty |
| :--- | :--- | :--- | :--- |
| `008_pudding_box` | Cuboid | Small | **Easy** (Highly stable, flat surfaces) |
| `009_gelatin_box` | Cuboid | Small | **Easy** (Highly stable, flat surfaces) |
| `005_tomato_soup_can` | Cylinder | Medium | **Easy** (Symmetrical top-down) |
| `007_tuna_fish_can` | Cylinder | Very Short | **Easy-Medium** (Requires precise height tracking) |
| `010_potted_meat_can`| Cylinder | Medium | **Easy-Medium** (Symmetrical) |
| `004_sugar_box` | Cuboid | Tall | **Medium** (Tipping risk if bumped off-center) |
| `003_cracker_box` | Cuboid | Very Tall | **Medium-Hard** (Large; limits gripper orientation) |
| `002_master_chef_can`| Cylinder | Large | **Medium-Hard** (Wide radius) |
| `024_bowl` | Concave | Very Wide (16cm)| **Hard** (Unstable rim grasping required) |
| `013_apple` | Spherical | Small | **Hard** (Rolling risk, low friction contact) |

### Selection Rationale:
1. **Demo Availability (Bootstrap)**: Every single object here already exists inside the 84 pre-configured ReplicaCAD Scene instances. This guarantees that **we do not need to manually generate new oracle trajectories immediately** to bootstrap Phase 1.
2. **Geometric Diversity**: Covers planar interactions (boxes), cylindrical rotational symmetry (cans), and complex irregular topology (bowl, apple).
3. **Semantic Coherence**: 100% ecological validity. Every item natively belongs inside the `fridge`, `wall_cabinet`, or on a `kitchen_counter`. 

---

## 2. Phase Transition Curriculum Rules

Once Phase 1 stabilizes, we need explicit programmatic metrics to manage the curriculum rollout and measure model plasticity.

### Transition Trigger Mechanism: *Success Rate Thresholding*
* **Trigger Formula**: A phase transitions when the evaluation success rate of the active object set surpasses **$\geq$ 85%** across a running window of 100 random validation episodes.
* **Why not Fixed Episodes or Time?**: Fixed triggers are hardware-dependent and arbitrarily punish harder phases that legitimately require more gradient steps to converge. Performance-based thresholding ensures the model actually learned the representation before the environment shifts.

### Addition Rates ($m$)
* **Rate**: $m = 2$ objects added per phase transition.
* **Selection Criteria (Curriculum Climbing)**: We iterate the remaining 68 YCB objects starting with **Harder Semantic Matches**, gradually bleeding into **Anomalous Out-of-Distribution (OOD)** meshes. 
  - *Phase 2 Additions*: `025_mug`, `065-a_cups` (Extending the kitchen theme with handle-based affordances).
  - *Phase 3 Additions*: `011_banana`, `021_bleach_cleanser` (Irregular kitchen/cleaning items, extremely high shape difficulty).
  - *Phase 4 Additions*: `033_spatula`, `048_hammer` (Tools representing total OOD geometric transfer). 

### Removal Rates ($n$) and Backward Transfer
* **Rate**: $n = 1$ object removed per phase transition.
* **Selection Criteria**: We remove the **easiest consistently solved object** from the active set (e.g., dropping `008_pudding_box`).
* **Reappearance Loop**: Removals are **NOT permanent**.
  - *Why*: To vigorously test **Catastrophic Forgetting** and Backward Transfer. After 3 phases (when the model hasn't seen a `pudding_box` in hundreds of thousands of gradient steps), we inject it back into the evaluation split. If the model fails on previously learned easy objects, our plasticity regularization is failing.


## Evaluation Protocol

### The Accuracy Matrix R

After each phase $t$, evaluate on every object seen so far:

$$R_{t,i} = \text{success rate on object } i \text{ after training through phase } t$$

This gives a lower-triangular matrix per task stream. Standard continual learning metrics derived from it:

| Metric | Formula | What it measures |
| :--- | :--- | :--- |
| **BWT** (Backward Transfer) | $\frac{1}{T-1}\sum_{i=1}^{T-1}(R_{T,i} - R_{i,i})$ | Catastrophic forgetting |
| **FWT** (Forward Transfer) | $\frac{1}{T-1}\sum_{i=2}^{T}(R_{i-1,i} - b_i)$ | Zero-shot generalization |
| **Intransigence** | $R_{i,i}^{\text{joint}} - R_{i,i}$ | Inability to learn new objects |
| **Plasticity** | $\frac{1}{T}\sum_{i=1}^{T} R_{i,i}$ | Peak performance at introduction |

Where $b_i$ is random-policy baseline for object $i$, and $R_{i,i}^{\text{joint}}$ is the upper bound from joint training on all objects.

### Baselines to Include

| Baseline | Description |
| :--- | :--- |
| **Joint training** | Train on all objects simultaneously — upper bound |
| **Fine-tuning (no replay)** | Sequential fine-tuning with no forgetting mitigation — lower bound |
| **EWC** (Elastic Weight Consolidation) | Standard regularization baseline |
| **PackNet / PNN** | Architecture-based CL baseline |
| **Replay buffer** | Fixed-size experience replay |
| **Your method** | Evolving-environment VLA curriculum |

---

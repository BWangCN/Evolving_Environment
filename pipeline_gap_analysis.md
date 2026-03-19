# 3DGS-EvoHome Pipeline Gap Analysis

> **Tool Decision:** Gaussian Grouping (core pipeline) + gsplat (batch rendering) — **方案 D**
> **Synced with:** [3DGS_Timeline.md](3DGS_Timeline.md) | [progress.md](progress.md)
> **Status Legend:** ✅ 工具覆盖 | ❌ 需自建 | 🔧 进行中 | ✔️ 已完成 | ~~删除~~ 不再需要
>
> **MAJOR UPDATE (Mar 16):** Replaced reference trajectory library + retargeting with
> AnyGrasp + motion planner pipeline. B5/B6 eliminated. True zero-demonstration.
>
> **UPDATE (Mar 17 PM):** A1/A2/A3 verified on 5090. B8/B9/B13 verified. AnyGrasp env ready.
> π0.5 inference verified. Stage A+/B-1/B-2 code complete (74 tests).

---

## Stage A: 场景重建与理解

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| A1 | 多视角拍摄 + COLMAP | ✔️ 已验证 | Gaussian Grouping (COLMAP) — bear scene 96 images, 63K points |
| A2 | 3DGS 重建 | ✔️ 已验证 | Gaussian Grouping — 30K iters, 892 MB PLY, ~30 it/s on 5090 |
| A3 | SAM 语义分割（哪些是物体） | ✔️ 已验证 | Gaussian Grouping (SAM + Identity Encoding) — 3D consistent segmentation |
| A4 | 物体语义识别（这是什么物体） | ❌ 缺失 | 候选：VLM (GPT-4V/Gemini) 或 CLIP 查询 |
| A5 | 物体物理属性推断（可抓取性等） | ❌ 缺失 | 候选：GaussianProperty (ICCV 2025) 或 VLM |
| A6 | 环境 Bank 注册 | ❌ 需自建 | 场景 ID → PLY + metadata 数据库 |

---

## Stage A+: Scene Object Registry (NEW)

> **Bridges A3→B1:** Unified data structure combining geometry (from 3DGS) + semantics + affordances.
> Drives task planning, language generation, place target computation, and collision checking.

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| A+ | SceneObject dataclass | ✔️ 已完成 | `src/scene/object.py` — geometry + category + description + affordances |
| A+ | SceneObjectRegistry | ✔️ 已完成 | `src/scene/registry.py` — per-env container, query by affordance/task type |
| A+ | AffordanceInference | ✔️ 已完成 | `src/scene/affordance.py` — category → valid tasks, is_container, is_surface, graspable |

---

## Stage B-1: 任务规划（做什么）

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| B1 | 物体-任务匹配 | ✔️ 已完成 | `src/task/planner.py` — TaskPlanner: registry → enumerate valid (task, object_pair) |
| B2 | Language instruction 生成 | ✔️ 已完成 | `src/task/language.py` — LanguageGenerator: template-based, LLM extension point |
| B3 | 任务组合编排 | ❌ 缺失 | 规则 + LLM；先单步，多步作为 extension |

---

## Stage B-2: 轨迹生成（怎么做）

> **Pipeline change:** 不再使用参考轨迹库 + 轨迹重定向。
> 改为：AnyGrasp 生成 grasp poses → Motion planner 生成轨迹。
> 这实现了真正的 zero-demonstration（不需要任何人工演示）。

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| B4 | Grasp pose generation | 🔧 环境就绪 | **AnyGrasp** SDK installed + compiled (license pending). Point cloud extraction from 3DGS ready |
| ~~B5~~ | ~~参考轨迹匹配~~ | ~~删除~~ | ~~被 AnyGrasp + motion planner 取代~~ |
| ~~B6~~ | ~~轨迹重定向~~ | ~~删除~~ | ~~被 AnyGrasp + motion planner 取代~~ |
| B6* | Motion planning | ✔️ 已完成 | `src/trajectory/generator.py` — phased waypoint trajectory, 165 trajectories tested |
| B7 | 碰撞检测 | ✔️ 已完成 | `src/trajectory/collision.py` — KDTree, phase-aware margins, 90.3% acceptance rate |

---

## Stage B-3: 场景编辑与渲染（生成图像）

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| B8 | 物体移除 | ✔️ 已验证 | Gaussian Grouping — bear removed, 764 MB PLY. **重定位未实现**（需自建 SE(3) 变换） |
| B9 | 空洞填充 (inpainting) | ✔️ 已验证 | Gaussian Grouping (LaMa + 10K fine-tune) — clean result, 2.8 GB PLY |
| B10 | 物体附着 EE (grasp 中) | ❌ 缺失 | 逐帧对物体 Gaussians 施加 SE(3) 变换 |
| B11 | 机器人模型渲染 | ❌ 可跳过 | wrist camera 视野中几乎看不到手臂 |
| B12 | 数据增强 | ❌ 需自建 | camera perturbation, SH modification, pose randomization |
| B13 | 批量渲染 → (I, a, l) | ✔️ 已验证 | gsplat 1.5.3 — loaded 9.5M Gaussians, rendered 640x480 from arbitrary pose |

---

## Stage C: 训练

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| C1 | Per-env LoRA + TFA 训练 | 🔧 推理+适配已验证 | openpi π0.5 — LIBERO format (7-dim delta EE pose) → ManiSkill adapter written. LoRA/TFA modification pending |
| C2 | 3DGS Environment Bank | ❌ 需自建 | 管理所有环境 3DGS + metadata + gsplat 渲染 |
| C3 | CARS replay 调度 | ✔️ 已完成 | `src/evaluation/cars.py` — CARSScheduler with decoupled VLM/decoder support |

---

## Stage D: 评估

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| D1 | EvoHome-Bench 评估管线 | ✔️ 已完成 | `src/evaluation/metrics.py` — FTS/FR/ZIC/CE/EvoHome Score + method comparison |

---

## 开发优先级

```
Week 1: 锁定决策 + A1/A2/A3 跑通 + 代码模块 + 组件安装验证  ← ALL DONE
  ✔️ A1/A2/A3 verified, B6*/B7/B1/B2/C3/D1 code complete (74 tests)
  ✔️ B8/B9 (removal+inpainting) verified, B13 (gsplat rendering) verified
  ✔️ AnyGrasp env ready (license pending), π0.5 inference verified
Week 2: B4 (AnyGrasp license → test) + B8 relocation (custom SE(3)) + first (I,a,l) triplets
Week 3: B10 (EE attach) + B12 (augmentation) → full E1 synthetic data
Week 4: C1 (LoRA+TFA on π0.5)
Week 5+: C2
```

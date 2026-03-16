# 3DGS-EvoHome Pipeline Gap Analysis

> **Tool Decision:** Gaussian Grouping (core pipeline) + gsplat (batch rendering) — **方案 D**
> **Synced with:** [3DGS_Timeline.md](3DGS_Timeline.md) | [progress.md](progress.md)
> **Status Legend:** ✅ 工具覆盖 | ❌ 需自建 | 🔧 进行中 | ✔️ 已完成

---

## Stage A: 场景重建与理解

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| A1 | 多视角拍摄 + COLMAP | ✅ GG 覆盖 | Gaussian Grouping (COLMAP) |
| A2 | 3DGS 重建 | ✅ GG 覆盖 | Gaussian Grouping |
| A3 | SAM 语义分割（哪些是物体） | ✅ GG 覆盖 | Gaussian Grouping (SAM + Identity Encoding) |
| A4 | 物体语义识别（这是什么物体） | ❌ 缺失 | 候选：VLM (GPT-4V/Gemini) 或 CLIP 查询 |
| A5 | 物体物理属性推断（可抓取性等） | ❌ 缺失 | 候选：GaussianProperty (ICCV 2025) 或 VLM |
| A6 | 环境 Bank 注册 | ❌ 需自建 | 场景 ID → PLY + metadata 数据库 |

---

## Stage B-1: 任务规划（做什么）

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| B1 | 物体-任务匹配 | ❌ 缺失 | LLM：物体列表 + 任务模板 → 可行组合 |
| B2 | Language instruction 生成 | ❌ 缺失 | LLM 模板扩展，多样化表述 |
| B3 | 任务组合编排 | ❌ 缺失 | 规则 + LLM；先单步，多步作为 extension |

---

## Stage B-2: 轨迹生成（怎么做）

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| B4 | Grasp pose proposal | ❌ 缺失 | GraspSplats / AnyGrasp / Contact-GraspNet |
| B5 | 参考轨迹匹配 | ❌ 缺失 | Grasp type + task type 相似度匹配 |
| B6 | 轨迹重定向 (retargeting) | ❌ 缺失 | 两个 grasp pose 间 SE(3) 相对变换 |
| B7 | 碰撞检测 | ❌ 缺失 | Gaussian 点云近似碰撞检查 或 trimesh |

---

## Stage B-3: 场景编辑与渲染（生成图像）

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| B8 | 物体移除/重定位 | ✅ GG 覆盖 | Gaussian Grouping |
| B9 | 空洞填充 (inpainting) | ✅ GG 覆盖 | Gaussian Grouping (clone + LaMa + fine-tune) |
| B10 | 物体附着 EE (grasp 中) | ❌ 缺失 | 逐帧对物体 Gaussians 施加 SE(3) 变换 |
| B11 | 机器人模型渲染 | ❌ 可跳过 | wrist camera 视野中几乎看不到手臂 |
| B12 | 数据增强 | ❌ 需自建 | camera perturbation, SH modification, pose randomization |
| B13 | 批量渲染 → (I, a, l) | ✅ gsplat 覆盖 | gsplat batch API + DDP |

---

## Stage C: 训练

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| C1 | Per-env LoRA + TFA 训练 | ❌ 需自建 | 基于 openpi (JAX) π0.5 代码修改 |
| C2 | 3DGS Environment Bank | ❌ 需自建 | 管理所有环境 3DGS + metadata + gsplat 渲染 |
| C3 | CARS replay 调度 | ❌ 需自建 | 能力评估 + 动态采样概率 |

---

## Stage D: 评估

| ID | 模块 | 状态 | 工具/方案 |
|----|------|------|-----------|
| D1 | EvoHome-Bench 评估管线 | ❌ 需自建 | FTS/FR/ZIC/CE 计算 + 性能矩阵 |

---

## 开发优先级

```
Week 1: 锁定决策 + A1/A2/A3 跑通（Gaussian Grouping 验证）
Week 2: A4→B1→B2 (LLM驱动) + B4 (grasp pose)
Week 3: B5→B6→B10→B13 (轨迹+渲染) + B12 (增强)
Week 4: C1 (LoRA+TFA on π0.5)
Week 5+: C2, C3, D1
```

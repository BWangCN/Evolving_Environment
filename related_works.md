# Related Works — Draft for CoRL 2026

> **Usage:** This document organizes related works into subsections matching our paper structure.
> Each entry includes a brief summary and how it relates to our contribution.
> BibTeX keys match entries in `references.bib`.

---

## 1. Vision-Language-Action Models

The emergence of large-scale VLA models has transformed robot learning. These models combine vision-language understanding with action generation, enabling language-conditioned manipulation from visual observations.

**Foundation VLA models.** RT-1 \cite{brohan2023rt1} introduced a scalable transformer architecture for real-world robotic control, trained on 130k demonstrations across 700+ tasks. RT-2 \cite{brohan2023rt2} extended this by finetuning a vision-language model (PaLM-E) to directly output robot actions, demonstrating emergent capabilities like semantic reasoning about novel objects. Octo \cite{team2024octo} provided the first open-source generalist robot policy, trained on 800k episodes from the Open X-Embodiment dataset \cite{openxembodiment2024} with a transformer backbone and diffusion action head. OpenVLA \cite{kim2024openvla} built on the Prismatic VLM backbone with 7B parameters, achieving strong open-source VLA performance with LoRA finetuning support.

**The π-series.** Physical Intelligence introduced π0 \cite{black2024pi0}, a dual-expert architecture combining a PaliGemma VLM with a flow matching action decoder. π0.5 \cite{physicalintelligence2025pi05} improved on this with knowledge insulation — a gradient-stopping mechanism that prevents action training from corrupting VLM representations — and hierarchical inference (subtask prediction → action generation). π0.6 \cite{physicalintelligence2025pi06} further scaled the action expert to 860M parameters and introduced RECAP, an RL-based training framework. **Our work builds on π0.5 as the base VLA, leveraging its knowledge insulation as a natural complement to our decoupled continual learning framework.**

**Action decoders.** Modern VLAs employ diverse action generation strategies. Diffusion Policy \cite{chi2023diffusion} formulates action generation as a conditional denoising process, achieving strong multi-modal action distributions. ACT \cite{zhao2023act} uses action chunking with transformers for bimanual manipulation. Flow matching \cite{lipman2023flow}, used in π0/π0.5, learns a velocity field that transforms noise into action sequences via ODE integration. FAST \cite{pertsch2025fast} introduced a discrete tokenizer for robot actions, enabling autoregressive action prediction. **Our TFA (Trajectory Flow Anchoring) method operates specifically on flow matching decoders, learning residual velocity fields for environment-specific adaptation.**

**VLA adaptation.** Despite impressive zero-shot generalization, current VLAs struggle with personalized deployment. The BEHAVIOR-1K 2025 Challenge \cite{li2024behavior} showed π0 achieving only ~26% success rate across 50 household tasks. OpenVLA-OFT \cite{kim2025openvlaoft} systematically studied finetuning recipes, boosting success from 76.5% to 97.1%. TraceVLA \cite{zheng2025tracevla} improved generalization via visual trace prompting. TwinRL-VLA \cite{wang2025twinrl} proposed RL-based adaptation but requires reward engineering and a digital twin. SpatialVLA \cite{qu2025spatialvla} incorporated 3D spatial understanding into VLAs for better zero-shot transfer. **However, none of these methods address continual adaptation to evolving environments — they assume a fixed target domain. Our work fills this gap by proposing zero-interaction adaptation via 3DGS-synthesized data with anti-forgetting guarantees.**

---

## 2. 3D Gaussian Splatting for Robotics

3D Gaussian Splatting (3DGS) \cite{kerbl2023gaussian} represents scenes as collections of anisotropic 3D Gaussians, enabling real-time, photorealistic novel-view synthesis. Its explicit, editable representation makes it particularly suitable for robotic data synthesis.

**3DGS-based manipulation data generation.** RoboSplat \cite{yang2025robosplat} is the most directly related work, generating manipulation demonstrations by editing 3DGS scenes — replacing objects, varying viewpoints, and swapping embodiments. It achieves 87.8% success from one-shot demonstrations. However, RoboSplat requires **per-task human demonstrations** for each new object and does not address continual learning. RoboGSim \cite{li2024robogsim} builds a real-to-sim-to-real pipeline using 3DGS for robotic simulation. GSWorld \cite{jiang2025gsworld} provides closed-loop evaluation with a GSDF scene format. SplatSim \cite{qureshi2024splatsim} enables zero-shot sim-to-real transfer by replacing simulation backgrounds with 3DGS renderings. GigaWorld-0 \cite{gigaworld2025} scales 3DGS to a large-scale data engine for embodied AI. GWM \cite{lu2025gwm} learns Gaussian world models for action-conditioned prediction. **Unlike all these works, our method combines 3DGS data synthesis with continual learning, uses a generative replay buffer (3DGS Environment Bank) instead of raw image storage, and requires zero per-object demonstrations.**

**Object-level 3DGS editing.** Gaussian Grouping \cite{ye2024gaussiangrouping} augments each Gaussian with a 16D identity encoding supervised by SAM masks, enabling object-level segmentation, removal, relocation, and inpainting via LaMa \cite{suvorov2022lama}. GaussianEditor \cite{chen2024gaussianeditor} and \cite{wang2024gaussianeditor2} enable text-guided 3D editing. 3DGS-Drag \cite{dong2025drag} supports point-based dragging of Gaussians. Feature 3DGS \cite{zhou2024feature3dgs} distills arbitrary 2D foundation model features into Gaussians. LangSplat \cite{qin2024langsplat} embeds CLIP language features for open-vocabulary 3D querying. **Our pipeline uses Gaussian Grouping for scene editing and inpainting, combined with gsplat \cite{ye2025gsplat} for efficient batch rendering.**

**3DGS inpainting.** When an object is removed from a 3DGS scene, the previously occluded surface has no Gaussian coverage. GScream \cite{wang2024gscream} addresses this with depth-guided cross-attention feature propagation. InFusion \cite{liu2024infusion} uses diffusion priors for depth completion. 3DGS-CD \cite{lu2025gscd} duplicates and optimizes Gaussians near the boundary. **RoboSplat notably avoids the inpainting problem entirely by using replacement rather than removal — our method addresses inpainting directly.**

**3DGS for robotic perception.** GaussianGrasper \cite{zheng2024gaussiangrasper} distills language features into Gaussians for open-vocabulary grasping with a normal-guided grasp selection module. GraspSplats \cite{ji2024graspsplats} combines feature splatting with grasp detection. POGS \cite{yu2025pogs} tracks object poses using persistent Gaussian representations. **We use 3DGS reconstruction for perception but delegate grasp pose generation to AnyGrasp \cite{fang2023anygrasp}, which operates on point clouds extracted from the Gaussian positions.**

---

## 3. Continual Learning for Robotics

Continual learning (CL) addresses the challenge of learning new tasks without forgetting previously acquired knowledge. While extensively studied in classification, its application to robotic manipulation — particularly for VLA models — remains underexplored.

**Classic CL methods.** Elastic Weight Consolidation (EWC) \cite{kirkpatrick2017ewc} penalizes changes to parameters important for previous tasks via the Fisher Information Matrix. Progressive Neural Networks \cite{rusu2016progressive} add new columns for new tasks while freezing old ones. PackNet \cite{mallya2018packnet} identifies and freezes important weights through iterative pruning. Experience Replay \cite{rolnick2019replay} stores and replays data from previous tasks during new learning. **We compare against EWC and experience replay as baselines, showing that standard CL methods are suboptimal for dual-architecture VLAs where the VLM backbone and action decoder have fundamentally different forgetting dynamics.**

**LoRA-based continual learning.** Low-Rank Adaptation (LoRA) \cite{hu2022lora} enables parameter-efficient finetuning by learning low-rank weight updates. Recent works explore LoRA for CL: InfLoRA \cite{liang2024inflora} introduces interference-free LoRA that reduces inter-task interference. O-LoRA \cite{wang2023olora} learns orthogonal LoRA subspaces for different tasks. Per-task LoRA adapters avoid forgetting by design — each task has its own adapter while the base model is frozen. **Our approach uses per-environment LoRA adapters on the VLM backbone, ensuring zero cross-environment forgetting in the perception component.**

**Continual learning for manipulation.** LIBERO \cite{liu2024libero} provides a lifelong robot learning benchmark with continual task suites. ConTinTin \cite{gao2025contintin} studies continual instruction tuning for VLAs. CL-SLAM-VLA explores simultaneous localization and VLA adaptation.

**Concurrent work: Continual learning for VLAs (Feb–Mar 2026).** Three concurrent papers address continual learning specifically for VLA models, but all operate at the task level (new manipulation objectives with the same physical environment):

CRL-VLA \cite{zeng2026crlvla} proposes a dual-critic RL architecture for task-stream CL. A frozen Goal-Conditioned Value critic anchors old task semantics while a trainable MC critic drives adaptation. Using OpenVLA-OFT on LIBERO, they achieve 0.98 FAR with 0.03 forgetting in single-task CL. The approach is purely RL-algorithmic — no parameter isolation or adapters.

LifeLong-RFT \cite{liu2026lifelongrft} replaces SFT with Group Relative Policy Optimization (GRPO, from DeepSeek-Math) for VLA continual fine-tuning. Using NORA-Long (3B, discrete actions) with a Multi-Dimensional Process Reward combining token accuracy, trajectory alignment, and format compliance, they achieve +22% AUC improvement over SFT on LIBERO CL tasks. They still require 10 demonstrations per new task plus 5 replay demos per old task.

Hu et al. \cite{hu2026simplerecipe} ("Simple Recipe Works") show that simple sequential LoRA fine-tuning with RL already achieves near-zero forgetting across OpenVLA-OFT, π0, and OpenVLA on LIBERO + RoboCasa + ManiSkill. Their explanation — large pretrained models + LoRA's low-rank constraint + on-policy RL = implicit anti-forgetting — supports our use of LoRA as a strong CL primitive.

**However, all three address task-level CL only — the physical environment remains fixed and only the manipulation objective changes. None addresses the harder, more realistic problem of environment-level CL where objects are added, removed, or rearranged. Our EvoHome-Bench is the first benchmark to formalize this problem, and our 3DGS-based data synthesis pipeline is the first to enable zero-interaction adaptation to evolving environments.**

**Generative replay.** Instead of storing raw data, generative replay uses a generative model to synthesize replay samples. This approach has been explored with VAEs \cite{shin2017generativereplay} and GANs for classification CL. DDGR \cite{gao2023ddgr} demonstrated that diffusion models produce higher-quality replay samples than VAEs/GANs for class-incremental learning. **Our 3DGS Environment Bank extends generative replay to 3D scenes — previous environments are stored as compact 3DGS representations (~50-200MB each) that can render unlimited novel-view replay samples with photorealistic quality, avoiding the storage cost of raw images and the distribution collapse of replaying identical training data.**

**Anti-forgetting in foundation models.** Recent work has explored continual learning specifically for large vision-language models. MoE-Adapters4CL \cite{yu2024moeadapters} uses mixture-of-experts adapters for VLM continual learning. A comprehensive survey \cite{shi2025llmcl} reviews CL strategies for LLMs. Biderman et al. \cite{biderman2024lora} showed that LoRA learns less but also forgets less than full finetuning, providing empirical support for our per-environment LoRA strategy. **Our work is the first to analyze forgetting patterns in dual-architecture VLAs where the VLM backbone and action decoder have fundamentally different forgetting dynamics, motivating our decoupled CL framework.**

---

## 4. Grasp Pose Estimation

Reliable grasp pose estimation is essential for our trajectory generation pipeline.

**6-DoF grasp detection.** AnyGrasp \cite{fang2023anygrasp} generates accurate, dense, temporally-smooth 7-DoF grasp poses robust to large depth noise, trained on the GraspNet-1Billion dataset \cite{fang2020graspnet}. Contact-GraspNet \cite{sundermeyer2021contactgraspnet} treats 3D points as potential grasp contacts for end-to-end 6-DoF grasp generation. GraspGen \cite{nvidia2025graspgen} uses diffusion-based grasp synthesis to surpass prior methods. **We use AnyGrasp to generate candidate grasp poses from point clouds extracted from our 3DGS reconstructions. All physically valid grasps are retained as diverse training data rather than selecting a single best grasp — this diversity is a feature of our synthetic data pipeline.**

---

## 5. Benchmarks and Simulation Environments

**Manipulation benchmarks.** BEHAVIOR-1K \cite{li2024behavior} defines 1,000 everyday household activities with 3,484 object categories in 51 scenes, built on OmniGibson/NVIDIA Omniverse. LIBERO \cite{liu2024libero} provides lifelong learning benchmarks for manipulation with 130 tasks across 10 task suites. RoboCasa \cite{nasiriany2024robocasa} offers large-scale simulation for everyday household tasks. MetaWorld \cite{yu2020metaworld} and RLBench \cite{james2020rlbench} provide multi-task manipulation benchmarks. ManiSkill \cite{gu2023maniskill2} supports GPU-parallelized manipulation evaluation. **Our EvoHome-Bench builds on BEHAVIOR-1K's infrastructure but introduces a fundamentally new evaluation dimension: sequential environment evolution with continual adaptation, tracked through a performance matrix measuring both forward transfer and backward forgetting.**

**Synthetic data for robot learning.** Domain randomization \cite{tobin2017domainrandomization} and sim-to-real transfer remain central to robot learning. NVIDIA's GR00T \cite{nvidia2025groot} blueprint generates 780K synthetic trajectories from 11 hours of demonstrations using GR00T-Mimic + Cosmos Transfer. RoboPaint \cite{chen2025robopaint} uses 2D inpainting for visual data augmentation but lacks 3D geometric consistency. **Our approach differs from all these by using 3DGS as the rendering engine, which provides photorealistic, geometrically consistent multi-view images while supporting object-level scene editing for manipulation data synthesis.**

---

## Summary of Positioning

| Aspect | Closest Prior Work | Our Contribution |
|--------|-------------------|------------------|
| 3DGS for manipulation data | RoboSplat (RSS 2025) | + Zero per-object demos + Continual learning + 3DGS Environment Bank |
| VLA adaptation | TwinRL-VLA | + No RL reward engineering + 3DGS-based synthetic data |
| Continual manipulation | LIBERO | + Evolving environments (not just new tasks) + EvoHome-Bench |
| CL for VLAs | CRL-VLA, LifeLong-RFT, Simple Recipe | + Environment-level CL (not task-level) + Zero demos + 3DGS data pipeline |
| CL mechanism | InfLoRA, O-LoRA | + Decoupled VLM/decoder strategy + TFA for flow matching |
| Manipulation benchmark | BEHAVIOR-1K | + Continual adaptation protocol + Forgetting metrics |

**Key positioning vs concurrent CL-for-VLA work (Feb–Mar 2026):**

| | CRL-VLA | LifeLong-RFT | Simple Recipe | **Ours** |
|--|---------|-------------|--------------|---------|
| CL level | Task | Task | Task | **Environment** |
| What changes | Language goal | Language goal | Language goal | **Physical world** |
| New demos needed | RL rollouts | 10/task | RL rollouts | **Zero** |
| 3DGS | No | No | No | **Core contribution** |
| VLA backbone | OpenVLA-OFT | NORA-Long 3B | OpenVLA/π0 | **π0.5** |

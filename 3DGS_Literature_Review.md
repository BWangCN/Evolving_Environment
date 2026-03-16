# Literature Review: 3D Gaussian Splatting for Object-Level Scene Editing and Robot Manipulation Data Generation

**Prepared for:** CoRL 2026 Submission
**Date:** March 16, 2026
**Scope:** Semantic segmentation of Gaussians, object removal/relocation, inpainting, scene composition, and 3DGS-based robot manipulation data generation

---

## Table of Contents
1. [Semantic Segmentation & Clustering of Gaussians](#1-semantic-segmentation--clustering-of-gaussians)
2. [Object Removal from 3DGS Scenes](#2-object-removal-from-3dgs-scenes)
3. [Object Relocation & Manipulation in 3DGS](#3-object-relocation--manipulation-in-3dgs)
4. [Inpainting / Filling the Hole After Object Removal](#4-inpainting--filling-the-hole-after-object-removal)
5. [Scene Composition: Adding New Objects](#5-scene-composition-adding-new-objects)
6. [3DGS + Robot Manipulation Data Generation](#6-3dgs--robot-manipulation-data-generation)
7. [Foundational & Peripheral Works](#7-foundational--peripheral-works)
8. [Summary Table](#8-summary-table)
9. [Key Takeaways for Our Project](#9-key-takeaways-for-our-project)

---

## 1. Semantic Segmentation & Clustering of Gaussians

### 1.1 Gaussian Grouping: Segment and Edit Anything in 3D Scenes
- **Authors:** Mingqiao Ye, Martin Danelljan, Fisher Yu, Lei Ke
- **Venue:** ECCV 2024
- **arXiv:** 2312.00732
- **GitHub:** https://github.com/lkeab/gaussian-grouping

**Key Idea:** Augments each 3D Gaussian with a compact 16-dimensional Identity Encoding that enables grouping by object instance or stuff class. Supervised via differentiable rendering using 2D masks from SAM, combined with 3D spatial consistency regularization (KL divergence over k=5 nearest neighbors).

**Segmentation approach:** SAM-based. SAM generates per-view 2D masks; a cross-entropy classification loss trains the identity encodings. A 3D spatial regularization loss ensures Gaussians in occluded or under-supervised regions maintain consistent identity with their neighbors.

**Hole/inpainting:** After deleting target Gaussians, ~200K new Gaussians are cloned near deletion regions and fine-tuned using LaMa 2D inpainting results as supervision. Takes minutes vs. hours for NeRF methods.

**Open-source:** Yes (full code, training, rendering, editing applications).

**Relevance:** **Extremely high.** This is the most directly applicable method for our pipeline. It provides (a) per-Gaussian semantic labels via SAM, (b) object-level decoupling, (c) direct scene recomposition by exchanging 3D locations between Gaussian groups, and (d) inpainting via LaMa + Gaussian fine-tuning.

---

### 1.2 SAGA: Segment Any 3D Gaussians
- **Authors:** Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, Qi Tian
- **Venue:** AAAI 2025
- **arXiv:** 2312.00860
- **GitHub:** https://github.com/Jumpat/SegAnyGAussians

**Key Idea:** Attaches a scale-gated affinity feature to each 3D Gaussian. Uses a scale-aware contrastive training strategy to distill SAM segmentation capability into the 3D Gaussian affinity features. Achieves 3D promptable segmentation in 4ms.

**Segmentation approach:** SAM-based distillation with scale-gated affinity features. Supports 2D visual prompts (points, boxes, text) mapped to 3D Gaussian subsets.

**Hole/inpainting:** Not directly addressed.

**Open-source:** Yes.

**Relevance:** High. Provides extremely fast (~4ms) segmentation of Gaussian subsets, useful for interactive object selection.

---

### 1.3 GaussianCut: Interactive Segmentation via Graph Cut for 3D Gaussian Splatting
- **Authors:** Umangi Jain, Ashkan Mirzaei, Igor Gilitschenski
- **Venue:** NeurIPS 2024
- **arXiv:** 2411.07555
- **GitHub:** https://umangi-jain.github.io/gaussiancut/

**Key Idea:** Represents the 3DGS scene as a graph and uses graph-cut energy minimization to partition Gaussians into foreground/background. Accepts point clicks, scribbles, or text as input.

**Segmentation approach:** Graph-cut on Gaussians with energy function combining user inputs (from 2D/video segmentation models) and 3D scene properties. Not purely SAM or CLIP -- uses them as initial coarse signals refined by graph structure.

**Hole/inpainting:** Not addressed.

**Open-source:** Yes.

**Relevance:** Moderate. Alternative segmentation approach that could be useful when SAM-based methods produce noisy boundaries.

---

### 1.4 LangSplat: 3D Language Gaussian Splatting
- **Authors:** Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, Hanspeter Pfister
- **Venue:** CVPR 2024 (Highlight)
- **arXiv:** 2312.16084
- **GitHub:** https://github.com/minghanqin/LangSplat

**Key Idea:** Embeds CLIP language features into 3D Gaussians via a scene-specific autoencoder. Uses SAM to learn hierarchical semantics. Enables open-vocabulary 3D object localization and semantic segmentation at 199x faster than LERF.

**Segmentation approach:** CLIP + SAM. Trains a scene-wise language autoencoder; learns language features on scene-specific latent space. SAM provides hierarchical semantic scales.

**Hole/inpainting:** Not addressed (focused on querying/localization).

**Open-source:** Yes.

**Relevance:** High for semantic querying of objects in the scene. Could be used upstream to identify which Gaussians to manipulate via natural language.

---

### 1.5 LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS
- **Authors:** Chongjie Li, Yujie Zhao, Minghan Qin, Wanhua Li, Haoqian Wang, Hanspeter Pfister
- **Venue:** NeurIPS 2025
- **arXiv:** 2507.07136
- **GitHub:** https://github.com/ZhaoYujie2002/LangSplatV2

**Key Idea:** Treats each Gaussian as a sparse code within a global dictionary, learning a 3D sparse coefficient field that eliminates the need for a decoder. Achieves 476 FPS feature splatting (42x faster than LangSplat) with improved query accuracy.

**Segmentation approach:** CLIP-based sparse coefficient field. Maintains the same open-vocabulary capabilities as LangSplat.

**Open-source:** Yes.

**Relevance:** Moderate-high. Real-time language queries useful if interactive object selection is needed.

---

### 1.6 Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields
- **Authors:** Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, Achuta Kadambi
- **Venue:** CVPR 2024 (Highlight)
- **arXiv:** 2312.03203
- **GitHub:** https://github.com/ShijieZhou-UCLA/feature-3dgs

**Key Idea:** Develops a Parallel N-dimensional Gaussian Rasterizer that renders arbitrarily high-dimensional features (SAM, CLIP-LSeg, etc.) without sacrificing speed. A speed-up convolutional module maintains efficiency.

**Segmentation approach:** Distills arbitrary 2D foundation model features (SAM, CLIP) into per-Gaussian feature fields. Supports novel view semantic segmentation and language-guided editing.

**Hole/inpainting:** Not directly addressed.

**Open-source:** Yes.

**Relevance:** High. Provides the general-purpose feature distillation backbone that could complement any segmentation or semantic querying approach.

---

### 1.7 LEGS: Language-Embedded Gaussian Splats
- **Authors:** Justin Yu, Kush Hari, Karim El-Refai, et al.
- **Venue:** IROS 2024
- **GitHub:** https://github.com/uynitsuj/LEGS

**Key Idea:** Incrementally builds room-scale language-embedded Gaussian splats using a mobile robot. Trains online as the robot traverses the environment. 3.5x faster training than LERF with comparable query accuracy (66% open-vocabulary accuracy).

**Segmentation approach:** CLIP features distilled into Gaussians during online training. Supports open-vocabulary queries.

**Open-source:** Yes.

**Relevance:** Moderate. Interesting for building the initial scene representation with a mobile robot, though less relevant for static manipulation scenes.

---

### 1.8 LERF: Language Embedded Radiance Fields
- **Authors:** Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, Matthew Tancik
- **Venue:** ICCV 2023
- **arXiv:** 2303.09553
- **Website:** https://www.lerf.io/

**Key Idea:** Grounds CLIP embeddings into NeRF via volume rendering. Enables zero-shot, pixel-aligned open-vocabulary 3D queries. Foundational work that motivated all subsequent language-embedded Gaussian methods.

**Segmentation approach:** CLIP volume rendering in NeRF (not 3DGS). Slow but pioneering.

**Hole/inpainting:** Not addressed.

**Open-source:** Yes (via nerfstudio).

**Relevance:** Low-moderate (NeRF-based predecessor; superseded by LangSplat/Feature 3DGS for our purposes).

---

## 2. Object Removal from 3DGS Scenes

### 2.1 GScream: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal
- **Authors:** Yuxin Wang, Qianyi Wu, Guofeng Zhang, Dan Xu
- **Venue:** ECCV 2024
- **GitHub:** https://github.com/W-Ted/GScream

**Key Idea:** Optimizes Gaussian primitive positioning for geometric consistency across removed and visible areas using online registration from monocular depth estimation (Marigold). Uses cross-attention feature propagation for texture coherence.

**Segmentation approach:** Uses SAM-based masks for identifying removal targets.

**Hole/inpainting:** Uses SD-inpainting and LaMa for 2D inpainted images; Marigold for depth guidance. Cross-attention between target and surrounding anchor Gaussians propagates features into the removal region.

**Open-source:** Yes (built on Scaffold-GS).

**Relevance:** High. Directly addresses the hole-filling problem with geometry-aware inpainting, which is better than Gaussian Grouping's simpler LaMa-only approach.

---

### 2.2 Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360 Scenes
- **Authors:** Taein Kwon et al.
- **Venue:** arXiv 2025 (2511.06457)

**Key Idea:** Extends 3DGS inpainting to complex 360-degree scenes with multi-object removal. Distills 2D segmentation masks into a 3D Gaussian field to assign per-Gaussian object labels. Addresses challenges specific to unbounded scenes.

**Segmentation approach:** Distills 2D segmentation masks into per-Gaussian labels in 3D.

**Hole/inpainting:** Multi-object aware inpainting pipeline optimized for 360-degree consistency.

**Open-source:** Not yet confirmed.

**Relevance:** Moderate. Useful if our scenes require full 360-degree consistency (e.g., tabletop from multiple angles).

---

### 2.3 Clean-Splat: Context-Aware Real-Time Object Removal in AR via Generative 3D Gaussian Inpainting
- **Authors:** (Preprint December 2025)
- **Venue:** Preprint 2025

**Key Idea:** Real-time object removal for augmented reality. Integrates a View-Consistent Diffusion Prior to hallucinate occluded background. Updates 3D scene representation in near real-time (>30 FPS).

**Hole/inpainting:** View-Consistent Diffusion Prior for hallucinating background geometry and texture.

**Open-source:** Not yet confirmed.

**Relevance:** Moderate. Real-time capability interesting but the AR focus may not directly apply.

---

## 3. Object Relocation & Manipulation in 3DGS

### 3.1 Gaussian Grouping (Recomposition)
- **Venue:** ECCV 2024 (see Section 1.1 above)

**Object Relocation:** After segmentation, exchanging 3D locations between two Gaussian groups is trivial -- "no parameter tuning" required. The decoupled explicit representation means one can simply apply a rigid transformation (translation + rotation) to all Gaussians in a group.

**Relevance:** **Critical.** This is exactly what we need: segment object Gaussians, apply SE(3) transform, render the new scene.

---

### 3.2 3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing
- **Authors:** Jiahua Dong, Yu-Xiong Wang
- **Venue:** ICLR 2025
- **GitHub:** https://github.com/Dongjiahua/3DGS-Drag

**Key Idea:** Extends DragGAN-style editing to 3D. Uses 3D handle and target points as input. Combines deformation guidance (3DGS) with diffusion guidance (content correction). Supports motion change, shape adjustment, inpainting, and content extension.

**How relocation works:** Users specify handle points on the object and target points for desired position. Deformation is applied to Gaussians with a progressive editing strategy. Takes 10-20 minutes on a single RTX 4090.

**Hole/inpainting:** Integrated -- diffusion guidance corrects visual artifacts and fills partially observed areas created by the deformation.

**Open-source:** Yes.

**Relevance:** Moderate. More suited for deformation/shape editing than rigid object relocation. The diffusion-based hole filling is interesting but slow.

---

### 3.3 3DGS-CD: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement
- **Authors:** Ziqi Lu et al.
- **Venue:** IEEE RA-L 2025
- **arXiv:** 2411.03706
- **GitHub:** https://github.com/520xyxyzq/3DGS-CD

**Key Idea:** Detects physical object rearrangements by comparing two sets of unaligned images (before/after). Leverages 3DGS rendering + EfficientSAM for 2D change detection, fused across views for 3D changes. Key insight: transforms pre-optimized Gaussians according to estimated object movements, then duplicates Gaussians near newly un-occluded regions and optimizes only those duplicates.

**How relocation works:** Gaussians representing changed objects are rigidly transformed per estimated SE(3) movement. New Gaussians are cloned and optimized for newly exposed regions.

**Hole/inpainting:** Duplicates Gaussians near un-occluded regions; freezes all pre-trained parameters; optimizes only the duplicates to fill in missing parts.

**Open-source:** Yes.

**Relevance:** High. Their approach to handling un-occluded regions after object movement is directly relevant to our pipeline.

---

### 3.4 3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting
- **Authors:** Qihang Zhang, Yinghao Xu, Chaoyang Wang, Hsin-Ying Lee, Gordon Wetzstein, Bolei Zhou, Ceyuan Yang
- **Venue:** ICLR 2025
- **arXiv:** 2405.18424
- **GitHub:** https://github.com/zqh0253/3DitScene

**Key Idea:** Disentangles Gaussians per object via CLIP language features. Users query objects by language prompt, then can change camera viewpoint and manipulate objects flexibly. Initializes 3DGS by lifting pixels to 3D, expands over novel views via RGB and depth inpainting.

**How relocation works:** Language-guided object selection + free manipulation of disentangled Gaussian groups.

**Hole/inpainting:** Uses RGB and depth inpainting to expand the scene to novel views.

**Open-source:** Yes (includes HuggingFace demo).

**Relevance:** Moderate-high. Language-guided disentanglement is appealing for flexible object selection. However, designed more for single-image scene generation than multi-view captured scenes.

---

## 4. Inpainting / Filling the Hole After Object Removal

### 4.1 SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields
- **Authors:** Ashkan Mirzaei, Tristan Aumentado-Armstrong, et al. (Samsung Labs)
- **Venue:** CVPR 2023
- **arXiv:** 2211.12254
- **GitHub:** https://github.com/SamsungLabs/SPIn-NeRF

**Key Idea:** Pioneering work on 3D inpainting. Given posed images and sparse annotations, obtains 3D segmentation masks for target objects, then uses perceptual optimization leveraging 2D image inpainters, distilling their information into 3D while ensuring view consistency.

**How inpainting works:** 2D inpainters provide per-view completions; perceptual loss distills this into the NeRF volume for multi-view consistency.

**Open-source:** Yes. Also released a benchmark dataset of real-world scenes with/without target objects.

**Relevance:** Moderate (NeRF-based, but the benchmark dataset and evaluation protocol are reusable; the 3DGS successors like GScream build on this).

---

### 4.2 InFusion: Inpainting 3D Gaussians via Learning Depth Completion from Diffusion Prior
- **Authors:** Zhiheng Liu, Hao Ouyang, et al. (Alibaba)
- **Venue:** arXiv 2024 (2404.11613)
- **GitHub:** https://github.com/ali-vilab/Infusion

**Key Idea:** Guides Gaussian point initialization for inpainting regions with an image-conditioned depth completion model learned from diffusion priors. Restores depth at aligned scale with the original scene, then initializes and optimizes new Gaussians.

**How inpainting works:** (1) Depth completion from diffusion prior at aligned scale, (2) Point initialization from completed depth, (3) Gaussian optimization for visual harmony. Supports user-specified textures and novel object insertion.

**Open-source:** Yes (inference code + pretrained checkpoint).

**Relevance:** High. The depth-guided initialization is a principled approach to filling holes that avoids the "floating Gaussians" problem.

---

### 4.3 SplatFill: 3D Scene Inpainting via Depth-Guided Gaussian Splatting
- **Authors:** (arXiv 2509.07809, September 2025)
- **Venue:** arXiv 2025

**Key Idea:** 7-stage pipeline: select reference view, perform 2D inpainting, estimate monocular depth, group pixels by depth, incorporate object supervision, detect depth inconsistencies, repeat until all views meet consistency requirements.

**How inpainting works:** Iterative depth-guided consistency checking across views. Depth grouping prevents incorrect Gaussian placement.

**Open-source:** Not confirmed.

**Relevance:** Moderate. Thorough but potentially slow due to iterative nature.

---

### 4.4 RePaintGS: Reference-Guided Gaussian Splatting for Realistic and View-Consistent 3D Scene Inpainting
- **Authors:** Ji Hyun Seo et al.
- **Venue:** arXiv 2025 (2507.08434)

**Key Idea:** Given an inpainted reference view, estimates inpainting similarity of other views to adjust their contribution. Warps reference inpainting to other views as pseudo-ground truth for optimization.

**How inpainting works:** Reference-guided approach: one view is inpainted well, then geometric warping + similarity weighting propagates the appearance to all other views.

**Open-source:** Not confirmed.

**Relevance:** Moderate-high. The reference-guided approach could be practical for our pipeline -- inpaint one view well, propagate to others.

---

### 4.5 VISTA: Visibility-Uncertainty-guided 3D Gaussian Inpainting via Scene Conceptional Learning
- **Authors:** Mingxuan Cui et al.
- **Venue:** ICLR 2025 submission (arXiv 2504.17815)
- **GitHub:** https://github.com/Aswhalefall/VISTA

**Key Idea:** Measures visibility uncertainties of 3D points across views and uses them to guide inpainting. Learns a semantic concept of the scene without the masked object, then uses a diffusion model to fill based on the learned concept.

**How inpainting works:** Visibility-uncertainty weighting + scene concept learning + diffusion-based filling.

**Open-source:** Yes.

**Relevance:** Moderate. Principled approach to handling inconsistent occlusions.

---

## 5. Scene Composition: Adding New Objects

### 5.1 GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting
- **Authors:** Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, Guosheng Lin
- **Venue:** CVPR 2024
- **arXiv:** 2311.14521
- **GitHub:** https://github.com/buaacyw/GaussianEditor

**Key Idea:** Introduces Gaussian Semantic Tracing for precise editing control and Hierarchical Gaussian Splatting (HGS) for stable results under stochastic generative guidance from 2D diffusion models. Supports object addition, removal, and modification.

**Segmentation approach:** Gaussian Semantic Tracing identifies Gaussians requiring editing at each training step.

**Scene composition:** 3D inpainting algorithm enables adding new objects into scenes. Only intended areas are modified.

**Hole/inpainting:** Integrated inpainting for both removal (hole filling) and addition (generating new content).

**Open-source:** Yes (WebUI available).

**Relevance:** High. Full-featured editor with both removal and addition capabilities. 5-10 minute editing sessions.

---

### 5.2 GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions
- **Authors:** Junjie Wang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, Qi Tian
- **Venue:** CVPR 2024
- **arXiv:** 2311.16037

**Key Idea:** Note: This is a different paper from 5.1 with the same name. Extracts region of interest (RoI) from text instructions, aligns text RoI to 3D Gaussian space via grounding segmentation, then edits Gaussians with constraints within the RoI. Runs in <20 minutes on a V100.

**Segmentation approach:** Text-guided grounding segmentation module maps text to 2D masks, projected to 3D Gaussian space.

**Open-source:** Available.

**Relevance:** Moderate. Text-guided approach useful if we want natural language control over editing.

---

### 5.3 GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing
- **Authors:** Jing Wu, Jia-Wang Bian, Xinghui Li, Guangrun Wang, Ian Reid, Philip Torr, Victor Prisacariu
- **Venue:** ECCV 2024
- **arXiv:** 2403.08733
- **GitHub:** https://github.com/ActiveVisionLab/gaussctrl

**Key Idea:** Edits all rendered images together (not iteratively) using ControlNet. Multi-view consistency via: (a) depth-conditioned editing for geometric consistency, (b) attention-based latent code alignment for unified appearance across views.

**How it works:** Render views from 3DGS, edit all simultaneously with ControlNet + cross-view attention, then optimize 3D model from edited images.

**Open-source:** Yes.

**Relevance:** Moderate. Good for appearance/style editing but not specifically designed for rigid object manipulation.

---

### 5.4 DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing
- **Authors:** Minghao Chen, Iro Laina, Andrea Vedaldi
- **Venue:** ECCV 2024
- **arXiv:** 2404.18929
- **GitHub:** https://github.com/silent-chen/DGE

**Key Idea:** Two-stage process: (1) multi-view consistent 2D editing via training-free modification of InstructPix2Pix with 3D geometry cues, (2) direct optimization of 3DGS from edited images. Avoids incremental edits.

**Open-source:** Yes.

**Relevance:** Low-moderate. More for appearance editing than object manipulation.

---

## 6. 3DGS + Robot Manipulation Data Generation

### 6.1 RoboSplat: Novel Demonstration Generation with Gaussian Splatting
- **Authors:** Sizhe Yang, Wenye Yu, Jia Zeng, Jun Lv, Kerui Ren, Cewu Lu, Dahua Lin, Jiangmiao Pang
- **Venue:** RSS 2025
- **arXiv:** 2504.13175
- **GitHub:** https://github.com/OpenRobotLab/RoboSplat

**Key Idea:** From a single expert demonstration + multi-view images, generates diverse manipulation demonstrations by directly editing 3D Gaussians. Achieves 87.8% success rate in one-shot settings vs. 57.2% for policies trained on hundreds of real demos with 2D augmentation.

**Segmentation:** Grounded-SAM for objects; URDF point cloud for robot arm links.

**Object relocation:** Rigid transformations applied to target object Gaussians. End-effector poses transformed equivariantly. Motion planning generates trajectories between keyframe poses.

**Hole/inpainting:** **Not explicitly addressed.** Uses replacement strategy (substitute background, swap objects) rather than filling holes. This is a notable gap.

**Pipeline:**
1. Multi-view capture + COLMAP + Depth Anything + 3DGS training
2. Grounded-SAM segmentation + URDF-based robot segmentation
3. Six augmentation types: object pose (rigid transform), object type (3D generation), camera view (novel view synthesis), embodiment (replace robot Gaussians), scene appearance (swap background), lighting (modify Gaussian colors)
4. Behavioral cloning policy training

**Open-source:** Yes.

**Relevance:** **Extremely high.** This is the closest existing work to our project. Key differences for our submission: (1) they avoid the inpainting problem through replacement, (2) they focus on single-arm tabletop, (3) we could extend with proper inpainting + humanoid manipulation.

---

### 6.2 RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator
- **Authors:** Xinhai Li, Jialin Li, Ziheng Zhang, Rui Zhang, Fan Jia, Tiancai Wang, Haoqiang Fan, Kuo-Kun Tseng, Ruiping Wang
- **Venue:** arXiv 2024 (2411.11839)
- **Website:** https://robogsim.github.io/

**Key Idea:** Full Real2Sim2Real pipeline with four modules: Gaussian Reconstructor (3DGS from multi-view + robot MDH parameters), Digital Twins Builder (mesh reconstruction + Isaac Sim), Scene Composer (combine scene + robot + objects), Interactive Engine (demonstration synthesis + policy evaluation).

**Segmentation:** Robot arm segmented via point cloud + MDH kinematics. General object segmentation method not detailed.

**Scene composition:** Coordinate transformation of Gaussian attributes with rotation matrix normalization and scale adjustment to place objects and robots into scenes.

**Hole/inpainting:** Not addressed.

**Data generation:** Physics-based trajectory generation in Isaac Sim, rendered through Gaussian Splatting. Zero-shot performance on real robot comparable to real data models.

**Open-source:** Not confirmed.

**Relevance:** **High.** Demonstrates the full Real2Sim2Real loop with physics integration. Their Digital Twins Builder + physics engine approach is complementary to our Gaussian-only editing approach.

---

### 6.3 GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation
- **Authors:** Guangqi Jiang, Haoran Chang, Ri-Zhao Qiu, Yutong Liang, Mazeyu Ji, Jiyue Zhu, Zhao Dong, Xueyan Zou, Xiaolong Wang
- **Venue:** arXiv 2025 (2510.20813)
- **GitHub:** https://github.com/luccachiang/GSWorld
- **Website:** https://3dgsworld.github.io/

**Key Idea:** Proposes GSDF (Gaussian Scene Description File) format combining Gaussian-on-Mesh with robot URDF. Curates database of 3 robot embodiments + 40+ objects. Supports zero-shot sim2real policies, DAgger data collection, reproducible benchmarking, virtual teleoperation, and visual RL.

**Key innovation:** GSDF format unifies visual (Gaussian) and physical (mesh) representations. Streamlined reconstruction pipeline for creating new assets.

**Open-source:** Yes.

**Relevance:** **Very high.** The GSDF format and asset database approach is highly relevant. Their closed-loop evaluation capability addresses a key limitation of open-loop data generation approaches.

---

### 6.4 SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies
- **Authors:** (CMU)
- **Venue:** arXiv 2024 (2409.10161)
- **Website:** https://splatsim.github.io/

**Key Idea:** Replaces mesh rendering in simulators with Gaussian Splat rendering. Each rigid body's Gaussians are segmented, and homogeneous transformations relative to the simulator are identified. Achieves 86.25% success rate vs. 97.5% for real-world trained policies.

**Segmentation:** Rigid body-level Gaussian segmentation aligned with simulator objects.

**Object manipulation:** Rigid body transforms applied in simulator; corresponding Gaussians rendered at new poses.

**Hole/inpainting:** Not needed -- simulator controls all object positions; background is always visible.

**Open-source:** Partially.

**Relevance:** **High.** Demonstrates the viability of Gaussian-based rendering for sim2real transfer. However, requires an existing simulator (not purely Gaussian-based).

---

### 6.5 RoboSimGS: High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation
- **Authors:** Haoyu Zhao, Cheng Zeng, Linghao Zhuang, et al.
- **Venue:** arXiv 2025 (2510.10637)

**Key Idea:** Hybrid reconstruction: 3DGS for photorealistic appearance + mesh primitives for interactive objects. Pioneers using a Multimodal LLM (MLLM) to automatically infer physical properties (density, stiffness) and kinematic structures (hinges, rails) from multi-view images.

**Open-source:** Not confirmed.

**Relevance:** High. The MLLM-based automatic asset creation is an interesting alternative to manual specification.

---

### 6.6 GigaWorld-0: World Models as Data Engine to Empower Embodied AI
- **Authors:** (Open-GigaAI team)
- **Venue:** arXiv 2025 (2511.19861)
- **GitHub:** https://github.com/open-gigaai/giga-world-0

**Key Idea:** Unified world model framework as a data engine for VLA learning. Integrates GigaWorld-0-Video (large-scale video generation) and GigaWorld-0-3D (3D generative modeling + 3DGS reconstruction + physics + motion planning). Uses 3DGS-FG (Trellis-based latent diffusion for foreground) and 3DGS-BG (sparse-view 3DGS for background).

**Key innovation:** Scales data generation via FP8-precision and sparse attention (GigaTrain framework). VLA models (GigaBrain-0) trained purely on generated data achieve strong real-world performance.

**Open-source:** Yes.

**Relevance:** **High.** Demonstrates that large-scale 3DGS-based data generation can train effective real-world policies without any real interaction. The video + 3D dual approach is complementary to ours.

---

### 6.7 GWM: Gaussian World Models for Robotic Manipulation
- **Authors:** Zhixuan Lu et al.
- **Venue:** ICCV 2025
- **arXiv:** 2508.17600
- **GitHub:** https://github.com/Gaussian-World-Model/gaussianwm

**Key Idea:** Latent Diffusion Transformer (DiT) + 3D VAE predicts future scene states as Gaussian propagations conditioned on robot actions. Enables action-conditioned 3D video prediction, visual representation learning for imitation learning, and serves as a neural simulator for model-based RL.

**Open-source:** Yes.

**Relevance:** Moderate-high. World model approach is complementary; predicts future Gaussian states rather than explicitly editing them.

---

### 6.8 ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation
- **Authors:** Guanxing Lu et al.
- **Venue:** ECCV 2024
- **arXiv:** 2403.08321
- **GitHub:** https://github.com/GuanxingLu/ManiGaussian

**Key Idea:** End-to-end behavior cloning agent with dynamic Gaussian Splatting framework + Gaussian world model. Infers semantics propagation in Gaussian embedding space; predicts optimal robot actions via future scene reconstruction. Outperforms prior SOTA by 13.1% on 10 RLBench tasks.

**Open-source:** Yes.

**Relevance:** Moderate. Focused on policy learning architecture rather than data generation. The scene dynamics modeling via Gaussians is conceptually related.

---

### 6.9 Object-Aware Gaussian Splatting for Robotic Manipulation
- **Authors:** (2025)
- **Venue:** OpenReview 2025
- **Website:** https://object-aware-gaussian.github.io/

**Key Idea:** Injects "objectness" into semantic 3D Gaussians -- Gaussians with the same semantic label initialize and update together. Captures dynamic manipulation scenes at 30 Hz with only 3 cameras. Enables language-conditioned dynamic grasping.

**Segmentation:** Pretrained foundation models at initial step; object-aware grouping for fast updates.

**Open-source:** Not confirmed.

**Relevance:** High. Real-time dynamic scene understanding during manipulation is valuable for closed-loop data generation.

---

### 6.10 GaussianGrasper: 3D Language Gaussian Splatting for Open-Vocabulary Robotic Grasping
- **Authors:** Yuhang Zheng, Xiangyu Chen, et al.
- **Venue:** RA-L 2024
- **arXiv:** 2403.09637
- **GitHub:** https://github.com/MrSecant/GaussianGrasper

**Key Idea:** Creates language-embedded Gaussian scene from limited RGB-D views. Efficient Feature Distillation (EFD) module uses contrastive learning for CLIP embedding distillation. Normal-guided grasp module selects collision-free grasp poses.

**Open-source:** Yes.

**Relevance:** Moderate. Focused on grasping rather than data generation, but the language-Gaussian pipeline could inform our semantic querying.

---

### 6.11 GraspSplats: Efficient Manipulation with 3D Feature Splatting
- **Authors:** Mazeyu Ji, Ri-Zhao Qiu, Xueyan Zou, Xiaolong Wang
- **Venue:** CoRL 2024
- **arXiv:** 2409.02084
- **GitHub:** https://github.com/jimazeyu/GraspSplats

**Key Idea:** High-quality scene representations in <60 seconds using depth supervision and reference feature computation. Explicit optimized geometry supports real-time grasp sampling and dynamic manipulation. 10x faster than existing GS methods.

**Open-source:** Yes.

**Relevance:** Moderate. Fast scene reconstruction useful for rapid prototyping.

---

## 7. Foundational & Peripheral Works

### 7.1 Robo-GS: A Physics Consistent Spatial-Temporal Model for Robotic Arm
- **Authors:** (Beijing Institute of Technology)
- **Venue:** ICRA 2025
- **arXiv:** 2408.14873

**Key Idea:** Hybrid mesh-Gaussian-pixel binding for robotic arms. Gaussian-Mesh-Pixel binding creates isomorphic mapping between mesh vertices and Gaussian models. Fully differentiable rendering + physics simulation.

**Relevance:** Moderate. The mesh-Gaussian binding is useful for representing articulated robots.

---

### 7.2 Physically Embodied Gaussian Splatting: A Realtime Correctable World Model for Robotics
- **Authors:** Jad Abou-Chakra, Krishan Rana, Feras Dayoub, Niko Suenderhauf
- **Venue:** CoRL 2024
- **GitHub:** https://github.com/bdaiinstitute/embodied_gaussians

**Key Idea:** Dual "Gaussian-Particle" representation. Particles for geometry + physics simulation; attached Gaussians for rendering. "Visual forces" correct particle positions by comparing predicted and observed images. Runs at 30Hz with 3 cameras.

**Relevance:** Moderate. The physics-visual coupling is interesting for realistic manipulation simulation.

---

### 7.3 GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs
- **Authors:** Xinli Xu et al.
- **Venue:** ICCV 2025
- **arXiv:** 2412.11258
- **GitHub:** https://github.com/EnVision-Research/Gaussian-Property

**Key Idea:** Training-free framework assigning physical material properties to Gaussians. Uses SAM + GPT-4V for global-local physical property reasoning in 2D, then projects to 3D via voting. Enables physics simulation and grasp force estimation.

**Relevance:** Moderate. Physical properties useful if we need realistic physics-based manipulation.

---

### 7.4 POGS: Persistent Object Gaussian Splat for Tracking
- **Authors:** Justin Yu, Kush Hari, et al. (UC Berkeley)
- **Venue:** ICRA 2025
- **arXiv:** 2503.05189
- **GitHub:** https://github.com/uynitsuj/pogs

**Key Idea:** Compact Gaussian representation embedding semantics, self-supervised visual features, and object grouping for continuous object pose estimation. Supports sequential resets and tool servoing without rescanning.

**Relevance:** Moderate. Object tracking during manipulation complements our data generation pipeline.

---

### 7.5 EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation
- **Authors:** Gunjan Chhablani et al.
- **Venue:** ICCV 2025
- **arXiv:** 2509.17430
- **GitHub:** https://github.com/gchhablani/embodied-splat-v1

**Key Idea:** iPhone-captured scenes reconstructed via DN-Splatter Gaussian Splatting for personalized navigation policy training. 20-40% success rate improvement over zero-shot baselines.

**Relevance:** Low-moderate. Navigation focus, but the mobile capture + GS reconstruction pipeline is relevant.

---

### 7.6 SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition
- **Authors:** (arXiv 2401.17857)
- **Venue:** arXiv 2024

**Key Idea:** Addresses boundary ambiguity in 3DGS segmentation by decomposing boundary Gaussians using the ellipsoidal structure of 3D Gaussians.

**Relevance:** Moderate. Improved boundary segmentation could reduce artifacts at object boundaries during manipulation.

---

## 8. Summary Table

| Paper | Venue | Segmentation | Object Removal | Relocation | Inpainting | Robotics | Open Source |
|-------|-------|-------------|----------------|------------|------------|----------|-------------|
| **Gaussian Grouping** | ECCV 2024 | SAM + Identity Encoding | Yes (delete Gaussians) | Yes (swap positions) | LaMa + fine-tune Gaussians | No | Yes |
| **SAGA** | AAAI 2025 | SAM distillation | Implicit (select subset) | No | No | No | Yes |
| **GaussianCut** | NeurIPS 2024 | Graph-cut + SAM/video | Implicit | No | No | No | Yes |
| **LangSplat** | CVPR 2024 | CLIP + SAM | No | No | No | No | Yes |
| **LangSplatV2** | NeurIPS 2025 | CLIP sparse codes | No | No | No | No | Yes |
| **Feature 3DGS** | CVPR 2024 | Any 2D foundation model | No | No | No | No | Yes |
| **LEGS** | IROS 2024 | CLIP (online) | No | No | No | No | Yes |
| **GScream** | ECCV 2024 | SAM | Yes | No | SD-inpainting + LaMa + depth | No | Yes |
| **Inpaint360GS** | arXiv 2025 | 2D mask distillation | Yes (360) | No | Multi-object 360 | No | No |
| **Clean-Splat** | Preprint 2025 | Not detailed | Yes (real-time) | No | Diffusion prior | No | No |
| **3DGS-Drag** | ICLR 2025 | Point-based | No | Yes (drag) | Diffusion guidance | No | Yes |
| **3DGS-CD** | RA-L 2025 | EfficientSAM | Yes | Yes (rigid SE3) | Duplicate + optimize | No | Yes |
| **3DitScene** | ICLR 2025 | CLIP language | No | Yes (free manip) | RGB + depth inpainting | No | Yes |
| **SPIn-NeRF** | CVPR 2023 | Manual + NeRF | Yes | No | Perceptual 2D distillation | No | Yes |
| **InFusion** | arXiv 2024 | N/A | Yes | No | Depth completion + diffusion | No | Yes |
| **RePaintGS** | arXiv 2025 | N/A | Yes | No | Reference-guided warping | No | No |
| **VISTA** | ICLR 2025 sub | N/A | Yes | No | Visibility-uncertainty + diffusion | No | Yes |
| **GaussianEditor (Chen)** | CVPR 2024 | Semantic tracing | Yes | Yes (add/remove) | Diffusion-guided HGS | No | Yes |
| **GaussianEditor (Wang)** | CVPR 2024 | Text grounding | Yes | Yes | Text-guided RoI editing | No | Yes |
| **GaussCtrl** | ECCV 2024 | Depth-conditioned | No | No (appearance edit) | N/A | No | Yes |
| **DGE** | ECCV 2024 | InstructPix2Pix | No | No (appearance edit) | N/A | No | Yes |
| **RoboSplat** | RSS 2025 | Grounded-SAM + URDF | Replacement only | Yes (rigid SE3) | **Not addressed** | **Yes** | Yes |
| **RoboGSim** | arXiv 2024 | Not detailed | No | Coordinate transform | No | **Yes** | No |
| **GSWorld** | arXiv 2025 | Not detailed | No | GSDF format | No | **Yes** | Yes |
| **SplatSim** | arXiv 2024 | Rigid body align | N/A (simulator) | Via simulator | N/A | **Yes** | Partial |
| **RoboSimGS** | arXiv 2025 | MLLM-based | No | Hybrid GS+mesh | No | **Yes** | No |
| **GigaWorld-0** | arXiv 2025 | Trellis FG/BG | N/A | 3D generation | N/A | **Yes** | Yes |
| **GWM** | ICCV 2025 | Gaussian world model | N/A | DiT prediction | N/A | **Yes** | Yes |
| **ManiGaussian** | ECCV 2024 | Dynamic GS | N/A | Dynamic prediction | N/A | **Yes** | Yes |
| **Object-Aware GS** | 2025 | Foundation models | N/A | Real-time update | N/A | **Yes** | No |
| **GaussianGrasper** | RA-L 2024 | CLIP + EFD | No | No | No | **Yes** | Yes |
| **GraspSplats** | CoRL 2024 | Feature splatting | No | No | No | **Yes** | Yes |
| **Robo-GS** | ICRA 2025 | Mesh-Gaussian bind | No | Physics-based | No | **Yes** | No |
| **Embodied Gaussians** | CoRL 2024 | Particle-Gaussian | No | Physics sim | No | **Yes** | Yes |
| **GaussianProperty** | ICCV 2025 | SAM + GPT-4V | No | No | No | **Yes** | Yes |
| **POGS** | ICRA 2025 | Self-supervised | No | Pose tracking | No | **Yes** | Yes |

---

## 9. Key Takeaways for Our Project

### 9.1 The Core Pipeline Should Follow Gaussian Grouping + RoboSplat

The most directly relevant combination for our project is:
1. **Gaussian Grouping** (ECCV 2024) for per-Gaussian semantic segmentation via SAM identity encodings
2. **RoboSplat** (RSS 2025) for the data augmentation pipeline (rigid transforms, novel views, embodiment swaps)

### 9.2 The Inpainting Problem is the Key Open Challenge

**Critical finding:** RoboSplat (the closest prior work) explicitly avoids the inpainting problem by using replacement strategies rather than true hole-filling. This is a significant limitation because:
- When an object is moved to a new position, the surface beneath it was occluded and has no Gaussian coverage
- Simply deleting object Gaussians leaves a visible hole
- Replacement (swapping backgrounds) is not always feasible for realistic rearrangements

**Best inpainting approaches for our pipeline (ranked):**
1. **Gaussian Grouping's approach:** Delete + clone + LaMa supervision + fine-tune (fast, practical, minutes)
2. **GScream's approach:** Depth-guided (Marigold) + cross-attention feature propagation (better geometry)
3. **InFusion's approach:** Depth completion from diffusion prior (principled, handles novel geometry)
4. **3DGS-CD's approach:** Duplicate Gaussians near un-occluded regions, optimize only duplicates (most relevant for object relocation specifically)

### 9.3 Our Novel Contribution Space

Based on this review, the following areas represent potential novel contributions for CoRL 2026:

1. **Joint object relocation + inpainting for manipulation data generation:** No existing work cleanly handles "move object A from position X to position Y, fill the hole at X, and render the result." RoboSplat avoids this; Gaussian Grouping demonstrates it but not in a robotics context.

2. **Humanoid/dual-arm manipulation data synthesis:** All existing works (RoboSplat, SplatSim, GSWorld) focus on single-arm tabletop. Extending to humanoid whole-body manipulation is novel.

3. **Physics-aware Gaussian manipulation:** Combining Gaussian editing with physical plausibility (GaussianProperty, Embodied Gaussians) for generating physically realistic manipulation data.

4. **Scalable scene rearrangement:** Moving multiple objects simultaneously with proper occlusion handling and mutual inpainting -- not addressed by any existing work.

### 9.4 Recommended Technical Approach

Based on the landscape, we recommend:

| Step | Method | Source |
|------|--------|--------|
| Scene reconstruction | 3DGS with Depth Anything priors | RoboSplat pipeline |
| Semantic segmentation | SAM + 16D Identity Encodings | Gaussian Grouping |
| Object selection | Grounded-SAM or language query | RoboSplat / LangSplat |
| Object relocation | Rigid SE(3) transform on Gaussian group | Gaussian Grouping |
| Hole filling | Clone + depth-guided initialization + LaMa/diffusion supervision | GScream + InFusion |
| Robot insertion | URDF-guided Gaussian placement | RoboSplat / RoboGSim |
| Novel view rendering | Standard 3DGS rasterization | Baseline |
| Policy training | Behavioral cloning on augmented demonstrations | RoboSplat |

### 9.5 Key Competitors for CoRL 2026

These papers are most likely to be concurrent or published before our submission:
- **RoboSplat** (RSS 2025) -- established baseline
- **GSWorld** (arXiv Oct 2025) -- closed-loop evaluation
- **GigaWorld-0** (arXiv Nov 2025) -- large-scale data engine
- **GWM** (ICCV 2025) -- Gaussian world models
- Any follow-up works from the RoboSplat/GSWorld teams

---

## References (by first appearance)

1. Ye et al., "Gaussian Grouping: Segment and Edit Anything in 3D Scenes," ECCV 2024. [arXiv:2312.00732](https://arxiv.org/abs/2312.00732)
2. Cen et al., "Segment Any 3D Gaussians," AAAI 2025. [arXiv:2312.00860](https://arxiv.org/abs/2312.00860)
3. Jain et al., "GaussianCut: Interactive Segmentation via Graph Cut for 3DGS," NeurIPS 2024. [arXiv:2411.07555](https://arxiv.org/abs/2411.07555)
4. Qin et al., "LangSplat: 3D Language Gaussian Splatting," CVPR 2024. [arXiv:2312.16084](https://arxiv.org/abs/2312.16084)
5. Li et al., "LangSplatV2: High-dimensional 3D Language Gaussian Splatting," NeurIPS 2025. [arXiv:2507.07136](https://arxiv.org/abs/2507.07136)
6. Zhou et al., "Feature 3DGS: Supercharging 3DGS to Enable Distilled Feature Fields," CVPR 2024. [arXiv:2312.03203](https://arxiv.org/abs/2312.03203)
7. Yu et al., "LEGS: Language-Embedded Gaussian Splats," IROS 2024. [arXiv:2409.18108](https://arxiv.org/abs/2409.18108)
8. Kerr et al., "LERF: Language Embedded Radiance Fields," ICCV 2023. [arXiv:2303.09553](https://arxiv.org/abs/2303.09553)
9. Wang et al., "GScream: Learning 3D Geometry and Feature Consistent GS for Object Removal," ECCV 2024.
10. Kwon et al., "Inpaint360GS," arXiv 2025. [arXiv:2511.06457](https://arxiv.org/abs/2511.06457)
11. "Clean-Splat," Preprint Dec 2025.
12. Dong & Wang, "3DGS-Drag: Dragging Gaussians for Point-Based 3D Editing," ICLR 2025.
13. Lu et al., "3DGS-CD: 3DGS-based Change Detection for Physical Object Rearrangement," RA-L 2025. [arXiv:2411.03706](https://arxiv.org/abs/2411.03706)
14. Zhang et al., "3DitScene: Editing Any Scene via Language-guided Disentangled GS," ICLR 2025. [arXiv:2405.18424](https://arxiv.org/abs/2405.18424)
15. Mirzaei et al., "SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with NeRF," CVPR 2023. [arXiv:2211.12254](https://arxiv.org/abs/2211.12254)
16. Liu et al., "InFusion: Inpainting 3D Gaussians via Learning Depth Completion from Diffusion Prior," arXiv 2024. [arXiv:2404.11613](https://arxiv.org/abs/2404.11613)
17. Seo et al., "RePaintGS: Reference-Guided GS for 3D Scene Inpainting," arXiv 2025. [arXiv:2507.08434](https://arxiv.org/abs/2507.08434)
18. Cui et al., "VISTA: Visibility-Uncertainty-guided 3D Gaussian Inpainting," arXiv 2025. [arXiv:2504.17815](https://arxiv.org/abs/2504.17815)
19. Chen et al., "GaussianEditor: Swift and Controllable 3D Editing with GS," CVPR 2024. [arXiv:2311.14521](https://arxiv.org/abs/2311.14521)
20. Wang et al., "GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions," CVPR 2024. [arXiv:2311.16037](https://arxiv.org/abs/2311.16037)
21. Wu et al., "GaussCtrl: Multi-View Consistent Text-Driven 3DGS Editing," ECCV 2024. [arXiv:2403.08733](https://arxiv.org/abs/2403.08733)
22. Chen et al., "DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing," ECCV 2024. [arXiv:2404.18929](https://arxiv.org/abs/2404.18929)
23. Yang et al., "RoboSplat: Novel Demonstration Generation with GS," RSS 2025. [arXiv:2504.13175](https://arxiv.org/abs/2504.13175)
24. Li et al., "RoboGSim: A Real2Sim2Real Robotic GS Simulator," arXiv 2024. [arXiv:2411.11839](https://arxiv.org/abs/2411.11839)
25. Jiang et al., "GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation," arXiv 2025. [arXiv:2510.20813](https://arxiv.org/abs/2510.20813)
26. "SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using GS," arXiv 2024. [arXiv:2409.10161](https://arxiv.org/abs/2409.10161)
27. Zhao et al., "RoboSimGS: High-Fidelity Simulated Data Generation for Zero-Shot Robotic Manipulation," arXiv 2025. [arXiv:2510.10637](https://arxiv.org/abs/2510.10637)
28. "GigaWorld-0: World Models as Data Engine to Empower Embodied AI," arXiv 2025. [arXiv:2511.19861](https://arxiv.org/abs/2511.19861)
29. Lu et al., "GWM: Towards Scalable Gaussian World Models for Robotic Manipulation," ICCV 2025. [arXiv:2508.17600](https://arxiv.org/abs/2508.17600)
30. Lu et al., "ManiGaussian: Dynamic GS for Multi-task Robotic Manipulation," ECCV 2024. [arXiv:2403.08321](https://arxiv.org/abs/2403.08321)
31. "Object-Aware Gaussian Splatting for Robotic Manipulation," 2025.
32. Zheng et al., "GaussianGrasper: 3D Language GS for Open-Vocabulary Robotic Grasping," RA-L 2024. [arXiv:2403.09637](https://arxiv.org/abs/2403.09637)
33. Ji et al., "GraspSplats: Efficient Manipulation with 3D Feature Splatting," CoRL 2024. [arXiv:2409.02084](https://arxiv.org/abs/2409.02084)
34. "Robo-GS: A Physics Consistent Spatial-Temporal Model for Robotic Arm," ICRA 2025. [arXiv:2408.14873](https://arxiv.org/abs/2408.14873)
35. Abou-Chakra et al., "Physically Embodied Gaussian Splatting," CoRL 2024. [arXiv:2406.10788](https://arxiv.org/abs/2406.10788)
36. Xu et al., "GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs," ICCV 2025. [arXiv:2412.11258](https://arxiv.org/abs/2412.11258)
37. Yu et al., "POGS: Persistent Object Gaussian Splat for Tracking," ICRA 2025. [arXiv:2503.05189](https://arxiv.org/abs/2503.05189)
38. Chhablani et al., "EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation," ICCV 2025. [arXiv:2509.17430](https://arxiv.org/abs/2509.17430)
39. He et al., "A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation," arXiv 2025. [arXiv:2508.09977](https://arxiv.org/abs/2508.09977)
40. "SplatFill: 3D Scene Inpainting via Depth-Guided Gaussian Splatting," arXiv 2025. [arXiv:2509.07809](https://arxiv.org/abs/2509.07809)

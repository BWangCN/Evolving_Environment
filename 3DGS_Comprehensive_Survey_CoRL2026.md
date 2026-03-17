# Comprehensive Survey: 3D Gaussian Splatting for Robotics and Synthetic Data Generation

**Target Venue:** CoRL 2026
**Date Compiled:** March 16, 2026
**Scope:** 3DGS for manipulation data synthesis, scene editing, inpainting, grasp/perception, foundational 3DGS works, and non-3DGS synthetic data methods.

---

## Category 1: 3DGS for Robot Manipulation Data Synthesis

### 1.1 RoboSplat
- **Title:** Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation
- **Authors:** Yang, S. et al.
- **Venue:** RSS 2025
- **Year:** 2025
- **arXiv:** 2504.13175
- **Summary:** RoboSplat generates diverse, visually realistic demonstrations by directly manipulating 3D Gaussians from a single real-world demonstration. It augments data across six types of generalization using five techniques: 3D Gaussian replacement (object types, scene appearance, robot embodiments), equivariant transformations (object poses), visual attribute editing (lighting), novel view synthesis (camera perspectives), and 3D content generation (diverse objects). Achieves 87.8% one-shot success rate vs. 57.2% for policies trained on hundreds of real demonstrations with 2D augmentation.

```bibtex
@inproceedings{yang2025robosplat,
  title={Novel Demonstration Generation with Gaussian Splatting Enables Robust One-Shot Manipulation},
  author={Yang, Sizhe and Yu, Wenye and Zeng, Jia and Lv, Jun and Ren, Kerui and Lu, Cewu and Lin, Dahua and Pang, Jiangmiao},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```

### 1.2 RoboGSim
- **Title:** RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator
- **Authors:** Li, X. et al.
- **Venue:** arXiv preprint
- **Year:** 2024
- **arXiv:** 2411.11839
- **Summary:** RoboGSim is a real-to-sim-to-real robotic simulator powered by 3DGS and physics engines. It includes four components: Gaussian Reconstructor, Digital Twins Builder, Scene Composer, and Interactive Engine. The system synthesizes data with novel views, objects, trajectories, and scenes. RoboGSim-generated data achieves zero-shot real-robot performance comparable to real robot data, and in novel scenes/perspectives, outperforms real-data-trained models.

```bibtex
@article{li2024robogsim,
  title={RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator},
  author={Li, Xinhai and Li, Jialin and Zhang, Ziheng and Zhang, Rui and Jia, Fan and Wang, Tiancai and Fan, Haoqiang and Tseng, Kuo-Kun and Wang, Ruiping},
  journal={arXiv preprint arXiv:2411.11839},
  year={2024}
}
```

### 1.3 GSWorld
- **Title:** GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation
- **Authors:** Jiang, G. et al.
- **Venue:** arXiv preprint
- **Year:** 2025
- **arXiv:** 2510.20813
- **Summary:** GSWorld is a robust, photo-realistic closed-loop simulator for robotic manipulation combining 3DGS with physics engines. It proposes a new asset format called GSDF (Gaussian Scene Description File) that infuses Gaussian-on-Mesh representation with robot URDF, with a database containing 3 robot embodiments and 40+ objects. Enables effective sim-to-real transfer for imitation learning, reinforcement learning, and DAgger-style data collection.

```bibtex
@article{jiang2025gsworld,
  title={GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation},
  author={Jiang, Guangqi and Chang, Haoran and Qiu, Ri-Zhao and Liang, Yutong and Ji, Mazeyu and Zhu, Jiyue and Dong, Zhao and Zou, Xueyan and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2510.20813},
  year={2025}
}
```

### 1.4 SplatSim
- **Title:** SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting
- **Authors:** Qureshi, M.N. et al.
- **Venue:** ICRA 2025
- **Year:** 2024
- **arXiv:** 2409.10161
- **Summary:** SplatSim replaces traditional mesh representations with Gaussian Splats in physics simulators (PyBullet) to produce photorealistic synthetic RGB data for training manipulation policies. Expert demonstrations collected in simulation are rendered through simulator-aligned splat models, transforming 3D Gaussians to extract photorealistic images at novel joint and object poses. Achieves 86.25% real-world success rate in zero-shot transfer using only RGB.

```bibtex
@inproceedings{qureshi2025splatsim,
  title={SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting},
  author={Qureshi, Mohammad Nomaan and Garg, Sparsh and Yandun, Francisco and Held, David and Kantor, George and Silwal, Abhisesh},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}
```

### 1.5 GigaWorld-0
- **Title:** GigaWorld-0: World Models as Data Engine to Empower Embodied AI
- **Authors:** GigaWorld Team et al.
- **Venue:** arXiv preprint
- **Year:** 2025
- **arXiv:** 2511.19861
- **Summary:** GigaWorld-0 is a unified world model framework designed as a data engine for Vision-Language-Action (VLA) learning. It integrates GigaWorld-0-Video (large-scale video generation using DiT with sparse attention and MoE) and GigaWorld-0-3D (3DGS reconstruction, physically differentiable system identification, and executable motion planning). The companion VLA model GigaBrain-0, trained exclusively on GigaWorld-0 data, achieves strong performance on physical robots across complex manipulation tasks.

```bibtex
@article{gigaworld2025,
  title={GigaWorld-0: World Models as Data Engine to Empower Embodied AI},
  author={{GigaWorld Team} and Ye, Angen and Wang, Boyuan and Ni, Chaojun and Huang, Guan and Zhao, Guosheng and Li, Haoyun and Zhu, Jiagang and Li, Kerui and Xu, Mengyuan and others},
  journal={arXiv preprint arXiv:2511.19861},
  year={2025}
}
```

### 1.6 GWM (Gaussian World Models)
- **Title:** GWM: Towards Scalable Gaussian World Models for Robotic Manipulation
- **Authors:** Lu, G. et al.
- **Venue:** ICCV 2025
- **Year:** 2025
- **arXiv:** 2508.17600
- **Summary:** GWM proposes a Gaussian World Model that reconstructs future scene states by inferring the propagation of Gaussian primitives under robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D-aware variational autoencoder for precise scene-level future state reconstruction. GWM can enhance visual representation for imitation learning via self-supervised future prediction training and serve as a neural simulator supporting model-based reinforcement learning. Shows strong data scaling potential.

```bibtex
@inproceedings{lu2025gwm,
  title={GWM: Towards Scalable Gaussian World Models for Robotic Manipulation},
  author={Lu, Guanxing and Jia, Baoxiong and Li, Puhao and Chen, Yixin and Wang, Ziwei and Tang, Yansong and Huang, Siyuan},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

### 1.7 ManiGaussian
- **Title:** ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation
- **Authors:** Lu, G. et al.
- **Venue:** ECCV 2024
- **Year:** 2024
- **arXiv:** 2403.08321
- **Summary:** ManiGaussian is a dynamic Gaussian Splatting method for multi-task robotic manipulation that mines scene dynamics via future scene reconstruction. It formulates a dynamic 3DGS framework that infers semantics propagation in the Gaussian embedding space, using a Gaussian world model to parameterize the distribution and provide supervision via future scene reconstruction. Outperforms state-of-the-art by 13.1% in average success rate across 10 RLBench tasks with 166 variations.

```bibtex
@inproceedings{lu2024manigaussian,
  title={ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation},
  author={Lu, Guanxing and Zhang, Shiyi and Wang, Ziwei and Liu, Changliu and Lu, Jiwen and Tang, Yansong},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### 1.8 RL-GSBridge
- **Title:** RL-GSBridge: 3D Gaussian Splatting Based Real2Sim2Real Method for Robotic Manipulation Learning
- **Authors:** Wu, Y. et al.
- **Venue:** arXiv preprint
- **Year:** 2024
- **arXiv:** 2409.20291
- **Summary:** RL-GSBridge incorporates 3DGS into the conventional RL simulation pipeline for zero-shot sim-to-real transfer of vision-based deep RL policies. It introduces mesh-based 3DGS with soft binding constraints and physics-aware GS editing that synchronizes rendering with the physics simulator. Achieves 96.88% and 100% real-world success rates on grasping tasks vs. significant drops (87% and 73%) for the baseline RL-sim method.

```bibtex
@article{wu2024rlgsbridge,
  title={RL-GSBridge: 3D Gaussian Splatting Based Real2Sim2Real Method for Robotic Manipulation Learning},
  author={Wu, Yuxuan and Pan, Lei and Wu, Wenhua and Wang, Guangming and Miao, Yanzi and Wang, Hesheng},
  journal={arXiv preprint arXiv:2409.20291},
  year={2024}
}
```

### 1.9 Physically Embodied Gaussian Splatting
- **Title:** Physically Embodied Gaussian Splatting: A Realtime Correctable World Model for Robotics
- **Authors:** Abou-Chakra, J. et al.
- **Venue:** CoRL 2024
- **Year:** 2024
- **arXiv:** 2406.10788
- **Summary:** Proposes a dual Gaussian-Particle representation where particles capture object geometry for physics simulation (anticipating physically plausible future states) and attached 3D Gaussians render images from any viewpoint via splatting. By comparing predicted and observed images, visual forces correct particle positions while respecting physical constraints. Runs in realtime at 30Hz with only 3 cameras.

```bibtex
@inproceedings{abouchakra2024physically,
  title={Physically Embodied Gaussian Splatting: A Realtime Correctable World Model for Robotics},
  author={Abou-Chakra, Jad and Rana, Krishan and Dayoub, Feras and S{\"u}nderhauf, Niko},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2024}
}
```

### 1.10 Splatting Physical Scenes (SplatMesh)
- **Title:** Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data
- **Authors:** Moran, B. et al.
- **Venue:** arXiv preprint
- **Year:** 2025
- **arXiv:** 2506.04120
- **Summary:** Introduces SplatMesh, a hybrid scene representation merging photorealistic 3DGS rendering with explicit physics-ready triangle meshes. An end-to-end optimization pipeline leverages differentiable rendering and differentiable physics within MuJoCo to jointly refine all scene components -- from object geometry and appearance to robot poses and physical parameters -- directly from raw, imprecise robot trajectories.

```bibtex
@article{moran2025splatting,
  title={Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data},
  author={Moran, Ben and Comi, Mauro and Byravan, Arunkumar and Bohez, Steven and Erez, Tom and Li, Zhibin and Hasenclever, Leonard},
  journal={arXiv preprint arXiv:2506.04120},
  year={2025}
}
```

### 1.11 RoboPaint
- **Title:** RoboPaint: From Human Demonstration to Any Robot and Any View
- **Authors:** Fan, J. et al.
- **Venue:** arXiv preprint
- **Year:** 2026
- **arXiv:** 2602.05325
- **Summary:** RoboPaint is a Real-Sim-Real data pipeline that transforms human demonstrations into robot-executable training data without direct teleoperation. It uses a Data-Acquisition Room with instrumented gloves, introduces Dex-Tactile joint retargeting for embodiment transfer (3.86 mm average contact error), and leverages 3DGS to reconstruct photorealistic digital twins for "painting" demonstrations onto target robots. VLA models trained on painted data achieve 80% manipulation success rate.

```bibtex
@article{fan2026robopaint,
  title={RoboPaint: From Human Demonstration to Any Robot and Any View},
  author={Fan, Jiacheng and Zhao, Zhiyue and Zhang, Yiqian and Chen, Chao and Wang, Peide and Zhang, Hengdi and Cheng, Zhengxue},
  journal={arXiv preprint arXiv:2602.05325},
  year={2026}
}
```

---

## Category 2: 3DGS Scene Editing (Object Manipulation)

### 2.1 Gaussian Grouping
- **Title:** Gaussian Grouping: Segment and Edit Anything in 3D Scenes
- **Authors:** Ye, M. et al.
- **Venue:** ECCV 2024
- **Year:** 2024
- **arXiv:** 2312.00732
- **Summary:** Extends 3DGS to jointly reconstruct and segment anything in open-world 3D scenes by augmenting each Gaussian with a compact Identity Encoding for grouping by object instance or stuff membership. Supervised via differentiable rendering using SAM 2D mask predictions with 3D spatial consistency regularization. Enables versatile scene editing: 3D object removal, inpainting, colorization, style transfer, and scene recomposition via a local Gaussian Editing scheme.

```bibtex
@inproceedings{ye2024gaussiangrouping,
  title={Gaussian Grouping: Segment and Edit Anything in 3D Scenes},
  author={Ye, Mingqiao and Danelljan, Martin and Yu, Fisher and Ke, Lei},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### 2.2 GaussianEditor (Chen et al.)
- **Title:** GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting
- **Authors:** Chen, Y. et al.
- **Venue:** CVPR 2024
- **Year:** 2024
- **arXiv:** 2311.14521
- **Summary:** The first 3D editing algorithm based on 3DGS. Proposes Gaussian semantic tracing to track editing targets throughout training and Hierarchical Gaussian Splatting (HGS) for stabilized results under stochastic generative guidance from 2D diffusion models. Enables swift and controllable 3D editing (2--7 minutes) including object removal and integration, substantially faster than NeRF-based alternatives like Instruct-NeRF2NeRF.

```bibtex
@inproceedings{chen2024gaussianeditor,
  title={GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting},
  author={Chen, Yiwen and Chen, Zilong and Zhang, Chi and Wang, Feng and Yang, Xiaofeng and Wang, Yikai and Cai, Zhongang and Yang, Lei and Liu, Huaping and Lin, Guosheng},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### 2.3 GaussianEditor (Wang et al.)
- **Title:** GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions
- **Authors:** Wang, J. et al.
- **Venue:** CVPR 2024
- **Year:** 2024
- **arXiv:** 2311.16037
- **Summary:** Proposes a systematic framework for editing 3D scenes via Gaussians with text instructions. Extracts the region of interest (RoI) corresponding to text instructions, aligns it to 3D Gaussians, and uses the Gaussian RoI to control the editing process. Achieves editing within 20 minutes on a single V100 GPU, more than twice as fast as Instruct-NeRF2NeRF.

```bibtex
@inproceedings{wang2024gaussianeditor,
  title={GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions},
  author={Wang, Junjie and Fang, Jiemin and Zhang, Xiaopeng and Xie, Lingxi and Tian, Qi},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### 2.4 3DGS-Drag
- **Title:** 3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing
- **Authors:** Dong, J. et al.
- **Venue:** ICLR 2025
- **Year:** 2025
- **arXiv:** 2601.07963
- **Summary:** A point-based 3D editing framework providing efficient, intuitive drag manipulation of real 3D scenes. Leverages two key innovations: deformation guidance utilizing 3DGS for consistent geometric modifications, and diffusion guidance for content correction and visual quality enhancement. Enables motion change, shape adjustment, inpainting, and content extension in 10--20 minutes on a single RTX 4090 GPU.

```bibtex
@inproceedings{dong2025threedgsdrag,
  title={3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing},
  author={Dong, Jiahua and Wang, Yu-Xiong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

### 2.5 3DitScene
- **Title:** 3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting
- **Authors:** Zhang, Q. et al.
- **Venue:** ICLR 2025
- **Year:** 2025
- **arXiv:** 2405.18424
- **Summary:** A unified scene editing framework leveraging language-guided disentangled Gaussian Splatting. Incorporates 3D Gaussians refined through generative priors and optimization, then uses CLIP language features to introduce semantics for object disentanglement. Users can query objects via language prompts and manipulate them in a flexible manner, enabling both global scene editing and individual object manipulation from 2D to 3D.

```bibtex
@inproceedings{zhang2025threeditsscene,
  title={3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting},
  author={Zhang, Qihang and Xu, Yinghao and Wang, Chaoyang and Lee, Hsin-Ying and Wetzstein, Gordon and Zhou, Bolei and Yang, Ceyuan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## Category 3: 3DGS Inpainting (Filling Holes After Object Removal)

### 3.1 GScream
- **Title:** GScream: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal
- **Authors:** Wang, Y. et al.
- **Venue:** ECCV 2024
- **Year:** 2024
- **arXiv:** 2404.13679
- **Summary:** Tackles object removal in 3DGS scenes while preserving geometric consistency and texture coherence despite the discrete nature of Gaussian primitives. Optimizes positioning of Gaussian primitives guided by monocular depth estimation via online registration, and employs a novel cross-attention-based feature propagation mechanism to bolster texture coherence. Uses SD-inpainting, LaMa, and Marigold depth estimation.

```bibtex
@inproceedings{wang2024gscream,
  title={GScream: Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal},
  author={Wang, Yuxin and Wu, Qianyi and Zhang, Guofeng and Xu, Dan},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### 3.2 InFusion
- **Title:** InFusion: Inpainting 3D Gaussians via Learning Depth Completion from Diffusion Prior
- **Authors:** Liu, Z. et al.
- **Venue:** arXiv preprint
- **Year:** 2024
- **arXiv:** 2404.11613
- **Summary:** Addresses the inpainting task of supplementing an incomplete set of 3D Gaussians with additional points for visually harmonious rendering. Guides point initialization with an image-conditioned depth completion model that learns to restore depth maps from diffusion priors. First inpaints depth in the reference view, then unprojects points into 3D for optimal initialization. Achieves 20x faster inpainting than baselines while maintaining superior visual quality.

```bibtex
@article{liu2024infusion,
  title={InFusion: Inpainting 3D Gaussians via Learning Depth Completion from Diffusion Prior},
  author={Liu, Zhiheng and Ouyang, Hao and Wang, Qiuyu and Cheng, Ka Leong and Xiao, Jie and Zhu, Kai and Xue, Nan and Liu, Yu and Shen, Yujun and Cao, Yang},
  journal={arXiv preprint arXiv:2404.11613},
  year={2024}
}
```

### 3.3 SPIn-NeRF
- **Title:** SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields
- **Authors:** Mirzaei, A. et al.
- **Venue:** CVPR 2023
- **Year:** 2023
- **arXiv:** 2211.12254
- **Summary:** Addresses 3D inpainting -- removing unwanted objects from a NeRF scene such that the replaced region is visually plausible and view-consistent. Given posed images and sparse annotations, first obtains a 3D segmentation mask, then uses a perceptual optimization approach that leverages learned 2D image inpainters while ensuring multi-view consistency. Introduces a benchmark dataset of real-world scenes with and without target objects.

```bibtex
@inproceedings{mirzaei2023spinnerf,
  title={SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields},
  author={Mirzaei, Ashkan and Aumentado-Armstrong, Tristan and Derpanis, Konstantinos G and Kelly, Jonathan and Brubaker, Marcus A and Gilitschenski, Igor and Levinshtein, Alex},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={20669--20679},
  year={2023}
}
```

### 3.4 3DGS-CD
- **Title:** 3DGS-CD: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement
- **Authors:** Lu, Z. et al.
- **Venue:** IEEE RA-L 2025
- **Year:** 2025
- **arXiv:** 2411.03706
- **Summary:** The first 3DGS-based method for detecting physical object rearrangements in 3D scenes. Estimates 3D object-level changes by comparing two sets of unaligned images taken at different times, leveraging 3DGS novel view rendering and EfficientSAM zero-shot segmentation. Detects changes in cluttered environments using as few as one post-change image within 18 seconds, achieving up to 14% higher accuracy and 1000x faster than prior radiance-field-based methods.

```bibtex
@article{lu2025threedgscd,
  title={3DGS-CD: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement},
  author={Lu, Ziqi and Ye, Jianbo and Leonard, John},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}
```

---

## Category 4: 3DGS + Grasp/Perception

### 4.1 GaussianGrasper
- **Title:** GaussianGrasper: 3D Language Gaussian Splatting for Open-vocabulary Robotic Grasping
- **Authors:** Zheng, Y. et al.
- **Venue:** IEEE RA-L 2024
- **Year:** 2024
- **arXiv:** 2403.09637
- **Summary:** Represents scenes as Gaussian primitives from limited RGB-D views, using tile-based splatting to create a feature field. Proposes an Efficient Feature Distillation (EFD) module using contrastive learning to distill CLIP language embeddings, and a normal-guided grasp module to filter unfeasible candidates and select the best grasp pose. Enables language-instructed open-vocabulary robotic grasping on a real robot.

```bibtex
@article{zheng2024gaussiangrasper,
  title={GaussianGrasper: 3D Language Gaussian Splatting for Open-vocabulary Robotic Grasping},
  author={Zheng, Yuhang and Chen, Xiangyu and Zheng, Yupeng and Gu, Songen and Yang, Runyi and Jin, Bu and Li, Pengfei and Zhong, Chengliang and Wang, Zengmao and Liu, Lina and Yang, Chao and Wang, Dawei and Chen, Zhen and Long, Xiaoxiao and Wang, Meiqing},
  journal={IEEE Robotics and Automation Letters},
  volume={9},
  pages={7827--7834},
  year={2024}
}
```

### 4.2 GraspSplats
- **Title:** GraspSplats: Efficient Manipulation with 3D Feature Splatting
- **Authors:** Ji, M. et al.
- **Venue:** CoRL 2024
- **Year:** 2024
- **arXiv:** 2409.02084
- **Summary:** Constructs high-fidelity scene representations as explicit Gaussian ellipsoids via 3DGS from posed RGBD frames in under 60 seconds. The explicit optimized geometry natively supports real-time grasp sampling and dynamic/articulated object manipulation with point trackers. Significantly outperforms NeRF-based methods (F3RM, LERF-TOGO) and 2D detection methods across diverse manipulation task settings on a Franka robot.

```bibtex
@inproceedings{ji2024graspsplats,
  title={GraspSplats: Efficient Manipulation with 3D Feature Splatting},
  author={Ji, Mazeyu and Qiu, Ri-Zhao and Zou, Xueyan and Wang, Xiaolong},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2024}
}
```

### 4.3 LERF
- **Title:** LERF: Language Embedded Radiance Fields
- **Authors:** Kerr, J. et al.
- **Venue:** ICCV 2023
- **Year:** 2023
- **arXiv:** 2303.09553
- **Summary:** Grounds CLIP language embeddings into NeRF by volume rendering CLIP embeddings along training rays, learning a dense multi-scale language field. After optimization, extracts 3D relevancy maps for a broad range of language prompts in real-time without relying on region proposals, masks, or fine-tuning. Supports long-tail open-vocabulary queries hierarchically across the 3D volume, with applications in robotics and vision-language understanding.

```bibtex
@inproceedings{kerr2023lerf,
  title={LERF: Language Embedded Radiance Fields},
  author={Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

### 4.4 LangSplat
- **Title:** LangSplat: 3D Language Gaussian Splatting
- **Authors:** Qin, M. et al.
- **Venue:** CVPR 2024 (Highlight)
- **Year:** 2024
- **arXiv:** 2312.16084
- **Summary:** Constructs a 3D language field using a collection of 3D Gaussians each encoding CLIP-distilled language features, enabling precise and efficient open-vocabulary querying in 3D. Uses tile-based splatting for rendering language features, circumventing the costly rendering inherent in NeRF. Significantly outperforms LERF on open-vocabulary 3D object localization and semantic segmentation with a 119x speedup.

```bibtex
@inproceedings{qin2024langsplat,
  title={LangSplat: 3D Language Gaussian Splatting},
  author={Qin, Minghan and Li, Wanhua and Zhou, Jiawei and Wang, Haoqian and Pfister, Hanspeter},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={20051--20060},
  year={2024}
}
```

### 4.5 Feature 3DGS
- **Title:** Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields
- **Authors:** Zhou, S. et al.
- **Venue:** CVPR 2024 (Highlight)
- **Year:** 2024
- **arXiv:** 2312.03203
- **Summary:** The first feature field distillation technique based on 3DGS, enabling arbitrary-dimension semantic features via 2D foundation model distillation. Develops a Parallel N-dimensional Gaussian Rasterizer with an optional convolutional speed-up module. Works with CLIP-LSeg, SAM, and others, extending 3DGS beyond novel view synthesis to semantic segmentation, language-guided editing, and promptable segmentation.

```bibtex
@inproceedings{zhou2024feature3dgs,
  title={Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields},
  author={Zhou, Shijie and Chang, Haoran and Jiang, Sicheng and Fan, Zhiwen and Zhu, Zehao and Xu, Dejia and Chari, Pradyumna and You, Suya and Wang, Zhangyang and Kadambi, Achuta},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={21676--21685},
  year={2024}
}
```

### 4.6 POGS
- **Title:** Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects
- **Authors:** Yu, J. et al.
- **Venue:** ICRA 2025
- **Year:** 2025
- **arXiv:** 2503.05189
- **Summary:** Embeds semantics, self-supervised visual features, and object grouping features into a compact Gaussian representation that is continuously updated to estimate scanned object poses. Supports grasping, reorientation, and natural language-driven manipulation without requiring expensive rescanning or prior CAD models. Achieves up to 12 consecutive successful object resets and recovers from 80% of in-grasp tool perturbations of up to 30 degrees.

```bibtex
@inproceedings{yu2025pogs,
  title={Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects},
  author={Yu, Justin and Hari, Kush and El-Refai, Karim and Dalal, Arnav and Kerr, Justin and Kim, Chung Min and Cheng, Richard and Irshad, Muhammad Zubair and Goldberg, Ken},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}
```

---

## Category 5: Original 3DGS and Key Extensions

### 5.1 3D Gaussian Splatting (Original)
- **Title:** 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **Authors:** Kerbl, B. et al.
- **Venue:** ACM Transactions on Graphics (SIGGRAPH 2023)
- **Year:** 2023
- **arXiv:** 2308.04079
- **Summary:** The foundational work representing scenes with 3D Gaussians -- anisotropic volumetric primitives with position, covariance, opacity, and spherical harmonics color coefficients. Performs interleaved optimization and density control of 3D Gaussians with a fast visibility-aware rendering algorithm supporting anisotropic splatting. Achieves state-of-the-art visual quality with real-time (>=30 fps) novel-view synthesis at 1080p resolution.

```bibtex
@article{kerbl2023threedgs,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  volume={42},
  number={4},
  year={2023},
  publisher={ACM}
}
```

### 5.2 gsplat
- **Title:** gsplat: An Open-Source Library for Gaussian Splatting
- **Authors:** Ye, V. et al.
- **Venue:** JMLR (MLOSS)
- **Year:** 2025
- **arXiv:** 2409.06765
- **Summary:** An open-source library for training and developing 3DGS methods, featuring a Python/PyTorch front-end and highly optimized CUDA back-end kernels. Offers optimization improvements for speed, memory, and convergence times. Achieves up to 10% less training time and 4x less memory than the original 3DGS implementation. Widely adopted in research projects and actively maintained.

```bibtex
@article{ye2025gsplat,
  title={gsplat: An Open-Source Library for Gaussian Splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Kanazawa, Angjoo},
  journal={Journal of Machine Learning Research},
  volume={26},
  year={2025}
}
```

### 5.3 Mip-Splatting
- **Title:** Mip-Splatting: Alias-free 3D Gaussian Splatting
- **Authors:** Yu, Z. et al.
- **Venue:** CVPR 2024 (Best Student Paper)
- **Year:** 2024
- **arXiv:** 2311.16493
- **Summary:** Addresses aliasing artifacts in 3DGS when changing sampling rate (focal length or camera distance). Introduces a 3D smoothing filter constraining Gaussian primitive sizes based on maximal sampling frequency from input views (eliminating high-frequency artifacts when zooming in), and a 2D Mip filter replacing 2D dilation to mitigate aliasing. Won the CVPR 2024 Best Student Paper award.

```bibtex
@inproceedings{yu2024mipsplatting,
  title={Mip-Splatting: Alias-free 3D Gaussian Splatting},
  author={Yu, Zehao and Chen, Anpei and Huang, Binbin and Sattler, Torsten and Geiger, Andreas},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={19447--19456},
  year={2024}
}
```

### 5.4 2D Gaussian Splatting (2DGS)
- **Title:** 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- **Authors:** Huang, B. et al.
- **Venue:** ACM SIGGRAPH 2024
- **Year:** 2024
- **arXiv:** 2403.17888
- **Summary:** Collapses 3D Gaussian volumes into 2D oriented planar Gaussian disks for geometrically accurate surface reconstruction from multi-view images. Introduces perspective-accurate 2D splatting via ray-splat intersection and rasterization, providing view-consistent geometry while modeling surfaces intrinsically. Enables noise-free, detailed geometry reconstruction while maintaining competitive appearance quality, fast training, and real-time rendering.

```bibtex
@inproceedings{huang20242dgs,
  title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
  author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
  booktitle={ACM SIGGRAPH Conference Proceedings},
  year={2024}
}
```

---

## Category 6: Synthetic Data for Robot Learning (Non-3DGS)

### 6.1 Domain Randomization
- **Title:** Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
- **Authors:** Tobin, J. et al.
- **Venue:** IROS 2017
- **Year:** 2017
- **arXiv:** 1703.06907
- **Summary:** The foundational work on domain randomization for sim-to-real transfer. Trains models on simulated images with randomized rendering (textures, colors, camera positions, distractor objects, noise) so the real world appears as just another variation. Achieves the first successful transfer of a deep neural network trained only on simulated RGB images to real-world robotic control, with object localization accurate to 1.5 cm.

```bibtex
@inproceedings{tobin2017domain,
  title={Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World},
  author={Tobin, Josh and Fong, Rachel and Ray, Alex and Schneider, Jonas and Zaremba, Wojciech and Abbeel, Pieter},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2017}
}
```

### 6.2 ROSIE (Scaling Robot Learning with Semantically Imagined Experience)
- **Title:** Scaling Robot Learning with Semantically Imagined Experience
- **Authors:** Yu, T. et al.
- **Venue:** RSS 2023
- **Year:** 2023
- **arXiv:** 2302.11550
- **Summary:** Leverages text-to-image diffusion models for aggressive data augmentation of robotic manipulation datasets, inpainting various unseen objects, backgrounds, and distractors with text guidance. Policies trained on augmented data solve completely unseen tasks with new objects and behave more robustly to novel distractors. A key precursor to 3DGS-based data augmentation approaches.

```bibtex
@inproceedings{yu2023rosie,
  title={Scaling Robot Learning with Semantically Imagined Experience},
  author={Yu, Tianhe and Xiao, Ted and Stone, Austin and Tompson, Jonathan and Brohan, Anthony and Wang, Su and Singh, Jaspiar and Tan, Clayton and M, Dee and Peralta, Jodilyn and Ichter, Brian and Hausman, Karol and Xia, Fei},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```

### 6.3 GenAug
- **Title:** GenAug: Retargeting behaviors to unseen situations via Generative Augmentation
- **Authors:** Chen, Z. et al.
- **Venue:** RSS 2023
- **Year:** 2023
- **arXiv:** 2302.06671
- **Summary:** Uses pre-trained generative text-to-image models for "semantic" data augmentation to improve policy generalization. Automatically generates augmented RGBD images for different realistic environments with visual realism and complexity. Demonstrates 40% improvement in generalization to novel scenes and objects on real-world tabletop manipulation tasks with only marginal amounts of real-world data.

```bibtex
@inproceedings{chen2023genaug,
  title={GenAug: Retargeting behaviors to unseen situations via Generative Augmentation},
  author={Chen, Zoey and Kiami, Sho and Gupta, Abhishek and Kumar, Vikash},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}
```

### 6.4 NVIDIA Cosmos
- **Title:** Cosmos World Foundation Model Platform for Physical AI
- **Authors:** NVIDIA et al.
- **Venue:** arXiv preprint
- **Year:** 2025
- **arXiv:** 2501.03575
- **Summary:** A platform of state-of-the-art generative world foundation models (WFMs), advanced tokenizers, guardrails, and accelerated data pipelines. Trained on 9,000 trillion tokens from 20 million hours of real-world data. Includes both diffusion and autoregressive transformer models. WFMs can generate synthetic data for training and be fine-tuned conditioned on rendering metadata for sim-to-real applications. Open-weight with permissive licenses.

```bibtex
@article{nvidia2025cosmos,
  title={Cosmos World Foundation Model Platform for Physical AI},
  author={Agarwal, Niket and Ali, Arslan and Bala, Maciej and Balaji, Yogesh and Barker, Erik and Cai, Tiffany and Chattopadhyay, Prithvijit and Chen, Yongxin and Cui, Yin and Ding, Yifan and others},
  journal={arXiv preprint arXiv:2501.03575},
  year={2025}
}
```

### 6.5 NVIDIA GR00T N1
- **Title:** GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- **Authors:** NVIDIA et al.
- **Venue:** arXiv preprint
- **Year:** 2025
- **arXiv:** 2503.14734
- **Summary:** An open Vision-Language-Action (VLA) foundation model for humanoid robots with a dual-system architecture: a vision-language module (System 2) interprets environment via vision and language instructions, while a diffusion transformer module (System 1) generates fluid motor actions in real-time. Trained on a heterogeneous mixture of real-robot trajectories, human videos, and synthetically generated datasets. GR00T-N1-2B has 2.2B parameters total.

```bibtex
@article{nvidia2025groot,
  title={GR00T N1: An Open Foundation Model for Generalist Humanoid Robots},
  author={{NVIDIA}},
  journal={arXiv preprint arXiv:2503.14734},
  year={2025}
}
```

---

## Supplementary: 3DGS in Robotics Survey

### S.1 3D Gaussian Splatting in Robotics: A Survey
- **Title:** 3D Gaussian Splatting in Robotics: A Survey
- **Authors:** Zhu, S. et al.
- **Venue:** arXiv preprint
- **Year:** 2024
- **arXiv:** 2410.12262
- **Summary:** The first comprehensive survey specifically focused on 3DGS applications within robotics. Reviews and categorizes existing research on 3DGS in scene reconstruction, segmentation, editing, SLAM, manipulation, and navigation. Covers both the application of 3DGS and advancements in 3DGS techniques, providing a structured overview of a domain where real-time performance and robust scene understanding are critical.

```bibtex
@article{zhu2024threedgsroboticssurvey,
  title={3D Gaussian Splatting in Robotics: A Survey},
  author={Zhu, Siting and Wang, Guangming and Kong, Dezhi and Wang, Hesheng},
  journal={arXiv preprint arXiv:2410.12262},
  year={2024}
}
```

---

## Quick Reference Summary Table

| Paper | Category | Venue | Year | arXiv |
|-------|----------|-------|------|-------|
| 3D Gaussian Splatting (Kerbl) | Foundation | SIGGRAPH 2023 | 2023 | 2308.04079 |
| Mip-Splatting (Yu) | Foundation | CVPR 2024 | 2024 | 2311.16493 |
| 2DGS (Huang) | Foundation | SIGGRAPH 2024 | 2024 | 2403.17888 |
| gsplat (Ye) | Foundation | JMLR 2025 | 2025 | 2409.06765 |
| RoboSplat (Yang) | Manip. Data | RSS 2025 | 2025 | 2504.13175 |
| RoboGSim (Li) | Manip. Data | arXiv | 2024 | 2411.11839 |
| GSWorld (Jiang) | Manip. Data | arXiv | 2025 | 2510.20813 |
| SplatSim (Qureshi) | Manip. Data | ICRA 2025 | 2024 | 2409.10161 |
| GigaWorld-0 (Team) | Manip. Data | arXiv | 2025 | 2511.19861 |
| GWM (Lu) | Manip. Data | ICCV 2025 | 2025 | 2508.17600 |
| ManiGaussian (Lu) | Manip. Data | ECCV 2024 | 2024 | 2403.08321 |
| RL-GSBridge (Wu) | Manip. Data | arXiv | 2024 | 2409.20291 |
| Phys. Embodied GS (Abou-Chakra) | Manip. Data | CoRL 2024 | 2024 | 2406.10788 |
| SplatMesh (Moran) | Manip. Data | arXiv | 2025 | 2506.04120 |
| RoboPaint (Fan) | Manip. Data | arXiv | 2026 | 2602.05325 |
| Gaussian Grouping (Ye) | Editing | ECCV 2024 | 2024 | 2312.00732 |
| GaussianEditor (Chen) | Editing | CVPR 2024 | 2024 | 2311.14521 |
| GaussianEditor (Wang) | Editing | CVPR 2024 | 2024 | 2311.16037 |
| 3DGS-Drag (Dong) | Editing | ICLR 2025 | 2025 | 2601.07963 |
| 3DitScene (Zhang) | Editing | ICLR 2025 | 2025 | 2405.18424 |
| GScream (Wang) | Inpainting | ECCV 2024 | 2024 | 2404.13679 |
| InFusion (Liu) | Inpainting | arXiv | 2024 | 2404.11613 |
| SPIn-NeRF (Mirzaei) | Inpainting | CVPR 2023 | 2023 | 2211.12254 |
| 3DGS-CD (Lu) | Inpainting | RA-L 2025 | 2025 | 2411.03706 |
| GaussianGrasper (Zheng) | Grasp/Percep. | RA-L 2024 | 2024 | 2403.09637 |
| GraspSplats (Ji) | Grasp/Percep. | CoRL 2024 | 2024 | 2409.02084 |
| LERF (Kerr) | Grasp/Percep. | ICCV 2023 | 2023 | 2303.09553 |
| LangSplat (Qin) | Grasp/Percep. | CVPR 2024 | 2024 | 2312.16084 |
| Feature 3DGS (Zhou) | Grasp/Percep. | CVPR 2024 | 2024 | 2312.03203 |
| POGS (Yu) | Grasp/Percep. | ICRA 2025 | 2025 | 2503.05189 |
| Domain Randomization (Tobin) | Syn. Data | IROS 2017 | 2017 | 1703.06907 |
| ROSIE (Yu) | Syn. Data | RSS 2023 | 2023 | 2302.11550 |
| GenAug (Chen) | Syn. Data | RSS 2023 | 2023 | 2302.06671 |
| Cosmos (NVIDIA) | Syn. Data | arXiv | 2025 | 2501.03575 |
| GR00T N1 (NVIDIA) | Syn. Data | arXiv | 2025 | 2503.14734 |
| 3DGS Robotics Survey (Zhu) | Survey | arXiv | 2024 | 2410.12262 |

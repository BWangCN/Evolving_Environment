# Continual Learning Methods for Robotics and VLA Adaptation: A Comprehensive Survey

**Prepared for CoRL 2026 submission**

---

## 1. Classic CL Methods Applied to Robotics

### 1.1 EWC: Elastic Weight Consolidation

**Title:** Overcoming Catastrophic Forgetting in Neural Networks
**Authors:** Kirkpatrick et al.
**Venue:** PNAS, 2017
**arXiv:** 1612.00796

EWC selectively slows down learning on weights important for previously learned tasks by adding a quadratic penalty based on the Fisher Information Matrix. This regularization-based approach allows networks to learn new tasks while preserving performance on old ones. EWC has become the foundational regularization method against which all subsequent CL approaches are compared, and has been widely applied to RL and robotic settings.

```bibtex
@article{kirkpatrick2017overcoming,
  title={Overcoming catastrophic forgetting in neural networks},
  author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={114},
  number={13},
  pages={3521--3526},
  year={2017},
  publisher={National Acad Sciences},
  note={arXiv:1612.00796}
}
```

---

### 1.2 Progressive Neural Networks

**Title:** Progressive Neural Networks
**Authors:** Rusu et al.
**Venue:** arXiv preprint, 2016
**arXiv:** 1606.04671

Progressive Neural Networks are immune to catastrophic forgetting by design: they retain a pool of pretrained models and allocate new capacity (lateral connections) for each new task. This architecture enables forward transfer via lateral connections to previously learned features while completely preventing interference with old task parameters. Originally developed for RL settings including Atari and robotic manipulation.

```bibtex
@article{rusu2016progressive,
  title={Progressive neural networks},
  author={Rusu, Andrei A and Rabinowitz, Neil C and Desjardins, Guillaume and Soyer, Hubert and Kirkpatrick, James and Kavukcuoglu, Koray and Pascanu, Razvan and Hadsell, Raia},
  journal={arXiv preprint arXiv:1606.04671},
  year={2016}
}
```

---

### 1.3 PackNet

**Title:** PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning
**Authors:** Mallya et al.
**Venue:** CVPR, 2018
**arXiv:** 1711.05769

PackNet exploits redundancies in large deep networks by iteratively pruning and re-training to sequentially "pack" multiple tasks into a single network. Unlike prior work using proxy losses, PackNet always optimizes directly for the task at hand. It assigns fixed parameter subsets to each task and achieves near-independent-network performance with minimal storage overhead. In the Continual World robotics benchmark, PackNet outperformed other CL methods in forward transfer.

```bibtex
@inproceedings{mallya2018packnet,
  title={PackNet: Adding multiple tasks to a single network by iterative pruning},
  author={Mallya, Arun and Lazebnik, Svetlana},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={7765--7773},
  year={2018},
  note={arXiv:1711.05769}
}
```

---

### 1.4 Continual World Benchmark

**Title:** Continual World: A Robotic Benchmark For Continual Reinforcement Learning
**Authors:** Wolczyk et al.
**Venue:** NeurIPS, 2021
**arXiv:** 2105.10919

Continual World is a benchmark of 10 realistic robotic manipulation tasks built on Meta-World, designed to evaluate CL methods in RL. The paper provides an in-depth empirical evaluation of existing CL methods (EWC, PackNet, Perfect Memory, etc.) in robotic settings, revealing that no single method dominates across all metrics and that balancing forgetting prevention with forward transfer remains an open challenge in robotic CL.

```bibtex
@inproceedings{wolczyk2021continual,
  title={Continual World: A Robotic Benchmark for Continual Reinforcement Learning},
  author={Wo{\l}czyk, Maciej and Zaj{\k{a}}c, Micha{\l} and Pascanu, Razvan and Kuci{\'n}ski, {\L}ukasz and Mi{\l}o{\'s}, Piotr},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={34},
  year={2021},
  note={arXiv:2105.10919}
}
```

---

### 1.5 LEGION: Lifelong Robotic RL

**Title:** Preserving and Combining Knowledge in Robotic Lifelong Reinforcement Learning
**Authors:** Meng et al.
**Venue:** Nature Machine Intelligence, 2025

LEGION (Language Embedding based Generative Incremental Off-policy RL with Non-parametric Bayes) uses Dirichlet process mixture models to dynamically preserve and integrate knowledge from sequential robotic tasks. The framework integrates LLM-based language embeddings for instruction understanding and enables knowledge recombination for long-horizon tasks by intelligently sequencing previously learned skills like pushing, opening drawers, or pressing buttons.

```bibtex
@article{meng2025preserving,
  title={Preserving and combining knowledge in robotic lifelong reinforcement learning},
  author={Meng, Yuan and Bing, Zhenshan and Yao, Xiangtong and Chen, Kejia and Huang, Kai and Gao, Yang and Sun, Fuchun and Knoll, Alois},
  journal={Nature Machine Intelligence},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s42256-025-00983-2}
}
```

---

## 2. LoRA-Based Continual Learning

### 2.0 LoRA (Foundational Method)

**Title:** LoRA: Low-Rank Adaptation of Large Language Models
**Authors:** Hu et al.
**Venue:** ICLR, 2022
**arXiv:** 2106.09685

LoRA freezes pretrained model weights and injects trainable low-rank decomposition matrices into Transformer layers, reducing trainable parameters by 10,000x and GPU memory by 3x compared to full fine-tuning. LoRA achieves on-par or better performance than full fine-tuning with no additional inference latency. This foundational PEFT method underpins all subsequent LoRA-based CL approaches.

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  note={arXiv:2106.09685}
}
```

---

### 2.1 O-LoRA: Orthogonal Subspace LoRA

**Title:** Orthogonal Subspace Learning for Language Model Continual Learning
**Authors:** Wang et al.
**Venue:** Findings of EMNLP, 2023
**arXiv:** 2310.14152

O-LoRA learns each new task in a low-rank vector subspace that is kept orthogonal to all previously learned subspaces, directly minimizing inter-task interference. The method incrementally updates parameters for new tasks by constraining them to lie in the orthogonal subspace of parameters from previous tasks, while freezing LoRA parameters learned from previous tasks. O-LoRA requires no user data storage for replay and excels at preserving LLM generalization on unseen tasks.

```bibtex
@inproceedings{wang2023olora,
  title={Orthogonal Subspace Learning for Language Model Continual Learning},
  author={Wang, Xiao and others},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  year={2023},
  note={arXiv:2310.14152}
}
```

---

### 2.2 InfLoRA: Interference-Free Low-Rank Adaptation

**Title:** InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning
**Authors:** Liang et al.
**Venue:** CVPR, 2024
**arXiv:** 2404.00228

InfLoRA injects a small number of parameters and shows that fine-tuning them is equivalent to fine-tuning pretrained weights within a subspace. It precomputes the LoRA-A matrix via SVD of the input covariance and trains only the B matrix, designing the subspace to eliminate interference of new tasks on old tasks. This makes an effective trade-off between stability (preventing forgetting) and plasticity (learning new tasks), outperforming state-of-the-art CL methods on multiple benchmarks.

```bibtex
@inproceedings{liang2024inflora,
  title={InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning},
  author={Liang, Yan-Shuo and Li, Wu-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  note={arXiv:2404.00228}
}
```

---

### 2.3 CL-LoRA: Continual Low-Rank Adaptation (Dual-Adapter)

**Title:** CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning
**Authors:** He et al.
**Venue:** CVPR, 2025
**arXiv:** 2505.24816

CL-LoRA introduces a dual-adapter architecture combining task-shared adapters (using random orthogonal matrices with knowledge distillation and gradient reassignment) and task-specific adapters (with learnable block-wise weights). The shared adapters learn cross-task knowledge while the task-specific adapters capture unique features, mitigating inter-task interference while maintaining plasticity. Achieves state-of-the-art with reduced training and inference computation.

```bibtex
@inproceedings{he2025clora,
  title={CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning},
  author={He, Jiangpeng and Duan, Zhihao and Zhu, Fengqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  note={arXiv:2505.24816}
}
```

---

### 2.4 C-LoRA: Continual LoRA with Routing

**Title:** C-LoRA: Continual Low-Rank Adaptation for Pre-trained Models
**Authors:** Zhang et al.
**Venue:** arXiv preprint, 2025
**arXiv:** 2502.17920

C-LoRA uses a learnable routing matrix to dynamically manage parameter updates across tasks, ensuring efficient reuse of learned subspaces while enforcing orthogonality to minimize interference. Unlike existing methods that rely on multiple adapter modules (increasing overhead), C-LoRA provides a single evolving adapter with theoretical insights into routing matrix behavior for knowledge retention and transfer.

```bibtex
@article{zhang2025clora,
  title={C-LoRA: Continual Low-Rank Adaptation for Pre-trained Models},
  author={Zhang, Xin and Bai, Liang and Yang, Xian and Liang, Jiye},
  journal={arXiv preprint arXiv:2502.17920},
  year={2025}
}
```

---

### 2.5 Online-LoRA: Task-Free Online CL

**Title:** Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation
**Authors:** Wei et al.
**Venue:** arXiv preprint, 2024
**arXiv:** 2411.05663

Online-LoRA addresses task-free online continual learning where there are no well-defined task boundaries. It finetunes pretrained ViT models in real-time by expanding the model with additional LoRA parameters when the loss surface plateaus (detecting distribution shifts). A novel online weight regularization strategy identifies and consolidates important model parameters, enabling robust continual adaptation without task identifiers.

```bibtex
@article{wei2024onlinelora,
  title={Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation},
  author={Wei, Xiwen and Li, Guihong and Marculescu, Radu},
  journal={arXiv preprint arXiv:2411.05663},
  year={2024}
}
```

---

### 2.6 CoDyRA: Dynamic Rank-Selective LoRA

**Title:** Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA
**Authors:** Lu et al.
**Venue:** arXiv preprint, 2024
**arXiv:** 2412.01004

CoDyRA reveals a fundamental tension in LoRA rank for CL: higher rank improves plasticity but increases forgetting, while lower rank enhances stability but limits adaptation. It adaptively assigns ranks to LoRA modules based on their relevance to current data, performing efficient CL as a sequence of LoRA updates without storing past data, adding no inference overhead, and achieving state-of-the-art on VLM continual learning benchmarks.

```bibtex
@article{lu2024adaptive,
  title={Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA},
  author={Lu, Haodong and others},
  journal={arXiv preprint arXiv:2412.01004},
  year={2024}
}
```

---

### 2.7 LoRA Learns Less and Forgets Less

**Title:** LoRA Learns Less and Forgets Less
**Authors:** Biderman et al.
**Venue:** TMLR, 2024
**arXiv:** 2405.09673

This empirical study compares LoRA and full fine-tuning on programming and math domains, showing that LoRA substantially underperforms full fine-tuning in learning but better maintains base model performance on out-of-domain tasks. LoRA provides implicit regularization against forgetting that exceeds common techniques like weight decay and dropout. Full fine-tuning learns perturbations with 10-100x greater rank than typical LoRA configurations, explaining the performance gap.

```bibtex
@article{biderman2024lora,
  title={LoRA Learns Less and Forgets Less},
  author={Biderman, Dan and others},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024},
  note={arXiv:2405.09673}
}
```

---

### 2.8 LoraHub: Dynamic LoRA Composition

**Title:** LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
**Authors:** Huang et al.
**Venue:** COLM, 2024
**arXiv:** 2307.13269

LoraHub composes multiple LoRA modules trained on different tasks into a single unified module using learned weight coefficients. A gradient-free algorithm refines the composition weights using just a few examples from an unseen task (Adapt stage), achieving near in-context-learning performance on BIG-Bench Hard with zero-shot-level inference throughput. This demonstrates practical multi-task LoRA composition without additional parameters.

```bibtex
@inproceedings{huang2024lorahub,
  title={LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition},
  author={Huang, Chengsong and Liu, Qian and Lin, Bill Yuchen and Pang, Tianyu and Du, Chao and Lin, Min},
  booktitle={Conference on Language Modeling (COLM)},
  year={2024},
  note={arXiv:2307.13269}
}
```

---

### 2.9 LoRA Soups: Merging LoRAs for Skill Composition

**Title:** LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks
**Authors:** Prabhakar et al.
**Venue:** COLING Industry Track, 2025
**arXiv:** 2410.13025

LoRA Soups studies how individually trained LoRA modules can be merged to achieve skill composition. Concatenation of LoRAs (CAT), which optimally weights per-skill LoRAs, outperforms existing model-merging and data-mixing techniques by 43% and 12% respectively on math-word problems. This is the first work demonstrating the superiority of model merging over data mixing for binary skill composition tasks.

```bibtex
@inproceedings{prabhakar2025lorasoups,
  title={LoRA Soups: Merging LoRAs for Practical Skill Composition Tasks},
  author={Prabhakar, Akshara and Li, Yuanzhi and Narasimhan, Karthik and Kakade, Sham and Malach, Eran and Jelassi, Samy},
  booktitle={Proceedings of the 31st International Conference on Computational Linguistics (COLING), Industry Track},
  year={2025},
  note={arXiv:2410.13025}
}
```

---

## 3. Continual Learning for Manipulation

### 3.1 LIBERO Benchmark

**Title:** LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning
**Authors:** Liu et al.
**Venue:** NeurIPS (Datasets and Benchmarks), 2023
**arXiv:** 2306.03310

LIBERO is the definitive simulation benchmark for lifelong learning in robot manipulation, providing 130 tasks across four suites with human-teleoperated demonstrations. It highlights five key research topics: transfer of declarative/procedural knowledge, effective policy architectures, robustness to task ordering, and the effect of model pretraining. Key finding: sequential fine-tuning outperforms existing CL methods in forward transfer, and no single visual encoder excels at all types of knowledge transfer.

```bibtex
@inproceedings{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  volume={36},
  year={2023},
  note={arXiv:2306.03310}
}
```

---

### 3.2 LOTUS: Continual Imitation Learning via Skill Discovery

**Title:** LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery
**Authors:** Wan et al.
**Venue:** ICRA, 2024
**arXiv:** 2311.02058

LOTUS constructs an ever-growing skill library from sequential tasks using an open-vocabulary vision model to extract recurring skill patterns from unsegmented demonstrations. Continual skill discovery updates existing skills (preventing forgetting) and adds new skills for novel tasks, while a meta-controller flexibly composes skills for vision-based manipulation. LOTUS outperforms baselines by over 11% in success rate, demonstrating superior knowledge transfer for lifelong robot learning.

```bibtex
@inproceedings{wan2024lotus,
  title={LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery},
  author={Wan, Weikang and Zhu, Yifeng and Shah, Rutav and Zhu, Yuke},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024},
  note={arXiv:2311.02058}
}
```

---

### 3.3 Task-Agnostic Lifelong Robot Learning

**Title:** Task-agnostic Lifelong Robot Learning with Retrieval-based Weighted Local Adaptation
**Authors:** Yang et al.
**Venue:** arXiv preprint, 2024
**arXiv:** 2410.02995

This work addresses task-free lifelong robot learning where task IDs and boundaries are unavailable. The method stores a subset of data from previous tasks and leverages it through experience replay and a novel Retrieval-based Local Adaptation technique with a selective weighting mechanism that focuses on the most "forgotten" skill segments. Experiments across diverse manipulation tasks demonstrate a scalable paradigm for lifelong learning in open-ended, task-free scenarios.

```bibtex
@article{yang2024taskagnostic,
  title={Task-agnostic Lifelong Robot Learning with Retrieval-based Weighted Local Adaptation},
  author={Yang, Pengzhi and Wang, Xinyu and Zhang, Ruipeng and Wang, Cong and Oliehoek, Frans A and Kober, Jens},
  journal={arXiv preprint arXiv:2410.02995},
  year={2024}
}
```

---

### 3.4 Diffusion Policy (Foundational for Manipulation)

**Title:** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
**Authors:** Chi et al.
**Venue:** RSS, 2023 (also IJRR, 2024)
**arXiv:** 2303.04137

Diffusion Policy represents robot visuomotor policies as conditional denoising diffusion processes, predicting high-dimensional action sequences with receding-horizon control. It gracefully handles multimodal action distributions, is suitable for high-dimensional action spaces, and achieves an average 46.9% improvement over prior methods across 15 tasks. This has become the de facto policy class for manipulation, making diffusion-based replay and continual learning highly relevant.

```bibtex
@inproceedings{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023},
  note={arXiv:2303.04137}
}
```

---

### 3.5 OpenVLA

**Title:** OpenVLA: An Open-Source Vision-Language-Action Model
**Authors:** Kim et al.
**Venue:** CoRL, 2024
**arXiv:** 2406.09246

OpenVLA is a 7B-parameter open-source VLA trained on 970k real-world robot demonstrations, built on Llama 2 with DINOv2 and SigLIP visual encoders. It can be fine-tuned on consumer GPUs via LoRA and served efficiently via quantization. OpenVLA has become the primary testbed for VLA continual learning research, with multiple subsequent papers (CLARE, ExpReS-VLA, CRL-VLA) using it as a base model.

```bibtex
@inproceedings{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiao, Ted and Balakrishna, Ashwin and Nair, Suraj and Rafailov, Rafael and Foster, Ethan and Lam, Grace and Nasiriany, Mohan and others},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2024},
  note={arXiv:2406.09246}
}
```

---

### 3.6 pi-zero: VLA Flow Model

**Title:** pi-zero: A Vision-Language-Action Flow Model for General Robot Control
**Authors:** Black et al.
**Venue:** arXiv preprint, 2024
**arXiv:** 2410.24164

pi-zero is a novel flow matching architecture built on a pretrained VLM to inherit internet-scale semantic knowledge, trained on diverse data from single-arm robots, dual-arm robots, and mobile manipulators. As the first large-scale generalist VLA from Physical Intelligence, it demonstrates cross-embodiment generalization and has spurred interest in continual adaptation methods for deployed VLA systems.

```bibtex
@article{black2024pi0,
  title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```

---

## 4. Replay-Based Methods

### 4.1 Deep Generative Replay (Foundational)

**Title:** Continual Learning with Deep Generative Replay
**Authors:** Shin et al.
**Venue:** NeurIPS, 2017
**arXiv:** 1705.08690

The foundational generative replay paper proposes a cooperative dual-model architecture: a deep generative model ("generator") and a task-solving model ("solver"). Training data for previous tasks is sampled from the generator and interleaved with new task data, avoiding the need to store raw past data. Inspired by the hippocampus as a short-term memory system in the primate brain, this approach eliminates storage requirements while effectively mitigating catastrophic forgetting.

```bibtex
@inproceedings{shin2017continual,
  title={Continual Learning with Deep Generative Replay},
  author={Shin, Hanul and Lee, Jung Kwon and Kim, Jaehong and Kim, Jiwon},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={30},
  year={2017},
  note={arXiv:1705.08690}
}
```

---

### 4.2 DDGR: Diffusion-Based Generative Replay

**Title:** DDGR: Continual Learning with Deep Diffusion-based Generative Replay
**Authors:** Gao et al.
**Venue:** ICML, 2023

DDGR adopts a diffusion model as the generator and computes an instruction-operator through the classifier to guide sample generation for replay. It addresses a key limitation: conventional GR methods focus on a single generator-to-classifier instruction relationship and reuse generated samples to update the generator, causing regenerated samples to deviate from previous task distributions. DDGR introduces bidirectional instruction to maintain distribution fidelity.

```bibtex
@inproceedings{gao2023ddgr,
  title={DDGR: Continual Learning with Deep Diffusion-based Generative Replay},
  author={Gao, Rui and Liu, Weiwei},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={10744--10763},
  year={2023}
}
```

---

### 4.3 Generative Distillation for Diffusion Model CL

**Title:** Continual Learning of Diffusion Models with Generative Distillation
**Authors:** Masip et al.
**Venue:** Conference on Lifelong Learning Agents (CoLLAs), 2025
**arXiv:** 2311.14028

Standard generative replay applied to diffusion models results in catastrophic loss of denoising capabilities. This work proposes generative distillation, which distills the entire reverse process of a diffusion model rather than just replaying outputs. This substantially improves CL performance with only a modest increase in computational costs, making it the first effective approach for applying generative replay to diffusion-based policies.

```bibtex
@inproceedings{masip2025continual,
  title={Continual Learning of Diffusion Models with Generative Distillation},
  author={Masip, Sergi and Rodriguez, Pau and Tuytelaars, Tinne and van de Ven, Gido M},
  booktitle={Proceedings of The 3rd Conference on Lifelong Learning Agents (CoLLAs)},
  year={2025},
  note={arXiv:2311.14028}
}
```

---

### 4.4 Diffusion-Based Dual Generative Replay for Offline RL

**Title:** Continual Offline Reinforcement Learning via Diffusion-based Dual Generative Replay
**Authors:** Liu et al.
**Venue:** arXiv preprint, 2024
**arXiv:** 2404.10662

This paper decouples the continual offline RL policy into a diffusion-based generative behavior model and a multi-head action evaluation model. A task-conditioned diffusion model generates synthetic states from past tasks, paired with behavior generator responses for high-fidelity replay. The approach achieves better forward transfer with less forgetting than methods using ground-truth past data, due to its high-fidelity sample space replay.

```bibtex
@article{liu2024continual,
  title={Continual Offline Reinforcement Learning via Diffusion-based Dual Generative Replay},
  author={Liu, Jinmei and Li, Wenbin and Yue, Xiangyu and Zhang, Shilin and Chen, Chunlin and Wang, Zhi},
  journal={arXiv preprint arXiv:2404.10662},
  year={2024}
}
```

---

### 4.5 t-DGR: Trajectory-Based Deep Generative Replay

**Title:** t-DGR: A Trajectory-Based Deep Generative Replay Method for Continual Learning in Decision Making
**Authors:** Yue et al.
**Venue:** Conference on Lifelong Learning Agents (CoLLAs), 2025
**arXiv:** 2401.02576

t-DGR proposes a simple, scalable, non-autoregressive method for CL in decision-making using a generative model that produces task samples conditioned on trajectory timestep. Unlike autoregressive approaches that suffer from compounding errors in generated trajectories, t-DGR generates each timestep independently. Achieves state-of-the-art average success rate on the Continual World benchmark among CL methods.

```bibtex
@inproceedings{yue2025tdgr,
  title={t-DGR: A Trajectory-Based Deep Generative Replay Method for Continual Learning in Decision Making},
  author={Yue, William and Liu, Bo and Stone, Peter},
  booktitle={Proceedings of The 3rd Conference on Lifelong Learning Agents (CoLLAs)},
  year={2025},
  note={arXiv:2401.02576}
}
```

---

### 4.6 Stable CRL via Diffusion Trajectory Replay

**Title:** Stable Continual Reinforcement Learning via Diffusion-based Trajectory Replay
**Authors:** Chen et al.
**Venue:** ICLR Workshop on Generative Models for Decision Making, 2024
**arXiv:** 2411.10809

This work applies diffusion-based generative replay to continual RL, using generative models to replay data distributions of past tasks to circumvent growing storage overhead and data privacy concerns. The method generates synthetic trajectories that maintain the distributional properties of past task data while enabling stable policy updates across task sequences.

```bibtex
@article{chen2024stable,
  title={Stable Continual Reinforcement Learning via Diffusion-based Trajectory Replay},
  author={Chen, Feng and Han, Fuguang and Guan, Cong and Yuan, Lei and Zhang, Zhilong and Yu, Yang and Zhang, Zongzhang},
  journal={arXiv preprint arXiv:2411.10809},
  year={2024}
}
```

---

### 4.7 Replay Scheduling

**Title:** Learn the Time to Learn: Replay Scheduling in Continual Learning
**Authors:** Klasson et al.
**Venue:** TMLR, 2023
**arXiv:** 2209.08660

Rather than treating replay as a fixed policy, this work proposes learning when and what to replay. Using Monte Carlo tree search to find optimal replay schedules, the paper demonstrates that learned schedules significantly outperform fixed scheduling policies. When storing historical data is cheap but processing time is constrained, intelligent scheduling of which tasks to replay at different time steps matters more than the raw replay method.

```bibtex
@article{klasson2023learn,
  title={Learn the Time to Learn: Replay Scheduling in Continual Learning},
  author={Klasson, Marcus and Kjellstr{\"o}m, Hedvig and Zhang, Cheng},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2023},
  note={arXiv:2209.08660}
}
```

---

## 5. Anti-Forgetting in Foundation Models

### 5.1 Survey: CL of Large Language Models

**Title:** Continual Learning of Large Language Models: A Comprehensive Survey
**Authors:** Shi et al.
**Venue:** ACM Computing Surveys, 2025
**arXiv:** 2404.16789

The definitive survey on continual learning for LLMs, covering vertical continuity (general-to-specific adaptation) and horizontal continuity (cross-time/domain adaptation). Organizes methods into three stages: Continual Pre-Training (CPT), Domain-Adaptive Pre-training (DAP), and Continual Fine-Tuning (CFT). Provides a comprehensive taxonomy of anti-forgetting strategies including replay, regularization, architectural, and PEFT-based approaches for foundation models.

```bibtex
@article{shi2025continual,
  title={Continual Learning of Large Language Models: A Comprehensive Survey},
  author={Shi, Haizhou and Xu, Zihao and Wang, Hengyi and Qin, Weiyi and Wang, Wenyuan and Wang, Yibin and Wang, Zifeng and Ebrahimi, Sayna and Wang, Hao},
  journal={ACM Computing Surveys},
  year={2025},
  publisher={ACM},
  note={arXiv:2404.16789}
}
```

---

### 5.2 Survey: Forgetting in Deep Learning Beyond CL

**Title:** A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning
**Authors:** Wang et al.
**Venue:** IEEE TPAMI, 2024
**arXiv:** 2307.09218

This survey broadens the scope beyond traditional CL, classifying forgetting in deep learning into harmful and beneficial forms across multiple domains: foundation models, domain adaptation, meta-learning, generative models, RL, and federated learning. Argues that forgetting is a double-edged sword that can be desirable in privacy-preserving scenarios. Particularly relevant for understanding forgetting dynamics in dual-architecture systems and multi-component models.

```bibtex
@article{wang2024comprehensive,
  title={A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning},
  author={Wang, Zhenyi and Yang, Enneng and Shen, Li and Huang, Heng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE},
  note={arXiv:2307.09218}
}
```

---

### 5.3 MoE-Adapters for VLM Continual Learning

**Title:** Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters
**Authors:** Yu et al.
**Venue:** CVPR, 2024
**arXiv:** 2403.11549

This work dynamically expands a pretrained CLIP model through Mixture-of-Experts (MoE) adapters in response to new tasks, with a Distribution Discriminative Auto-Selector (DDAS) that routes in-distribution inputs to MoE Adapters and out-of-distribution inputs to original CLIP. Achieves state-of-the-art while reducing parameter training burdens by 60%. Directly addresses the adapter-based approach to preventing forgetting in VLMs.

```bibtex
@inproceedings{yu2024boosting,
  title={Boosting Continual Learning of Vision-Language Models via Mixture-of-Experts Adapters},
  author={Yu, Jiazuo and Zhuge, Yunzhi and Zhang, Lu and Hu, Ping and Wang, Dong and Lu, Huchuan and He, You},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
  note={arXiv:2403.11549}
}
```

---

### 5.4 Aligned Model Merging for VLM CL

**Title:** Continual Learning in Vision-Language Models via Aligned Model Merging
**Authors:** Sokar et al.
**Venue:** arXiv preprint, 2025
**arXiv:** 2506.03189

From Google DeepMind, this work proposes merging newly trained task parameters with previously learned ones while promoting learning aligned weights to avoid interference when merging. The approach reduces forgetting, increases robustness to various task orders and similarities, and improves generalization. Identifies cross-modal feature drift and shared module interference as core failure modes in VLM CL.

```bibtex
@article{sokar2025continual,
  title={Continual Learning in Vision-Language Models via Aligned Model Merging},
  author={Sokar, Ghada and Dziugaite, Gintare Karolina and Arnab, Anurag and Iscen, Ahmet and Castro, Pablo Samuel and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2506.03189},
  year={2025}
}
```

---

### 5.5 VLM2VLA: Actions as Language

**Title:** Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting
**Authors:** Hancock et al.
**Venue:** arXiv preprint, 2025
**arXiv:** 2509.22195

VLM2VLA resolves the distribution mismatch between VLM pretraining and robotics fine-tuning by representing low-level actions with natural language. This alignment makes it possible to train VLAs solely with LoRA, minimally modifying the VLM backbone and averting catastrophic forgetting. Over 800 real-world experiments show that VLM2VLA preserves VLM core capabilities, enabling zero-shot generalization to novel tasks requiring open-world semantic reasoning and multilingual instruction following.

```bibtex
@article{hancock2025actions,
  title={Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting},
  author={Hancock, Asher J and others},
  journal={arXiv preprint arXiv:2509.22195},
  year={2025}
}
```

---

### 5.6 Preserving Pretrained Representations in VLAs

**Title:** Enhancing Generalization in Vision-Language-Action Models by Preserving Pretrained Representations
**Authors:** Grover et al.
**Venue:** arXiv preprint, 2025
**arXiv:** 2509.11417

Demonstrates that pretrained visual representations deteriorate when VLA models are fine-tuned on action data. Proposes a dual-encoder design (one frozen, one trainable), a string-based action tokenizer aligning actions with language tokens, and a co-training strategy mixing robot data with VL datasets emphasizing spatial reasoning. This directly addresses the dual-architecture forgetting problem -- separate backbone degradation during decoder training.

```bibtex
@article{grover2025enhancing,
  title={Enhancing Generalization in Vision-Language-Action Models by Preserving Pretrained Representations},
  author={Grover, Shresth and Gopalkrishnan, Akshay and Ai, Bo and Christensen, Henrik I and Su, Hao and Li, Xuanlin},
  journal={arXiv preprint arXiv:2509.11417},
  year={2025}
}
```

---

### 5.7 CLARE: Continual Learning for VLAs via Adapter Routing

**Title:** CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion
**Authors:** Romer et al.
**Venue:** arXiv preprint, 2026
**arXiv:** 2601.09512

CLARE introduces lightweight modular adapters into VLA feedforward layers and autonomously expands the model only where necessary (guided by layer-wise feature similarity). An autoencoder-based routing mechanism dynamically activates the most relevant adapters during deployment without task labels. Evaluated on the LIBERO benchmark, CLARE achieves high performance on new tasks without catastrophic forgetting, significantly outperforming even exemplar-based methods.

```bibtex
@article{romer2026clare,
  title={CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion},
  author={R{\"o}mer, Ralf and Zhang, Yi and Schoellig, Angela P},
  journal={arXiv preprint arXiv:2601.09512},
  year={2026}
}
```

---

### 5.8 CRL-VLA: Continual VLA Learning

**Title:** CRL-VLA: Continual Vision-Language-Action Learning
**Authors:** Zeng et al.
**Venue:** arXiv preprint, 2026
**arXiv:** 2602.03445

Identifies that the fundamental driver of forgetting in continual VLA learning is the goal-conditioned advantage magnitude, which directly links policy divergence to performance degradation on prior tasks. CRL-VLA resolves this through asymmetric regulation: constraining advantage magnitudes on prior tasks while enabling controlled growth on new tasks, using a dual-critic architecture with a frozen critic for semantic consistency and a trainable estimator for adaptation.

```bibtex
@article{zeng2026crlvla,
  title={CRL-VLA: Continual Vision-Language-Action Learning},
  author={Zeng, Qixin and Zhang, Shuo and Zhang, Hongyin and Wang, Renjie and Zhao, Han and Zhao, Libang and Li, Runze and Wang, Donglin and Huang, Chao},
  journal={arXiv preprint arXiv:2602.03445},
  year={2026}
}
```

---

### 5.9 ExpReS-VLA: Experience Replay for VLA Specialization

**Title:** ExpReS-VLA: Specializing Vision-Language-Action Models Through Experience Replay and Retrieval
**Authors:** Syed et al.
**Venue:** arXiv preprint, 2025
**arXiv:** 2511.06202

ExpReS-VLA continuously collects experiences during deployment, stores them as compressed feature representations from the frozen vision backbone (reducing memory by ~97%), and retrieves relevant past experiences to guide future adaptation. Combines compressed memory, retrieval-augmented generation, and contrastive learning to avoid past failures. Transforms OpenVLA from a generalist into a specialist in its deployment environment while preventing catastrophic forgetting.

```bibtex
@article{syed2025expresvla,
  title={ExpReS-VLA: Specializing Vision-Language-Action Models Through Experience Replay and Retrieval},
  author={Syed, Shahram Najam and Ahuja, Yatharth and Jakobsson, Arthur and Ichnowski, Jeff},
  journal={arXiv preprint arXiv:2511.06202},
  year={2025}
}
```

---

### 5.10 Stellar VLA: Continually Evolving Skill Knowledge

**Title:** Continually Evolving Skill Knowledge in Vision Language Action Model
**Authors:** Wu et al.
**Venue:** arXiv preprint, 2025
**arXiv:** 2511.18085

Stellar VLA proposes a knowledge-driven continual learning framework with two variants: T-Stellar (modeling task-centric knowledge space) and TS-Stellar (capturing hierarchical task-skill structure). The work addresses how VLA models can continually evolve their skill repertoire without forgetting previously acquired capabilities, providing a structured approach to skill knowledge management in continually learning VLAs.

```bibtex
@article{wu2025stellar,
  title={Continually Evolving Skill Knowledge in Vision Language Action Model},
  author={Wu, Yuxuan and Wang, Guangming and Yang, Zhiheng and Yao, Maoqing and Sheil, Brian and Wang, Hesheng},
  journal={arXiv preprint arXiv:2511.18085},
  year={2025}
}
```

---

### 5.11 LifeLong-RFT: Continual VLA via Reinforcement Fine-Tuning

**Title:** Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning
**Authors:** Liu et al.
**Venue:** arXiv preprint, 2026
**arXiv:** 2602.10503

LifeLong-RFT integrates chunking-level on-policy reinforcement learning with a Multi-Dimensional Process Reward (MDPR) mechanism that quantifies heterogeneous contributions of intermediate action chunks across three dimensions. The method achieves 22% gain in average success rate over SFT on the LIBERO continual learning benchmark while adapting to new tasks using only 20% of training data, demonstrating that RL-based fine-tuning is more data-efficient for continual VLA adaptation than supervised approaches.

```bibtex
@article{liu2026lifelong,
  title={Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning},
  author={Liu, Yuan and Li, Haoran and Tian, Shuai and Qin, Yuxing and Chen, Yuhui and Zheng, Yupeng and Huang, Yongzhen and Zhao, Dongbin},
  journal={arXiv preprint arXiv:2602.10503},
  year={2026}
}
```

---

## Summary Table

| # | Paper | Category | Method Type | Year | Key Contribution |
|---|-------|----------|-------------|------|-----------------|
| 1 | EWC (Kirkpatrick) | Classic CL | Regularization | 2017 | Fisher Information penalty |
| 2 | Progressive Nets (Rusu) | Classic CL | Architecture | 2016 | Lateral connections, no forgetting |
| 3 | PackNet (Mallya) | Classic CL | Pruning | 2018 | Iterative pruning packs tasks |
| 4 | Continual World (Wolczyk) | Benchmark | Evaluation | 2021 | Robotic CL benchmark |
| 5 | LEGION (Meng) | Classic CL | Bayesian NP | 2025 | Dirichlet process for lifelong RL |
| 6 | LoRA (Hu) | PEFT | Foundation | 2022 | Low-rank adaptation |
| 7 | O-LoRA (Wang) | LoRA CL | Orthogonal | 2023 | Orthogonal subspace per task |
| 8 | InfLoRA (Liang) | LoRA CL | Subspace | 2024 | SVD-based interference-free LoRA |
| 9 | CL-LoRA (He) | LoRA CL | Dual adapter | 2025 | Shared + specific adapters |
| 10 | C-LoRA (Zhang) | LoRA CL | Routing | 2025 | Learnable routing matrix |
| 11 | Online-LoRA (Wei) | LoRA CL | Task-free | 2024 | Loss plateau detection |
| 12 | CoDyRA (Lu) | LoRA CL | Dynamic rank | 2024 | Rank-plasticity tradeoff |
| 13 | Biderman | LoRA CL | Empirical | 2024 | LoRA implicit regularization |
| 14 | LoraHub (Huang) | LoRA Merge | Composition | 2024 | Gradient-free LoRA composition |
| 15 | LoRA Soups (Prabhakar) | LoRA Merge | Merge | 2025 | Optimal LoRA concatenation |
| 16 | LIBERO (Liu) | Manipulation | Benchmark | 2023 | Lifelong manipulation benchmark |
| 17 | LOTUS (Wan) | Manipulation | Skill library | 2024 | Continual skill discovery |
| 18 | Yang et al. | Manipulation | Retrieval | 2024 | Task-free lifelong robot learning |
| 19 | Diffusion Policy (Chi) | Manipulation | Foundation | 2023 | Action diffusion for manipulation |
| 20 | OpenVLA (Kim) | VLA | Foundation | 2024 | Open-source 7B VLA |
| 21 | pi-zero (Black) | VLA | Foundation | 2024 | Flow-based generalist VLA |
| 22 | Deep Gen. Replay (Shin) | Replay | Generative | 2017 | Generator-solver dual model |
| 23 | DDGR (Gao) | Replay | Diffusion | 2023 | Diffusion-based generative replay |
| 24 | Gen. Distillation (Masip) | Replay | Distillation | 2025 | Distill reverse diffusion process |
| 25 | Dual Gen. Replay (Liu) | Replay | Diffusion+RL | 2024 | Dual diffusion replay for offline RL |
| 26 | t-DGR (Yue) | Replay | Trajectory | 2025 | Non-autoregressive trajectory replay |
| 27 | SDTR (Chen) | Replay | Diffusion+RL | 2024 | Diffusion trajectory replay for CRL |
| 28 | Replay Scheduling (Klasson) | Replay | Scheduling | 2023 | MCTS for optimal replay scheduling |
| 29 | LLM CL Survey (Shi) | Foundation | Survey | 2025 | LLM continual learning taxonomy |
| 30 | Forgetting Survey (Wang) | Foundation | Survey | 2024 | Forgetting beyond CL |
| 31 | MoE-Adapters (Yu) | Foundation | MoE + Adapter | 2024 | MoE adapters for CLIP CL |
| 32 | Aligned Merging (Sokar) | Foundation | Model merge | 2025 | Aligned weight merging for VLM CL |
| 33 | VLM2VLA (Hancock) | Foundation | Data alignment | 2025 | Actions as language to prevent forgetting |
| 34 | Gen-VLA (Grover) | Foundation | Dual encoder | 2025 | Frozen + trainable encoder for VLA |
| 35 | CLARE (Romer) | VLA CL | Adapter routing | 2026 | Autonomous adapter expansion for VLA |
| 36 | CRL-VLA (Zeng) | VLA CL | Dual critic | 2026 | Advantage-based forgetting analysis |
| 37 | ExpReS-VLA (Syed) | VLA CL | Replay+Retrieval | 2025 | Compressed experience replay for VLA |
| 38 | Stellar VLA (Wu) | VLA CL | Skill knowledge | 2025 | Task-skill hierarchical CL |
| 39 | LifeLong-RFT (Liu) | VLA CL | RL fine-tuning | 2026 | RL-based continual VLA adaptation |

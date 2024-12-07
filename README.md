# Soft Tensor Product Representations for Fully Continuous, Compositional Visual Representations

Code for *Soft Tensor Product Representations for Fully Continuous, Compositional Visual Representations* to appear in the [38th Annual Conference on Neural Information Processing Systems](https://neurips.cc/Conferences/2024) (NeurIPS 2024). 

| 📖 [Arvix Preprint](<i>coming soon</i>) | 🧵 [OpenReview](https://openreview.net/forum?id=oEVsxVdush) | 🪧 [Poster](https://neurips.cc/virtual/2024/poster/93635) | 🛝 [Slides](https://neurips.cc/media/neurips-2024/Slides/93635.pdf) |
| - | - | - | - | 

## Table of Contents
1. [Paper Overview](#paper_tldr)
2. [Installation](#install)
3. [Code Overview](#code_tldr)
4. [Misc](#misc)


## 1. Paper Overview <a name="paper_tldr"></a>

Learning explicitly compositional representations has both theorised [1, 2] and empiricial benefits [4-12].  

A predominant approach is that of disentanglement, where the underlying factors of variation (FoVs) are *isolated* into *distinct parts* (coloured blocks of RHS in Fig 1) of the representation, $\psi(x)$, which corresponds to the Jacobian disentanglement requirement of [13].
<p align="center" width="100%">
<img width="40%" alt="image" src="https://github.com/user-attachments/assets/93c743e9-ecde-4ea9-9b6d-8a9edb73e334">
</p>

However, this enforces a fundamentally **symbolic** treatment of compositional structure, where the FoVs are discretely allocated to distinct representational slots, which are concatenated together to form a **string-like** compositional representation. 
<p align="center" width="100%">
<img width="20%" alt="image" src="https://github.com/user-attachments/assets/199d9e10-5dde-4f39-9625-e4509ac7e97d">
</p>

We argue this symbolic approach of treating compositional structure is fundamentally incompatible with the *continuous* vector spaces of deep learning (please see paper, main body Section 1. paragraphs 4-5, and appendix A.3 for more details). 
<p align="center" width="100%">
<img width="55%" alt="image" src="https://github.com/user-attachments/assets/b57302cd-355b-4794-872a-4b099ef7f8f0">
</p>

To align compositional structure with continuous vector spaces, we formulate a fundamentally **continuous** compositional representation. Such an approach *smoothly interweaves* FoVs into $\psi(x)$, similar to the *continuous superimposition* of multiple waves into an aggregate waveform. 
<p align="center" width="100%">
<img width="30%" alt="image" src="https://github.com/user-attachments/assets/0a8cfb09-4628-42ad-a2de-309e045e50b3">
</p>
Our approach, **Soft TPR**, builds upon Smolensky's established Tensor Product Representation (TPR) [3], but provides enhanced ease of learning and representational flexibility compared to the conventional TPR. 

We additionally introduce **Soft TPR Autoencoder**, a theoretically-principled method to learn Soft TPRs that learns elements of the Soft TPR form by leveraging the mathematical properties of the Soft TPR/TPR framework. 

![image](https://github.com/user-attachments/assets/b07d8eba-35ea-4b3c-b550-73a6db9fb4d8)


Our results empiricially suggest that the enhanced vector space alignment produced by Soft TPRs is broadly beneficial for DL models (both representation learners & downstream models). In particular, Soft TPRs are (please see our main paper and Appendix C for full results): 

1. **Structural**: Soft TPRs are more explicitly compositional than baselines (quantified using disentanglement metrics), achieving SoTA performance (DCI boosts of 29%+, 74%+ on Cars3D/MPI3D)
2. **Representation Learner Convergence**: Soft TPRs learn representations useful for downstream tasks more quickly than baselines (e.g., for FoV regression on MPI3D and the task of abstract visual reasoning, the performance of downstream models using Soft TPRs produced at 100 iterations of representation learner training is equivalent to the performance of the most competitive baseline at 2 orders' more training iterations (i.e., $10^4$ iterations))
3. **Downstream Performance**: Soft TPRs have superior downstream efficiency (e.g., 93+%) and low-sample regime performance (e.g., 138%+, 168%+).

###### [1] Noam Chomsky. Syntactic Structures. The Hague: Mouton, 1957. <br> [2] Jerry A. Fodor. The Language of Thought: A Theory of Mental Representation. Cambridge, MA: Harvard University Press, 1975. <br> [3] Paul Smolensky. “Tensor product variable binding and the representation of symbolic structures in connectionist systems”. In: Artificial Intelligence 46.1 (1990), pp. 159–216. <br> [4] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. “beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework”. In: International Conference on Learning Representations. 2017. <br> [5] Tameem Adel, Zoubin Ghahramani, and Adrian Weller. “Discovering Interpretable Representations for Both Deep Generative and Discriminative Models”. In: Proceedings of the 35th International Conference on Machine Learning. Vol. 80. Proceedings of Machine Learning Research. PMLR, 2018, pp. 50–59.<br> [6] Sjoerd van Steenkiste, Francesco Locatello, Jürgen Schmidhuber, and Olivier Bachem. “Are disentangled representations helpful for abstract visual reasoning?” In: Proceedings of the 33rd International Conference on Neural Information Processing Systems. 2019 <br> [7] F. Locatello, B. Poole, G. Rätsch, B. Schölkopf, O. Bachem, and M. Tschannen. “Weakly-Supervised Disentanglement Without Compromises”. In: Proceedings of the 37th International Conference on Machine Learning (ICML). Vol. 119. Proceedings of Machine Learning Research. PMLR, 2020, pp. 6348– 6359 <br> [8] Elliot Creager, David Madras, Joern-Henrik Jacobsen, Marissa Weis, Kevin Swersky, Toniann Pitassi, and Richard Zemel. “Flexibly Fair Representation Learning by Disentanglement”. In: Proceedings of the 36th International Conference on Machine Learning. 2019. <br> [9] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. “Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations”. In: Proceedings of the 36th International Conference on Machine Learning. Vol. 97. Proceedings of Machine Learning Research. PMLR, 2019, pp. 4114–4124. <br> [10] Sungho Park, Sunhee Hwang, Dohyung Kim, and Hyeran Byun. “Learning Disentangled Representation for Fair Facial Attribute Classification via Fairness-aware Information Alignment”. In: Proceedings of the AAAI Conference on Artificial Intelligence 35 (2021), pp. 2403–2411 <br> [11] H. Zhang, Y.-F. Zhang, W. Liu, A. Weller, B. Schölkopf, and E. Xing. “Towards Principled Disentanglement for Domain Generalization”. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022, pp. 8024–8034 <br> [12] Haoyang Li, Xin Wang, Zeyang Zhang, Haibo Chen, Ziwei Zhang, and Wenwu Zhu. “Disentangled Graph Self-supervised Learning for Out-of-Distribution Generalization”. In: Forty-first International Conference on Machine Learning. 2024.<br> [13] Frederik Träuble, Elliot Creager, Niki Kilbertus, Francesco Locatello, Andrea Dittadi, Anirudh Goyal, Bernhard Schölkopf, and Stefan Bauer. “On Disentangled Representations Learned from Correlated Data”. In: Proceedings of the 38th International Conference on Machine Learning. Vol. 139. Proceedings of Machine Learning Research. 2021, pp. 10401–10412.

## 2. Installation <a name="install"></a>
### 2.1. Environments
There are 2 environments enclosed within 1. requirements.txt (conda venv format) and 2. requirements-gadi.txt (standard format). This is to accommodate training on different hardware as each model has differing computational costs. 

The environment of requirements.txt was deployed on a RTX4090 (Pytorch and Python listed in conda venv format). 
The environment of requirements-gadi.txt was deployed on a V100 (Pytorch 1.9.0, Python 3.9.2). 

Please refer to the following table to determine which env/hardware combination is required to train the model of your interest: 

| Model Type | Model | Requirements File | Hardware Type|
| ---------- | ------ | ----------------- | --------------------| 
| Ours | Soft TPR Autoencoder | requirements.txt | RTX4090 | 
| Symbolic scalar-tokened baseline | Ada-GVAE, GVAE, MLVAE, Shu, SlowVAE | requirements.txt | RTX4090 | 
| Symbolic vector-tokened baselines | COMET, VCT | requirements-gadi.txt | V100 | 

### 2.2. Datasets

| 🟩 [Shapes3D](https://console.cloud.google.com/storage/browser/3d-shapes?pli=1) | 🦾 [MPI3D (real)](https://github.com/CUN-bjy/mpi3d_real?tab=readme-ov-file) | 🚗 [Cars3D](http://www.scottreed.info/files/nips2015-analogy-data.tar.gz) | 
| - | - | - |

## 3. Code Overview <a name="code_tldr"></a>
### 3.1. Directory Structure

| Dir | Contents |
| --- | --------- |
| bash/ <a name="bash"></a>| Shell scripts for running experiments |
| src/ <a name="src"></a> | Code |
| src/data/ <a name="data"></a> | Generic dataset/dataloader related logic |
| src/eval/ <a name="eval"></a> | Generic evaluation code that can be used for any scalar-tokened VAE-based model (i.e., Ada-GVAE, GVAE, MLVAE, SlowVAE) and the Soft TPR Autoencoder |
| src/eval/avr/ <a name="avr"></a> | Abstract visual reasoning evaluation | 
| src/eval/dis/ <a name="dis"></a> | Disentanglement metric evaluation | 
| src/eval/fov_regression/ <a name="fov"></a> | FoV regression evaluation | 
| src/logger/ <a name="logger"></a> | Logging | 
| src/repn_learners/ <a name="repn_learners"></a> | Code for implementing and training the representation learner models |
| src/repn_learners/baselines/comet/ <a name="comet"></a> | Code for training **and** evaluating COMET. Note that COMET could not be easily integrated with src/eval/, so there is an eval subdir under this directory containing COMET-specific evaluation logic |
| src/repn_learners/baselines/vct/ <a name="vct"></a> | Code for training **and** evaluating VCT. Similar to COMET, there is an eval subdir under this directory containing VCT-specific evaluation logic |
| src/repn_learners/baselines/scalar_tokened/ <a name="scalar_tokened"></a> | Code for all scalar-tokened baselines: Ada-GVAE, GVAE, MLVAE, SlowVAE (VAE-based) and Shu (GAN-based) | 
| src/repn_learners/baselines/scalar_tokened/shu <a name="shu"></a> | Code for the Shu model. Similar to COMET, VCT, there is an eval subdir under this directory containing Shu-specific evaluation logic |
| src/repn_learners/baselines/scalar_tokened/vae_based <a name="vae_based"></a> | Code for the scalar-tokened VAE-baselines. Evaluation is performed using src/eval/ | 
| src/repn_learners/tpr_ae <a name="tpr_ae"></a> | Code for the Soft TPR Autoencoder. Evaluation is performed using src/eval/ | 

### 3.2. Representation Learner Training
To train the representation learner of interest, simply create a script that runs the relevant train.py file with the required command-line arguments. Please refer to Appendix B.4 for our model hyperparameters, and the original papers of the baseline models for hyperparameter settings.

| Model | Train File | 
| ---- | ------- | 
| Scalar-tokened VAE baselines (i.e., Ada-GVAE, GVAE, MLVAE, SlowVAE) | src/repn_learners/baselines/scalar_tokened/vae_based/train.py |
| Scalar-tokened GAN baseline (Shu) | src/repn_learners/baselines/scalar_tokened/shu/train.py |
| COMET | src/repn_learners/baselines/comet/train.py
| VCT |  src/repn_learners/baselines/vct/main_vqvae.py (to train VQ-VAE backbone) src/repn_learners/baselines/vct/main_vct.py (to train VCT) | 
| Soft TPR Autoencoder | src/repn_learners/soft_tpr_ae/train.py |

We train *all* models for $2 \times 10^{5}$ iterations. 


### 3.3. Evaluation
1. To evaluate **representation structure**, we simply evaluate disentanglement metrics on *fully* trained representation learners. 
2. To evaluate **representation learner convergence** we checkpoint representation learners at different stages of representation learner training (e.g., $10^{2}, 2.5 \times 10^{2}, 5 \times 10^{2}, 10^{3}, 10^{4}, 10^{5}, 5 \times 10^{5}$ and evaluate:
   * Disentanglement performance: simply run disentanglement metric evaluation on checkpointed representation learners
   * Downstream performance: simply run FoV regression evaluation or abstract visual reasoning evaluation on checkpointed representation learners
4. To evaluate **downstream model sample efficiency/low sample regime performance**, provide the path to the representation learner of interest (this can be a fully trained representation learner, or checkpointed one), and run either the FoV regression evaluation, or the abstract visual reasoning evaluation while specifying the total number of samples the representation learner should be trained with using `--n_sample_list`. The provided representation learner will be used to generate representations for the downstream model to use.

| Evaluation Type | Model | File Location |
| ---- | ------- | ------- |
| Disentanglement | All scalar-tokened baselines (i.e., Ada-GVAE, GVAE, MLVAE, SlowVAE, Shu) | src/eval/dis/run_dis_eval.py (also possible to pass flag ``--compute_dis_metrics`` in train.py |
| Disentanglement | COMET | src/repn_learners/baselines/comet/eval/eval_dis_metrics.py | 
| Disentanglement | VCT |  Invoked during training | 
| Disentanglement | Soft TPR Autoencoder | src/eval/dis/run_dis_eval.py (also possible to pass flag ``--compute_dis_metrics`` in train.py | 
| Abstract Vis Reasoning | All scalar-tokened baselines (i.e., Ada-GVAE, GVAE, MLVAE, SlowVAE, Shu) | src/eval/avr/run_downstream_avr.py | 
| Abstract Vis Reasoning | COMET | src/repn_learners/baselines/comet/eval/run_downstream_avr.py | 
| Abstract Vis Reasoning | VCT | src/repn_learners/baselines/vct/eval/run_downstream_avr.py | 
| Abstract Vis Reasoning | Soft TPR Autoencoder | src/eval/avr/run_downstream_avr.py | 
| FoV Regression | All scalar-tokened baselines (i.e., Ada-GVAE, GVAE, MLVAE, SlowVAE, Shu) | src/eval/fov/run_downstream_fov.py | 
| FoV Regression | COMET | src/repn_learners/baselines/comet/eval/run_downstream_fov.py | 
| FoV Regression | VCT | src/repn_learners/baselines/vct/eval/run_downstream_fov.py | 
| FoV Regression | Soft TPR Autoencoder | src/eval/fov/run_downstream_fov.py | 


## 4. Misc <a name="misc"></a>
### 4.1. Contact 
Please feel free to contact me at bethia.sun@gmail.com if you have any questions!

The idea for the paper randomly stumbled in my mind around 5 weeks before the Neurips deadline, so you can imagine the state (🍝🍝🍝) of my original repo 😹. Quite a bit of code tidying up was required prior to open-sourcing this project. All original results seem reproduceable with the tidied-up repo, but if you happen to notice any issues because of accidental minor bugs I may have introduced, please contact me as I have access to the original repo.
### 4.2. Bibtex 
If you find our paper useful, please feel free to cite it at: 
```BibTeX
@inproceedings{
sun2024soft,
title={Soft Tensor Product Representations for Fully Continuous, Compositional Visual Representations},
author={Bethia Sun and Maurice Pagnucco and Yang Song},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=oEVsxVdush}
}
```
### 4.3. Acknowledgements
For evaluation and baseline impleemntation, this code has built upon original implementations of the authors listed below. A big thank-you to these authors for making their code open-source 🤗! 

| Author | Repo Link | Paper | Code Use | 
| ------- | --------  | ----- | --------| 
| Schott et al. | [OpenReview Supp](https://openreview.net/attachment?id=9RUHPlladgh&name=supplementary_material) | Visual Representation Learning Does Not Generalize Strongly Within the Same Domain, ICLR 2022 | Evaluation, implementation of most weakly-supervised baselines  |
| Zhu et al. | [Github Repo](https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch) | Commutative Lie Group VAE for Disentanglement Learning, ICML 2021 | Evaluation | 
| Yang et al. | [Github Repo](https://github.com/ThomasMrY/VCT) | Visual Concepts Tokenization, NeurIPS 2022 | Implementation of VCT |
| Du et al. | [Github Repo](https://github.com/yilundu/comet) | Unsupervised Learning of Compositional Energy Concepts, NeurIPS 2021 | Implementation of COMET |
| Locatello et al. | [Github Repo](https://github.com/google-research/disentanglement_lib/tree/master/disentanglement_lib) | Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations, ICML 2019 | Evaluation |

"""
The below code is adapted from the code for the paper 'Commutative Lie Group VAE for Disentanglement Learning' (ICML, 2021)
at: https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch/blob/main/metrics/utils.py

@inproceedings{Xinqi_liegroupvae_icml21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Commutative Lie Group VAE for Disentanglement Learning},
booktitle={ICML},
year={2021}
}
"""

import numpy as np
import sklearn
from sklearn.decomposition import PCA 
import torch

from src.data.datasets import DisLibDataset
from src.shared.constants import QUANTISED_FILLERS, SOFT_FILLERS, TPR_BINDINGS, FILLER_IDXS, Z_TPR, Z_SOFT_TPR

def get_pca(latents: np.array) -> torch.Tensor: 
    pca_list = [] 
    for i in range(latents.shape[1]): 
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(latents[:, i, :])
        pca_list.append(pca_result)
    pca_rep = np.concatenate(pca_list, axis=1)
    return pca_rep 

def generate_batch_factor_code(dataset, repn_fn, n_points, random_state, batch_size,
                               use_multidim_latents: bool=False, key: str=SOFT_FILLERS): 
    representations = None
    factors = None
    i = 0
    while i < n_points:
        num_points_iter = min(n_points - i, batch_size)
        current_factors, current_observations = \
            dataset.sample(num_points_iter, random_state, tensorify=True)
        current_factors = current_factors.cpu()
        current_observations = current_observations.contiguous()
        if i == 0:
            factors = current_factors.cpu()
            representations = repn_fn(current_observations, key).cpu() # (N_{R}, D_{F})
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations,
                                    repn_fn(
                                        current_observations, key).cpu()))
        i += num_points_iter
    if use_multidim_latents:                                        # (N_{B}, N_{R}, D_{F}) -> (N_{R}, N_{B}, D_{F})
        if key in [SOFT_FILLERS, QUANTISED_FILLERS, TPR_BINDINGS]:  # (N_{B}, N_{R}, D_{F}), (N_{B}, N_{R}, D_{F}*D_{R})
            representations = representations.transpose(1, 0, 2)
        if key in [FILLER_IDXS, Z_TPR, Z_SOFT_TPR]:                 # (N_{B}, N_{R}) or (N_{B}, D_{R}*D_{F})
            representations = representations.transpose(1, 0)
        return representations, np.transpose(factors) 
    return np.transpose(representations), np.transpose(factors)     # (N_{B}, n_latents) -> (n_latents, N_{B})

def sample_batch(repn_fn, n_points: int, dataset: DisLibDataset, random_state, key: str=QUANTISED_FILLERS): 
    """ 
    Returns: 
        latent_reps (B, latent_dim)
        factors (B, n_factors)
    """
    factors, imgs = dataset.sample(num=n_points, random_state=random_state, tensorify=True)
    i = 0 
    while i < n_points: 
        n_points_iter = min(n_points - i, 64)
        curr_imgs = imgs[i: i+n_points_iter]
        if i == 0: 
            representations = repn_fn(curr_imgs.contiguous(), key=key)
        else: 
            representation = repn_fn(curr_imgs.contiguous(), key=key)
            if isinstance(representation, torch.Tensor):
                representations = torch.vstack([
                    representations, repn_fn(curr_imgs.contiguous(), key=key)
                ])
            else: 
                representations = np.vstack((representations, 
                                             repn_fn(curr_imgs, key=key)))
        i += n_points_iter

    if isinstance(representations, torch.Tensor): 
        if len(representations.shape) > 2: 
            representations = representations.permute(1, 0, 2) # (N_{B}, N_{R}, D_{F}) -> (N_{R}, N_{B}, D_{F})
        else: 
            representations = representations.t()
        return np.array(representations.cpu()), np.array(factors.cpu().t()) 
    return np.transpose(representations), np.transpose(factors)


def histogram_discretize(target, num_bins=20):
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h

def normalize_data(data, mean=None, stddev=None):
    if mean is None:
        mean = np.mean(data, axis=1)
    if stddev is None:
        stddev = np.std(data, axis=1)
    return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev

import torch.nn as nn 
from src.logger import Logger 

def compute_dis_metrics(aggregator, repn_fn, step_idx: int, split_type: str,
                        wandb_logger: Logger, eval: bool=False):
    metric_outputs = aggregator(repn_fn, eval=eval) 
    wandb_logger.log_scalars({**metric_outputs, 
                              f'dmetric/{split_type}_step': step_idx},
                             prefix_to_append=f'dmetric/{split_type}/')
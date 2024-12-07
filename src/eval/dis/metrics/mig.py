"""
The below code is adapted from the code for the paper 'Commutative Lie Group VAE for Disentanglement Learning' (ICML, 2021)
at: https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch/blob/main/metrics/mig.py

@inproceedings{Xinqi_liegroupvae_icml21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Commutative Lie Group VAE for Disentanglement Learning},
booktitle={ICML},
year={2021}
}
"""

"""Implementation of MIG Metric.

Based on "Isolating Sources of Disentanglement in VAEs" .
Implementation based on https://github.com/google-research/disentanglement_lib
"""

import numpy as np
import sklearn
import logging 
from src.data.datasets import DisLibDataset
from src.eval.dis.metrics import utils

logging.basicConfig(level=logging.INFO)

def _histogram_discretize(target, num_bins=20):
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


class MigMetric:
    def __init__(self, ds: DisLibDataset, random_state, use_multidim_latents: bool=False, num_points=10000, bs: int=16):
        """ MIG Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to evaluate on
        """
        super().__init__()
        self.ds = ds
        self.bs = bs
        self.num_points = num_points
        self.random_state = random_state
        self.use_multidim_latents = use_multidim_latents


    def __call__(self, repn_fn, eval: bool=False):
        logging.info('...Generating batch code...')
        if not self.use_multidim_latents: 
            reps, facs = utils.generate_batch_factor_code(dataset=self.ds, repn_fn=repn_fn,
                                                      n_points=self.num_points, random_state=self.random_state, 
                                                      batch_size=self.bs,
                                                      use_multidim_latents=self.use_multidim_latents)
        else: 
            discretized, facs = utils.generate_batch_factor_code(dataset=self.ds, 
                                                                 repn_fn=repn_fn, 
                                                                 n_points=self.num_points, 
                                                                 random_state=self.random_state,
                                                                 batch_size=self.bs, 
                                                                 use_multidim_latents=self.use_multidim_latents, 
                                                                 key='filler_idxs')
        logging.info('****DONE****')
        logging.info('...Computing entropy....')
        if not self.use_multidim_latents:  
            discretized = _histogram_discretize(reps)
        m = discrete_mutual_info(discretized, facs)
        assert m.shape[0] == discretized.shape[0]
        assert m.shape[1] == facs.shape[0]
        entropy = discrete_entropy(facs)
        sorted_m = np.sort(m, axis=0)[::-1]
        logging.info('****DONE****')
        discrete_mig = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
        logging.info(f'Discrete MIG: {discrete_mig:3f}')

        return {'discrete_mig': discrete_mig}


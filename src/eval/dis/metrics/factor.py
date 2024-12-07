"""
The below code is adapted from the code for the paper 'Commutative Lie Group VAE for Disentanglement Learning' (ICML, 2021)
at: https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch/blob/main/metrics/factor.py

@inproceedings{Xinqi_liegroupvae_icml21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Commutative Lie Group VAE for Disentanglement Learning},
booktitle={ICML},
year={2021}
}
"""

#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: factor.py
# --- Creation Date: 16-01-2021
# --- Last Modified: Sat 12 Jun 2021 22:24:28 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Implementation of FactorVAE Metric.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).
Implementation based on https://github.com/google-research/disentanglement_lib
"""

import torch.linalg as LA 
import numpy as np
from src.data.datasets import DisLibDataset
import torch 
from tqdm import tqdm
from scipy import stats 
from src.shared.constants.models import *

class FactorVAEMetric:
    def __init__(self, dataset: DisLibDataset, 
                random_state1: np.random.RandomState,
                random_state2: np.random.RandomState, 
                use_multidim_latents: bool=False,
                use_discrete_repn: bool=False,
                n_train: int=10000, n_eval:int=5000, batch_size:int=64, n_var_est: int=10000):
        """ FactorVAE Metric

        Args:
            dataset (Dataset): torch dataset on which to evaluate
            n_train (int): Number of points to train on
            n_eval (int): Number of points to evaluate on
            batch_size (int): batch size
            n_var_est (int): Number of examples to estimate global variance.
        """
        super().__init__()
        self.dataset = dataset
        self.n_train = n_train
        self.n_eval = n_eval
        self.batch_size = batch_size
        self.n_var_est = 20000 if use_multidim_latents else n_var_est
        self.random_state1 = random_state1
        self.random_state2 = random_state2  
        self.n_factors = len(dataset.factor_sizes)
        self.use_multidim_latents = use_multidim_latents
        self.use_discrete_repn = use_discrete_repn


    def __call__(self, repn_fn, eval: bool=False):
        global_var = self._compute_variances(repn_fn)
        active_dims = self._prune_dims(global_var)
        scores_dict = {}

        if not active_dims.any():
            scores_dict["fac_train"] = 0.
            scores_dict["fac_eval"] = 0.
            scores_dict["fac_num_act"] = 0
            return scores_dict

        train_votes = self._get_train_votes(repn_fn, 64, self.n_train,
                                            global_var, active_dims, random_state=self.random_state1)
        print('Train votes:', train_votes)
        classifier = np.argmax(train_votes, axis=0)
        other_index = np.arange(train_votes.shape[1])
        train_accuracy = np.sum(
            train_votes[classifier, other_index]) * 1. / np.sum(train_votes)

        eval_votes = self._get_train_votes(repn_fn, 64, self.n_eval,
                                           global_var, active_dims, random_state=self.random_state2)
        print('Eval_votes:', eval_votes)
        eval_accuracy = np.sum(
            eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
        scores_dict["fac_train"] = train_accuracy
        scores_dict["fac_eval"] = eval_accuracy
        scores_dict["fac_num_act"] = active_dims.astype(int).sum()
        print(f'Factor scores {scores_dict}')
        return scores_dict

    def _get_train_votes(self, rep_fn, bs, num_points, global_var, active_dims, random_state):
        votes = np.zeros((self.n_factors, global_var.shape[0]),
                         dtype=np.int64)
        print(f'Shape of votes is {votes.shape}')
        for _ in tqdm(range(num_points)):
            factor_index, argmin = self._generate_training_sample(rep_fn, bs,
                                                             global_var, active_dims,
                                                             random_state)
            votes[factor_index, argmin] += 1
        return votes

    def _generate_training_sample(self, rep_fn, bs, global_var, active_dims, random_state):
        factor_index = random_state.randint(self.n_factors)
        # maybe it is better to use approx filler bindings or strict filler bindings
        factors = self.dataset.sample_factors(num=bs if self.use_discrete_repn else 6, 
                                              random_state=random_state)
        # Fix the selected factor across mini-batch.
        factors[:, factor_index] = factors[0, factor_index]
        # Obtain the observations.
        observations = self.dataset.sample_observations_from_factors(
            factors, random_state=random_state, tensorify=True)
        
        if self.use_multidim_latents: 
            if not self.use_discrete_repn:
                representations = rep_fn(observations.contiguous()) # (N_{B}, N_{R}, D_{F})
                local_variances = LA.vector_norm(
                    (representations-torch.mean(representations, dim=0))**2, 2, -1).cpu()
                argmin = torch.argmin(local_variances[:, active_dims] / global_var[active_dims], dim=-1)
            else: 
                local_vars = []
                representations = rep_fn(observations.contiguous(), key=FILLER_IDXS)
                for fac in range(representations.shape[1]): 
                    local_vars.append(self.get_discrete_variance(filler_idxs=representations[:, fac], 
                                                                 minlength=57))
                local_vars = np.array(local_vars)
                argmin = np.argmin(local_vars[active_dims]/global_var[active_dims])
        else: 
            representations = np.array(rep_fn(observations).cpu())
            local_variances = np.var(representations, axis=0, ddof=1, dtype=np.float64)
            argmin = np.argmin(local_variances[active_dims] /
                                global_var[active_dims])
        return factor_index, argmin

    def _prune_dims(self, variances, threshold=0.1):
        """Mask for dimensions collapsed to the prior."""
        print(f'Variances has shape {variances.shape}')
        scale_z = np.sqrt(variances)
        print(f'Scaled variances {scale_z}')
        return scale_z >= threshold
    
    def get_discrete_variance(self, filler_idxs: torch.Tensor, minlength: int): 
        pk = torch.bincount(filler_idxs.ravel().to(torch.int32))/filler_idxs.ravel().shape[0]
        mask = torch.where(pk != 0)[0]
        pk = np.array(pk[mask].cpu(), dtype=np.float64)
        xk_lim = torch.unique(filler_idxs.ravel()).shape[0]
        xk = np.arange(xk_lim)
        return stats.rv_discrete(values=(xk, pk)).var()
    
    def _compute_variances(self,
                        representation_function,
                        eval_batch_size: int=64):
        """Computes the variance for each dimension of the representation.
        We consider interpreting the deviation as the euclidean distance (equal to the trace of the covariance matrix)
        https://stats.stackexchange.com/questions/225434/a-measure-of-variance-from-the-covariance-matrix 

        Args:
            ground_truth_data: GroundTruthData to be sampled from.
            representation_function: Function that takes observation as input and
            outputs a representation/
            random_state: Numpy random state used for randomness.
            eval_batch_size: Batch size used to eval representation.

        Returns:
            Vector with the variance of each dimension.
        """
        observations = self.dataset.sample_observations_n_large(self.n_var_est, 
                                                                self.random_state1)
        representations = self._obtain_representation(observations,
                                                        representation_function,
                                                        eval_batch_size,
                                                        obtain_discretised_repn=self.use_discrete_repn)
        if self.use_multidim_latents:
            if not self.use_discrete_repn: 
                #cosine_sim = F.cosine_similarity(representations, torch.mean(representations, dim=0), dim=-1)
                #return np.array(cosine_sim.mean(dim=0).cpu(), dtype=np.float64)
                self.empiricial_mean = torch.mean(representations, dim=0)
                l2_dist = LA.vector_norm((representations - self.empiricial_mean)**2, 2, dim=-1) # (N_{B}, N_{R}, D_{F}) -> (N_{B}, N_{R})
                l2_dist = l2_dist.mean(dim=0)
                return np.array(l2_dist.cpu(), dtype=np.float64)
            else: 
                vars = []
                for fac in range(representations.shape[1]):
                    vars.append(self.get_discrete_variance(filler_idxs=representations[:, fac], 
                                                           minlength=57)) 
                return np.array(vars, dtype=np.float64)
        representations = np.transpose(np.array(representations))
        assert representations.shape[0] == self.n_var_est
        print(f'Representations shape is {representations.shape}, type is {type(representations)}')
        return np.var(representations, axis=0, ddof=1, dtype=np.float64)

    def _obtain_representation(self, obs, rep_fn, bs,
                               obtain_discretised_repn: bool=False):
        representations = None
        num_points = obs.shape[0]
        key = FILLER_IDXS if obtain_discretised_repn else SOFT_FILLERS
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, bs)
            current_observations = obs[i:i + num_points_iter]
            current_observations = torch.from_numpy(current_observations).cuda().contiguous()
            if i == 0:
                representations = rep_fn(current_observations, key)
            else:
                representations = torch.vstack([representations, 
                                                   rep_fn(current_observations, key)])
            i += num_points_iter
        if self.use_multidim_latents: 
            return representations 
        return np.transpose(representations.cpu())

"""
The below code is adapted from the code for the paper 'Commutative Lie Group VAE for Disentanglement Learning' (ICML, 2021)
at: https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch/blob/main/metrics/beta.py

@inproceedings{Xinqi_liegroupvae_icml21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Commutative Lie Group VAE for Disentanglement Learning},
booktitle={ICML},
year={2021}
}
"""

"""Implementation of BetVAE Metric.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" .
Implementation based on https://github.com/google-research/disentanglement_lib
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import linear_model
from tqdm import tqdm 

from src.data.datasets import DisLibDataset
import logging 

logging.basicConfig(level=logging.INFO)

class BetaVAEMetric:
    def __init__(self, ds: DisLibDataset, random_state1, random_state2, num_points=10000, 
                 use_multidim_latents: bool=False, bs=64):
        """ BetaVAE Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to evaluate on
            bs (int): batch size
            paired (bool): If True expect the dataset to output symmetry paired images
            fixed_shape (bool): If fix shape in dsprites.
        """
        super().__init__()
        self.ds = ds
        self.num_points = num_points
        self.bs = bs
        self.random_state1 = random_state1 
        self.random_state2 = random_state2
        self.use_multidim_latents = use_multidim_latents

    def _generate_training_sample(self, rep_fn):
        index = self.random_state1.randint(self.ds.num_factors)
        # Sample two mini batches of latent variables.
        factors1 = self.ds.sample_factors(self.bs, random_state=self.random_state1)
        factors2 = self.ds.sample_factors(self.bs, random_state=self.random_state2)
        # Ensure sampled coordinate is the same across pairs of samples.
        factors2[:, index] = factors1[:, index]
        # Transform latent variables to observation space.
        observation1 = self.ds.sample_observations_from_factors(
            factors1, random_state=self.random_state1)
        observation2 = self.ds.sample_observations_from_factors(
            factors2, random_state=self.random_state2)
        # Compute representations based on the observations.
        representation1 = rep_fn(torch.from_numpy(observation1.astype(np.float32)).cuda().contiguous()).cpu()
        representation2 = rep_fn(torch.from_numpy(observation2.astype(np.float32)).cuda().contiguous()).cpu()
        
        if self.use_multidim_latents: 
            feature_vector = F.cosine_similarity(representation1, representation2, dim=-1) # (N_{B}, N_{R}, D_{F}) -> (N_{B}, N_{R})
            feature_vector = np.mean(np.array(feature_vector, dtype=np.float64), axis=0) 
        else: 
            representation1 = np.array(representation1, dtype=np.float64)
            representation2 = np.array(representation2, dtype=np.float64)
            feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
        return index, feature_vector

    def _generate_training_batch(self, rep_fn, eval: bool=False):
        if eval: 
            n_points = 5000
        else: 
            n_points = self.num_points
        labels = np.zeros(n_points, dtype=np.int64)
        points = None  # Dimensionality depends on the representation function.

        for i in tqdm(range(n_points)):
            labels[i], feats = self._generate_training_sample(rep_fn)
            if points is None:
              points = np.zeros((n_points, feats.shape[0]))
            points[i, :] = feats
        return points, labels

    def __call__(self, repn_fn, eval: bool=False):
        logging.info("....Generating training batch....")
        train_points, train_labels = self._generate_training_batch(repn_fn, eval)
        logging.info("****Done*****")
        
        logging.info("....Fitting model....")
        regressor = linear_model.LogisticRegression(penalty=None, multi_class='multinomial', solver='newton-cg')
        regressor.fit(train_points, train_labels)
        logging.info("****Done*****")

        train_accuracy = regressor.score(train_points, train_labels)
        train_accuracy = np.mean(regressor.predict(train_points) == train_labels)

        eval_points, eval_labels = self._generate_training_batch(repn_fn, eval)
        eval_accuracy = regressor.score(eval_points, eval_labels)
        
        logging.info(f"****BETA TRAIN ACC: {train_accuracy:3f}. BETA VAL ACC: {eval_accuracy:3f}")

        return {'beta_train_acc': train_accuracy, 'beta_val_acc': eval_accuracy}

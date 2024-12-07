"""
The below code is adapted from the code for the paper 'Commutative Lie Group VAE for Disentanglement Learning' (ICML, 2021)
at: https://github.com/zhuxinqimac/CommutativeLieGroupVAE-Pytorch/blob/main/metrics/dci.py

@inproceedings{Xinqi_liegroupvae_icml21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Commutative Lie Group VAE for Disentanglement Learning},
booktitle={ICML},
year={2021}
}
"""

# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
Implementation based on https://github.com/google-research/disentanglement_lib

"""

import numpy as np
import scipy
import logging 
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier 
from src.eval.dis.metrics import utils

from src.data.datasets import DisLibDataset
from src.shared.constants.models import QUANTISED_FILLERS, FILLER_IDXS 
logging.basicConfig(level=logging.INFO)

class DciMetric:
    def __init__(self, dataset: DisLibDataset, 
                 random_state1, random_state2, use_multidim_latents: bool=False, 
                 use_discrete_repn: bool=False, n_points: int=1000):
        """ DCI Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to use in metric calculation
        """
        self.dataset = dataset
        self.n_points = n_points
        self.random_state1 = random_state1 
        self.random_state2 = random_state2 
        self.use_multidim_latents = use_multidim_latents
        self.use_discrete_repn = use_discrete_repn

    def __call__(self, repn_fn, eval: bool=False):
        if eval: 
            n_points = min(self.n_points, 2000) 
        else: 
            n_points = self.n_points
        logging.info('...Generating training batch....')
        latents_train, factors_train = utils.sample_batch(repn_fn, n_points, self.dataset,
                                                          random_state=self.random_state1,
                                                          key=FILLER_IDXS if self.use_discrete_repn else QUANTISED_FILLERS)
        logging.info('***DONE***')
        print(f'Latents shape {latents_train.shape}')
        assert latents_train.shape[1] == n_points
        assert factors_train.shape[1] == n_points
        logging.info('...Generating test batch....')
        latents_test, factors_test = utils.sample_batch(repn_fn, n_points, self.dataset,
                                                        random_state=self.random_state2,
                                                        key=FILLER_IDXS if self.use_discrete_repn else QUANTISED_FILLERS)
        logging.info('***DONE***')
        logging.info('...Computing DCI scores...')
        scores = _compute_dci(latents_train, factors_train, 
                              latents_test, factors_test, use_multidim_latents=self.use_multidim_latents,
                              use_discrete_repn=self.use_discrete_repn)
        logging.info('***DONE***')
        logging.info(f'Scores {scores}')
        return scores


def _compute_dci(latents_train, factors_train, latents_test, factors_test, use_multidim_latents: bool=False,
                 use_discrete_repn: bool=False):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(
      latents_train, factors_train, latents_test, factors_test, use_multidim_latents,
      use_discrete_repn=use_discrete_repn)
  print(f'Importance matrix is {importance_matrix}')
  assert importance_matrix.shape[0] == latents_train.shape[0]
  assert importance_matrix.shape[1] == factors_train.shape[0]
  scores["dci_informativeness_train"] = train_err
  scores["dci_informativeness_test"] = test_err
  scores["dci_disentanglement"] = disentanglement(importance_matrix)
  scores["dci_completeness"] = completeness(importance_matrix)
  return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test, use_multidim_latents,
                           use_discrete_repn: bool=False):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  batch_size = y_train.shape[1]
  
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  if use_multidim_latents:
      if not use_discrete_repn:  
        x_train = x_train.transpose(1, 0, 2).reshape(batch_size, -1) 
        x_test = x_test.transpose(1, 0, 2).reshape(batch_size, -1)
      else: 
         x_train = x_train.transpose(1, 0)
         x_test = x_test.transpose(1, 0)
  else: 
      x_train = x_train.T
      x_test = x_test.T 
  for i in range(num_factors):
    logging.info(f'...Fitting importance gbt for factor {i}/{num_factors}...')
    model = GradientBoostingClassifier() 
    model.fit(x_train, y_train[i, :])
    logging.info('***DONE***')
    if use_multidim_latents and not use_discrete_repn:                      # (N_{R}, N_{B}, D_{F}) -> (N_{B}, N_{R}, D_{F})
        for j in range(num_codes):
            splits = np.split(np.abs(model.feature_importances_), num_codes)
            print(f'Splits[j] has shape {splits[j].shape}, {len(splits)}')
            importance_matrix[j, i] = np.mean(splits[j], axis=0)
    else: 
        importance_matrix[:, i] = np.abs(model.feature_importances_) 

    train_loss.append(np.mean(model.predict(x_train) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test) == y_test[i, :]))
    logging.info(f'***DONE***')
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  print(f'Entropy of importance matrix is {scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])}')
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_code(importance_matrix):
  """Compute completeness of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)

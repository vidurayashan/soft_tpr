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

"""Tests for mig.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from src.eval.dis.metrics.tests import dummy_data
from src.eval.dis import mig
import numpy as np

class FakeModel(): 
    def __init__(self, repn_fn): 
        self.repn_fn = repn_fn

class MIGTest(absltest.TestCase):

  def test_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    mig_metric = mig.MigMetric(ds=ground_truth_data, random_state=random_state, 
                               num_points=3000, bs=16)
    scores = mig_metric(FakeModel(representation_function))
    self.assertBetween(scores["dmetric/discrete_mig"], 0.9, 1.0)

  def test_bad_metric(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = np.zeros_like
    random_state = np.random.RandomState(0)
    mig_metric = mig.MigMetric(ds=ground_truth_data, random_state=random_state, 
                               num_points=3000, bs=16)
    scores = mig_metric(FakeModel(representation_function))
    self.assertBetween(scores["dmetric/discrete_mig"], 0.0, 0.2)

  def test_duplicated_latent_space(self):
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    mig_metric = mig.MigMetric(ds=ground_truth_data, random_state=random_state, 
                               num_points=3000, bs=16)
    scores = mig_metric(FakeModel(representation_function))
    self.assertBetween(scores["dmetric/discrete_mig"], 0.0, 0.1)

if __name__ == "__main__":
  absltest.main()

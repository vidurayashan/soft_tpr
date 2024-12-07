"""
Implementation of the Shu model baseline uses the open-source repo provided by the authors': https://github.com/google-research/google-research/tree/master/weak_disentangle
The code below has been adapted from: https://github.com/google-research/google-research/blob/master/weak_disentangle/viz.py

@inproceedings{
Shu2020Weakly,
title={Weakly Supervised Disentanglement with Guarantees},
author={Rui Shu and Yining Chen and Abhishek Kumar and Stefano Ermon and Ben Poole},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJgSwyBKvr}
}
"""

# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Visualization utility functions."""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
import numpy as np
from scipy.stats import norm


def add_padding(images, pad):
    n, h, w, ch = images.shape
    new_images = np.ones((n, h + 2 * pad, w + 2 * pad, ch)) * 0.5
    for i in range(len(images)):
        new_images[i, pad:-pad, pad:-pad] = images[i]
    return new_images

def grid(images, num_cols, pad=1):
    if pad > 0:
        images = add_padding(images, pad)

    n, ch, h, w = images.shape
    c = num_cols
    r = n // num_cols
    images = images[:r * c].reshape(r, c, ch, h, w).transpose(2, 0, 3, 1, 4)
    images = images.reshape(ch, r * h, c * w)

    if ch == 3: return images
    if ch == 1: return images[Ellipsis, 0]

def generate(x1, x2, gen, z_dim, num_rows_per_set, num_cols):
    xs = [] 
    x = x1.detach().cpu().numpy()[:num_rows_per_set * num_cols]
    xs += [x]
    xs += [x[:num_cols] * 0 + 0.5]  # Black border

    x = x2.detach().cpu().numpy()[:num_rows_per_set * num_cols]
    xs += [x]
    xs += [x[:num_cols] * 0 + 0.5]  # Black border

    for dim in range(z_dim):
        for _ in range(num_rows_per_set):
            z = np.tile(np.random.randn(1, z_dim), (num_cols, 1)).astype(np.float32)
            z[:, dim] = norm.ppf(np.linspace(0.01, 0.99, num_cols))
            x = gen(torch.from_numpy(z).cuda()).detach().cpu().numpy()
            #print(f'X has shape {x.shape}')
            xs += [x]
            xs += [x * 0 + 0.5]
    del xs[-1]

    return np.concatenate(xs, 0)

def ablation_visualization(x1, x2, gen, z_dim):
    per_latent_trs = generate(x1, x2, gen, z_dim, 3, 12)
    return make_grid(torch.from_numpy(per_latent_trs))

import torch.nn as nn 
import torch 
from torchvision.utils import make_grid 

def decode(x: torch.Tensor, enc_repn_fn, generator: nn.Module) -> torch.Tensor: 
    with torch.no_grad():
        batch_end = x.shape[0] if x.shape[0] <= 25 else 25
        x = x[:batch_end]
        z = enc_repn_fn(x)
        x_hat = generator(z)
        pairs = torch.stack([x.detach().cpu(), x_hat.detach().cpu()], dim=1)
        g = make_grid(pairs.view(batch_end*2, *pairs.shape[2:]))
        return g 
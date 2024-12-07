"""
Implementation of the Shu model baseline uses the open-source repo provided by the authors': https://github.com/google-research/google-research/tree/master/weak_disentangle
The code below has been adapted from: https://github.com/google-research/google-research/blob/master/weak_disentangle/networks.py

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

"""Models."""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
# pylint: disable=g-wrong-blank-lines, missing-super-argument


import torch 
import torch.nn as nn
import torch.nn.functional as F 
from src.shared.components import View

class Encoder(nn.Module):
    def __init__(self, z_dim, n_channels: int, width: int=1, spectral_norm: bool=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 32*width, 4, 2, 1), # 62 -> 32
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32*width, 32*width, 4, 2, 1), # 32 -> 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(32*width, 64*width, 4, 2, 1), # 16 -> 8 
            nn.LeakyReLU(0.2),
            nn.Conv2d(64*width, 64*width, 4, 2, 1), # 8 -> 4
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4*4*64*width, 128*width), 
            nn.LeakyReLU(0.2),
            nn.Linear(128*width, 2 * z_dim)
            )

        if spectral_norm:
            self.net.apply(
                lambda x: nn.utils.spectral_norm(x) if isinstance(x, nn.Linear) else x)

        self.kwargs_for_loading = {
            'z_dim': z_dim, 
            'spectral_norm': spectral_norm,
            'n_channels': n_channels, 
            'width': width
        }

    def repn_fn(self, x: torch.Tensor, key: str=None): 
        return self(x)[0] # take first chunk as mean

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return torch.chunk(h, 2, dim=-1)
        
class LabelDiscriminator(nn.Module): 
    def __init__(self, n_channels: int, y_dim: int, width: int=1, 
              share_dense: bool=False, bias: bool=False):
        super().__init__() 
        self.y_dim = y_dim 
        self.body = [
            nn.Conv2d(n_channels, 32*width, 4, 2, 1), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32*width, 32*width, 4, 2, 1),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32*width, 64*width, 4, 2, 1),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(64*width, 64*width, 4, 2, 1),
            nn.LeakyReLU(0.2), 
            nn.Flatten() ]
        
        self.aux = [
            nn.Linear(4*4*64*width, 128*width),
            nn.LeakyReLU(0.2) ]

        if share_dense: 
            self.body.extend([nn.Linear(128*width, 128*width), 
                    nn.LeakyReLU])
            self.aux.extend([nn.Linear(128*width, 128*width), 
                   nn.LeakyReLU(0.2)])
            
        self.body = nn.Sequential(*self.body)
        self.aux = nn.Sequential(*self.aux)

        self.head = nn.Sequential(
            nn.Linear(128*width, 128*width), 
            nn.LeakyReLU(0.2), 
            nn.Linear(128*width, 128*width),
            nn.LeakyReLU(0.2),
            nn.Linear(128*width, 1, bias=bias)
        )

        self.kwargs_for_loading = {
            'n_channels': n_channels, 
            'y_dim': y_dim, 
            'width': width, 
            'share_dense': share_dense, 
            'bias': bias
        }

        # apply spectral norm to linear layers
        for m in (self.body, self.aux, self.head):
            m.apply(
                lambda x: nn.utils.spectral_norm(x) if isinstance(x, nn.Linear) else x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        hx = self.body(x) 
        hy = self.aux(y) 
        o = self.head(torch.concatenate((hx, hy), axis=-1))
        return o
    
class Discriminator(nn.Module): 
    def __init__(self, n_channels: int, y_dim: int, 
           add_weight_norm: bool=True, width: int=2, 
           share_dense: bool=True, uncond_bias: bool=False, 
           cond_bias: bool=False):
        super().__init__()  
        self.y_dim = y_dim 
        self.body = [
            nn.Conv2d(n_channels, 32*width, 4, 2, 1), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32*width, 32*width, 4, 2, 1),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(32*width, 64*width, 4, 2, 1),
            nn.LeakyReLU(0.2), 
            nn.Conv2d(64*width, 64*width, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        ]
        if share_dense: 
            self.body.extend([
                nn.Linear(4*4*64*width, 128*width),
                nn.LeakyReLU(0.2)])
        input_dim = 4*4*64*width if not share_dense else 128*width
        # only consider match mask type 
        input_dim *= 2
        self.neck = nn.Sequential(
            nn.Linear(input_dim, 128*width),
            nn.LeakyReLU(0.2), 
            nn.Linear(128*width, 128*width),
            nn.LeakyReLU(0.2) 
        )
        #print(f'Self neck is {self.neck}')
        self.head_uncond = nn.Linear(128*width, 1, bias=uncond_bias)
        self.head_cond = nn.Linear(y_dim, 128*width, bias=cond_bias)
        self.body = nn.Sequential(*self.body)
        for m in [self.body, self.neck, self.head_uncond]:
            m.apply(lambda x: nn.utils.spectral_norm(x) if isinstance(x, nn.Linear) else x)
        
        if add_weight_norm: 
            nn.utils.weight_norm(self.head_cond)
        self.kwargs_for_loading = {
            'n_channels': n_channels, 
            'y_dim': y_dim, 
            'add_weight_norm': add_weight_norm, 
            'width': width, 
            'share_dense': share_dense,
            'uncond_bias': uncond_bias, 
            'cond_bias': cond_bias,
        }
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        h = self.body(torch.concatenate((x1, x2), axis=0))
        (h1, h2) = torch.chunk(h, 2, dim=0)
        #print(f'Concatenation of h1, h2 has shape {torch.concatenate((h1, h2), axis=-1).shape}')
        h = self.neck(torch.concatenate((h1, h2), axis=-1))
        o_uncond = self.head_uncond(h)
        
        w = self.head_cond(F.one_hot(y.long(), self.y_dim).to(dtype=torch.float32).cuda().squeeze())
        o_cond = torch.sum(h*w, dim=-1, keepdim=True)
        return o_uncond + o_cond 

        
    def expose_encoder(self, x: torch.Tensor): 
        h = self.body(x) 
        _, z = torch.split(h, (1, self.y_dim), dim=-1)
        return z

class Generator(nn.Module): 
    def __init__(self, n_channels: int, z_dim: int, width: int, batch_norm: bool=True): 
        super().__init__() 
        self.kwargs_for_loading = {
            'n_channels': n_channels, 
            'z_dim': z_dim, 
            'width': width, 
            'batch_norm': batch_norm
        }
        if not batch_norm:
            self.net = nn.Sequential(
                nn.Linear(z_dim, 128*width),
                nn.ReLU(), 
                nn.Linear(128*width, 4*4*64*width),
                nn.ReLU(), 
                View((-1, 64*width, 4, 4)),
                nn.ConvTranspose2d(64*width, 64*width, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64*width, 32*width, 4, 2, 1),
                nn.LeakyReLU(0.2), 
                nn.ConvTranspose2d(32*width, 32*width, 4, 2, 1),
                nn.LeakyReLU(0.2), 
                nn.ConvTranspose2d(32*width, n_channels, 4, 2, 1),
                nn.Sigmoid()
            )
        else: 
            self.net = nn.Sequential(
                nn.Linear(z_dim, 128*width),
                nn.ReLU(), 
                nn.Linear(128*width, 4*4*64*width),
                nn.ReLU(), 
                View((-1, 64*width, 4, 4)),
                nn.ConvTranspose2d(64*width, 64*width, 4, 2, 1),
                nn.BatchNorm2d(64*width),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64*width, 32*width, 4, 2, 1),
                nn.BatchNorm2d(32*width),
                nn.LeakyReLU(0.2), 
                nn.ConvTranspose2d(32*width, 32*width, 4, 2, 1),
                nn.BatchNorm2d(32*width),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32*width, n_channels, 4, 2, 1),
                nn.Sigmoid()
            )

    def forward(self, z): 
        return self.net(z)
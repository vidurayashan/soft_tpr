"""
Implementation of the Shu model baseline uses the open-source repo provided by the authors': https://github.com/google-research/google-research/tree/master/weak_disentangle
The code below has been adapted from: https://github.com/google-research/google-research/blob/master/weak_disentangle/datasets.py

@inproceedings{
Shu2020Weakly,
title={Weakly Supervised Disentanglement with Guarantees},
author={Rui Shu and Yining Chen and Abhishek Kumar and Stefano Ermon and Ben Poole},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJgSwyBKvr}
}
"""

import numpy as np 
import torch 

def make_masks(n_factors: int): 
    masks = torch.eye(n_factors, dtype=torch.float32)
    return masks.cuda()

def paired_randn(batch_size: int, z_dim: int, masks: np.array) -> np.array: 
    return match_randn(batch_size, z_dim, masks)

def match_randn(batch_size, z_dim, masks):
    # Note that masks.shape[-1] = s_dim and we assume s_dim <= z-dim
    n_dim = z_dim - masks.shape[-1]

    assert n_dim == 0, f'N dim {n_dim} != 0'
    z1 = torch.randn((batch_size, z_dim)).cuda()
    z2 = torch.randn((batch_size, z_dim)).cuda() 

    mask_idx = torch.randint(low=0, high=len(masks), size=(batch_size, 1), dtype=torch.int32).cuda()
    mask = torch.gather(masks, 0, mask_idx.to(dtype=torch.int64))
    z2 = z2 * mask + z1 * (1 - mask)

    return z1, z2, mask_idx
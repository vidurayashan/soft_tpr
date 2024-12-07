"""
This code has been adapted from the supplementary material linked to the paper 
'Visual Representation Learning Does Not Generalize Strongly Within the Same Domain' (ICLR 2022) 
accessible at https://openreview.net/forum?id=9RUHPlladgh

@inproceedings{
schott2022visual,
title={Visual Representation Learning Does Not Generalize Strongly Within the Same Domain},
author={Lukas Schott and Julius Von K{\"u}gelgen and Frederik Tr{\"a}uble and Peter Vincent Gehler and Chris Russell and Matthias Bethge and Bernhard Sch{\"o}lkopf and Francesco Locatello and Wieland Brendel},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=9RUHPlladgh}
}
"""

from typing import List

import torch

def log_metrics_averaged_over_epoch(nb_metrics: List[torch.Tensor], name: str,  
                wandb_logger=None, prefix_to_append: str=None): 
    epoch_metric = torch.stack(nb_metrics, dim=0).mean(dim=0)
    if wandb_logger is not None: 
        wandb_logger.log_scalars({name: epoch_metric}, prefix_to_append)

def collect_per_factor(per_factors, name: str,
                       factor_names: List[str],
                       aggregate_fct=torch.mean, wandb_logger=None, 
                       prefix_to_append: str=None):
    per_factor = torch.stack(per_factors, dim=0).mean(dim=0)
    if 'excl_cat' not in name: 
        split_prefix = prefix_to_append.split('/') 
        prefix_to_append_factor = '/'.join([split_prefix[0], 'factor_wise', split_prefix[1]])
        for factor_i, factor_name in zip(per_factor, factor_names):
            if wandb_logger is not None: 
                wandb_logger.log_scalars({f'{name}_{factor_name}': factor_i},
                                        prefix_to_append=prefix_to_append_factor)
    if wandb_logger is not None: 
        wandb_logger.log_scalars({f'{name}': aggregate_fct(per_factor)},
                                     prefix_to_append=prefix_to_append)
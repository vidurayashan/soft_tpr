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

import torch

from tqdm import tqdm 
from src.data.datasets import DisLibDataset
from src.eval.fov_regression.optim import RSquared
from src.eval.fov_regression.models import ConcatModels
from src.eval.fov_regression.utils import collect_per_factor, log_metrics_averaged_over_epoch

def eval_clf(model: ConcatModels, data_loader, factor_sizes, mode='test',
                   wandb_logger=None, log_name_prefix: str=''): 
    model.eval()
    log = {'comb_acc': [], 'factor_acc': [],
           'cat_comb_acc': [], 'cat_factor_acc': [],
           'loss': []}

    with torch.no_grad():
        for iteration, (batch, targets) in tqdm(enumerate(data_loader)):
            batch = batch.cuda()
            targets = DisLibDataset.convert_to_one_hot(targets, factor_sizes).cuda()
            clf_out = model.forward_clf(batch, targets)
            predictions = clf_out['state']['preds']
            loss = clf_out['loss']['unweighted_loss']
            comb_acc = torch.mean((predictions == targets).all(dim=1).to(torch.float16))
            factor_acc = torch.mean((predictions == targets).to(torch.float16))

            # bookkeeping
            log['comb_acc'].append(comb_acc.detach())
            log['factor_acc'].append(factor_acc.detach())
            log['loss'].append(loss.detach())
            if wandb_logger is not None: 
                wandb_logger.log_scalars({'comb_acc': comb_acc.detach(), 
                                          'factor_acc': factor_acc.detach(),
                                          'step': iteration}, 
                prefix_to_append=f'clf_mlp/{mode}')

    log_metrics_averaged_over_epoch(log['comb_acc'], f'{log_name_prefix}_final_comb_acc', 
                                    wandb_logger=wandb_logger, 
                                    prefix_to_append=f'clf_mlp/{mode}')
    log_metrics_averaged_over_epoch(log['factor_acc'], f'{log_name_prefix}_final_factor_acc',
                                    wandb_logger=wandb_logger, 
                                    prefix_to_append=f'clf_mlp/{mode}')
    log_metrics_averaged_over_epoch(log['loss'], f'{log_name_prefix}_final_loss',
                                    wandb_logger=wandb_logger, 
                                    prefix_to_append=f'clf_mlp/{mode}')
    final_comb_acc, final_fac_acc, final_loss = tuple(map(lambda x: torch.stack(x, dim=0).mean(dim=0), [log['comb_acc'], 
                                                                                                  log['factor_acc'],
                                                                                                  log['loss']]))
    

    return final_comb_acc, final_fac_acc, final_loss


def eval_regressor(model, data_loader, factor_names, categorical_factors, rsquared: RSquared=None,
               mode='test', wandb_logger=None,
               log_name_prefix: str=''):
    model.eval()
    log = {'rsquared': [], 'mse': [], 'rsquared_excl_cat': [], 'mse_excl_cat': []}
    with torch.no_grad():
        for _, (batch, targets) in enumerate(data_loader):
            batch = batch.cuda()
            targets = targets.cuda()
            predictions = model(batch)
            squared_diff = (targets - predictions).pow(2)
            r_squared_per_factor = rsquared(predictions, targets)

            # bookkeeping
            log['rsquared'].append(r_squared_per_factor.detach())
            r_squared_per_factor_exc = r_squared_per_factor[~categorical_factors]
            log['rsquared_excl_cat'].append(r_squared_per_factor_exc.detach())
            mse = squared_diff.mean(dim=0)
            log['mse'].append(mse.detach())
            mse_exc = squared_diff[:, ~categorical_factors].mean(dim=0)
            log['mse_excl_cat'].append(mse_exc.detach())

    collect_per_factor(log['rsquared'], name=f'{log_name_prefix}_rsquared',
                       factor_names=factor_names, wandb_logger=wandb_logger,
                       prefix_to_append=f'reg_mlp/{mode}')
    collect_per_factor(log['mse'], name=f'{log_name_prefix}_mse',
                       factor_names=factor_names, 
                       aggregate_fct=torch.sum, wandb_logger=wandb_logger, 
                       prefix_to_append=f'reg_mlp/{mode}')
    
    factor_names_exc_categorical = [fac_name for i, fac_name in enumerate(factor_names) if categorical_factors[i] != True]

    collect_per_factor(log['rsquared_excl_cat'], name=f'{log_name_prefix}_rsquared_excl_cat',
                       factor_names=factor_names_exc_categorical, 
                       wandb_logger=wandb_logger, prefix_to_append=f'reg_mlp/{mode}')
    collect_per_factor(log['mse_excl_cat'], f'{log_name_prefix}_mse_excl_cat',
                       factor_names=factor_names_exc_categorical, 
                       aggregate_fct=torch.sum,
                       wandb_logger=wandb_logger, prefix_to_append=f'reg_mlp/{mode}')

    final_rsq, final_rsq_exc = tuple(map(lambda x: torch.mean(torch.stack(x, dim=0)), 
                                         [log['rsquared'], log['rsquared_excl_cat']]))
    final_mse, final_mse_exc = tuple(map(lambda x: torch.mean(torch.stack(x, dim=0), dim=0).sum(), 
                                         [log['mse'], log['mse_excl_cat']]))
    
    return final_rsq, final_mse, final_rsq_exc, final_mse_exc

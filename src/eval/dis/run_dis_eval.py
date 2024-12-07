import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{path}/src/')
print(f'{path}/src/')

import numpy as np
import argparse
import torch

from src.data.datasets import get_dataset
from src.eval.dis.aggregator import MetricAggregator
from src.shared.constants import *
from src.logger.logger import Logger
from src.data import get_dataloaders
from src.logger.load import load_model_from_path, load_corresponding_args_from_path

import logging 

logging.basicConfig(level=logging.INFO)

TO_KEEP = [
    'no_wandb',
    'wandb_proj_name',
    'wandb_run_name',
    'no_factor',
    'no_mig',
    'no_beta', 
    'no_dci',
    'load_dir',
    'discretise_soft_tpr_repn',
    'data_dir',
    'saved_iter',
]

def populate_args_with_orig_args(args, orig_args): 
    args_dict = vars(args)
    orig_args_dict = vars(orig_args)
    for k in orig_args_dict.keys(): 
        if k not in TO_KEEP: 
            args_dict[k] = orig_args_dict[k]
    
    return argparse.Namespace(**args_dict)

def set_wandb_run_name(args, orig_args): 
    if orig_args.model == SOFT_TPR_AE:
        args.wandb_run_name = f'{args.saved_iter}_{orig_args.model}_{args.repn_fn_key}'
    elif orig_args.model in WS_SCALAR_BASELINES: 
        args.wandb_run_name = f'{args.saved_iter}_{orig_args.model}_{args.repn_fn_key}'
    else: 
        raise NotImplementedError(f'Have not yet implemented run name for {orig_args.model}')


def main(args):

    if not torch.cuda.is_available():
        raise Exception("Cuda not available!!!")

    logging.info(f'****LOADING MODEL FROM PATH {args.load_dir}')
    model = load_model_from_path(model_path=args.load_dir)
    orig_args = load_corresponding_args_from_path(model_path=args.load_dir)
    saved_iter = args.load_dir.split('iter_')[1].split('_')[0]
    args.saved_iter = int(saved_iter)

    dataset, number_factors, number_channels, test_ratio_per_factor = get_dataset(orig_args.dataset,
                                                                                      args.data_dir)
    dataloader_train, dataloader_test, dataloader_full = get_dataloaders(orig_args, full_dataset=dataset)


    args = populate_args_with_orig_args(args=args, orig_args=orig_args)
    print(f'Args.seed is {args.seed}')
    logger = Logger(args, model)

    model.eval() 
    if args.use_all: 
        args.use_factor = True 
        args.use_beta = True 
        args.use_dci = True 
        args.use_mig = True 

    aggregator = MetricAggregator(val_dataset=dataloader_full.dataset, 
                                  n_points=10000, 
                                  random_state1=np.random.RandomState(args.seed), 
                                  random_state2=np.random.RandomState(args.seed+10),
                                  verbose=True, 
                                  use_multidim_latents=(args.model == SOFT_TPR_AE),
                                  use_beta=args.use_beta, 
                                  use_factor=args.use_factor,
                                  use_dci=args.use_dci,
                                  use_mig=args.use_mig,
                                  discretise_soft_tpr_repn=args.discretise_soft_tpr_repn)
    model.eval() 
    metric_outputs = aggregator(model.repn_fn) 
    logger.log_scalars({**metric_outputs, 
                              f'dmetric/train_step': args.n_iters},
                             prefix_to_append=f'dmetric/')

    

def parse_args():
    parser = argparse.ArgumentParser(description='')
    # loading 
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--data_dir', default='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets/', type=str)
    parser.add_argument('--gadi', action='store_true')

    # logging
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_proj_name', type=str, default='compute_dis_metrics_convergence')
    parser.add_argument('--wandb_run_name', default=None, type=str)

    # downstream evaluation 
    parser.add_argument('--use_factor', action='store_true')
    parser.add_argument('--use_mig', action='store_true')
    parser.add_argument('--use_beta', action='store_true')
    parser.add_argument('--use_dci', action='store_true')
    parser.add_argument('--use_all', action='store_true')
    parser.add_argument('--discretise_soft_tpr_repn', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

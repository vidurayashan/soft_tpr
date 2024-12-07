import numpy as np
import argparse
import torch

from src.data.datasets import get_dataset
from src.data.utils import get_dataloaders
from src.eval.dis.aggregator import MetricAggregator
from src.shared.constants import *
from src.logger.logger import Logger
from src.eval.dis import compute_dis_metrics
from src.repn_learners.baselines.comet.utils import load_comet, get_latents
from src.repn_learners.baselines.comet.models import LatentEBM
from sklearn.decomposition import PCA

import logging 

logging.basicConfig(level=logging.INFO)

TO_KEEP = [
    'wandb_proj_name',
    'wandb_run_name',
    'no_factor',
    'no_mig',
    'no_beta', 
    'no_dci',
    'checkpoint_dir',
    'discretise_soft_tpr_repn',
    'data_dir',
    'save_dir',
    'gadi'
]


def set_defaults(args): 
    args.lr = 1e-4
    args.repn_fn_key = QUANTISED_FILLERS_CONCATENATED 
    args.components = 10
    args.latent_dim = 64 # keep as this quantity for loading COMET
    args.filter_dim = 64
    args.pos_embed = False 
    args.recurrent_model = False 
    args.model = COMET
    args.supervision_mode = UNSUPERVISED 
    args.modification = None 
    args.n_workers = 8
    args.ensembles = args.components
    args.n_iters = 200000
    args.hidden_dim = 64
    args.use_embed_layer = False
    
    
def populate_args_with_orig_args(args, orig_args): 
    args_dict = vars(args)
    orig_args_dict = vars(orig_args)
    for k in orig_args_dict.keys(): 
        if k not in TO_KEEP: 
            args_dict[k] = orig_args_dict[k]
    
    return argparse.Namespace(**args_dict)


def get_latents_for_dis_metric(args, x: torch.Tensor, model: LatentEBM) -> torch.Tensor: 
    latents = get_latents(args=args, proj_mat=None, model=model, 
                          x=x, reshape=False)
    if len(latents.shape) != 3: 
        latents = latents.view(latents.shape[0], 10, 64).cpu()
    pca_list = []
    for i in range(latents.shape[1]): 
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(latents[:, i, :].numpy())
        pca_list.append(pca_result)
        pca_rep = np.concatenate(pca_list, axis=1)
    return torch.from_numpy(pca_rep)
    
def main(args):

    if not torch.cuda.is_available():
        raise Exception("Cuda not available!!!")

    logging.info(f'****LOADING MODEL FROM PATH {args.checkpoint_dir}')
    comet = load_comet(args)
    logging.info(f'****COMET LOADED****')
    
    saved_iter = args.checkpoint_dir.split('model_')[1].split('.')[0]
    args.saved_iter = int(saved_iter)
    

    dataset, number_factors, number_channels, test_ratio_per_factor = get_dataset(args.dataset,
                                                                                      args.data_dir)
    args.test_ratio_per_factor = test_ratio_per_factor
    dataloader_train, dataloader_test, dataloader_full = get_dataloaders(args, full_dataset=dataset)
    args.gadi = True
    logger = Logger(args, None)
    

    model = comet[0]
    model.eval() 
    repn_fn = lambda x, key=None, args=args, model=model: get_latents_for_dis_metric(args=args, x=x, model=model)
    if args.use_all: 
        args.use_factor = True 
        args.use_beta = True 
        args.use_dci = True 
        args.use_mig = True 
    print(f'About to create metric aggregator....')
    aggregator = MetricAggregator(val_dataset=dataloader_full.dataset, 
                                  n_points=10000, 
                                  random_state1=np.random.RandomState(args.seed), 
                                  random_state2=np.random.RandomState(args.seed+10),
                                  verbose=True, 
                                  use_multidim_latents=False,                   
                                  use_beta=args.use_beta,                       # note that COMET and VCT, as per their respective papers, do NOT
                                  use_factor=args.use_factor,                   # use multidimensional latents to evaluate dis metrics, and instead use 
                                  use_dci=args.use_dci,                         # their own methods to accommodate dis metric eval for their vector-valued symbolic comp representations.
                                  use_mig=args.use_mig,                         
                                  discretise_soft_tpr_repn=False)
    
    compute_dis_metrics(aggregator=aggregator, 
                        repn_fn=repn_fn, 
                        step_idx=args.saved_iter, 
                        split_type='train', 
                        wandb_logger=logger,
                        eval=False)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    # loading 
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--data_dir', default='/g/data/po67/anonymous_cat/Data', type=str)
    parser.add_argument('--gadi', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--batch_size", type=int, default=64,
                help="config settings")
    parser.add_argument('--seed', type=int, default=123)

    # logging
    parser.add_argument('--wandb_proj_name', type=str, default='compute_dis_metrics_convergence')
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_log_dir', default='/g/data/po67/anonymous_cat/wandb_logs/dis_convergence')
    parser.add_argument('--save_dir', default='/g/data/po67/anonymous_cat/tmp/')
    parser.add_argument('--no_wandb', action='store_true')
    
    # downstream evaluation 
    parser.add_argument('--use_factor', action='store_true')
    parser.add_argument('--use_mig', action='store_true')
    parser.add_argument('--use_beta', action='store_true')
    parser.add_argument('--use_dci', action='store_true')
    parser.add_argument('--use_all', action='store_true')
    parser.add_argument('--discretise_soft_tpr_repn', action='store_true')

    args = parser.parse_args()
    set_defaults(args)
    args.use_all = True
    args.gadi = True
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

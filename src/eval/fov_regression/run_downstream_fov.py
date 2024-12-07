import numpy as np
import argparse
import random 
import logging 

import torch
from torch.utils.data import DataLoader

from src.data import get_dataset, get_dataloaders
from src.shared.constants import *
from src.logger.logger import Logger
from src.logger.load import load_model_from_path, load_corresponding_args_from_path
from src.eval.fov_regression.optim.train import ReadOutMLP, ConcatModels
from src.eval.fov_regression.optim import RSquared
from src.eval.fov_regression.eval import eval_regressor, eval_clf
from src.eval.fov_regression.optim.train import train_mlp_on_readout, train_clf_on_readout
from src.eval.fov_regression.models import BaseClf, ModularClf
from src.shared.components import EmbedLayer

logging.basicConfig(level=logging.INFO)    

TO_KEEP = [
    'no_wandb',
    'wandb_proj_name',
    'load_dir',
    'repn_fn_key', 
    'lr', 
    'n_epochs', 
    'patience', 
    'min_delta', 
    'n_test',
    'data_dir',
    'repn_fn_key',
    'wandb_run_name',
    'saved_iter',
    'use_embed_layer',
    'desired_output_dim'
]

def populate_args_with_orig_args(args, orig_args): 
    args_dict = vars(args)
    orig_args_dict = vars(orig_args)
    for k in orig_args_dict.keys(): 
        if k not in TO_KEEP: 
            args_dict[k] = orig_args_dict[k]
    return argparse.Namespace(**args_dict)

def set_defaults(args): 
    if args.model == SOFT_TPR_AE: 
        if args.repn_fn_key in [SOFT_FILLERS_CONCATENATED, QUANTISED_FILLERS_CONCATENATED]: 
            args.latent_dim = args.n_roles * args.filler_embed_dim 
        if args.repn_fn_key == FILLER_IDXS: 
            args.latent_dim = args.n_roles 
        if args.repn_fn_key in [Z_SOFT_TPR, Z_TPR]: 
            args.latent_dim = args.filler_embed_dim * args.role_embed_dim
        if args.repn_fn_key == TPR_BINDINGS_FLATTENED: 
            args.latent_dim = args.n_roles*args.filler_embed_dim*args.role_embed_dim
    if args.model in BASELINES:
        args.repn_fn_key = QUANTISED_FILLERS_CONCATENATED 
    if args.model in WS_SCALAR_BASELINES: 
        if args.use_embed_layer: 
            args.latent_dim = args.desired_output_dim

def set_wandb_run_name(args, orig_args): 
    if orig_args.model == SOFT_TPR_AE:
        args.wandb_run_name = f'{args.saved_iter}_{orig_args.model}_{args.repn_fn_key}'
    elif orig_args.model in WS_SCALAR_BASELINES: 
        wandb_run_name = f'{args.saved_iter}_{orig_args.model}_{args.repn_fn_key}'
        if args.use_embed_layer: 
            wandb_run_name += f'-embedding_layer_output_dim-{args.desired_output_dim}'
        args.wandb_run_name = wandb_run_name
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
    print(f'Saved iter is {args.saved_iter}')
    set_wandb_run_name(args, orig_args)

    if orig_args.model in WS_SCALAR_BASELINES: 
        if args.use_embed_layer: 
            embed_layer = EmbedLayer(output_dim=args.desired_output_dim,
                                     latent_dim=orig_args.latent_dim)
            model.eval()
            model = ConcatModels(repn_fn_first_model=model.repn_fn, second_model=embed_layer).cuda()        

    model.eval()
    dataset, number_factors, number_channels, test_ratio_per_factor = get_dataset(orig_args.dataset,
                                                                                      args.data_dir)
    dataloader_train, dataloader_test, dataloader_full = get_dataloaders(orig_args, full_dataset=dataset)

    args = populate_args_with_orig_args(args=args, orig_args=orig_args)
    set_defaults(args)
    if args.repn_fn_key in [QUANTISED_FILLERS_CONCATENATED, SOFT_FILLERS_CONCATENATED, TPR_BINDINGS_FLATTENED, Z_TPR, Z_SOFT_TPR]:
        repn_fn = lambda x, model=model, args=args: model.repn_fn(x, key=args.repn_fn_key)
    if args.repn_fn_key == FILLER_IDXS: 
        repn_fn = lambda x, model=model, args=args: model.repn_fn(x, key=args.repn_fn_key).to(torch.float32)
    logger = Logger(args, model)
    print(f'Latent dim {args.latent_dim}, seed {args.seed}, model {args.model}\nAll args {args}')
    print(f'Saved iter {args.saved_iter}, file name {args.load_dir}')

    model.eval() 
    if orig_args.model in WS_SCALAR_BASELINES and args.use_embed_layer: 
            latent_dim = embed_layer.output_dim 
    else: 
        latent_dim = args.latent_dim 

    labels_01 = dataset.get_normalized_labels() 
    r_sq = RSquared(labels_01, device='cuda')
    
    
    if args.dataset == CARS3D_DATASET:
        dataset_10000, val_dataset, test_dataset, dataset_1000, _ = torch.utils.data.random_split(dataset, 
                                                                             lengths=[10000, 5000, 1000, 1000, 
                                                                                      len(dataset)-(17000)])
        dataset_500, dataset_250, dataset_100, _ = torch.utils.data.random_split(dataset, lengths=[500, 250, 100, 
                                                                    len(dataset)-(850)])
    else: 
        dataset_10000, val_dataset, test_dataset, dataset_1000, dataset_500, dataset_250, dataset_100, _ = torch.utils.data.random_split(dataset, 
                                                                             lengths=[10000, 5000, 1000, 1000, 500, 250, 100, 
                                                                                      len(dataset)-(17850)])
    train_loader_10000_samples =  DataLoader(dataset_10000, batch_size=args.batch_size, 
                                             num_workers=args.n_workers, shuffle=True)
    train_loader_1000_samples = DataLoader(dataset_1000, batch_size=args.batch_size, 
                                             num_workers=args.n_workers, shuffle=True)
    train_loader_500_samples = DataLoader(dataset_500, batch_size=args.batch_size, 
                                             num_workers=args.n_workers, shuffle=True)
    train_loader_250_samples = DataLoader(dataset_250, batch_size=args.batch_size, 
                                          num_workers=args.n_workers, shuffle=True)
    train_loader_100_samples = DataLoader(dataset_100, batch_size=args.batch_size, 
                                             num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                             num_workers=args.n_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             num_workers=args.n_workers, shuffle=True)
    
    concat_trained_models = [] 

    prefixes = [] 
    models = []
    train_loaders = [] 
    patiences = [] 
    n_epochs = []
    
    if args.no_randomisation: 
        d1 = 256
        d2 = 256
        d3 = 256 
    else: 
        random.seed(args.seed)
        d1 = random.choice([256, 512])
        d2 = random.choice([256, 512])
        d3 = random.choice([128, 256])

    for n_samples_of_interest in args.n_sample_list.split(','):
        if n_samples_of_interest == 'full': 
            prefixes.append('full')
            models.append(ReadOutMLP(in_features=args.latent_dim, out_features=dataset.num_factors,
                                     d1=d1, d2=d2, d3=d3).cuda())
            train_loaders.append(dataloader_full)
            n_epochs.append(args.n_epochs)
            patiences.append(args.patience)
        if n_samples_of_interest == '10000': 
            prefixes.append('10000')
            models.append(ReadOutMLP(in_features=args.latent_dim, out_features=dataset.num_factors,
                                     d1=d1, d2=d2, d3=d3).cuda())
            train_loaders.append(train_loader_10000_samples) 
            patiences.append(args.patience + 2**1)
            n_epochs.append(args.n_epochs + 1*10)
        if n_samples_of_interest == '1000': 
            prefixes.append('1000')
            models.append(ReadOutMLP(in_features=args.latent_dim, out_features=dataset.num_factors,
                                     d1=d1, d2=d2, d3=d3).cuda())
            train_loaders.append(train_loader_1000_samples)
            patiences.append(args.patience + 2**2)
            n_epochs.append(args.n_epochs + 2*10)
        if n_samples_of_interest == '500': 
            prefixes.append('500')
            models.append(ReadOutMLP(in_features=args.latent_dim, out_features=dataset.num_factors,
                                     d1=d1, d2=d2, d3=d3).cuda())
            train_loaders.append(train_loader_500_samples)
            patiences.append(args.patience + 2**3)
            n_epochs.append(args.n_epochs + 3*10)
        if n_samples_of_interest == '250': 
            prefixes.append('250')
            models.append(ReadOutMLP(in_features=args.latent_dim, out_features=dataset.num_factors,
                                     d1=d1, d2=d2, d3=d3).cuda())
            train_loaders.append(train_loader_250_samples)
            patiences.append(args.patience + 2**3)
            n_epochs.append(args.n_epochs + 4*10)
        if n_samples_of_interest == '100':
            prefixes.append('100')
            models.append(ReadOutMLP(in_features=args.latent_dim, out_features=dataset.num_factors,
                                     d1=d1, d2=d2, d3=d3).cuda())
            train_loaders.append(train_loader_100_samples)
            patiences.append(args.patience + 2**3)
            n_epochs.append(args.n_epochs + 4*10)

    for i, (prefix, model, train_loader) in enumerate(zip(prefixes, models, train_loaders)): 
        concat_trained_models.append((prefix, train_mlp_on_readout(args,
                                            patience=patiences[i], 
                                            min_delta=args.min_delta, 
                                            repn_fn=repn_fn, 
                                            readout_model=model, 
                                            train_loader=train_loader, 
                                            val_loader=val_loader, 
                                            lr=args.lr, 
                                            n_epochs=n_epochs[i],
                                            wandb_logger=logger,
                                            log_prefix=prefix)))
        
    final_rsqs = [] 
    final_mses = []
    final_rsqs_exc = [] 
    final_mses_exc = [] 
    for prefix, concat_trained in concat_trained_models:
        final_rsq, final_mse, final_rsq_exc, final_mse_exc = eval_regressor(model=concat_trained, data_loader=test_loader, 
                                                                        factor_names=dataset.factor_names,
                                                                        categorical_factors=dataset.categorical,
                                                                        rsquared=r_sq, 
                                                                        mode='test', 
                                                                        wandb_logger=logger,
                                                                        log_name_prefix=prefix)
        final_rsqs.append((f'{prefix}_final_rsq', final_rsq))
        final_mses.append((f'{prefix}_final_mse', final_mse))
        final_rsqs_exc.append((f'{prefix}_final_rsq_exc', final_rsq_exc))
        final_mses_exc.append((f'{prefix}_final_mse_exc', final_mse_exc))
    # take combinations, only consider metric of model trained on j samples / metric of model trained w i samples where j > i
    ratios = [] 
    for final_scores in [final_rsqs, final_mses, final_rsqs_exc, final_mses_exc]:
        for i, (prefix_i, final_score_i) in enumerate(final_scores): # assume ordered
            for j, (prefix_j, final_score_j) in enumerate(final_scores): 
                if i <= j: 
                    ratios.append((f'{prefix_j}/{prefix_i}', final_score_j/final_score_i))
    print(f'Ratios {ratios}')
    for (prefix, score) in ratios: 
        logger.log_scalars({prefix: score}, prefix_to_append='reg_mlp/diff/')


def parse_args():
    parser = argparse.ArgumentParser(description='')
    # loading 
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--gadi', action='store_true')
    parser.add_argument('--data_dir', default='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets/', type=str)
    parser.add_argument('--n_sample_list', type=str, default="full,10000,1000,500,250,100")

    # downstream regressor model 
    parser.add_argument('--seed', type=int, default=987, help='Random seed used to sample layer dimensions for regressor')
    parser.add_argument('--no_randomisation', action='store_true', help='If specified, we do not randomise the layer dimensions for regressor')

    # representation function type
    parser.add_argument('--repn_fn_key', 
                        default=QUANTISED_FILLERS_CONCATENATED, 
                        choices=[FILLER_IDXS, 
                                 SOFT_FILLERS_CONCATENATED, 
                                 QUANTISED_FILLERS_CONCATENATED, 
                                 TPR_BINDINGS_FLATTENED, 
                                 Z_TPR, 
                                 Z_SOFT_TPR])
    
    parser.add_argument('--use_embed_layer', action='store_true', 
                        help='If specified, and the model produces an embedding dimensionality distinct' + 
                        'to our model, we add an additional embedding layer where each latent dimension is ' + 
                        'multiplied by a random embedding, and all multiplied embeddings concatenated')
    parser.add_argument('--desired_output_dim', type=int, default=None, 
                        help='The desired output dimensionality of the latent embedding (applicable only for scalar-valued compositional reps)')
    # logging
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_proj_name', type=str, default='compute_downstream-3dshapes')
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--save_dir', default='/media/bethia/F6D2E647D2E60C25/trained/readout/')
    parser.add_argument('--checkpoint_freq', default=1000, type=int,
                        help='Number of iterations between which we save the model')

    # downstream evaluation 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

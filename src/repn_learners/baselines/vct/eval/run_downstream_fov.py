import numpy as np
import math
import random 
import argparse
import torch
from sklearn.decomposition import PCA 
from torch.utils.data import DataLoader
import gin
from src.data.datasets import get_dataset
from src.data.utils import get_dataloaders

from src.shared.constants import *
from src.logger.logger import Logger
from src.eval.fov_regression.optim import RSquared
from src.eval.fov_regression.optim.train import train_clf_on_readout, train_mlp_on_readout
from src.eval.fov_regression.models import ReadOutMLP, BaseClf, ModularClf
from src.eval.fov_regression.eval import eval_clf, eval_regressor
from src.shared.components import EmbedLayer
from src.repn_learners.baselines.vct.models.visual_concept_tokeniser import VCT_Decoder, VCT_Encoder
import src.repn_learners.baselines.vct.utils as utils
from src.shared.utils import set_seed 

import logging 

logging.basicConfig(level=logging.INFO)    

TO_KEEP = [
    'no_wandb',
    'wandb_proj_name',
    'checkpoint_dir',
    'repn_fn_key', 
    'lr', 
    'n_epochs', 
    'patience', 
    'min_delta', 
    'n_test',
    'clf_hidden_dims',
    'data_dir',
    'repn_fn_key',
    'wandb_run_name',
    'saved_iter',
    'use_embed_layer',
    'desired_output_dim'
]


def main(args):

    if not torch.cuda.is_available():
        raise Exception("Cuda not available!!!")
 
    vct_enc, _, vq_vae_model, model_args = utils.load_vct_model(args)
    utils.set_defaults(args, model_args)
    set_seed(args.seed)
    saved_iter = args.checkpoint_dir.split('model_')[1].split('_')[0].split('.')[0]
    args.saved_iter = int(saved_iter)

    logging.info(f'Saved iter is {args.saved_iter}')
    utils.set_wandb_run_name(args)
    logging.info(f'Wandb run name is {args.wandb_run_name}')

    vct_enc.eval()
    vq_vae_model.eval()
    repn_fn = lambda x, args=args, vq_vae_model=vq_vae_model, vct_enc=vct_enc: utils.get_latents(args, x, vq_vae_model, vct_enc)

    logging.info(f'***LOADING {args.dataset} DATASET***')
    if args.dataset == 'mpi_real': 
        args.dataset = 'mpi3d'
    dataset, number_factors, number_channels, test_ratio_per_factor = get_dataset(args.dataset,
                                                                                      args.data_dir)
    args.test_ratio_per_factor = test_ratio_per_factor
    dataloader_train, dataloader_test, dataloader_full = get_dataloaders(args, full_dataset=dataset)
    logger = Logger(args, None)
    logging.info(f'Latent dim {args.latent_dim}, seed {args.seed}, model {args.model}\nAll args {args}')
    logging.info(f'Saved iter {args.saved_iter}, file name {args.checkpoint_dir}')

    latent_dim = args.latent_dim 

    labels_01 = dataset.get_normalized_labels() 
    r_sq = RSquared(labels_01, device='cuda')

    if args.dataset == CARS3D_DATASET:
        print(f'Len of dataset {len(dataset)}')
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
    
    # vct-specific 
    parser.add_argument('--wo_dis_loss', type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64,
                help="config settings")
    parser.add_argument("--concepts_num", type=int, default=20,
            help="config settings")
    
    # downstream regressor model 
    parser.add_argument('--seed', type=int, default=987, help='Random seed used to sample layer dimensions for regressor')
    parser.add_argument('--no_randomisation', action='store_true', help='If specified, we do not randomise the layer dimensions for regressor')

    # loading 
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--config', type=str)
    parser.add_argument('--gadi', action='store_true')
    parser.add_argument('--n_sample_list', type=str, default="full,10000,1000,500,250,100")
    parser.add_argument('--data_dir', default='/g/data/po67/anonymous_cat/Data', type=str)

    # representation function    
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
    parser.add_argument('--save_dir', default='/g/data/po67/anonymous_cat/tmp/')
    parser.add_argument('--checkpoint_freq', default=1000, type=int,
                        help='Number of iterations between which we save the model')
    parser.add_argument('--wandb_log_dir', default='/g/data/po67/anonymous_cat/wandb_logs')

    # downstream evaluation 
    parser.add_argument('--mode', type=str, default='regression', choices=['clf', 'regression'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--clf_hidden_dims', type=str, default='512,512,256,256')

    args = parser.parse_args()
    args.clf_hidden_dims = [int(item) for item in args.clf_hidden_dims.split(',')]
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

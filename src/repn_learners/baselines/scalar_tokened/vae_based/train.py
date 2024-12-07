import numpy as np
import argparse
import torch
from tqdm import tqdm
import time

from torch.optim import Adam
from src.shared.utils import get_filename, set_seed 
from src.logger.load import load_model_from_path, load_corresponding_args_from_path
from src.data.datasets import get_dataset
from src.data.datasets import DisLibDataset
from src.repn_learners.baselines.scalar_tokened.vae_based import get_model
from src.eval.dis import MetricAggregator, compute_dis_metrics
from src.shared.constants import *
from src.logger import Logger
from src.data import get_dataloaders

import logging 

logging.basicConfig(level=logging.INFO)

ARGS_TO_KEEP = ['vis_frequency',
                'eval_frequency',
                'no_wandb', 
                'wandb_proj_name',
                'save_ae', 
                'save_dir',
                'checkpoint_freq', 
                'file_name',
                'compute_dis_metrics',
                'wandb_run_name', 
                'n_iters', 
                'load_dir']

def combine_checkpoint_args_with_new_args(args: argparse.Namespace, checkpoint_args: argparse.Namespace) -> argparse.Namespace:
    old_args_to_keep = dict(filter(lambda x: x[0] not in ARGS_TO_KEEP, vars(checkpoint_args).items()))
    new_args_to_keep = dict(filter(lambda x: x[0] in ARGS_TO_KEEP, vars(args).items()))
    
    merged = argparse.Namespace(**old_args_to_keep, **new_args_to_keep)
    return merged 

def get_filename(args): 
    if args.file_name is not None: 
        return args.file_name 
    model_file_name = (f'{args.model}-latent_dim_{args.latent_dim}_{args.dataset}_k-{args.k}-{args.supervision_mode}' + 
                    f'_n_iters-{args.n_iters}_beta-{args.vae_beta}')
    if args.model == SLOWVAE: 
        model_file_name += f'_gamma-{args.slowvae_gamma}_rate-{args.slowvae_rate}'
    return model_file_name 

    
def train_one_epoch(args, model, optimiser, dataloader_train: DisLibDataset, epoch_idx: int, iteration: int, logger: Logger, pbar): 
    model.train()
    for batch, targets in dataloader_train:
        batch = batch.to(args.device)
        targets = targets.to(args.device)
        pbar.update(iteration - pbar.n)
        # get loss
        if args.model == PCL_MODEL:
            latents = model(batch)
            loss, infos = model.loss_f(latents)
        elif args.model == SLOWVAE:
            x_recon, mu, logvar = model(batch, use_stochastic=True)
            loss, infos = model.loss_f(batch, x_recon, mu, logvar)
        elif args.model == ADAGVAE or args.model == ADAGVAE_K_KNOWN:
            loss, infos, x_recon = model(batch, use_stochastic=True)
        elif args.model == GVAE or args.model == MLVAE: 
            loss, infos, x_recon = model(batch, targets, use_stochastic=True)
        else: 
            raise NotImplementedError(f'Unexpected model type {args.model}')
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        logs = dict(filter(lambda x: '_dim' not in x[0], infos.items()))
        logger.log_scalars(logs={
            **dict(filter(lambda x: 'recon' in x[0], logs.items())), 'step': iteration},
                           prefix_to_append='model/loss/recon/train')
        logger.log_scalars(logs={
            **dict(filter(lambda x: 'recon' not in x[0], logs.items())), 'step': iteration},
                           prefix_to_append='model/loss/specific/train')
        
        if (args.save_ae and iteration % args.checkpoint_freq == 0 and iteration != 0) or (iteration == 100):
            model_file_name = get_filename(args) 
            logger.save_model(args=args, model=model, model_file_name=model_file_name, iteration_id=iteration)
        
        iteration += 1
        if iteration == args.n_iters - 1:
            break

    if epoch_idx % args.vis_frequency == 0:
        if args.model != PCL_MODEL: # pcl has no reconstructions
            logger.log_reconstructions(x=batch, x_hat=x_recon, n_epoch=epoch_idx, training=True)                

    return iteration 


def eval_one_epoch(args, model, dataloader: DisLibDataset, 
                   epoch_idx: int, logger: Logger, 
                   mode: str='test', evaluate_avg_whole_dset: bool=False):
    model.eval() 
    logging.info(f"\n***EVAULATING MODEL AT EPOCH {epoch_idx}****") 
    mse_losses = []
    bce_losses = []
    with torch.no_grad(): 
        for batch_idx, (imgs, targets) in enumerate(tqdm(dataloader)): 
            if batch_idx > 200 and not evaluate_avg_whole_dset: 
                break 
            imgs = imgs.to(args.device) 
            targets = targets.to(args.device)
            if args.model == PCL_MODEL:
                latents = model(imgs)
                loss, infos = model.loss_f(latents)
            else: 
                if args.model == SLOWVAE:
                    x_recon, mu, logvar = model(imgs, use_stochastic=True)
                    loss, infos = model.loss_f(imgs, x_recon, mu, logvar)
                elif args.model == ADAGVAE or args.model == ADAGVAE_K_KNOWN:
                    loss, infos, x_recon = model(imgs, use_stochastic=True)
                elif args.model == GVAE or args.model == MLVAE: 
                    loss, infos, x_recon = model(imgs, targets, use_stochastic=True)
                else: 
                    raise NotImplementedError(f'Unexpected model type {args.model}')
                mse_losses.append(infos['mse_recon_loss'])
                bce_losses.append(infos['bce_recon_loss'])
            if not evaluate_avg_whole_dset: 
                logs = dict(filter(lambda x: '_dim' not in x[0], infos.items()))
                logger.log_scalars(logs={**dict(filter(lambda x: 'recon' in x[0], 
                                                       logs.items())), 
                                     'step': batch_idx + (epoch_idx // args.eval_frequency)*200}, 
                               prefix_to_append=f'model/loss/recon/{mode}')
                logger.log_scalars(logs={**dict(filter(lambda x: 'recon' not in x[0], 
                                                       logs.items())), 
                                     'step': batch_idx + (epoch_idx // args.eval_frequency)*200}, 
                               prefix_to_append=f'model/loss/specific/{mode}')
        if args.model != PCL_MODEL: # pcl has no reconstructions
            logger.log_reconstructions(x=imgs, x_hat=x_recon, n_epoch=epoch_idx, training=False) 

        if evaluate_avg_whole_dset and args.model != PCL_MODEL: # pcl model has no bce/mse loss
            avg_mse_loss = torch.tensor(mse_losses).mean()
            avg_bce_loss = torch.tensor(bce_losses).mean()
            logger.log_scalars(logs={'avg_mse': avg_mse_loss, 
                                     'avg_bce': avg_bce_loss}, 
                                     prefix_to_append=f'model/loss/final/{mode}')
            return avg_mse_loss, avg_bce_loss

def main(args):
    start_time = time.time()

    # paths
    if not torch.cuda.is_available():
        raise Exception("Cuda not available!!!")
    args.device = 'cuda'

    if args.load_dir is None: 
        # data
        set_seed(args.seed)
        dataset, number_factors, number_channels, test_ratio_per_factor = \
            get_dataset(args.dataset, args.data_dir)
        args.test_ratio_per_factor = test_ratio_per_factor  # for logging
        dataloader_train, dataloader_test, dataloader_full = get_dataloaders(args, full_dataset=dataset)

        # model
        args.nc = number_channels 
        model = get_model(args.model, number_factors, number_channels,
                        args.latent_dim, args, dataset).to(args.device)
        logging.info('***MODEL IS***', model)
        n_iters_total = args.n_iters 
    else: 
        logging.info(f"Loading model from path {args.load_dir}")
        model = load_model_from_path(model_path=args.load_dir)
        orig_args = load_corresponding_args_from_path(model_path=args.load_dir)
        n_iters_total = orig_args.saved_iter + args.n_iters 
        set_seed(seed=orig_args.seed)
        
        dataset, number_factors, number_channels, test_ratio_per_factor = \
        get_dataset(orig_args.dataset, orig_args.data_dir)
        dataloader_train, dataloader_test, dataloader_full = get_dataloaders(
            orig_args, full_dataset=dataset)
        args = combine_checkpoint_args_with_new_args(args=args, checkpoint_args=orig_args)
        
    # aggregator for dis metrics
    if args.compute_dis_metrics: 
        metric_aggregator_full = MetricAggregator(val_dataset=dataloader_full.dataset, 
                                                n_points=10000,
                                                random_state1=np.random.RandomState(args.seed),
                                                random_state2=np.random.RandomState(args.seed+10),
                                                verbose=True)
    # for the transfer learning we might only want to train the last layer
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    # logging 
    logger = Logger(args, model)

    # train model
    iteration = 0
    pbar = tqdm(total=args.n_iters)
    epoch_idx = 0
    while iteration < args.n_iters - 1:
        iteration = train_one_epoch(args=args, model=model, optimiser=optimizer, dataloader_train=dataloader_train, epoch_idx=epoch_idx, iteration=iteration,
                        logger=logger, pbar=pbar)
        if epoch_idx % args.eval_frequency == 0: 
            eval_one_epoch(args=args, model=model, dataloader=dataloader_test, epoch_idx=epoch_idx,
                          logger=logger)
                
        epoch_idx += 1
        if epoch_idx % 10 == 0: 
            logging.info(f'***EPOCH {epoch_idx}***')

    if args.save_ae: 
        logging.info("Saving model final iteration...")
        model_file_name = get_filename(args) 
        logger.save_model(args=args, model=model, 
                          model_file_name=model_file_name, 
                          iteration_id=n_iters_total)
        logging.info("****SAVING FINISHED****")

    if args.compute_dis_metrics: 
        compute_dis_metrics(aggregator=metric_aggregator_full, repn_fn=model.repn_fn,
        step_idx=epoch_idx//args.eval_frequency, split_type='train', 
        wandb_logger=logger, eval=True)
    
    test_avg_mse, test_avg_bce = eval_one_epoch(args=args, model=model, dataloader=dataloader_test, epoch_idx=epoch_idx, 
                   logger=logger, evaluate_avg_whole_dset=True)
    train_avg_mse, train_avg_bce = eval_one_epoch(args=args, model=model, dataloader=dataloader_train, epoch_idx=epoch_idx, 
                   logger=logger, mode='train', evaluate_avg_whole_dset=True)
    # log differences
    diffs = list(map(
                lambda x: 
                    (x[1] - x[0])/x[0], 
                    zip([train_avg_mse, train_avg_bce], [test_avg_mse, test_avg_bce])
                )
            )
    diffs = {'diff_mse': diffs[0], 'diff_bce': diffs[1]}
    logger.log_scalars(diffs, prefix_to_append='model/loss/final/')

    total_time = (time.time() - start_time) / 60
    logging.info(f'****TOTAL TIME****: {total_time}')


def parse_args():
    parser = argparse.ArgumentParser(description='')
    # reproduceability
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--gadi', action='store_true')

    # data
    parser.add_argument('--dataset', type=str, default=SHAPES3D_DATASET,
                        help='Dataset to use',
                        choices=DATASET_CHOICES)
    parser.add_argument('--data_dir', default='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets/', type=str)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--k', type=int, default=1,
                        help='k from locatello paper')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='number of workers')

    # train
    parser.add_argument('--n_iters', type=int, default=200000,
                        help='Number of training iterations')
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    # load from checkpoint
    parser.add_argument('--load_dir', default=None, type=str)

    # model
    parser.add_argument('--model', type=str, default=SLOWVAE,
                        choices=WS_SCALAR_BASELINES,
                        help='Which architecture to use')
    parser.add_argument('--latent_dim', type=int, default=10)
    
    # vae based models
    parser.add_argument('--vae_beta', type=float, default=1.,
                        help='Weighting factor for the KL[q(z|x)]||p(z) in '
                             'the elbo')
    parser.add_argument('--slowvae_gamma', type=float, default=10.,
                        help='Weighting factor for the Laplacian transition '
                             'prior')
    parser.add_argument('--slowvae_rate', type=float, default=6.,
                        help='Weighting factor for the Laplacian transition '
                             'prior')

    # logging
    parser.add_argument('--vis_frequency', default=1, type=int)
    parser.add_argument('--eval_frequency', default=10, type=int)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_proj_name', type=str, default='train_ws_models')
    parser.add_argument('--save_ae', action='store_true', 
                        help='If specified, the trained autoencoder model is saved')
    parser.add_argument('--save_dir', default='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines', type=str)
    parser.add_argument('--checkpoint_freq', default=50000, type=int)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--compute_dis_metrics', action='store_true')
    parser.add_argument('--wandb_run_name', default=None, type=str)

    args = parser.parse_args()
    args.supervision_mode = WEAKLY_SUPERVISED

    # each model comes with assumptions about the data generative process
    if args.model in [SLOWVAE, PCL_MODEL]:
        args.transition_prior = 'laplace'
    elif args.model == ADAGVAE or args.model == GVAE or args.model == MLVAE or args.model == ADAGVAE_K_KNOWN:
        args.transition_prior = 'locatello'
    else:
        args.transition_prior = None

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    args.maximally_distinct = False
    args.n_workers = 8

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

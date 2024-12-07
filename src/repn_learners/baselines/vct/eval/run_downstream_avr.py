import argparse
import torch
from tqdm import tqdm
import random

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from src.shared.constants import *
from src.logger.logger import Logger
from src.logger.load import load_model_from_path, load_corresponding_args_from_path
from src.shared.components import EmbedLayer
from src.eval.avr.data.pgm_data import get_pgm_dataset
from src.eval.avr.models import WReN
from src.shared.training import EarlyStopper
from src.eval.avr.shared.constants import *
import src.repn_learners.baselines.vct.utils as utils

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
    'data_dir',
    'repn_fn_key',
    'save_dir',
    'wandb_run_name',
    'saved_iter',
    'use_embed_layer',
    'desired_output_dim',
    'gadi'
]

def train_wildnet(args, prefix: str, patience: float,
                  min_delta: float, wildnet: WReN, 
                  train_loader: DataLoader, 
                  val_loader: DataLoader, 
                  test_loader: DataLoader,
                  n_epochs: int, 
                  wandb_logger=None): 
    optimiser = Adam(wildnet.parameters(), lr=args.lr)
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    wildnet.train() 

    for epoch in tqdm(range(n_epochs)): 
        for batch_idx, (features, targets) in enumerate(train_loader):
            # context (B, n_context, 3, 64, 64)
            # ans (B, n_ans, 3, 64, 64)
            batch = torch.cat((features["context"], 
                                      features["answers"]), dim=1).permute(
                                          0, 1, 4, 2, 3
                                      ) # (B, n_context+n_ans, 3, 64, 64)
            assert batch.shape == torch.Size([targets.shape[0], N_CONTEXT+N_ANS, 3, 64, 64])
            batch = batch.reshape(-1, 3, 64, 64).contiguous().cuda()
            targets = F.one_hot(targets, num_classes=N_ANS).cuda()
            optimiser.zero_grad() 
            out = wildnet.forward(batch, targets)
            loss = out['loss']
            loss.backward() 
            optimiser.step() 

            if wandb_logger is not None: 
                wandb_logger.log_scalars(logs={**out,
                                               'step': epoch*len(train_loader) + batch_idx}, 
                                               prefix_to_append=f'wildnet_n_samples_{prefix}/train')
            
        avg_val_loss = []
        wildnet.eval() 
        with torch.no_grad():
            for step, (features, targets) in enumerate(val_loader): 
                batch = torch.cat((features["context"], 
                                    features["answers"]), dim=1).permute(
                                        0, 1, 4, 2, 3
                                    ) # (B, n_context+n_ans, 3, 64, 64)
                assert batch.shape == torch.Size([targets.shape[0], N_CONTEXT+N_ANS, 3, 64, 64])
                batch = batch.reshape(-1, 3, 64, 64).contiguous().cuda()
                targets = F.one_hot(targets, num_classes=N_ANS).cuda()
                out = wildnet.forward(batch, targets)
                avg_val_loss.append(out['loss'])
                if wandb_logger is not None:
                    wandb_logger.log_scalars(logs={**out, 'step': step + epoch*1000},
                                            prefix_to_append=f'wildnet_n_samples_{prefix}/cross_val')
                if step > 500: 
                    break 
        if early_stopper(torch.stack(avg_val_loss).mean()):  
            break 

def test_wildnet(wildnet: WReN, prefix,
                 test_loader: DataLoader, 
                mode: str,
                wandb_logger=None, 
                global_step: int=0): 
    
    wildnet.eval() 
    with torch.no_grad():
        avg_loss = []
        avg_acc = []
        for batch_idx, (features, targets) in enumerate(test_loader): 
            batch = torch.cat((features['context'], 
                                    features['answers']), dim=1).permute(
                                        0, 1, 4, 2, 3)
            targets = F.one_hot(targets, num_classes=N_ANS).cuda()
            assert batch.shape == torch.Size([targets.shape[0], N_CONTEXT+N_ANS, 3, 64, 64])
            batch = batch.reshape(-1, 3, 64, 64).contiguous().cuda()

            with torch.no_grad():
                out = wildnet.forward(batch, targets)
                avg_loss.append(out['loss'])
                avg_acc.append(out['acc'])
                if wandb_logger is not None:
                    wandb_logger.log_scalars(logs={**out, 'step': batch_idx + global_step*len(test_loader)},
                                            prefix_to_append=f'wildnet_n_samples_{prefix}/test')
        
        avg_loss = torch.stack(avg_loss).mean()
        avg_acc = torch.stack(avg_acc).mean()
        if wandb_logger is not None: 
            wandb_logger.log_scalars(logs={f'avg_loss_n_samples_{prefix}_{mode}': avg_loss,
                                           'step': global_step})
            wandb_logger.log_scalars(logs={f'avg_acc_n_samples_{prefix}_{mode}': avg_acc, 
                                           'step': global_step})

        return avg_loss, avg_acc


def main(args):

    if not torch.cuda.is_available():
        raise Exception("Cuda not available!!!")

    
    vct_enc, vct_dec, vq_vae_model, model_args = utils.load_vct_model(args)
    
    vct_enc_params = sum(p.numel() for p in vct_enc.parameters() if p.requires_grad)
    vct_dec_params = sum(p.numel() for p in vct_dec.parameters() if p.requires_grad)
    vq_vae_model_params = sum(p.numel() for p in vq_vae_model.parameters() if p.requires_grad)
    
    
    logging.info(f'N PARAMETERS {vq_vae_model_params + vct_enc_params + vct_dec_params}')

    utils.set_defaults(args, model_args)
    
    saved_iter = args.checkpoint_dir.split('model_')[1].split('_')[0].split('.')[0]
    args.saved_iter = int(saved_iter)
        
    utils.set_wandb_run_name(args)
    vct_enc.eval()
    vq_vae_model.eval()
    repn_fn = lambda x, args=args, vq_vae_model=vq_vae_model, vct_enc=vct_enc: utils.get_latents(args, x, vq_vae_model, vct_enc)

    pgm_dataset = get_pgm_dataset(args.pgm_type, dataset=args.dataset, data_dir=args.data_dir)

    latent_dim = args.latent_dim 
    logger = Logger(args, None)
    print(f'Latent dim {args.latent_dim}, seed {args.seed}, model {args.model}\nAll args {args}')
    print(f'Saved iter {args.saved_iter}, file name {args.checkpoint_dir}')

    embedding_fn = lambda x, repn_fn=repn_fn, key=args.repn_fn_key: repn_fn(x)
                 
    # let's log the accuracy on the test set periodically so we can examine convergence of downstream model 


    
    dataset_100000, dataset_10000, dataset_1000, dataset_500, dataset_250, dataset_100, val_dataset, test_dataset, _ = torch.utils.data.random_split(pgm_dataset, 
                                                                             lengths=[100000, 10000, 1000, 500, 250, 100, 1000, 1000, 
                                                                                  len(pgm_dataset)-(113850)])

    train_loader_100000_samples = DataLoader(dataset_100000, batch_size=args.batch_size, 
                                             num_workers=args.n_workers)
    train_loader_10000_samples =  DataLoader(dataset_10000, batch_size=args.batch_size, 
                                             num_workers=args.n_workers)
    train_loader_1000_samples = DataLoader(dataset_1000, batch_size=args.batch_size, 
                                             num_workers=args.n_workers)
    train_loader_500_samples = DataLoader(dataset_500, batch_size=args.batch_size, 
                                             num_workers=args.n_workers)
    train_loader_250_samples = DataLoader(dataset_250, batch_size=args.batch_size, 
                                          num_workers=args.n_workers)
    train_loader_100_samples = DataLoader(dataset_100, batch_size=args.batch_size, 
                                             num_workers=args.n_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                             num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             num_workers=args.n_workers)
    
    prefixes = [] 
    models = []
    train_loaders = [] 
    patiences = [] 
    n_epochs = []
    
    if args.no_randomisation: 
        hidden_size_g = None 
        hidden_size_f = None 
    else: 
        random.seed(args.seed)
        hidden_size_g = random.choice([256, 512])
        hidden_size_f = random.choice([128, 256])
        args.wandb_run_name += f'hidden_dim_g-{hidden_size_g}_hidden_dim_f-{hidden_size_f}'

    # sorry I know this is GROSS 
    for n_samples_of_interest in args.n_sample_list.split(','):
        if n_samples_of_interest == '100000': 
            prefixes.append('100000')
            models.append(WReN(embedding_fn=embedding_fn, embedding_dim=latent_dim, 
                               hidden_size_g=hidden_size_g, hidden_size_f=hidden_size_f).cuda())
            train_loaders.append(train_loader_100000_samples)
            n_epochs.append(args.n_epochs + 10)
            patiences.append(args.patience)
        if n_samples_of_interest == '10000': 
            prefixes.append('10000')
            models.append(WReN(embedding_fn=embedding_fn, embedding_dim=latent_dim, 
                               hidden_size_g=hidden_size_g, hidden_size_f=hidden_size_f).cuda())
            train_loaders.append(train_loader_10000_samples) 
            patiences.append(args.patience + 2**1)
            n_epochs.append(args.n_epochs + 1*100)
        if n_samples_of_interest == '1000': 
            prefixes.append('1000')
            models.append(WReN(embedding_fn=embedding_fn, embedding_dim=latent_dim, 
                               hidden_size_g=hidden_size_g, hidden_size_f=hidden_size_f).cuda())
            train_loaders.append(train_loader_1000_samples)
            patiences.append(args.patience + 2**2)
            n_epochs.append(args.n_epochs + 2*100)
        if n_samples_of_interest == '500': 
            prefixes.append('500')
            models.append(WReN(embedding_fn=embedding_fn, embedding_dim=latent_dim, 
                               hidden_size_g=hidden_size_g, hidden_size_f=hidden_size_f).cuda())
            train_loaders.append(train_loader_500_samples)
            patiences.append(args.patience + 2**3)
            n_epochs.append(args.n_epochs + 3*100)
        if n_samples_of_interest == '250': 
            prefixes.append('250')
            models.append(WReN(embedding_fn=embedding_fn, embedding_dim=latent_dim, 
                               hidden_size_g=hidden_size_g, hidden_size_f=hidden_size_f).cuda())
            train_loaders.append(train_loader_250_samples)
            patiences.append(args.patience + 2**3)
            n_epochs.append(args.n_epochs + 4*100)
        if n_samples_of_interest == '100':
            prefixes.append('100')
            models.append(WReN(embedding_fn=embedding_fn, embedding_dim=latent_dim, 
                               hidden_size_g=hidden_size_g, hidden_size_f=hidden_size_f).cuda())
            train_loaders.append(train_loader_100_samples)
            patiences.append(args.patience + 2**3)
            n_epochs.append(args.n_epochs + 4*100)


    for i, (prefix, model, train_loader) in enumerate(zip(prefixes, models, train_loaders)):
        train_wildnet(args, f'{prefix}+{args.pgm_type}',
                    patience=patiences[i], 
                    min_delta=args.min_delta, 
                    wildnet=model,
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    test_loader=test_loader,
                    n_epochs=n_epochs[i],
                    wandb_logger=logger)
    
        for mode_prefix, loader in zip(['train', 'test', 'val'], 
                                       [train_loader, val_loader, test_loader]): 
            test_wildnet(wildnet=model, prefix=f'{prefix}+{args.pgm_type}', test_loader=loader,
                                            mode=mode_prefix,
                                            wandb_logger=logger,
                                            global_step=n_epochs[i])
        

def parse_args():
    parser = argparse.ArgumentParser(description='')
    # loading 
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--gadi', action='store_true')
    parser.add_argument('--data_dir', default='/g/data/po67/anonymous_cat/Data/', type=str)
    parser.add_argument('--n_sample_list', type=str, default="100000,10000,1000,500,250,100")
    parser.add_argument('--dataset', default=SHAPES3D_DATASET)
    parser.add_argument('--wandb_log_dir', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--seed', type=int, default=987, help='Random seed used to sample layer dimensions for WReN')
    parser.add_argument('--no_randomisation', action='store_true', help='If specified, we do not randomise the layer dimensions for WReN')


    # representation function type    
    parser.add_argument('--use_embed_layer', action='store_true', 
                        help='If specified, and the model produces an embedding dimensionality distinct' + 
                        'to our model, we add an additional embedding layer where each latent dimension is ' + 
                        'multiplied by a random embedding, and all multiplied embeddings concatenated')
    parser.add_argument('--desired_output_dim', type=int, default=None, 
                        help='The desired output dimensionality of the latent embedding (applicable only for scalar-valued compositional reps)')
    
    # vct specific
    parser.add_argument('--wo_dis_loss', type=bool, default=False)
    parser.add_argument("--concepts_num", type=int, default=20,
            help="config settings")
    
    parser.add_argument('--seed', type=int, default=123)
    
    # pgm specific arguments
    parser.add_argument('--pgm_type', default='easy3', type=str, 
                        choices=['easy1', 'easy2', 'easy3', 'hard1', 'hard2', 'hard3'])
    
    # logging
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_proj_name', type=str, default='compute_downstream-abstract_visual_reasoning')
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--save_dir', default='/g/data/po67/anonymous_cat/tmp/')
    parser.add_argument('--checkpoint_freq', default=1000, type=int,
                        help='Number of iterations between which we save the model')

    # downstream evaluation 
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_delta', type=float, default=0.00001)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

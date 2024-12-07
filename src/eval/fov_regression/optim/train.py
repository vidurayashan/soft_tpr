from tqdm import tqdm 
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.data.datasets import DisLibDataset
from src.shared.training import EarlyStopper 
from src.shared.constants import *
from src.eval.fov_regression.models import ConcatModels, ReadOutMLP 

def train_clf_on_readout(args, patience: float, 
                         min_delta: float,
                         readout_model, 
                         repn_fn, 
                         train_loader: DataLoader, 
                         val_loader: DataLoader, 
                         factor_sizes: np.array, 
                         lr: float, 
                         n_epochs=8, 
                         wandb_logger=None,
                         log_prefix: str=''): 
    optimiser = Adam(readout_model.parameters(), lr=lr)
    early_stopper = EarlyStopper(patience=patience, 
                                 min_delta=min_delta)
    step = 0

    print(f'CLF READ OUT MODEL:\n{readout_model}\n')
 
    for epoch in tqdm(range(n_epochs)): 
        readout_model.train()
        for batch_idx, (batch, targets) in enumerate(train_loader):
            if args.model == VCT and args.use_embed_layer: 
                if batch.shape[0] < args.reduced_hidden_dim: 
                    sampled_idxs = torch.randint(batch.shape[0], (args.reduced_hidden_dim,))
                    batch = batch[sampled_idxs].cuda()
                    targets = DisLibDataset.convert_to_one_hot(targets[sampled_idxs], factor_sizes).cuda()
                else: 
                    batch, targets = batch.cuda(), DisLibDataset.convert_to_one_hot(targets, factor_sizes).cuda()
                assert batch.shape[0] >= args.reduced_hidden_dim
            else: 
                batch, targets = batch.cuda(), DisLibDataset.convert_to_one_hot(targets, factor_sizes).cuda()
            with torch.no_grad(): 
                model_latents = repn_fn(batch)
            out = readout_model(latent_reps=model_latents, tgts=targets)
            optimiser.zero_grad() 
            loss = out['loss']['total_loss']
            loss.backward()
            optimiser.step() 
            if wandb_logger is not None: 
                wandb_logger.log_scalars(logs={**dict((f'k_{log_prefix}', v) for k, v in out['loss'].items()), 
                                               'step': epoch*len(train_loader) + batch_idx},
                                         prefix_to_append='clf_mlp/train')

        avg_val_loss = []
        readout_model.eval()
        for batch_idx, (batch, targets) in enumerate(val_loader): 
            if args.model == VCT and args.use_embed_layer: 
                if batch.shape[0] < args.reduced_hidden_dim: 
                    sampled_idxs = torch.randint(batch.shape[0], (args.reduced_hidden_dim,))
                    batch = batch[sampled_idxs].cuda()
                    targets = DisLibDataset.convert_to_one_hot(targets[sampled_idxs], factor_sizes).cuda()
                else: 
                    batch, targets = batch.cuda(), DisLibDataset.convert_to_one_hot(targets, factor_sizes).cuda()
                assert batch.shape[0] >= args.reduced_hidden_dim
            else: 
                batch, targets = batch.cuda(), DisLibDataset.convert_to_one_hot(targets, factor_sizes).cuda()
            with torch.no_grad(): 
                model_latents = repn_fn(batch)
            out = readout_model(latent_reps=model_latents, tgts=targets)
            optimiser.zero_grad() 
            loss = out['loss']['total_loss']
            avg_val_loss.append(loss)
            loss.backward()
            optimiser.step() 
            if wandb_logger is not None: 
                wandb_logger.log_scalars(logs={**dict((f'k_{log_prefix}', v) for k, v in out['loss'].items()), 
                                               'step': step},
                                         prefix_to_append='clf_mlp/cross_val')
            if batch_idx > 1000: 
                break
            step += batch_idx  
        if early_stopper(torch.stack(avg_val_loss).mean()): 
            break 
            
    supervised_model = ConcatModels(repn_fn, readout_model).eval() 
    return supervised_model


def train_mlp_on_readout(args, patience: float, 
                         min_delta: float, 
                         repn_fn, 
                         readout_model: ReadOutMLP, 
                         train_loader: DataLoader, 
                         val_loader: DataLoader, 
                         lr: float, 
                         n_epochs=8,
                         wandb_logger=None, 
                         log_prefix: str=''):
    
    print(f'MLP READ OUT MODEL:\n{readout_model}\n')

    optimizer = Adam(readout_model.parameters(), lr=lr)
    early_stopper = EarlyStopper(patience=patience, 
                                 min_delta=min_delta)
    step = 0 
    for epoch in tqdm(range(n_epochs)):
        readout_model.train() 
        for batch_idx, (batch, targets) in tqdm(enumerate(train_loader)):
            if args.model == VCT and args.use_embed_layer: 
                if batch.shape[0] < args.reduced_hidden_dim: 
                    sampled_idxs = torch.randint(batch.shape[0], (args.reduced_hidden_dim,))
                    batch = batch[sampled_idxs].cuda()
                    targets = targets[sampled_idxs].cuda()
                else: 
                    batch, targets = batch.cuda(), targets.cuda()
            else: 
                batch, targets = batch.cuda(), targets.cuda()
            with torch.no_grad():
                #print(f'Batch size before entering repn fn is {batch.shape}\n')
                model_latents = repn_fn(batch)
            predicted_factor = readout_model(model_latents)
            squared_diff = (targets - predicted_factor).pow(2)
            loss = squared_diff.sum(dim=1).mean()  # mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if wandb_logger is not None: 
                wandb_logger.log_scalars(logs={f'loss (mse) {log_prefix}': loss,
                                               'step': epoch*len(train_loader) + batch_idx}, 
                                         prefix_to_append='reg_mlp/train')

        avg_val_loss = []
        for batch_idx, (batch, targets) in enumerate(val_loader): 
            if args.model == VCT and args.use_embed_layer: 
                if batch.shape[0] < args.reduced_hidden_dim: 
                    sampled_idxs = torch.randint(batch.shape[0], (args.reduced_hidden_dim,))
                    batch = batch[sampled_idxs].cuda()
                    targets = targets[sampled_idxs].cuda()
                else: 
                    batch, targets = batch.cuda(), targets.cuda()
            else: 
                batch, targets = batch.cuda(), targets.cuda()
            with torch.no_grad(): 
                model_latents = repn_fn(batch)
            predicted_factor = readout_model(model_latents)
            squared_diff = (targets - predicted_factor).pow(2)
            loss = squared_diff.sum(dim=1).mean()  # mse
            avg_val_loss.append(loss)
            if wandb_logger is not None: 
                wandb_logger.log_scalars(logs={f'loss (mse) {log_prefix}': loss, 
                                               'step': step},
                                               prefix_to_append='reg_mlp/cross_val')
            step += batch_idx 
            if batch_idx > 1000: 
                break 
        if early_stopper(torch.stack(avg_val_loss).mean()): 
            break 

    supervised_model = ConcatModels(repn_fn, readout_model).eval()
    return supervised_model


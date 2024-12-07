"""
Implementation of the Shu model baseline uses the open-source repo provided by the authors': https://github.com/google-research/google-research/tree/master/weak_disentangle
The code below has been adapted from: https://github.com/google-research/google-research/blob/master/weak_disentangle/main.py

@inproceedings{
Shu2020Weakly,
title={Weakly Supervised Disentanglement with Guarantees},
author={Rui Shu and Yining Chen and Abhishek Kumar and Stefano Ermon and Ben Poole},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HJgSwyBKvr}
}
"""

# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script for all experiments.
"""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
# pylint: disable=no-value-for-parameter, unused-argument
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
import argparse 

from src.eval.dis import compute_dis_metrics
from src.eval.dis import MetricAggregator
from src.shared.constants import *
from src.shared.utils import get_filename, set_seed 
from src.data import get_dataset, get_dataloaders
from src.logger.logger import Logger

from src.repn_learners.baselines.scalar_tokened.shu.model import Discriminator, Generator, Encoder, LabelDiscriminator 
import src.repn_learners.baselines.scalar_tokened.shu.visualise as viz
import src.repn_learners.baselines.scalar_tokened.shu.datasets as shu_datasets

from torch.distributions.normal import Normal 

def train(args, n_factors, dataloader_train, 
          model_type="gen"):
    s_dim = n_factors # don't consider for now any datasets with nuisance factors 
    n_dim = 0 

    masks = shu_datasets.make_masks(n_factors) # s dim number of GT facs
    z_dim = s_dim + n_dim # n_dim is nuisance dims 
    enc_lr_mul = args.enc_lr_mul 
    gen_lr = args.gen_lr 
    dis_lr = args.dis_lr 
    enc_lr = enc_lr_mul * gen_lr

    # Load data

      # Networks
    y_dim = len(masks) # number of factors we change 
    dis = Discriminator(args.n_channels, y_dim, width=args.dis_width, 
                        share_dense=args.dis_share_dense, 
                        uncond_bias=args.dis_uncond_bias, 
                        cond_bias=args.dis_cond_bias).cuda()
    gen = Generator(args.n_channels, z_dim,
                    width=args.gen_width).cuda()
    enc = Encoder(s_dim, args.n_channels, width=args.enc_width).cuda()  # Encoder ignores nuisance param

    # Create optimizers
    if model_type in {"gen", "van"}:
        gen_opt = optim.Adam(params=gen.parameters(), betas=(0.5, 0.999), lr=gen_lr)
        dis_opt = optim.Adam(params=dis.parameters(), betas=(0.5, 0.999), lr=dis_lr)
        enc_opt = optim.Adam(params=enc.parameters(), betas=(0.5, 0.999), lr=enc_lr)
    elif model_type == "enc":
        enc_opt = optim.Adam(params=enc.parameters(), betas=(0.5, 0.999), lr=enc_lr)


    def train_gen_step(x1_real, x2_real, y_real):
        targets_real = torch.ones((x1_real.shape[0], 1)).cuda()
        targets_fake = torch.zeros((x1_real.shape[0], 1)).cuda()
        targets = torch.concatenate((targets_real, targets_fake), axis=0)
        gen.train()
        dis.train()
        enc.train()
        # Alternate discriminator step and generator step

        # Generate
        z1, z2, y_fake = shu_datasets.paired_randn(x1_real.shape[0], z_dim, masks)
        with torch.no_grad():
            x1_fake = gen(z1.detach())
            x2_fake = gen(z2.detach())
        # Discriminate
        x1 = torch.concatenate((x1_real, x1_fake), 0)
        x2 = torch.concatenate((x2_real, x2_fake), 0)
        #print(f'X1 real shape {x1_real.shape}, x2 real shape {x2_real.shape}, x1 fake shape {x1_fake.shape}, x2 fake shape {x2_fake.shape}, targets shape {targets.shape}')
        y = torch.concatenate((y_real, y_fake), 0).long()
        logits = dis(x1, x2, y)
        # Encode
        mu_logvar = enc(x1_fake)	
        #print(f'Targets shape {targets.shape}')	
        dis_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                target=targets, reduction='mean')
        # Encoder ignores nuisance parameters (if they exist)
        dist = Normal(mu_logvar[0], (mu_logvar[1]*0.5).exp())
        enc_loss = -torch.mean(dist.log_prob(z1[:, :s_dim]))

        dis_opt.zero_grad() 
        enc_opt.zero_grad() 
        
        dis_loss.backward() 
        enc_loss.backward()

        dis_opt.step() 
        enc_opt.step()

        # Generate
        z1, z2, y_fake = shu_datasets.paired_randn(x1_real.shape[0], z_dim, masks)
        x1_fake = gen(z1)
        x2_fake = gen(z2)

        # Discriminate
        logits_fake = dis(x1_fake, x2_fake, y_fake)
        gen_loss = F.binary_cross_entropy_with_logits(input=logits_fake, 
                                                target=targets_real, 
                                                reduction='mean')

        gen_opt.zero_grad() 
        gen_loss.backward() 
        gen_opt.step() 

        return dict(gen_loss=gen_loss, dis_loss=dis_loss, enc_loss=enc_loss)


    def train_enc_step(x1_real, x2_real, y_real):
        enc.train()
        z1 = enc(x1_real)[0] # take first chunk as mean
        z2 = enc(x2_real)[0] # take first chunk as mean
        logits = torch.gather(z1 - z2, masks.to(dtype=torch.int64), dim=-1)
        loss = F.binary_cross_entropy_with_logits(
          input=logits, target=y_real, reduction='mean')
        
        enc_opt.zero_grad() 
        loss.backward() 
        enc_opt.step()

        return dict(gen_loss=0, dis_loss=0, enc_loss=loss)

    def gen_eval(z):
        gen.eval()
        return gen(z)

    def enc_eval(x):
        enc.eval()
        return enc(x)[0] # take first chunk as mean

    train_time = 0
    logger = Logger(args, None)
    it = 0 
    pb = tqdm(total=args.n_iters, unit_scale=True, smoothing=0.1, ncols=70)
    
    while it < args.n_iters:
        for batch_idx, (imgs, labels) in enumerate(dataloader_train):
            imgs = imgs.cuda() 
            labels = labels.cuda()
            stopwatch = time.time()
            x1 = imgs[::2] # interwoven pairs in batch dim 
            x2 = imgs[1::2] 
            y = torch.argwhere((labels[::2] != labels[1::2]))[:, 1].unsqueeze(1)
            if model_type == "gen":
                vals = train_gen_step(x1, x2, y)
            elif model_type == "enc":
                vals = train_enc_step(x1, x2, y)
            train_time += time.time() - stopwatch

            logger.log_scalars(logs=vals, prefix_to_append='train/specific/')

            # Log visualizations and evaluations
            if (batch_idx) % args.eval_frequency == 0: 
                if model_type == "gen":
                    ablation_img = viz.ablation_visualization(x1, x2, gen_eval, z_dim)
                    decoded = viz.decode(x1, enc_eval, gen_eval)
                    #print(f'Ablation img has shape {ablation_img.shape}')
                    logger.log_img(grid=ablation_img, step=batch_idx % args.eval_frequency, 
                    state='generate')
                    logger.log_img(grid=decoded, step=batch_idx, state='decode')
            # Save model
            if args.save_ae and ((it  % args.checkpoint_freq == 0) or (it == 100)):
                model_file_name = get_filename(args) 
                logger.save_model(args=args, model=(enc, gen), 
                        model_file_name=model_file_name, iteration_id=it)
            it += 1 
            pb.update()
    if args.save_ae and ((batch_idx  % args.checkpoint_freq == 0) or (batch_idx == 100)):
                model_file_name = get_filename(args) 
                logger.save_model(args=args, model=(enc, gen), 
                        model_file_name=model_file_name, iteration_id=batch_idx)
    if args.compute_dis_metrics: 
        metric_aggregator_full = MetricAggregator(val_dataset=dataloader_full.dataset, 
                                            n_points=10000, 
                                            random_state1=np.random.RandomState(args.seed), 
                                            random_state2=np.random.RandomState(args.seed+10), 
                                            verbose=True)
        compute_dis_metrics(aggregator=metric_aggregator_full, model=enc, 
                            step_idx=args.n_iters, split_type='train', 
                            wandb_logger=logger, eval=False)
 
def main(args):
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    # specific arguments
    parser.add_argument('--dis_width', default=1, type=int, choices=[1,2])
    parser.add_argument('--gen_width', default=1, type=int, choices=[1,2,3])
    parser.add_argument('--enc_width', default=1, type=int, choices=[1,2,3])
    parser.add_argument('--gen_lr', type=float, default=1e-3)
    parser.add_argument('--dis_lr', type=float, default=1e-3)
    parser.add_argument('--dis_share_dense', default='False', type=str, choices=['True', 'False']) # wandb sweep
    parser.add_argument('--dis_cond_bias', default='False', type=str, choices=['True', 'False']) 
    parser.add_argument('--dis_uncond_bias', default='False', type=str, choices=['True', 'False'])
    parser.add_argument('--enc_lr_mul', type=float, default=1.)
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', type=str, default=SHAPES3D_DATASET, choices=DATASET_CHOICES)
    parser.add_argument('--seed', type=int, default=9876, help="config settings")
    parser.add_argument('--data_dir', default='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets/', type=str)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--modification', type=str, default=None,
                        choices=['composition', 'random'],
                        help='data set modification')
    parser.add_argument('--test_split_ratio', type=float, default=None)
    parser.add_argument('--k', type=int, default=1,
                        help='k from locatello paper')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--gadi', action='store_true')

    parser.add_argument('--n_iters', type=int, default=200000,
                        help='Number of training iterations')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)	

    # logging 
    parser.add_argument('--eval_frequency', default=500, type=int)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_proj_name', type=str, default='train_models')
    parser.add_argument('--save_ae', action='store_true', 
                        help='If specified, the trained autoencoder model is saved')
    parser.add_argument('--save_dir', default='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe2/trained/baselines/shu/', type=str)
    parser.add_argument('--checkpoint_freq', default=50000, type=int)
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--wandb_run_name', default=None, type=str)
 
    # eval
    parser.add_argument('--compute_dis_metrics', action='store_true')

    # checkpoints
    parser.add_argument('--load_dir', type=str, default=None)
    args = parser.parse_args() 
    args.model = SHU 
    args.supervision_mode = WEAKLY_SUPERVISED
    args.transition_prior = 'locatello'
    args.k = 1
    args.n_channels = 3 
    args.maximally_distinct = False 
    args.dis_share_dense = True if args.dis_share_dense == 'True' else False 
    args.dis_cond_bias = True if args.dis_cond_bias == 'True' else False 
    args.dis_uncond_bias = True if args.dis_uncond_bias == 'True' else False 

    train_range = args.n_iters 
    set_seed(args.seed)
    dataset, number_factors, number_channels, test_ratio_per_factor = \
            get_dataset(args.dataset, args.data_dir)
    args.test_ratio_per_factor = test_ratio_per_factor  # for logging
    dataloader_train, dataloader_test, dataloader_full = get_dataloaders(args, full_dataset=dataset)
    args.latent_dim = dataset.num_factors	

    train(args=args, n_factors=dataloader_train.dataset.num_factors, 
       dataloader_train=dataloader_train)
import wandb 
import numpy as np
from typing import Dict, List, Tuple
import math 
import os 
import logging 

import torch
from torchvision.utils import make_grid

from src.repn_learners import VQVAE 
from src.eval.fov_regression.models import ReadOutMLP, BaseClf
from src.shared.constants import *
from src.logger.load import save_downstream_model, save_representation_model, get_save_path 

logging.basicConfig(level=logging.INFO)

class Logger():
    def __init__(self, args, model):
        self.wandb_logger = self.init_wandb_logger(args, model)

        self.save_parent_dir_ae = get_save_path(args, model_type=AUTOENCODER)
        self.save_parent_dir_clf = get_save_path(args, model_type=CLF)
        self.save_parent_dir_reg = get_save_path(args, model_type=REGRESSOR)

        self.clf_saved_count = 0 
        self.reg_saved_count = 0
        self.ae_saved_count = 0 
        
        if isinstance(model, VQVAE):
            self.img_postprocessing = lambda x: x
        else:
            self.img_postprocessing = lambda x: x.sigmoid()

    def save_model(self, args, model, model_file_name: str, iteration_id: int) -> None: 
        epoch_prefixed_file_name = f'iter_{iteration_id}_{model_file_name}.pt'
        args.saved_iter = iteration_id
        if isinstance(model, BaseClf): 
            save_downstream_model(args, model, epoch_prefixed_file_name, save_path=self.save_parent_dir_clf)
            if self.clf_saved_count == 0: 
                args_prefix = model_file_name.split('-')[0]
                torch.save(args, os.path.join(self.save_parent_dir_clf, model_to_save_prefix[CLF], f'{args_prefix}_args.pt'))
            self.clf_saved_count +=1 
            logging.info(f"****SAVED CLF AT {os.path.join(self.save_parent_dir_clf, model_to_save_prefix[CLF], epoch_prefixed_file_name)}")
        if isinstance(model, ReadOutMLP):
            save_downstream_model(args, model, epoch_prefixed_file_name, save_path=self.save_parent_dir_reg)
            if self.reg_saved_count == 0: 
                args_prefix = model_file_name.split('-')[0]
                torch.save(args, os.path.join(self.save_parent_dir_reg, model_to_save_prefix[REGRESSOR], f'{args_prefix}_args.pt'))
            logging.info(f"****SAVED READOUTMLP AT {os.path.join(self.save_parent_dir_clf, model_to_save_prefix[REGRESSOR], epoch_prefixed_file_name)}")
        else: 
            assert args.model in [*BASELINES, SOFT_TPR_AE, VCT], f'Unrecognised representation model type {args.model}'
            save_representation_model(args, model, epoch_prefixed_file_name, 
                                      save_path=self.save_parent_dir_ae)
            if self.ae_saved_count == 0: 
                args_prefix = model_file_name.split('-')[0]
                torch.save(args, os.path.join(f'{self.save_parent_dir_ae}', model_to_save_prefix[AUTOENCODER], f'{args_prefix}_args.pt'))
            self.ae_saved_count += 1
            logging.info(f"****SAVED {args.model} AT {os.path.join(self.save_parent_dir_clf, model_to_save_prefix[AUTOENCODER], epoch_prefixed_file_name)}")
        
    def init_wandb_logger(self, args, model): 
        if not args.no_wandb: 
            if args.gadi: 
                os.environ["WANDB_MODE"] = 'offline' # log locally 
                wandb_logger = wandb.init(project=args.wandb_proj_name, config=self.parse_args_wandb_cfg(args),
                                    name=self.parse_args_wandb_run_name(args),
                                    mode='offline',
                                    dir=args.wandb_log_dir)
            else:
                wandb_logger = wandb.init(project=args.wandb_proj_name, config=self.parse_args_wandb_cfg(args),
                                    name=self.parse_args_wandb_run_name(args))
            wandb.define_metric('vis/train_step')
            wandb.define_metric('vis/test_step')
            wandb.define_metric('dmetric/train_step')
            wandb.define_metric('dmetric/test_step')
            wandb.define_metric('ae/test_step')
            wandb.define_metric('ae/train_step')
            wandb.define_metric('clf/test_step')
            wandb.define_metric('clf/train_step')
            wandb.define_metric('clf/cross_val_step')
            wandb.define_metric('ae/diff/step')
            wandb.define_metric('vis/recons-train', step_metric='vis/train_step')
            wandb.define_metric('vis/recons-val', step_metric='vis/test_step')
            
            if model is not None: 
                wandb.watch(model, log='all')
        else: 
            wandb_logger = None 
        return wandb_logger 

    def parse_args_wandb_run_name(self, args) -> str: 
        print(f'ARGS WANDB RUN NAME {args.wandb_run_name}')
        if args.wandb_run_name is not None: 
            return args.wandb_run_name
        name = f'{args.model}{args.n_iters}_dim-{args.latent_dim}_{args.dataset}_{args.supervision_mode}_'
        if args.supervision_mode == WEAKLY_SUPERVISED: 
            name += f'k-{args.k}_prior-{args.transition_prior}'
    
        if args.model in WS_SCALAR_BASELINES:
            if args.model == SHU: 
                name += f'_gen_lr-{args.gen_lr}_gen_width-{args.gen_width}_dis_lr-{args.dis_lr}_dis_width-{args.dis_width}_enc_lr-{args.dis_lr*args.enc_lr_mul}'
            else:  
                name += f'_beta-{args.vae_beta}'
                if args.model == SLOWVAE: 
                    name += f'_gamma-{args.slowvae_gamma}_rate-{args.slowvae_rate}'
        elif args.model == SOFT_TPR_AE:
            name += (f'_nr-{args.n_roles}-{args.role_embed_dim}_nf-{args.n_fillers}-{args.filler_embed_dim}' + 
                    f'_lvq-{args.lambda_vq}_lc-{args.lambda_commit}_lr-{args.lambda_recon}') 
            if args.supervision_mode == WEAKLY_SUPERVISED: 
                name += f'_lwsr-{args.lambda_ws_recon}_lwsd-{args.lambda_ws_r_embed_ce}'    
        elif args.model == COMET: 
            name += f'_components-{args.components}_hidden-{args.hidden_dim}'            
        args.wandb_run_name = name
        return name 

    def bin_frac(self, frac: float, granularity: float=0.05) -> str: 
        rounded_down = math.floor(frac/granularity) * granularity
        return f'{rounded_down}-{rounded_down + granularity}'

    def parse_args_wandb_cfg(self, args) -> Dict: 
        repn_fn_key = vars(args).get('repn_fn_key', 'default')
        saved_iter = vars(args).get('saved_iter', 'n/a')
        extra_cfg = {}
        if args.model in BASELINES and args.model != VCT: 
            repn_fn_key = 'default'
        common_cfg = {
            'model': args.model, 
            'n_iters': args.n_iters,
            'saved_iter': saved_iter,
            'latent_dim': args.latent_dim, 
            'seed': args.seed,
            'dataset': args.dataset, 
            'supervision_mode': args.supervision_mode,
            'repn_fn_key': repn_fn_key,
            'group_by_model_repn_fn_key': f'{args.model}_{repn_fn_key}',
            'group_by_model_type_saved_iter': f'{args.model}_{saved_iter}',
            'group_by_model_type_saved_iter_repn_fn_key': f'{args.model}_{saved_iter}_{repn_fn_key}'
        }
        if args.supervision_mode == WEAKLY_SUPERVISED: 
            common_cfg = {**common_cfg, 
                          'transition_prior': args.transition_prior,
                          'k': args.k}
        if args.model in WS_SCALAR_BASELINES and args.model != SHU: 
            extra_cfg = {
                'beta': args.vae_beta 
            }
            if args.model == SLOWVAE: 
                extra_cfg = {**extra_cfg, 
                                'gamma': args.slowvae_gamma, 
                                'rate': args.slowvae_rate}
        if args.model == SOFT_TPR_AE: 
            if 'lambda_ws_recon' not in vars(args).keys(): 
                args.lambda_ws_recon = args.lambda_ws_recon_loss # renaming stupid stuff
            extra_cfg = {
                'n_roles': args.n_roles, 
                'n_fillers': args.n_fillers, 
                'role_embed_dim': args.role_embed_dim, 
                'filler_embed_dim': args.filler_embed_dim,
                'lambda_vq': args.lambda_vq, 
                'lambda_recon': args.lambda_recon, 
                'lambda_commit': args.lambda_commit, 
                'lambda_ws_recon': args.lambda_ws_recon, 
                'lambda_ws_r_embed_ce': args.lambda_ws_r_embed_ce, 
                'hidden_dims': args.hidden_dims,
            }
        if args.model == COMET: 
            extra_cfg = {
                'latent_dim': args.latent_dim,
                'n_components': args.components, 
                'ensembles': args.ensembles
            }
        if args.model == VCT: 
            extra_cfg = {
                'k': args.k,
                'backbone': args.backbone, 
                'backbone_n_epochs': args.backbone_n_epochs, 
                'n_concepts': args.concepts_num, 
                'vq_vae_hidden_dim': args.vq_vae_hidden_dim
            }
        if args.model == SHU: 
            extra_cfg = {
                'dis_width': args.dis_width, 
                'gen_width': args.gen_width, 
                'gen_lr': args.gen_lr, 
                'dis_lr': args.dis_lr,
                'enc_lr': args.dis_lr * args.enc_lr_mul
            }
        return {**common_cfg, **extra_cfg}
    
    def shut(self): 
        wandb.finish()

    def append_prefix_to_logs(self, logs: Dict, prefix: str) -> Dict: 
        loss_logs = dict(
            map(
                lambda kv: (f'{prefix}_{kv[0]}', kv[1]), logs.items()
            )
        )
        return loss_logs

    def log_scalars(self, logs: Dict, prefix_to_append: str=None) -> None: 
        if self.wandb_logger is not None: 
            if prefix_to_append is not None: 
                logs = self.append_prefix_to_logs(logs, prefix_to_append)
            wandb.log(logs)
                
    def log_train_val_diff(self, train_losses: List[float], val_losses: List[float], 
                        n_epoch: int) -> None:
        if self.wandb_logger is not None: 
            train_losses = np.mean(train_losses)
            val_losses = np.mean(val_losses)
            percentage_diff = (train_losses - val_losses) / train_losses
            wandb.log({'ae/diff/recon_loss_diff': percentage_diff, 
                    'ae/diff/step': n_epoch}) 
            
    def log_img(self, grid: torch.Tensor, step: int, state: str='train') -> None: 
        if self.wandb_logger is not None: 
            wandb.log({f'vis/recons-{state}': wandb.Image(grid), 
                   f'vis/{state}_step': step})

    def log_reconstructions(self, x: torch.Tensor, x_hat: torch.Tensor, 
                            n_epoch: int, training: bool) -> None: 
        
        if self.wandb_logger is not None: 
            state = 'train' if training else 'test'
            batch_end = x.shape[0] if x.shape[0] <= 25 else 25
            pairs = torch.stack([x[0:batch_end].cpu(), self.img_postprocessing(x_hat[0:batch_end]).cpu()], dim=1) # (batch_size, 2, C, H, W)
            grid = make_grid(pairs.view(batch_end*2, *pairs.shape[2:]))
            
            wandb.log({f'vis/recons-{state}': wandb.Image(grid), 
                    f'vis/{state}_step': n_epoch//5})
            
    def log_swapped_reconstructions(self, x: torch.Tensor, x_hat: torch.Tensor, 
                            n_epoch: int, training: bool) -> None: 
        if self.wandb_logger is not None: 
            state = 'train' if training else 'test'
            
            x2_hat, x1_hat = torch.chunk(x_hat, 2, 0)
            x1, x2 = torch.chunk(x, 2, 0)
            batch_end = x1.shape[0] if x1.shape[0] <= 18 else 18
            print(f'Self.img postprocessing is {self.img_postprocessing}')
        
            pairs = torch.stack([x1[:batch_end].cpu(), self.img_postprocessing(x1_hat[:batch_end]).cpu(),
                                x2[:batch_end].cpu(), self.img_postprocessing(x2_hat[:batch_end]).cpu()], dim=1)
            #print(f'Pairs shape {pairs.shape}')
                                
            grid = make_grid(pairs.view(batch_end*4, *pairs.shape[2:]))
            
            wandb.log({f'vis/recons_swapped-{state}': wandb.Image(grid), 
                    f'vis/{state}_step': n_epoch//5})
                
    def log_latent_reps(self, wandb_logger, state: Dict, batch_idx: int) -> None: 
        if self.wandb_logger is not None: 
            if batch_idx == 0: 
                latent_reps = state['z'].squeeze().tolist() 
                gt_vals = state['gt_shape'].squeeze().tolist() 
                data = [[latent_rep, gt_val] for (latent_rep, gt_val) in zip(latent_reps, gt_vals)]
                table = wandb.Table(data=data, columns=['latent_rep', 'gt_val'])
                wandb.log({'latent_reps': wandb.plot.scatter(table, 'latent_rep', 'gt_val')})
                
    def format_dmetrics(self, metrics_out: Dict) -> Dict: 
        import plotly.express as px
        to_plot = {}
        for metric_name, metric_out in metrics_out.items(): 
            if 'matrix' in metric_name: 
                latent_dim = metric_out.shape[0]
                n_gen_factors = metric_out.shape[1]
                heatmap = px.imshow(metric_out,
                                    text_auto=True)
                heatmap = heatmap.update_layout(
                    yaxis={'title':'latent_dim',
                        'tickvals':list(range(latent_dim)),
                        'ticktext':list(str(i) for i in range(latent_dim))},
                    xaxis={'title':'factor_idx',
                        'tickvals': list(range(n_gen_factors)), 
                        'ticktext': list(str(i) for i in range(n_gen_factors))},
                    yaxis_nticks=latent_dim,
                    xaxis_nticks=n_gen_factors)
                #print(f'Metric out is {metric_out}')
                to_plot[metric_name] = heatmap
            else: 
                to_plot[metric_name] = metric_out
        return to_plot

    def log_dmetrics(self, logs, train_out: Tuple[torch.Tensor, torch.Tensor], test_out: Tuple[torch.Tensor, torch.Tensor]) -> None: 
        if self.wandb_logger is not None: 
            to_plot = self.format_dmetrics(metrics_out=logs)
            wandb.log(to_plot)
            
            latent_reps_tr, gt_vals_tr = train_out 
            latent_reps_tr = latent_reps_tr.squeeze().tolist() # (N)
            gt_vals_tr = gt_vals_tr.squeeze().tolist() 
            data = [[latent_rep, gt_val] for (latent_rep, gt_val) in zip(latent_reps_tr, gt_vals_tr)]
            table = wandb.Table(data=data, columns=['latent_rep', 'gt_shape'])
            wandb.log({'latent_reps': wandb.plot.scatter(table, 'latent_rep', 'gt_shape')})

    def log_heatmap(self, weights: torch.Tensor, x_labels: List[str], 
                    y_labels: List[str], panel_name: str) -> None: 
        if self.wandb_logger is not None: 
            wandb.log({panel_name: wandb.plots.HeatMap(x_labels=x_labels, y_labels=y_labels,
                                                        matrix_values=np.array(weights))})
""" 
Implementation of COMET adapted from the authors' repo: https://github.com/yilundu/comet
@inproceedings{du2021comet,
  title={Unsupervised Learning of Compositional Energy Concepts},
  author={Du, Yilun and Li, Shuang and Sharma, Yash and Tenenbaum, B. Joshua
  and Mordatch, Igor},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
""" 

import torch 
from torch.optim import Adam
import math 
from src.shared.constants import *

from src.repn_learners.baselines.comet.models import LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128

def init_model(FLAGS, device):
    models = [LatentEBM(FLAGS).to(device) for i in range(FLAGS.ensembles)]
    optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models]

    return models, optimizers

def load_comet(args): 
    checkpoint = torch.load(args.checkpoint_dir, map_location=torch.device('cpu'), weights_only=False)

    models, _ = init_model(args, torch.device('cuda'))
    for i, model in enumerate(models): 
        model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)])
    
    return models 

 
def adjust_latent_dim(args): # to accommodate loading
    args.hidden_dim = args.latent_dim
    if args.use_embed_layer: 
        args.reduced_hidden_dim = math.ceil(args.desired_output_dim / args.components)
        args.latent_dim = args.components * args.reduced_hidden_dim  
    else:
        args.latent_dim = args.components * args.hidden_dim 
    
def set_wandb_run_name(args): 
    wandb_run_name = f'{args.saved_iter}_COMET'
    if args.use_embed_layer: 
        wandb_run_name += f'-embedding_layer_output_dim-{args.desired_output_dim}'
    args.wandb_run_name = wandb_run_name
    
def load_comet(args): 
    checkpoint = torch.load(args.checkpoint_dir, map_location=torch.device('cpu'), weights_only=False)

    models, _ = init_model(args, torch.device('cuda'))
    for i, model in enumerate(models): 
        model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)])
    
    return models 
        
def get_latents(args, proj_mat: torch.Tensor, x: torch.Tensor, model: LatentEBM, reshape=True) -> torch.Tensor: 
    # use 0th model in list of models as the original source code does 
    model.eval()
    with torch.no_grad(): 
        latents = model.embed_latent(x)
        if args.use_embed_layer: 
            latents = torch.bmm(latents.view(latents.shape[0], args.components, args.hidden_dim), proj_mat.unsqueeze(0).expand(latents.shape[0], -1, -1))
    if reshape:
        return latents.view(x.shape[0], -1)
    return latents

def set_defaults(args): 
    args.lr = 1e-4
    args.repn_fn_key = QUANTISED_FILLERS_CONCATENATED 
    args.components = 10
    args.latent_dim = 64
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
    args.seed = 123
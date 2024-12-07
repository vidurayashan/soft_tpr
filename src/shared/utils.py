import numpy as np 
import torch 
from src.shared.constants import *

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_filename(args): 
    if args.file_name is not None: 
        return args.file_name
    if args.model == SOFT_TPR_AE: 
        if args.supervision_mode == WEAKLY_SUPERVISED: 
            model_file_name = (f'soft_tpr_ae_dataset-{args.dataset}_latent_dim-{args.latent_dim}_n_roles-{args.n_roles}_embed_dim-{args.role_embed_dim}_n_fillers-{args.n_fillers}_embed_dim-{args.filler_embed_dim}' + 
                            f'_{args.supervision_mode}-lvq_{args.lambda_vq}-lc_{args.lambda_commit}-lr_{args.lambda_recon}-lwsd_{args.lambda_ws_r_embed_ce}')
        else: 
            model_file_name = (f'soft_tpr_ae_dataset-{args.dataset}_latent_dim-{args.latent_dim}_n_roles-{args.n_roles}_embed_dim-{args.role_embed_dim}_n_fillers-{args.n_fillers}_embed_dim-{args.filler_embed_dim}' + 
                        f'_{args.supervision_mode}-lvq_{args.lambda_vq}-lc_{args.lambda_commit}-lr_{args.lambda_recon}')
    if args.model == VCT: 
        model_file_name = (f'vct_dataset-{args.dataset}_n_concepts-{args.n_concepts}_vq_vae_hidden-{args.vq_vae_hidden}' + 
                           f'latent_dim_{args.latent_dim}-backbone_n_epochs-{args.backbone_n_epochs}')
    if args.model == SHU: 
        model_file_name = (f'shu_dataset-{args.dataset}')
    return model_file_name 
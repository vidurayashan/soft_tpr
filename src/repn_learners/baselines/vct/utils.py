import logging 
import gin
import torch 
import math 
import numpy as np
from sklearn.decomposition import PCA

from src.shared.constants import VCT, QUANTISED_FILLERS_CONCATENATED, UNSUPERVISED, VQ_VAE
from src.repn_learners.baselines.vct.models.visual_concept_tokeniser import VCT_Decoder, VCT_Encoder
from src.repn_learners.baselines.vct.main_vct import get_args, get_model_args, models

def set_wandb_run_name(args): 
    wandb_run_name = f'{args.saved_iter}_VCT'
    if args.use_embed_layer: 
        wandb_run_name += f'-embedding_layer_output_dim-{args.desired_output_dim}'
    args.wandb_run_name = wandb_run_name
    
def set_defaults(args, model_args): 
    args.repn_fn_key = QUANTISED_FILLERS_CONCATENATED 
    if args.use_embed_layer: 
        args.reduced_hidden_dim = math.ceil(args.desired_output_dim / args.concepts_num)
        args.latent_dim = args.concepts_num * args.reduced_hidden_dim  
    else:
        args.latent_dim = args.concepts_num * model_args.hidden 
    args.concepts_num = 20
    args.model = VCT
    args.supervision_mode = UNSUPERVISED 
    args.modification = None 
    args.n_workers = 8
    args.n_iters = 200000
    args.k = model_args.k
    args.backbone = VQ_VAE 
    args.backbone_n_epochs = 200
    args.vq_vae_hidden_dim = model_args.hidden
    

def load_vct_model(meta_args): 
    logging.info(f'****LOADING MODEL FROM PATH {meta_args.checkpoint_dir}')
    gin.parse_config_file(meta_args.config)
    args = get_args()
    meta_args.dataset = args.dataset
    gin.constant('num_steps', args.num_steps)
    gin.constant('step_lr', args.step_lr)
    gin.constant('image_energy', args.image_energy)
    gin.parse_config_file(f"{meta_args.config[:-7]}_shared.gin")
    
    if not meta_args.wo_dis_loss:
        gin.bind_parameter("get_train_flags.dis_loss", True)
    else:
        gin.bind_parameter("get_train_flags.dis_loss", False)

    model_args = get_model_args()
    vq_vae_model = models[args.dataset][model_args.model](model_args.hidden, k=model_args.k, num_channels=model_args.num_channels)
    vq_vae_model.cuda()
    # load
    vq_vae_model.load_state_dict(torch.load(model_args.path, map_location='cpu', weights_only=False))

    vct_enc = VCT_Encoder(z_index_dim = meta_args.concepts_num)
    vct_dec = VCT_Decoder(index_num = model_args.k, z_index_dim=model_args.shape_num, ce_loss=True)

    vct_enc.cuda()
    vct_dec.cuda()

    vct_enc.load_state_dict(torch.load(meta_args.checkpoint_dir, weights_only=False)['encoder_model_state_dict'])
    vct_dec.load_state_dict(torch.load(meta_args.checkpoint_dir, weights_only=False)['decoder_model_state_dict'])

    logging.info('****SUCCESSFULLY LOADED****')
    vq_vae_model.eval()
    vct_enc.eval() 
    vct_dec.eval()
    
    return vct_enc, vct_dec, vq_vae_model, model_args

def get_latents(args, x: torch.Tensor, vq_vae_model, vct_enc: VCT_Encoder) -> torch.Tensor: 
    vq_vae_model.eval() 
    vct_enc.eval() 
    with torch.no_grad(): 
        im_code = vq_vae_model.encode(x) 
        im_emb, _ = vq_vae_model.emb(im_code.detach())
        im_code_new = im_emb.view(im_code.shape[0], im_code.shape[1], -1).permute(0, 2, 1)
        latents = vct_enc(im_code_new)
        #print(f'Latent shape is {latents.shape}')
        if args.use_embed_layer: 
            assert x.shape[0] >= args.reduced_hidden_dim
            # apply PCA concept-wise 
            pca_list = [] 
            for i in range(latents.shape[1]): 
                pca = PCA(n_components=args.reduced_hidden_dim) # performed by COMET authors
                pca_result = pca.fit_transform(latents[:, i, :].cpu().numpy())
                pca_list.append(pca_result)
            pca_rep = torch.from_numpy(np.concatenate(pca_list, axis=1)).cuda()
            latents = pca_rep 
    return latents.view(x.shape[0], -1)
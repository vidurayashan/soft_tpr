import argparse 
import torch 
import torch.nn.functional as F 
import logging 
import numpy as np
import sys
import os

# Add the root directory to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.shared.utils import get_filename
from src.shared.utils import set_seed
from src.shared.constants import *
from src.shared.model_constructors import *
from src.logger.logger import Logger
from src.logger.load import load_model_from_path, load_corresponding_args_from_path
from src.repn_learners.soft_tpr_ae.optim.train import train 
from src.eval.dis.aggregator import MetricAggregator
from src.data import get_dataset, get_dataloaders

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

# 0. HPC SYS
parser.add_argument('--gadi', action='store_true',
                    help='If specified, train on gadi HPC system')

# 1. DATA DIRECTORIES
parser.add_argument('--data_dir', type=str, 
            default='/scratch/jq77/vs9475/datasets/soft_tpr_datasets/', 
                    help='Location of datasets')

# 2. DATASET 
parser.add_argument('--dataset', type=str, 
                    default=SHAPES3D_DATASET, choices=DATASET_CHOICES,
                    help='Specifies the dataset to train the model on')
parser.add_argument('--supervision_mode', type=str, 
                    default=WEAKLY_SUPERVISED, choices=SUPERVISION_CHOICES,
                    help=('Specifies the style of supervision. If weakly supervised, the model is presented with '
                          'pairs of images, $(x, x^{*})$, with a subset, $S$, of FoV *types* that differ between $x$ and $x^{*}$')
                    )
parser.add_argument('--transition_prior', type=str, 
                    default=LOCATELLO, choices=TRANSITION_PRIOR_CHOICES,
                    help=('Specifies the transition prior dictating how the identities of the FOV types within $S$ is selected.' 
                          'If the choice is LOCATELLO, the FoV types in $S$ are uniformly selected. If LAPLACE, they are sampled from a Laplacian')
                    )
parser.add_argument('--k', type=int,
                    default=1,
                    help='Specifies the *number* of FoV types that change between $x$ and $x^{*}$, i.e. cardinality of $S$')

# 3. MODEL CONFIG
parser.add_argument('--model_encoder', type=str, 
                    default=VQ_VAE_ENCODER, choices=ENCODER_CHOICES,
                    help='Specifies the type of encoder, E, used in TPR Autoencoder')
parser.add_argument('--model_decoder', type=str, 
                    default=VQ_VAE_DECODER2, choices=DECODER_CHOICES,
                    help='Specifies the type of decoder, D, used in TPR Autoencoder')
parser.add_argument('--hidden_dims', type=str, default="1024, 512, 512", 
                    help='Specifies the hidden dims for the MLP component of the standard encoder, E')
parser.add_argument('--role_embed_dim', type=int, default=8,
                    help=('Specifies the embedding dimension, $D_{R}$, of the role embedding function.' 
                          'Each role will be of dimension $D_{R}$ once embedded.')
                    )
parser.add_argument('--filler_embed_dim', type=int, default=64,
                    help=('Specifies the embedding dimension, $D_{F}, of the filler embedding function.'
                          'Each filler will be of this dimension once embedded.')
                    )
parser.add_argument('--n_fillers', type=int, default=None,
                    help='Specifies the number of fillers, $N_{F}$')
parser.add_argument('--n_roles', type=int, default=None,
                    help='Specifies the number of roles, $N_{R}$'
                    )
parser.add_argument('--init_roles_orth', action='store_true',
                    help='If specified, the role embeddings are initialised from a semi orthogonal matrix')
parser.add_argument('--freeze_role_embeddings', action='store_true', 
                    help='If specified, we freeze our role embeddings')
# TODO: change to the inverse, since the standard action we take is to initialise the role embeddings and to freeze the role embeddings
parser.add_argument('--use_concatenated_rep', action='store_true')

# 4. LOSS FN 
# 4.1 Standard loss fn parameters across all models:
parser.add_argument('--lr', type=float, default=1e-4, 
                    help='Learning rate in loss function')
parser.add_argument('--recon_loss_fn', type=str, 
                    default=MSE, choices=RECON_LOSS_FN_CHOICES,
                    help='Loss function for recon loss')
parser.add_argument('--scheduler', type=str, 
                    default=STEP, choices=SCHEDULER_CHOICES, 
                    help='LR scheduler type')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='Gamma parameter used in the StepLR scheduler')
parser.add_argument('--step_size', type=int, default=30,
                    help='Step size parameter used in StepLR scheduler')
parser.add_argument('--n_iters', type=int, default=200000)
parser.add_argument('--batch_size', type=int, default=64)
# 4.2 Specific loss fn parameters for TPR Autoencoder: 
parser.add_argument('--lambda_vq', type=float, default=1, 
                    help='Weighting coefficient for VQ-VAE quantisation loss. Third term in Eq 6')
parser.add_argument('--lambda_commit', type=float, default=0.5,
                    help=('Weighting coefficient for commitment loss in VQ-VAE loss.'
                          'Corresponds to beta term in Eq 6')
                    )
parser.add_argument('--lambda_recon', type=float, default=1, 
                    help='Weighting coefficient for self-supervised reconstruction loss. Second term in Eq 6')
parser.add_argument('--lambda_ws_recon', type=float, default=1.25, 
                    help=('Weighting coefficient for weakly supervised reconstruction loss.'
                          'Corresponds to lambda 1 in Eq 7')
                    )
parser.add_argument('--lambda_ws_r_embed_ce', default=1, type=float,
                    help=('Weighting coefficient for weakly-supervised cross-entropy based loss on the role embeddings.'
                          'Corresponds to lambda 2 in Eq 7')
                    )

# 5. LOGGING, LOADING & SAVING
# 5.1 Loading
parser.add_argument('--load_dir', type=str, default=None, 
                    help='If specified, loads model from given checkpoint directory')
# 5.2 Logging 
parser.add_argument('--no_wandb', action='store_true',
                    help='If specified, logging is not conducted on wandb')
parser.add_argument('--wandb_proj_name', type=str, default='train',
                    help='Specifies the project name for wandb')
parser.add_argument('--wandb_run_name', default=None, type=str)
parser.add_argument('--vis_frequency', type=int, default=1, 
                    help='Specifies the frequency at which visualisations of the reconstructions should be made')
parser.add_argument('--eval_frequency', type=int, default=10, 
                    help='Specifies the frequency at which the model loss is computed against the unseen dataset split')
# 5.3 Saving
parser.add_argument('--save_ae', action='store_true', 
                    help='If specified, the trained autoencoder model is saved')
parser.add_argument('--save_dir', type=str, default='C:\\Users\\21361535\\Downloads\\soft_tpr_datasets\\output', 
                    help='Specifies the directory in which the trained autoencoder model is saved into')
parser.add_argument('--checkpoint_freq', default=50000, type=int,
                    help='Frequency at which autoencoder saved')
parser.add_argument('--file_name', type=str, default=None,
                    help='File name for saved model')

# 6. EVALUATION 
parser.add_argument('--compute_dis_metrics', action='store_true',
                    help='If specified, disentanglement metrics are computed at the end of training')

# 7. REPRODUCEABILITY 
parser.add_argument('--seed', type=int, default=9876)


ARGS_TO_KEEP = [
    'vis_frequency', 
    'eval_frequency', 
    'no_wandb',
    'wand_proj_name',
    'save_ae',
    'save_dir',
    'checkpoint_freq', 
    'file_name', 
    'wandb_run_name', 
    'n_iters', 
    'load_dir'
]

args = parser.parse_args()
    
def get_encoder(args): 
    args.hidden_dims = [int(item) for item in args.hidden_dims.split(',')]
    encoder = encoders_map[args.model_encoder](filler_embed_dim=args.filler_embed_dim, 
                                                   role_embed_dim=args.role_embed_dim, 
                                                   hidden_dims=args.hidden_dims, 
                                                   nc=args.nc)
    args.latent_dim = encoder.rep_dim
    return encoder
    
def combine_checkpoint_args_with_new_args(args: argparse.Namespace, 
                                          checkpoint_args: argparse.Namespace) -> argparse.Namespace:
    old_args_to_keep = dict(filter(lambda x: x[0] not in ARGS_TO_KEEP, vars(checkpoint_args).items()))
    new_args_to_keep = dict(filter(lambda x: x[0] in ARGS_TO_KEEP, vars(args).items()))
    
    merged = argparse.Namespace(**old_args_to_keep, **new_args_to_keep)
    return merged 

def populate_args(args): 
    args.model = SOFT_TPR_AE
    args.nc = 3
    args.n_workers = 8
    args.maximally_distinct = False 
    if args.supervision_mode == UNSUPERVISED: 
        args.lambda_ws_recon = 0 
        args.lambda_ws_r_embed_ce = 0

def run(args): 
    logging.info(f'**** ARGS: {args}\n ****')
    populate_args(args)
    
    if args.load_dir is None: 
        set_seed(seed=args.seed)
        full_dataset, _, _, _ = get_dataset(args.dataset, args.data_dir)
        
        dataloader_train, dataloader_test, dataloader_full = get_dataloaders(args, full_dataset)

        encoder = get_encoder(args)
        if args.n_fillers is None: 
            args.n_fillers = sum(full_dataset.factor_sizes)
        n_fillers = args.n_fillers
        if args.n_roles is None: 
            args.n_roles = len(full_dataset.factor_sizes)
        n_roles = args.n_roles
        
        model = SoftTPRAutoencoder(encoder=encoder, 
                               decoder=decoders_map[args.model_decoder](latent_dim=args.latent_dim,
                                                                        nc=args.nc),
                               n_roles=n_roles, 
                               n_fillers=n_fillers, 
                               role_embed_dim=args.role_embed_dim, 
                               filler_embed_dim=args.filler_embed_dim, 
                               lambdas_loss={VQ_PENALTY: args.lambda_vq, 
                                            ORTH_PENALTY_FILLER: 0,
                                            ORTH_PENALTY_ROLE: 0,
                                            COMMITMENT_PENALTY: args.lambda_commit,
                                            RECON_PENALTY: args.lambda_recon,
                                            WS_RECON_LOSS_PENALTY: args.lambda_ws_recon,
                                            WS_DIS_PENALTY: args.lambda_ws_r_embed_ce}, 
                                init_fillers_orth=False,
                                init_roles_orth=args.init_roles_orth,
                                freeze_role_embeddings=args.freeze_role_embeddings, 
                                recon_loss_fn=args.recon_loss_fn,
                                weakly_supervised=(args.supervision_mode == WEAKLY_SUPERVISED)).cuda()
        args.n_params = model.count_params()
        n_iters_total = args.n_iters
    else: 
        logging.info(f'**** Loading model from path {args.load_dir} ****')
        model = load_model_from_path(model_path=args.load_dir)
        orig_args = load_corresponding_args_from_path(model_path=args.load_dir)
        n_iters_total = orig_args.saved_iter + args.n_iters
        set_seed(seed=orig_args.seed)
        
        full_dataset, _, _, _ = get_dataset(orig_args.dataset, orig_args.data_dir)

        if 'maximally_distinct' not in vars(orig_args).keys(): 
            orig_args.maximally_distinct = False
            args.maximally_distinct = False
        if 'n_fillers' not in (vars(orig_args)).keys(): 
            n_fillers = sum(full_dataset.factor_sizes) 
            orig_args.n_fillers = n_fillers 
            args.n_fillers = n_fillers 
        if 'n_roles' not in vars(orig_args).keys(): 
            n_roles = len(full_dataset.factor_sizes)
            orig_args.n_roles = n_roles 
            args.n_roles = n_roles 
        
        dataloader_train, dataloader_test, dataloader_full = get_dataloaders(orig_args, full_dataset=full_dataset)
        args = combine_checkpoint_args_with_new_args(args, orig_args)
    
    logging.info("***Model is***")
    logging.info(model)
    logging.info(f"***Model n parameters: {args.n_params}***")

    if args.n_iters > 0:
        logger = Logger(args, model)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.scheduler == 'cosine': 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.lr)
        else: 
            scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.step_size, gamma=args.gamma)
        
        logging.info('Training model...')
        train_out = train(args, dataloader_train, dataloader_test, dataloader_full, model, optimiser, scheduler, logger)
        logging.info(f'****TRAINING FINISHED, final out: {train_out}****')
    
        if args.save_ae: 
            logging.info("Saving model...")
            model_file_name = get_filename(args)
            logger.save_model(args=args, model=model, 
                    model_file_name=model_file_name, 
                    iteration_id=n_iters_total)
            logging.info("****SAVING FINISHED****")

    if args.compute_dis_metrics:
        aggregator = MetricAggregator(val_dataset=dataloader_full.dataset, 
                                    n_points=10000, 
                                    random_state1=np.random.RandomState(args.seed),
                                    random_state2=np.random.RandomState(args.seed+10), 
                                    verbose=True, 
                                    use_multidim_latents=True,
                                    use_beta=True, 
                                    use_factor=True,
                                    use_dci=True,
                                    use_mig=True,
                                    discretise_soft_tpr_repn=True)
        model.eval() 
        metric_outputs = aggregator(model.repn_fn) 
        logger.log_scalars({**metric_outputs, 
                              'dmetric/train_step': args.n_iters},
                             prefix_to_append='dmetric/')
        
if __name__ == '__main__': 
    run(args)
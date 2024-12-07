import argparse
import torch
from tqdm import tqdm
import logging
import random

from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from src.shared.constants import *
from src.logger.logger import Logger
from src.logger.load import load_model_from_path, load_corresponding_args_from_path
from src.shared.components import EmbedLayer
from src.eval.avr.data.pgm_data import get_pgm_dataset
from src.eval.avr.models import WReN
from src.shared.training import EarlyStopper
from src.eval.fov_regression.optim.train import ConcatModels
from src.eval.avr.shared import *

logging.basicConfig(level=logging.INFO)    

parser = argparse.ArgumentParser()

# 0. HPC SYS 
parser.add_argument('--gadi', action='store_true',
                    help='If specified, train on gadi HPC system')

# 1. DATA DIRECTORIES 
parser.add_argument('--data_dir', type=str, 
                    default='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets/', 
                    help='Location of datasets')

# 2. DATASET 
parser.add_argument('--dataset', type=str, 
                    default=SHAPES3D_DATASET, choices=DATASET_CHOICES,
                    help='Specifies the dataset to train the model on')
parser.add_argument('--pgm_type', default='easy3', type=str, 
                    choices=['easy1', 'easy2', 'easy3', 'hard1', 'hard2', 'hard3'],
                    help='Specifies difficulty of PGM task')

# 3. REPRESENTATION LEARNER CONFIG 
parser.add_argument('--use_embed_layer', action='store_true', 
                    help='If specified, and the representation learner produces a representation that has a dimensionality' + 
                    'that is not identical to our TPR-based model, we add an additional embedding layer where each latent dimension is ' + 
                    'multiplied by a random embedding, and all multiplied embeddings are concatenated, to control for' +
                    'dimensionality.')
parser.add_argument('--desired_output_dim', type=int, default=None, 
                    help='The desired output dimensionality of the latent embedding (applicable only for scalar-valued compositional reps)')
parser.add_argument('--repn_fn_key', 
                    default=QUANTISED_FILLERS_CONCATENATED, 
                    choices=[FILLER_IDXS, 
                             SOFT_FILLERS_CONCATENATED, 
                             QUANTISED_FILLERS_CONCATENATED, 
                             TPR_BINDINGS_FLATTENED, 
                             Z_TPR, 
                             Z_SOFT_TPR])

# 4. WILDNET-SPECIFIC ARGUMENTS
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--min_delta', type=float, default=0.00001)
parser.add_argument('--n_test', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_sample_list', type=str, default="100000,10000,1000,500,250,100",
                    help=('Select a subset of elements from the above list to train |n_sample_list| unique WildNets' + 
                          ' where WildNet[i] is trained on n_sample_list[i] samples produced by the representation learner'))
parser.add_argument('--seed', type=int, default=987, help='Random seed used to sample layer dimensions for WReN')
parser.add_argument('--no_randomisation', action='store_true', help='If specified, we do not randomise the layer dimensions for WReN')

# 5. LOGGING, LOADING & SAVING
# 5.1 Loading
parser.add_argument('--load_dir', type=str, default=None, 
                    help=('Location where the representation learner model is loaded from.' + 
                          'Note that this file only trains the WildNet (downstream model), and **not**' + 
                          'the representation learner (assumed to be at location --load_dir)'))
# 5.2 Logging 
parser.add_argument('--no_wandb', action='store_true',
                    help='If specified, logging is not conducted on wandb')
parser.add_argument('--wandb_proj_name', type=str, default='train',
                    help='Specifies the project name for wandb')
parser.add_argument('--wandb_run_name', default=None, type=str)
# 5.3 Saving
parser.add_argument('--save_dir', type=str, default='/media/bethia/F6D2E647D2E60C251/trained_cleaning/own_model', 
                    help='Specifies the directory in which the trained autoencoder model is saved into')
parser.add_argument('--checkpoint_freq', default=1000, type=int,
                    help='Frequency at which WildNet model saved')

args = parser.parse_args()

TO_KEEP = [
    'no_wandb',
    'wandb_proj_name',
    'load_dir',
    'repn_fn_key', 
    'lr', 
    'n_epochs', 
    'patience', 
    'min_delta', 
    'n_test',
    'data_dir',
    'wandb_run_name',
    'saved_iter',
    'use_embed_layer',
    'desired_output_dim'
]

def populate_args_with_orig_args(args, orig_args): 
    args_dict = vars(args)
    orig_args_dict = vars(orig_args)
    for k in orig_args_dict.keys(): 
        if k not in TO_KEEP: 
            args_dict[k] = orig_args_dict[k]
    return argparse.Namespace(**args_dict)

def set_defaults(args): 
    if args.model == SOFT_TPR_AE: 
        if args.repn_fn_key in [SOFT_FILLERS_CONCATENATED, QUANTISED_FILLERS_CONCATENATED]: 
            args.latent_dim = args.n_roles * args.filler_embed_dim 
        if args.repn_fn_key == FILLER_IDXS: 
            args.latent_dim = args.n_roles 
        if args.repn_fn_key in [Z_SOFT_TPR, Z_TPR]: 
            args.latent_dim = args.filler_embed_dim * args.role_embed_dim
        if args.repn_fn_key == TPR_BINDINGS_FLATTENED: 
            args.latent_dim = args.n_roles*args.filler_embed_dim*args.role_embed_dim
    if args.model in BASELINES:
        args.repn_fn_key = QUANTISED_FILLERS_CONCATENATED
    if args.model in WS_SCALAR_BASELINES: 
        if args.use_embed_layer: 
            args.latent_dim = args.desired_output_dim

def set_wandb_run_name(args, orig_args): 
    if orig_args.model == SOFT_TPR_AE:
        args.wandb_run_name = f'{args.saved_iter}_{orig_args.model}_{args.repn_fn_key}'
    elif orig_args.model in WS_SCALAR_BASELINES: 
        wandb_run_name = f'{args.saved_iter}_{orig_args.model}_{args.repn_fn_key}'
        if args.use_embed_layer: 
            wandb_run_name += f'-embedding_layer_output_dim-{args.desired_output_dim}'
        args.wandb_run_name = wandb_run_name
    else: 
        raise NotImplementedError(f'Have not yet implemented run name for model type: {orig_args.model}')
    
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
            batch = torch.concatenate((features["context"], 
                                      features["answers"]), dim=1).permute(
                                          0, 1, 4, 2, 3)
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
                batch = torch.concatenate((features["context"], 
                                    features["answers"]), dim=1).permute(0, 1, 4, 2, 3)
                assert batch.shape == torch.Size([targets.shape[0], N_CONTEXT+N_ANS, 3, 64, 64])
                batch = batch.reshape(-1, 3, 64, 64).contiguous().cuda()
                targets = F.one_hot(targets, num_classes=N_ANS).cuda()
                out = wildnet.forward(batch, targets)
                avg_val_loss.append(out['loss'])
                if wandb_logger is not None:
                    wandb_logger.log_scalars(logs={**out, 'step': step+(epoch*500)},
                                            prefix_to_append=f'wildnet_n_samples_{prefix}/cross_val')
                if step > 500: 
                    break 
        if early_stopper(torch.stack(avg_val_loss).mean()):  
            break 
        if (epoch != args.n_epochs - 1) and (epoch % 10 == 0): 
            logging.info('****TESTING WILDNET****')
            test_wildnet(wildnet, prefix, test_loader=test_loader, mode='test', wandb_logger=wandb_logger, 
                        global_step=epoch)

            
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
            batch = torch.concatenate((features['context'], 
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
                    wandb_logger.log_scalars(logs={**out, 'step': batch_idx},
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

    logging.info(f'****LOADING MODEL FROM PATH {args.load_dir}****')
    model = load_model_from_path(model_path=args.load_dir)
    orig_args = load_corresponding_args_from_path(model_path=args.load_dir)
    assert orig_args.dataset == args.dataset, f'Preloaded embedded model trained on {orig_args.dataset}, but we require {args.dataset}!'
    saved_iter = args.load_dir.split('iter_')[1].split('_')[0]
    args.saved_iter = int(saved_iter)
    set_wandb_run_name(args, orig_args)

    if orig_args.model in WS_SCALAR_BASELINES: 
        if args.use_embed_layer: 
            embed_layer = EmbedLayer(output_dim=args.desired_output_dim,
                                     latent_dim=orig_args.latent_dim)
            model.eval()
            model = ConcatModels(repn_fn_first_model=model.repn_fn, second_model=embed_layer).cuda()        

    model.eval()
    pgm_dataset = get_pgm_dataset(args.pgm_type, dataset=args.dataset, data_dir=args.data_dir)
    args = populate_args_with_orig_args(args=args, orig_args=orig_args)
    set_defaults(args)

    logger = Logger(args, None)
    model.eval() 
    if orig_args.model in WS_SCALAR_BASELINES and args.use_embed_layer: 
            latent_dim = embed_layer.output_dim 
    else: 
        latent_dim = args.latent_dim 

    embedding_fn = lambda x, model=model, key=args.repn_fn_key: model.repn_fn(x, key=key)
    
                        
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
    
    if args.no_randomisation: 
        hidden_size_g = None 
        hidden_size_f = None 
    else: 
        random.seed(args.seed)
        hidden_size_g = random.choice([256, 512])
        hidden_size_f = random.choice([128, 256])
        args.wandb_run_name += f'hidden_dim_g-{hidden_size_g}_hidden_dim_f-{hidden_size_f}'
        
    prefixes = [] 
    models = []
    train_loaders = [] 
    patiences = [] 
    n_epochs = []

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
        train_wildnet(args, prefix,
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
            test_wildnet(wildnet=model, prefix=prefix, test_loader=loader,
                                            mode=mode_prefix,
                                            wandb_logger=logger)
        
if __name__ == "__main__":
    main(args)
import torch
import os

from src.eval.fov_regression.models import ReadOutMLP, Clf, ModularClf
from src.repn_learners import VQVAE, SoftTPRAutoencoder
from src.shared.constants import *
from src.repn_learners.baselines.vct.models.visual_concept_tokeniser import VCT_Encoder, VCT_Decoder
from src.shared.model_constructors import encoders_map, decoders_map, baseline_maps, clfs_map
from pathlib import Path 

def get_save_path(args, model_type: str):
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)
        subdir_suffix = '0'
        os.makedirs(f'{args.save_dir}{subdir_suffix}')
    else: 
        subdirs = list(map(lambda x: int(x), 
                           [d for d in os.listdir(args.save_dir) if 
                            d.isnumeric()]))
        subdirs.sort() 
        print(f'Subdirs[0:10] is {subdirs[0:10]}')
        if len(subdirs) > 0: 
            latest = subdirs[-1]
            subdir_suffix = int(latest)
            if os.path.exists(f'{args.save_dir}{subdir_suffix}/{model_to_save_prefix[model_type]}'):
                subdir_suffix += 1 
            elif model_type == AUTOENCODER and os.path.exists(f'{args.save_dir}{subdir_suffix}/{model_to_save_prefix[CLF]}'): 
                subdir_suffix +=1
        else: 
            subdir_suffix = 0
        if not os.path.exists(f'{args.save_dir}{subdir_suffix}'):
            os.mkdir(f'{args.save_dir}{subdir_suffix}')
    
    return f'{args.save_dir}{subdir_suffix}' 

def mkdirs(desired_path): 
    if not os.path.exists(desired_path): 
        os.makedirs(desired_path)

def save_downstream_model(model, model_file_name: str, save_path: str) -> None: 
    if isinstance(model, Clf): 
        mkdirs(os.path.join(save_path, model_to_save_prefix[CLF]))
        torch.save([CLF, model.state_dict(), 
                    model.kwargs_for_loading],
                    os.path.join(save_path, model_to_save_prefix[CLF], model_file_name))
    if isinstance(model, ModularClf): 
        mkdirs(os.path.join(save_path, model_to_save_prefix[CLF]))
        torch.save([MODULAR_CLF, model.state_dict(), 
                        model.kwargs_for_loading], 
                    os.path.join({save_path}, model_to_save_prefix[CLF], model_file_name))
    if isinstance(model, ReadOutMLP): 
        mkdirs(os.path.join(save_path, model_to_save_prefix[REGRESSOR]))
        torch.save([READOUT_MLP, model.state_dict(),
                    model.kwargs_for_loading],
                    os.path.join({save_path}, model_to_save_prefix[REGRESSOR], model_file_name))

def save_representation_model(args, model, model_file_name: str, save_path: str) -> None: 
    mkdirs(os.path.join(save_path, model_to_save_prefix[AUTOENCODER]))
    if args.model == SOFT_TPR_AE: 
        torch.save([SOFT_TPR_AE, 
                    model.state_dict(), 
                    model.kwargs_for_loading,
                    model.encoder.state_dict(), 
                    model.decoder.state_dict(), 
                    {'class': args.model_encoder, 'kwargs': model.encoder.kwargs_for_loading}, 
                    {'class': args.model_decoder, 'kwargs': model.decoder.kwargs_for_loading}],
                    os.path.join(save_path, 'ae', model_file_name))
    elif args.model == VQ_VAE: 
        torch.save([VQ_VAE, 
                    model.state_dict(), 
                    model.kwargs_for_loading], 
                    os.path.join(save_path, 'ae', model_file_name))
    elif args.model in BASELINES: 
        if args.model == VCT: 
            encoder, decoder = model
            torch.save([VCT, 
                    encoder.state_dict(), 
                    encoder.kwargs_for_loading,
                    decoder.state_dict(), 
                    decoder.kwargs_for_loading], 
                    os.path.join(save_path, 'ae', model_file_name))
        elif args.model == SHU: 
            enc, gen = model 
            torch.save([SHU, 
                        enc.state_dict(), enc.kwargs_for_loading,
                        gen.state_dict(), gen.kwargs_for_loading], 
                        os.path.join(save_path, 'ae', model_file_name))
        else: 
            torch.save([args.model, 
                    model.state_dict(),
                    model.kwargs_for_loading],
                    os.path.join(save_path, 'ae', model_file_name))
    else:
        raise NotImplementedError(f'Unknown model type {args.model}')

def get_most_recent_dir(save_dir: str, model_type: str, encoder_type: str=None) -> str: 
    dirs = [os.path.join(dp, f) for dp, dn, fn in os.walk(save_dir) for f in fn]
    relevant_dirs = list(filter(lambda x, model_type=model_type: model_type in x, dirs))
    if encoder_type is not None: 
        relevant_dirs = list(filter(lambda x, encoder_type=encoder_type: encoder_type in x, relevant_dirs))
    print(f'Relevant dirs {relevant_dirs}')
    model_dir, args_dir = relevant_dirs[-2:]
    return model_dir, args_dir

def load_most_recent_model_and_args(model_type: str, save_dir: str, encoder_type: str=None):
    if not os.path.exists(save_dir): 
        raise FileNotFoundError(f'Save dir {save_dir} does not exist') 
    else: 
        model_dir, args_dir = get_most_recent_dir(save_dir, model_to_save_prefix[model_type], encoder_type)
        return load_model_from_path(model_dir), load_args(args_dir)

def load_args(args_path: str): 
    return torch.load(args_path)

def get_corresponding_clf_from_model_path(model_path: str) -> Clf: 
    parent_dir = '/'.join(model_path.split('/')[:-2])
    path = os.path.join(parent_dir, 'clf')
    assert os.path.exists(path), f'Desired path does not exist {path}'
    child_files = [f for f in os.listdir(path) if 'args' not in f] # ignore args file 
    assert len(child_files) > 0, f'No child files in dir {path}'
    clf_path = os.path.join(path, child_files[0])
    print(f'CLF PATH IS {clf_path}')
    clf = load_model_from_path(model_path=clf_path)
    return clf

def load_corresponding_args_from_path(model_path: str): 
    parent_dir = os.path.dirname(model_path)
    paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(parent_dir) for f in fn]
    args_path = list(filter(lambda x: 'args' in x, paths))
    return load_args(args_path=args_path[0])

def load_model_from_path(model_path: str):
    print(f'Model path is {model_path}')
    model_type, *saved = torch.load(Path(model_path), map_location='cuda:0')
    if model_type == SOFT_TPR_AE: 
        model_state, model_kwargs, e_state, d_state, e_meta, d_meta = saved 
        encoder = encoders_map[e_meta['class']](**e_meta['kwargs'])
        decoder = decoders_map[d_meta['class']](**d_meta['kwargs'])
        encoder.load_state_dict(e_state)
        decoder.load_state_dict(d_state)
        if 'recon_loss_fn' not in model_kwargs.keys(): 
            model_kwargs = {**model_kwargs, 
                            'recon_loss_fn': 'mse'}
        model = SoftTPRAutoencoder(encoder=encoder, decoder=decoder, **model_kwargs)
    elif model_type == VQ_VAE: 
        model_state, model_kwargs = saved
        model = VQVAE(**model_kwargs)
    elif model_type in BASELINES:
        if model_type != VCT:  
            if model_type != SHU:
                if len(saved) == 1:
                    model_state = saved
                    model = baseline_maps[model_type]()
                else: 
                    model_state, model_kwargs = saved 
                    model = baseline_maps[model_type](**model_kwargs)
            else: 
                model_state, model_kwargs, _, _ = saved 
                model = baseline_maps[model_type](**model_kwargs)
        else: 
            encoder_state, encoder_kwargs, decoder_state, decoder_kwargs = saved 
            encoder = VCT_Encoder(**encoder_kwargs)
            decoder = VCT_Decoder(**decoder_kwargs)
            encoder.load_state_dict(encoder_state)
            decoder.load_state_dict(decoder_state)
            encoder = encoder.cuda()
            decoder = decoder.cuda() 
            return (encoder, decoder)
    elif model_type in [BASE_CLF, MODULAR_CLF]: 
        model_state, model_kwargs = saved 
        model = clfs_map[model_type](**model_kwargs)
    elif model_type == REGRESSOR: 
        model_state, model_kwargs = saved 
        model = ReadOutMLP(**model_kwargs)
    else: 
        raise NotImplementedError(f"Unknown model type {model_type}")
    
    model.load_state_dict(model_state)    
    return model.cuda()

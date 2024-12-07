import torch 
from torch.utils.data import DataLoader
from src.data.datasets import custom_collate, own_collate
from src.data.modify_datasets import \
    modify_dataset, convert_to_pairs_dataset
from src.shared.constants import *


def get_dataloaders(args, full_dataset): 
    collate_fn = custom_collate if args.model in BASELINES else own_collate
    dataset_train, dataset_test, datasets_evaluation = \
        modify_dataset(args, full_dataset,
                       None)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                   num_workers=args.n_workers, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                                  num_workers=args.n_workers, shuffle=True)
    dataloader_full = DataLoader(full_dataset, batch_size=args.batch_size,
                                  num_workers=args.n_workers, shuffle=True)
                                                       
    if args.supervision_mode == WEAKLY_SUPERVISED:
        # train with pairs of images
        dataset_train_pairs = \
            convert_to_pairs_dataset(dataset_train,
                                    args.transition_prior,
                                    args.k,
                                    maximally_distinct=args.maximally_distinct)
        dataloader_train = DataLoader(dataset_train_pairs,
                                    batch_size=args.batch_size,
                                    num_workers=args.n_workers,
                                    shuffle=True,
                                    collate_fn=collate_fn)
    return dataloader_train, dataloader_test, dataloader_full

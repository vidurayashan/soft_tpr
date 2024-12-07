"""
This code has been adapted from the supplementary material linked to the paper 
'Visual Representation Learning Does Not Generalize Strongly Within the Same Domain' (ICLR 2022) 
accessible at https://openreview.net/forum?id=9RUHPlladgh

@inproceedings{
schott2022visual,
title={Visual Representation Learning Does Not Generalize Strongly Within the Same Domain},
author={Lukas Schott and Julius Von K{\"u}gelgen and Frederik Tr{\"a}uble and Peter Vincent Gehler and Chris Russell and Matthias Bethge and Bernhard Sch{\"o}lkopf and Francesco Locatello and Wieland Brendel},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=9RUHPlladgh}
}
"""

from typing import Tuple

import numpy as np
import torch 

from src.data.datasets import DisLibDataset, \
    PairsDataset

def interpolation_indices(total, min_cut=0.1, keep_all=False):
    # drop indices as equally spaced as possible starting from the middle
    # drop at least one
    if keep_all:
        return []
    pad = int(np.ceil(total * min_cut))
    indices = np.linspace(0, total - 1, total - pad, endpoint=True)
    indices = np.round(indices).astype(np.int32).tolist()
    full_indices = range(total)
    indices_to_keep = list(set(full_indices) - set(indices))
    return indices_to_keep

def extrapolate_indices(total, min_cut=0.1, keep_all=False):
    if keep_all:
        return []
    pad = int(np.ceil(total * min_cut))
    indices = list(range(total - pad, total))
    return indices

def composition_indices(total, min_cut=0.1, keep_all=False):
    if keep_all:
        return []
    # for composition we switch persectives and treat it like
    # extrapolation because here we can defnine rules per
    # indice and the generalize across dimensions.
    # train and test switched from extrapolation --> floor instead of ceil
    pad = int(np.floor(total * min_cut))
    indices = list(range(0, pad))
    return indices

def get_mask(labels, all_factors):
    masks = []
    for i, factors_to_keep in enumerate(all_factors):
        for factor_i in factors_to_keep:
            masks.append(labels[:, i] == factor_i)
    final = np.any(masks, axis=0)
    return final

def get_mask(labels, all_factors):
    masks = []
    for i, factors_to_keep in enumerate(all_factors):
        for factor_i in factors_to_keep:
            masks.append(labels[:, i] == factor_i)
    final = np.any(masks, axis=0)
    return final

def get_all_magenta(labels): 
    masks = [] 
    masks.append(labels[:, 2] == 9)
    final = np.any(masks, axis=0)
    return final

def get_mask_per_factor(labels, all_factors):
    masks_per_factors = []
    for i, factors_to_keep in enumerate(all_factors):
        masks = []
        for factor_i in factors_to_keep:
            masks.append(labels[:, i] == factor_i)
        masks_per_factors.append(np.any(masks, axis=0))
    return masks_per_factors

def get_fixed_random_mask(positive_ratio, size, random_state=0):
    rng = np.random.RandomState(random_state)
    random_mask = np.zeros(size, dtype=np.bool)
    random_mask[:int(size * positive_ratio)] = True
    rng.shuffle(random_mask)  # in-place
    return random_mask

def modify_dataset(args, dataset: DisLibDataset, modification: str=None,
                   test_ratio_per_factor: float=0) \
        -> Tuple[DisLibDataset, DisLibDataset, dict]:
    if modification is None:
        print(f'****NO MODIFICATIONS****RETURNING DATASET, DATASET')
        return dataset, dataset, {}
    
    factor_sizes = dataset.factor_sizes
    is_test_and_train = dataset.test_and_train.tolist()

    number_varying_factors = len(is_test_and_train) - np.sum(is_test_and_train)
    factors_to_keep = []
    invert = False
    for factor_size_i, is_test_and_train_i in zip(factor_sizes,
                                                  is_test_and_train):
        if modification == 'interpolation':
            factors_to_keep_i = \
                interpolation_indices(factor_size_i,
                                      min_cut=test_ratio_per_factor,
                                      keep_all=is_test_and_train_i)
        elif (modification == 'extrapolation' or modification == 'random'):
            # for random mimic extrapolation to have the same
            # test / total split ratio
            factors_to_keep_i = \
                extrapolate_indices(factor_size_i,
                                    min_cut=test_ratio_per_factor,
                                    keep_all=is_test_and_train_i)
        elif modification == 'composition':
            # invert perspective: composition is the opposite of extrapolation
            q = 1 - (1. - (
                    1. - test_ratio_per_factor) ** number_varying_factors) ** (
                        1. / number_varying_factors)
            if dataset.__class__.__name__ == 'CelebGlowDataset':
                q = 0.2  # hardcode because otherwise 0 factors in train
            factors_to_keep_i = \
                composition_indices(factor_size_i,
                                    min_cut=q,
                                    keep_all=is_test_and_train_i)
            invert = True
        else:
            if not (args.dataset == 'shapes3d' and args.modification == 'magenta_unseen'):
                raise Exception('specified modification not available')
            else: 
                factors_to_keep_i = extrapolate_indices(factor_size_i, 
                                                        min_cut=0, 
                                                        keep_all=is_test_and_train_i)
        factors_to_keep.append(factors_to_keep_i)

    labels = dataset.labels

    print(labels)
    
    if modification != 'magenta_unseen':
        test_mask = get_mask(labels, factors_to_keep)
    else:
        test_mask = get_all_magenta(labels)

    if invert:
        test_mask = ~test_mask

    if modification == 'random':
        # random split with the same test train ration as inter- and
        # extrapolation
        test_ratio = args.test_split_ratio
        test_mask = get_fixed_random_mask(positive_ratio=test_ratio,
                                          size=np.prod(factor_sizes))
    # memory effecient index based data splitting:
    # each class shares data array but has disjiont set of indices
    print('sum test_mask', np.sum(test_mask.astype(np.float64)))
    train_mask = ~test_mask

    if modification in ['interpolation', 'extrapolation']:
        factor_evaluation_datasets = one_factor_ood(dataset, factors_to_keep)

        def get_missing_elements(full_list, sparse_list):
            return list(set(full_list) - set(sparse_list))

        all_factors = dataset.all_factors
        train_factors = [get_missing_elements(l1, l2) for l1, l2 in
                         zip(all_factors, factors_to_keep)]
    else:
        train_factors = None
        factor_evaluation_datasets = {}

    dataset_train = shared_memory_class_subset(dataset, train_mask,
                                               train_factors)
    dataset_test = shared_memory_class_subset(dataset, test_mask)

    # some sanity checks
    disjoint_mask = dataset_train.mask ^ dataset_test.mask
    assert np.sum(disjoint_mask) == np.sum(dataset.mask)
    assert np.sum(dataset.mask) == len(dataset) == len(dataset.indices_map)

    return dataset_train, dataset_test, factor_evaluation_datasets

def one_factor_ood(dataset, factors_to_keep):
    mask_per_factors = get_mask_per_factor(dataset.labels, factors_to_keep)
    number_of_factors_ood = np.sum(
        [m for m, c in zip(mask_per_factors, dataset.test_and_train) if not c],
        axis=0)
    # mask s.t. only one factor is ood
    only_one_ood = number_of_factors_ood == 1
    data_loaders = {}
    for mask_per_factor, is_test_and_train, name in zip(mask_per_factors,
                                                        dataset.test_and_train,
                                                        dataset.factor_names):
        if is_test_and_train:
            continue
        mask = np.all([mask_per_factor, only_one_ood], axis=0)
        evaluation_dataset = shared_memory_class_subset(dataset, mask)
        data_loaders[name] = evaluation_dataset
    return data_loaders

def shared_memory_class_subset(dataset: DisLibDataset,
                               mask,
                               available_factors=None) -> DisLibDataset:
    new_dataset = DisLibDataset(
        factor_sizes=dataset.factor_sizes,
        data=dataset._data,  # datasets share memory
        categorical=dataset.categorical.tolist(),
        test_and_train=dataset.test_and_train.tolist(),
        indices_map=dataset.indices_map[mask].copy(),  # avoid inplace changes
        factor_names=dataset.factor_names,
        transform=dataset.transform,
        available_factors=available_factors,
        root='')
    return new_dataset

def convert_to_pairs_dataset(dataset: DisLibDataset,
                             transition_prior: str,
                             max_number_changing_factors: int,
                             maximally_distinct: bool=False,
                             modifictation: str=None) -> DisLibDataset:
    if isinstance(dataset, torch.utils.data.Subset): 
        dataset = dataset.dataset
       
    if modifictation is None:
        available_factors = dataset.all_factors
    else:
        available_factors = dataset.available_factors

    new_dataset = PairsDataset(
        transition_prior=transition_prior,
        max_number_changing_factors=max_number_changing_factors,
        available_factors=available_factors,
        factor_sizes=dataset.factor_sizes,
        data=dataset._data,
        maximally_distinct=maximally_distinct,
        categorical=dataset.categorical.tolist(),
        test_and_train=dataset.test_and_train.tolist(),
        indices_map=dataset.indices_map.copy(),  # avoid inplace changes
        factor_names=dataset.factor_names,
        transform=dataset.transform,
        root='')
    new_dataset.indices_map = dataset.indices_map
    return new_dataset

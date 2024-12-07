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

import logging
import os
from typing import Tuple, List
from urllib import request

import h5py
import numpy as np
import torch
import PIL 
from PIL.Image import fromarray
from scipy.stats import laplace
import scipy.io as sio
from sklearn.utils.extmath import cartesian
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class IndexManger(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes: List[int]):
        """Index to latent (= features) space and vice versa.
        Args:
          factor_sizes: List of integers with the number of distinct values for each
            of the factors.
        """
        self.factor_sizes = np.array(factor_sizes)
        self.num_total = np.prod(self.factor_sizes)
        self.factor_bases = self.num_total / np.cumprod(self.factor_sizes)

        self.index_to_feat = cartesian(
            [np.array(list(range(i))) for i in self.factor_sizes])

    def features_to_index(self, features):
        """Returns the indices in the input space for given factor configurations.
        Args:
          features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the input space should be
            returned.
        """
        assert np.all((0 <= features) & (features <= self.factor_sizes))
        index = np.array(np.dot(features, self.factor_bases), dtype=np.int64)
        assert np.all((0 <= index) & (index < self.num_total))
        return index

    def index_to_features(self, index: int) -> np.ndarray:
        assert np.all((0 <= index) & (index < self.num_total))
        features = self.index_to_feat[index]
        assert np.all((0 <= features) & (features <= self.factor_sizes))
        return features


import torch.nn.functional as F 

class DisLibDataset(VisionDataset):
    _index_manager: IndexManger = None
    categorical: np.ndarray = None

    def __init__(self,
                 factor_sizes: List[int],
                 data: np.ndarray,
                 categorical: List[bool],
                 test_and_train: List[bool] = None,
                 factor_names: List[str] = None,
                 available_factors: List[int] = None,
                 indices_map: np.ndarray = None,
                 remove_inactives: bool = True,
                 **vision_dataset_kwargs):
        super().__init__(**vision_dataset_kwargs)
        assert len(categorical) == len(factor_sizes)
        assert len(factor_names) == len(factor_sizes)
        assert len(data) == np.prod(factor_sizes)

        self._factor_sizes = factor_sizes
        self._data = data
        self.categorical = np.array(categorical)
        self.test_and_train = np.array(
            categorical) if test_and_train is None else np.array(test_and_train)
        self.factor_names = factor_names

        self._index_manager = IndexManger(factor_sizes)
        if indices_map is None:
            self.indices_map = np.arange(len(data)).astype(np.int64)
        else:
            self.indices_map = indices_map
        self.all_factors = [list(range(f)) for f in factor_sizes]

        if available_factors is None:
            self.available_factors = \
                [np.unique(l_i).tolist() for l_i in self.labels.T]
        else:
            self.available_factors = available_factors

        # number of active factors required for metric evaluations s.a. mig
        # which cannot deal with constant inputs
        self.active_factors = [len(af) > 1 for af in self.available_factors]
        self.remove_inactives = remove_inactives

        # mask for test train splits
        mask = np.zeros(np.prod(self.factor_sizes), dtype=bool)
        mask[self.indices_map] = True
        self.mask = mask.reshape(*self.factor_sizes)

    @property
    def categorical_factor_index(self):
        return np.argwhere(self.categorical)[0, 0]

    @property
    def labels(self):
        return self._index_manager.index_to_feat[self.indices_map]
    
    @property 
    def one_hot_labels(self): 
        cumsum = np.cumsum([0] + list(self.factor_sizes[:-1]))
        offset_factor_classes_for_clf = torch.from_numpy(self.labels + cumsum).to(dtype=torch.int64)

        one_hot = F.one_hot(offset_factor_classes_for_clf, num_classes=np.sum(self.factor_sizes))
        return torch.sum(one_hot, dim=1)
    
    @staticmethod 
    def convert_to_one_hot(factors, factor_sizes, normalize: bool=True):
        if normalize: 
            factors = factors *  (np.array(factor_sizes) - 1)
        cumsum = np.cumsum([0] + list(factor_sizes[:-1]))
        offset_factor_classes_for_clf = (factors + cumsum).to(dtype=torch.int64)

        one_hot = F.one_hot(offset_factor_classes_for_clf, num_classes=np.sum(factor_sizes))
        return torch.sum(one_hot, dim=1)

    @property
    def _labels(self):
        return self._index_manager.index_to_feat

    @property
    def data(self):
        return self._data[self.indices_map]

    @property
    def factor_sizes(self):
        return self._factor_sizes

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, idx: int, normalize: bool = True):
        _indices = self.convert_indices(idx)
        data = self._data[_indices]
        targets = self._labels[_indices]
        if normalize:
            targets = targets / (np.array(self.factor_sizes) - 1)

        # allow for consistency with vision datasets
        image = self.to_img(data)
        if self.transform is not None:
            image = self.transform(image)
        #print(f'Len of one hot {self.one_hot_labels.shape}, of data {self._data.shape} of factors {self._labels.shape}')
        return image, targets

    @property
    def num_factors(self):
        return len(np.array(self.factor_sizes)[self.active_factors])

    def convert_indices(self, indices):
        return self.indices_map[indices]

    def get_normalized_labels(self):
        return self._labels / (np.array(self.factor_sizes) - 1)

    def to_img(self, tensor):
        assert len(tensor.shape) == 3
        if tensor.shape[0] == 1:
            return fromarray(tensor[0])
        elif tensor.shape[0] == 3:
            if tensor.dtype == np.uint8:
                return fromarray(np.transpose(tensor, (1, 2, 0)))
            return fromarray((tensor * 255).astype(np.uint8).transpose(1, 2, 0))
        elif tensor.shape[2] == 3 and tensor.dtype != np.uint8: 
            return fromarray((tensor * 255).astype(np.uint8))
        else:
            print(f'Tensor is {tensor.shape} with type {type(tensor)}')
            raise NotImplementedError

    def sample_factors(self, num, random_state, return_indices=False):
        """Sample a batch of observations X. Needed in dis. lib."""
        batch_size = num
        replace = (False if self.__len__() > batch_size else True)
        indices = random_state.choice(self.__len__(),
                                      batch_size,
                                      replace=replace)
        _indices = self.convert_indices(indices)
        factors = self._labels[_indices].astype(np.int32)

        if self.remove_inactives:
            factors = factors.T[self.active_factors].T

        if return_indices:
            return factors, indices
        else:
            return factors

    def sample_observations_from_factors(self, factors, random_state, tensorify: bool=False):
        factor_idxs = self._index_manager.features_to_index(factors)
        if tensorify: 
            return torch.from_numpy(self.get_observation_from_index(factor_idxs).astype(dtype=np.float32)).cuda()
        return self.get_observation_from_index(factor_idxs)

    def get_observation_from_index(self, indices):
        _indices = self.convert_indices(indices)
        samples = self._data[_indices]
        if len(samples.shape) == 3:  # set channel dim to 1
            samples = samples[:, None]
        if np.issubdtype(samples.dtype, np.uint8):
            samples = samples.astype(np.float32) / 255.
        if np.issubdtype(samples.dtype, np.float64):
            samples = samples.astype(np.float32) # weird sampling business??? 
        return samples

    def sample(self, num, random_state, tensorify: bool=False):
        # Sample a batch of factors Y and observations X
        factors, indices = self.sample_factors(num, random_state,
                                               return_indices=True)
        observations = self.get_observation_from_index(indices)
        if tensorify: 
            return torch.from_numpy(factors.astype(np.int32)).cuda(), torch.from_numpy(observations).cuda()
        return factors.astype(np.int32), observations

    def sample_observations(self, num, random_state, tensorify: bool=False):
        # Sample a batch of observations X
        return self.sample(num, random_state, tensorify)[1]
    
    def sample_n_large(self, num, random_state): 
        i = 0 
        obs = None 
        facs = None 
        while i < num: 
            n_points_increment = min(num - i, 216) 
            facs_batch, obs_batch = self.sample(num=n_points_increment)
            if i == 0: 
                obs = obs_batch 
                facs = facs_batch 
            else: 
                obs = np.vstack((obs, obs_batch))
                facs = np.vstack((facs, facs_batch)) 
            i += n_points_increment
            if i % 2160 == 0: 
                print(f'{i} out of {num} samples generated')
        return obs, facs
            
    
    def sample_observations_n_large(self, num, random_state): 
        i = 0 
        obs = None 
        while i < num: 
            n_points_increment = min(num - i, 512)
            ob = self.sample_observations(n_points_increment, random_state)
            if i == 0: 
               obs = ob 
            else: 
                obs = np.vstack((obs, ob))
            i += n_points_increment
            if i % 1280 == 0: 
                print(f'{i} out of {num} samples generated')
        return obs 


class PairsDataset(DisLibDataset):
    def __init__(self,
                 max_number_changing_factors=-1,  # uniform data
                 data_laplace_rate=1,
                 transition_prior='laplace',
                 maximally_distinct: bool=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_number_changing_factors = max_number_changing_factors
        self.transition_prior = transition_prior
        self.data_laplace_rate = data_laplace_rate
        self.maximally_distinct=maximally_distinct

    def get_available_factors(self, label):
        assert self.mask[tuple(label)]  # requested label not in split
        available_factors_cross = []
        for factor_args in range(len(self.factor_sizes)):
            label_list = label.copy().tolist()
            label_list[factor_args] = slice(None)  # get all values along axis
            mask_per_factor = self.mask[tuple(label_list)]  # index all dims
            available_factors = np.arange(len(mask_per_factor))[mask_per_factor]
            available_factors_cross.append(available_factors)
        return available_factors_cross

    def __getitem__(self, idx: int, normalize: bool = True):
        # first sample is sampled uniformly
        _indices = self.convert_indices(idx)
        first_image = self._data[_indices]
        first_factors = self._labels[_indices]
        if self.transition_prior == 'locatello':
            # only change up to k factors
            if self.max_number_changing_factors == -1:
                k = np.random.randint(1, self.num_factors)
            else:
                k = self.max_number_changing_factors
            second_factors = first_factors.copy()
            indices = np.random.choice(len(self.factor_sizes),
                                       len(self.factor_sizes), replace=False)
            factors_changed = 0
            for i in indices:
                avail = self.get_available_factors(np.array(second_factors))[i]
                avail = np.array(avail)

                if avail.shape[0] > 1:
                    factors_changed += 1
                    p = np.ones_like(avail) / (avail.shape[0] - 1)
                    p[avail == first_factors[i]] = 0  # dont pick same
                else:
                    p = np.ones(1)
                if self.maximally_distinct: 
                    second_factors[i] = avail[np.argmax(abs(avail - first_factors[i]))]
                else: 
                    second_factors[i] = np.random.choice(avail, 1, p=p)
                if factors_changed == k:
                    break
            if not np.equal(first_factors - second_factors,
                            0).sum() == self.num_factors - k:
                # this is very unlikely but can happen if no datapoints
                # are available along any axis in a random split.
                # For the systematic splits where we keep all categorical
                # variables, this cannot happen if k=1.
                print('k not matched', k, first_factors, second_factors)

        elif self.transition_prior == 'laplace':
            rate = self.data_laplace_rate
            second_factors = first_factors.copy()
            # sample each feature individually
            for i, (mean, upper, is_categorical) in enumerate(
                    zip(first_factors, self.factor_sizes, self.categorical)):
                if is_categorical:
                    continue
                avail = self.get_available_factors(np.array(second_factors))[i]
                p = laplace.pdf(avail, loc=mean, scale=np.log(upper) / rate)
                p /= np.sum(p)
                second_factors[i] = np.random.choice(avail, 1, p=p)[0]
            second_factors = np.array(second_factors).astype(np.int64)
        else:
            raise NotImplementedError
        _second_sample_index = self._index_manager.features_to_index(
            second_factors)
        second_image = self._data[_second_sample_index]
        # allow for consistency with vision datasets
        first_image = self.to_img(first_image)
        second_image = self.to_img(second_image)
        if self.transform:
            first_image = self.transform(first_image)
            second_image = self.transform(second_image)

        assert np.all(
            [f in t for t, f in zip(self.available_factors, first_factors)])
        assert np.all(
            [f in t for t, f in zip(self.available_factors, second_factors)])
        # requested label not in split
        assert self.mask[tuple(first_factors)], f'{first_factors}'
        assert self.mask[tuple(second_factors)], f'{second_factors}'

        if normalize:
            first_factors = first_factors / (np.array(self.factor_sizes) - 1)
            second_factors = second_factors / (np.array(self.factor_sizes) - 1)

        return first_image, second_image, first_factors, second_factors



def own_collate(batch: List[Tuple]): 
    x1, x2, factor_cls1, factor_cls2 = zip(*batch)
    return torch.tensor(np.stack([*x1, *x2], axis=0)), torch.tensor(np.stack([*factor_cls1, *factor_cls2], axis=0))

def custom_collate(sample):
    inputs, labels = [], []
    for s in sample:
        inputs.append(s[0])
        inputs.append(s[1])
        labels.append(s[2])
        labels.append(s[3])
    return torch.tensor(np.stack(inputs)), torch.tensor(np.stack(labels))


class SpriteDataset(DisLibDataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    # dim, type,     #values  avail.-range
    * 0, color       |  1 |     1-1
    * 1, shape       |  3 |     1-3
    * 2, scale       |  6 |     0.5-1.
    * 3, orientation | 40 |     0-2pi
    * 4, x-position  | 32 |     0-1
    * 5, y-position  | 32 |     0-1
    for details see https://github.com/deepmind/dsprites-dataset
    """
    url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    def __init__(self, path: str = '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets//dsprites', **vision_dataset_kwargs):
        factor_names = ['shape', 'scale', 'orientation', 'x-position',
                        'y-position']

        try:
            data = self.load_data('/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets//dsprites')
        except FileNotFoundError:
            raise FileNotFoundError
            #if not os.path.exists(path):
            #    os.makedirs(path, exist_ok=True)
            #LOGGER.info(
            #    f'downloading dataset ... saving to {os.path.join(path, "dsprites.npz")}'
            #)
            #request.urlretrieve(self.url, os.path.join(path, 'dsprites.npz'))
            #data = self.load_data(path)
        super().__init__(
            factor_sizes=[3, 6, 40, 32, 32],
            data=data,
            categorical=[True, False, False, False, False],
            root=path,
            factor_names=factor_names,
            **vision_dataset_kwargs)
        assert self._data.shape[1] == 1
        assert self._data.shape[2] == 64
        assert self._data.shape[3] == 64

    def load_data(self, path):
        dataset_zip = np.load(os.path.join(path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'),
                              encoding='latin1',
                              allow_pickle=True)
        return dataset_zip['imgs'].astype(np.uint8)[:, None, :, :] * 255

# use disentanglement lib code 
class ScreamDSprites(SpriteDataset):
    def __init__(self, path: str = '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets//dsprites', 
                 scream_path: str='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets//dsprites/scream.jpg',
                 **vision_dataset_kwargs):
       super().__init__(path, **vision_dataset_kwargs)
       scream = PIL.Image.open(scream_path)
       scream.thumbnail((350, 274, 3))
       self.scream = np.array(scream) * 1./255.

    def __getitem__(self, idx: int): 
        img, factor = super().__getitem__(idx)
        x_crop = np.randint(0, self.scream.shape[0] - 64)
        y_crop = np.randint(0, self.scream.shape[1] - 64)
        background = (self.scream[x_crop: x_crop+64, 
                                  y_crop: y_crop+64] + np.random.uniform(0, 1, size=3)) / 2
        mask = (img == 1)
        background[mask] = 1 - background[mask]
        img = background 
        return img, factor 


class CelebGlowDataset(DisLibDataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    # dim, type,     #values  avail.-range
    * 0, color       |  1 |     1-1
    * 1, shape       |  3 |     1-3
    * 2, scale       |  6 |     0.5-1.
    * 3, orientation | 40 |     0-2pi
    * 4, x-position  | 32 |     0-1
    * 5, y-position  | 32 |     0-1
    for details see https://github.com/deepmind/dsprites-dataset
    """
    # url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    def __init__(self, path: str = './data/celeb_glow/', **vision_dataset_kwargs):
        factor_names = ['person', 'smile', 'blond', 'age']

        # try:
        data = self.load_data(path)
        # except FileNotFoundError:
            # if not os.path.exists(path):
            #     os.makedirs(path, exist_ok=True)
            # LOGGER.info(
            #     f'downloading dataset ... saving to {os.path.join(path, "dsprites.npz")}'
            # )
            # request.urlretrieve(self.url, os.path.join(path, 'dsprites.npz'))
            # data = self.load_data(path)
        super().__init__(
            factor_sizes=[1000, 6, 6, 6],
            data=data,
            categorical=[True, False, False, False],
            root=path,
            factor_names=factor_names,
            **vision_dataset_kwargs)
        assert self._data.shape[1] == 3
        assert self._data.shape[2] == 64
        assert self._data.shape[3] == 64

    def load_data(self, path):
        dataset = np.load(os.path.join(path, 'celeba_glow_all_images_64.npy'))
        print('datasdt', np.max(dataset), dataset.dtype, np.min(dataset), dataset.shape)
        # dataset = np.swapaxes(dataset, 0, 3)
        # print('datasdt', np.max(dataset), dataset.dtype, np.min(dataset), dataset.shape)
        return dataset.reshape((1000*6*6*6, 64, 64, 3)).transpose(0, 3, 1, 2)


class SpriteDatasetRot90(DisLibDataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    # dim, type,     #values  avail.-range
    * 0, color       |  1 |     1-1
    * 1, shape       |  3 |     1-3
    * 2, scale       |  6 |     0.5-1.
    * 3, orientation | 10 |     0-pi/2
    * 4, x-position  | 32 |     0-1
    * 5, y-position  | 32 |     0-1
    for details see https://github.com/deepmind/dsprites-dataset
    """
    url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    def __init__(self, path: str = '/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets//dsprites', **vision_dataset_kwargs):
        factor_names = ['shape', 'scale', 'orientation', 'x-position',
                        'y-position']

        try:
            data = self.load_data('/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets//dsprites')
        except FileNotFoundError:
            raise FileNotFoundError
            #if not os.path.exists(path):
            #    os.makedirs(path, exist_ok=True)
            #LOGGER.info(
            #    f'downloading dataset ... saving to {os.path.join(path, "dsprites.npz")}'
            #)
            #request.urlretrieve(self.url, os.path.join(path, 'dsprites.npz'))
            #data = self.load_data(path)
        super().__init__(
            factor_sizes=[3, 6, 10, 32, 32],
            data=data,
            categorical=[True, False, False, False, False],
            root=path,
            factor_names=factor_names,
            **vision_dataset_kwargs)
        assert self._data.shape[1] == 1
        assert self._data.shape[2] == 64
        assert self._data.shape[3] == 64

    def load_data(self, path):
        print(f'PATH IS {path}')
        dataset_zip = np.load(os.path.join(path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'),
                              encoding='latin1',
                              allow_pickle=True)
        dataset_zip = dataset_zip['imgs'].astype(np.uint8)[:, None, :, :] * 255
        print('dataset shape', dataset_zip.shape)
        dataset_zip = dataset_zip.reshape(3, 6, 40, 32, 32, 1, 64, 64)
        print('dataset shape', dataset_zip.shape)
        dataset_zip = dataset_zip[:, :, :10, :, :, ...]  # only keep 10 out 40 possible rotations
        dataset_zip = dataset_zip.reshape(3*6*10*32*32, 1, 64, 64)

        print('dataset shape no rot', dataset_zip.shape)

        return dataset_zip

class Shapes3D(DisLibDataset):
    """Shapes3D dataset.
    self.factor_sizes = [10, 10, 10, 8, 4, 15]

    The data set was originally introduced in "Disentangling by Factorising".
    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    """
    # url = 'https://liquidtelecom.dl.sourceforge.net/project/shapes3d/Shapes3D.zip'
    # fname = 'shapes3d.pkl'
    url = 'https://storage.googleapis.com/3d-shapes/3dshapes.h5'
    fname = '3dshapes.h5'

    def __init__(self,
                 path='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets/',
                 **vision_dataset_kwargs):
        factor_names = [
            'floor color', 'wall color', 'object color', 'object size',
            'object type', 'azimuth']

        file_path = os.path.join(path, 'shapes3d', self.fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} not found')
            #self.download(path)

        with h5py.File(file_path, 'r') as dataset:
            images = dataset['images'][()]
        data = np.transpose(images, (0, 3, 1, 2))  # np.uint8

        super().__init__(
            factor_sizes=[10, 10, 10, 8, 4, 15],
            data=data,
            categorical=[False, False, False, False, True, False],
            root=path,
            factor_names=factor_names,
            **vision_dataset_kwargs)
        assert self._data.shape[1] == 3
        assert self._data.shape[2] == 64
        assert self._data.shape[3] == 64

    def download(self, path):
        LOGGER.info('downloading shapes3d')
        os.makedirs(path, exist_ok=True)
        request.urlretrieve(self.url, os.path.join(path, self.fname))


class MPI3DReal(DisLibDataset):
    """
    object_color	white=0, green=1, red=2, blue=3, brown=4, olive=5
    object_shape	cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5
    object_size	small=0, large=1
    camera_height	top=0, center=1, bottom=2
    background_color	purple=0, sea green=1, salmon=2
    horizontal_axis	0,...,39
    vertical_axis	0,...,39
    """
    url = 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz'
    fname = 'mpi3d_real.npz'

    def __init__(self, path='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets/', **vision_dataset_kwargs):
        factor_names = ['color', 'shape', 'size', 'height', 'bg color',
                        'x-axis', 'y-axis']

        file_path = os.path.join(path, 'mpi3d', self.fname)
        if not os.path.exists(path):
            raise FileNotFoundError
        
        data = np.load(file_path)['images']

        # np.uint8
        #data = np.transpose(data.reshape([-1, 64, 64, 3]), (0, 3, 1, 2))
        super().__init__(
            factor_sizes=[6, 6, 2, 3, 3, 40, 40],
            data=data,
            categorical=[True, True, False, False, True, False, False],
            test_and_train=[False, True, True, False, False, False, False],
            root=path,
            factor_names=factor_names,
            **vision_dataset_kwargs)
        #assert self._data.shape[1] == 3
        #assert self._data.shape[2] == 64
        #assert self._data.shape[3] == 64

    def download(self, path):
        os.makedirs(path, exist_ok=True)
        LOGGER.info('downloading MPI3D real')
        request.urlretrieve(self.url, os.path.join(path, self.fname))
        LOGGER.info('download complete')


class MPI3DSimplistic(DisLibDataset):
    """
    object_color	white=0, green=1, red=2, blue=3, brown=4, olive=5
    object_shape	cone=0, cube=1, cylinder=2, hexagonal=3, pyramid=4, sphere=5
    object_size	small=0, large=1
    camera_height	top=0, center=1, bottom=2
    background_color	purple=0, sea green=1, salmon=2
    horizontal_axis	0,...,39
    vertical_axis	0,...,39
    """
    url = 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz'
    fname = 'mpi3d_toy.npz'

    def __init__(self, path='./data/mpi3d_real/', **vision_dataset_kwargs):
        factor_names = ['color', 'shape', 'size', 'height', 'bg color',
                        'x-axis', 'y-axis']

        if not os.path.exists(os.path.join(path, self.fname)):
            self.download(path)

        load_path = os.path.join(path, self.fname)
        data = np.load(load_path)['images']

        # np.uint8
        data = np.transpose(data.reshape([-1, 64, 64, 3]), (0, 3, 1, 2))
        super().__init__(
            factor_sizes=[6, 6, 2, 3, 3, 40, 40],
            data=data,
            categorical=[False, True, False, False, False, False, False],
            test_and_train=[False, True, True, False, False, False, False],
            root=path,
            factor_names=factor_names,
            **vision_dataset_kwargs)
        assert self._data.shape[1] == 3
        assert self._data.shape[2] == 64
        assert self._data.shape[3] == 64

    def download(self, path):
        os.makedirs(path, exist_ok=True)
        LOGGER.info('downloading MPI3D real')
        request.urlretrieve(self.url, os.path.join(path, self.fname))
        LOGGER.info('download complete')

# from VCT code 

def _features_to_state_space_index(features):
    """Returns the indices in the atom space for given factor configurations.

    Args:
    features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the atom space should be
        returned.
    """
    factor_sizes = [4, 24, 183]
    num_total_atoms = np.prod(factor_sizes)
    factor_bases = num_total_atoms / np.cumprod(factor_sizes)
    if (np.any(features > np.expand_dims(factor_sizes, 0)) or
        np.any(features < 0)):
        raise ValueError("Feature indices have to be within [0, factor_size-1]!")
    return np.array(np.dot(features, factor_bases), dtype=np.int64)


def features_to_index(features):
    """Returns the indices in the input space for given factor configurations.

    Args:
        features: Numpy matrix where each row contains a different factor
        configuration for which the indices in the input space should be
        returned.
    """
    factor_sizes = [4, 24, 183]
    num_total_atoms = np.prod(factor_sizes)
    lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
    global_features = cartesian([np.array(list(range(i))) for i in factor_sizes])
    feature_state_space_index = _features_to_state_space_index(global_features)
    lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
    state_space_to_save_space_index = lookup_table
    state_space_index = _features_to_state_space_index(features)
    return state_space_to_save_space_index[state_space_index]


def _load_mesh(filename):
    """Parses a single source file and rescales contained images."""
    with open(filename, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail((64, 64), PIL.Image.LANCZOS)
        rescaled_mesh[i, :, :, :] = np.array(pic)
    return rescaled_mesh * 1. / 255

def _load_data(dset_folder):
    dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
    all_files = [x for x in os.listdir(dset_folder) if ".mat" in x]
    for i, filename in enumerate(all_files):
        data_mesh = _load_mesh(os.path.join(dset_folder, filename))
        factor1 = np.array(list(range(4)))
        factor2 = np.array(list(range(24)))
        all_factors = np.transpose([
          np.tile(factor1, len(factor2)),
          np.repeat(factor2, len(factor1)),
          np.tile(i,
                  len(factor1) * len(factor2))
        ])
        indexes = features_to_index(all_factors)
        dataset[indexes] = data_mesh
    return dataset


class Cars3D(DisLibDataset):
    # from VCT code 
    """Cars3D data set.

    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.   
    The ground-truth factors of variation are:
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)
    """ 
    def __init__(self, path: str='/media/bethia/aba5749c-a217-4cbb-8a86-2d95002d9fe21/Data/datasets//', 
                 **vision_dataset_kwargs):
        data = _load_data(os.path.join(path, 'cars3d', 'data')).transpose(0, 3, 1, 2)
        factor_names = ['elevation', 'azimuth', 'object_type']
        categorical = [False, False, True]
        factor_sizes = [4, 24, 183]

        super().__init__(
            factor_sizes=factor_sizes, 
            data=data, 
            categorical=categorical, 
            factor_names=factor_names, 
            root=path,
            **vision_dataset_kwargs
        )

class DummyDataset(DisLibDataset):
    def __init__(self, path: str = '', **vision_dataset_kwargs):
        data = np.zeros((self.factor_sizes + [1, 64, 64]), dtype=np.uint8)

        super().__init__(
            factor_sizes=[10, 10, 10, 8, 4, 15],
            data=data,
            categorical=[True, False, False, False, False],
            root=path,
            factor_names=[str(i) for i in range(5)],
            **vision_dataset_kwargs)


class SmallDummyDataset(DisLibDataset):
    def __init__(self, path: str = '', **vision_dataset_kwargs):
        fs = [10, 10, 10]
        data = np.zeros([np.prod(fs)] + [1, 64, 64], dtype=np.uint8)
        super().__init__(
            factor_sizes=fs,
            data=data,
            categorical=[True, False, False],
            root=path,
            factor_names=[str(i) for i in range(3)],
            **vision_dataset_kwargs)


def get_dataset(dataset_name: str, data_path: str) -> \
        Tuple[DisLibDataset, int, int, float]:
    transform = ToTensor()
    if dataset_name.lower() == 'dsprites':
        raise NotImplementedError('Do not use dsprites without controlling for injectivity')
        #dataset_class = SpriteDataset
    elif dataset_name.lower() == 'dsprites_rot90':
        dataset_class = SpriteDatasetRot90
    elif dataset_name.lower() == 'scream_dsprites': 
        dataset_class = ScreamDSprites
    elif dataset_name.lower() == 'shapes3d':
        dataset_class = Shapes3D
    elif dataset_name.lower() == 'mpi3d':
        dataset_class = MPI3DReal
    elif dataset_name.lower() == 'mpi3d_toy':
        dataset_class = MPI3DSimplistic
    elif dataset_name.lower() == 'celeb_glow':
        dataset_class = CelebGlowDataset
    elif dataset_name.lower() == 'dummy':
        dataset_class = DummyDataset
    elif dataset_name.lower() == 'smalldummy':
        dataset_class = SmallDummyDataset
    elif dataset_name.lower() == 'cars3d': 
        dataset_class = Cars3D
    else:
        print('hrere')
        raise Exception(f'Dataset {dataset_name.lower()} not implemented')
    
    dataset = dataset_class(path=data_path, transform=transform)
    number_channels = dataset.data.shape[1]


    number_factors = len(dataset.factor_sizes)
    # adjust drop ration per factor to dataset to have comparable test / train
    # splits
    dataset_to_test_factor = {'dsprites': 0.2, 
                              'dsprites_scream': 0.2, 
                              'dsprites_rot90':0.2,
                              'shapes3d': 0.15,
                              'mpi3d': 0.1, 'mpi3d_toy':0.1,
                              'celeb_glow': 0.3,
                              'dummy': 0.2, 'smalldummy': 0.4, 
                              'cars3d': 0.1}
    test_ratio_per_factor = dataset_to_test_factor[dataset_name]
    return dataset, number_factors, number_channels, test_ratio_per_factor

""" 
Implementation of the VCT model baseline uses the open-source repo provided by the authors':https://github.com/ThomasMrY/VCT
The below code is adapted from https://github.com/ThomasMrY/VCT/blob/master/dataset.py

@article{yang2022visual,
  title={Visual Concepts Tokenization},
  author={Yang, Tao and Wang, Yuwang and Lu, Yan and Zheng, Nanning},
  journal={NeurIPS},
  year={2022}
}
"""

# Custum Dataset loader
from matplotlib import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
import h5py
from PIL import Image
from imageio import imread
from glob import glob
import PIL
import scipy.io as sio
from sklearn.utils.extmath import cartesian
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def _load_mesh(filename):
    """Parses a single source file and rescales contained images."""
    with open(filename, "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])
    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))
    for i in range(flattened_mesh.shape[0]):
        pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail((64, 64), PIL.Image.ANTIALIAS)
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

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform, eval_flag = False):
        self.data_tensor = data_tensor
        self.transform = transform
        self.eval_flag = eval_flag

    def __getitem__(self, index):
        data = self.data_tensor[index]
        if data.shape[0] == 3: 
            data = data.transpose(1, 2, 0)
        img = Image.fromarray(data)
        return self.transform(img), torch.tensor(1.0)
        
    def __len__(self):
        return self.data_tensor.shape[0]


class ImagesFilesDataset(Dataset):
    def __init__(self, paths, transform, labels = None, eval_flag = False):
        self.paths = paths
        self.transform = transform
        self.lables = labels
        self.eval_flag = eval_flag

    def __getitem__(self, index):
        data_img = imread(self.paths[index])[:, :, :3]
        img = Image.fromarray(data_img)
        return self.transform(img), torch.tensor(1.0)
        
    def __len__(self):
        return len(self.paths)

class ImagesFilesDataset_eval(Dataset):
    def __init__(self, paths, transform, labels, eval_flag = True):
        self.paths = paths
        self.transform = transform
        self.labels = labels
        self.eval_flag = eval_flag

    def __getitem__(self, index):
        data_img = imread(self.paths[index])[:, :, :3]
        img = Image.fromarray(data_img)
        return self.transform(img), self.labels[index]
        
    def __len__(self):
        return len(self.paths)
    

class ClipDataset(Dataset):
    def __init__(self, data_tensor, transform, clip_transform):
        self.data_tensor = data_tensor
        self.transform = transform
        self.clip_transform = clip_transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data_tensor[index])
        return self.transform(img), self.clip_transform(img)
        
    def __len__(self):
        return self.data_tensor.shape[0]

def custum_dataset(
    dset_dir = "./",
    Dataset_fc = CustomTensorDataset,
    transform = None,
    name = "shapes3d",
    eval_flag = False
):

    if name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = data['imgs']
        train_kwargs = {'data_tensor':data, 'transform':transform, 'eval_flag': True}
        dset = Dataset_fc
    
    elif name.lower() == 'shapes3d':
        data = h5py.File(os.path.join(dset_dir, '3dshapes.h5'), 'r')
        train_kwargs = {'data_tensor':data["images"][()],"transform":transform, 'eval_flag': True}
        dset = Dataset_fc
    
    elif name.lower() == "mpi_toy":
        data  = np.load(os.path.join(dset_dir,"mpi3d_toy.npz"), "r")
        images = data['images']
        train_kwargs = {'data_tensor':images,'transform':transform, 'eval_flag': True}
        dset = Dataset_fc
        

    elif name.lower() == "mpi_mid":
        data  = np.load(os.path.join(dset_dir,"mpi3d_realistic.npz"), "r")
        images = data['images']
        train_kwargs = {'data_tensor':images,'transform':transform, 'eval_flag': True}
        dset = Dataset_fc


    elif name.lower() == "mpi_real":
        data  = np.load(os.path.join(dset_dir, 'mpi3d', "mpi3d_real.npz"), "r")
        images = data['images']
        train_kwargs = {'data_tensor':images,'transform':transform, 'eval_flag': True}
        dset = Dataset_fc

    elif name.lower() == "clevr":
        if eval_flag:
            path = os.path.join(dset_dir.replace("clevr",""),"clevr_test/images/*.png")
            images = sorted(glob(path))
            masks = np.load(os.path.join(dset_dir.replace("clevr",""),"clevr_test/masks.npz"))['masks']
            train_kwargs = {'paths':images,'transform':transform, 'labels':masks, 'eval_flag': True}
            dset = ImagesFilesDataset_eval
        else:
            path = os.path.join(dset_dir.replace("clevr",""),"images_clevr/*.png")
            images = sorted(glob(path))
            train_kwargs = {'paths':images,'transform':transform}
            dset = ImagesFilesDataset
    
    elif name.lower() == "mnist":
        path = os.path.join(dset_dir,"mnist_5.npz")
        data = np.load(path,"r")['data']
        train_kwargs = {'data_tensor':data[:,0,:,:],"transform":transform}
        dset = Dataset_fc

    elif name.lower() == "celebanpz":
        path = os.path.join(dset_dir,"celeba.npz")
        data = np.load(path,"r")['imgs']
        train_kwargs = {'data_tensor':data,"transform":transform}
        dset = Dataset_fc
    
    elif name.lower() == "celeba":
        path = os.path.join(dset_dir,"data.npz")
        data = np.load(path,"r")['data']
        train_kwargs = {'data_tensor':np.uint8(data*255),'transform':transform}
        dset = Dataset_fc
    
    elif name.lower() == 'cars3d':
        dataset = _load_data(os.path.join(dset_dir, 'cars3d', 'data'))
        train_kwargs = {'data_tensor':np.uint8(dataset*255), 'transform':transform, 'eval_flag': True}
        dset = Dataset_fc
    
    elif name.lower() == 'tetrominoes_npz':
        data = np.load(os.path.join(dset_dir.replace("_npz",""), 'data.npz'),"r")['images']
        train_kwargs = {'data_tensor':np.uint8(data),'transform':transform}
        dset = Dataset_fc
    
    elif name.lower() == 'object_room_npz':
        data = np.load(os.path.join(dset_dir.replace("object_room_npz","objects_room"), 'data.npz'),"r")['images']
        train_kwargs = {'data_tensor':np.uint8(data),'transform':transform}
        dset = Dataset_fc
    elif name.lower() == 'coco':
        file_list = np.load(os.path.join(dset_dir,'file_np.npy')).tolist()
        train_kwargs = {'file_list':file_list, 'transform':transform}
        dset = ImagesFilesDataset
    
    elif name.lower() == "kitti":
        txt_file = open(os.path.join(dset_dir,'file_paths.txt'),"r")
        row_file_list = txt_file.readlines()
        file_list = []
        for line in row_file_list:
            file_list.append(os.path.join(dset_dir,line).replace("\n",""))
        train_kwargs = {'paths':file_list, 'transform':transform}
        dset = ImagesFilesDataset
    else:
        raise NotImplementedError


    return dset(**train_kwargs)


def custum_dataset_clip(
    dset_dir = "./",
    Dataset_fc = ClipDataset,
    transform = None,
    clip_transform = None,
    name = "shapes3d"
):

    if name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='bytes')
        data = data['imgs']
        train_kwargs = {'data_tensor':data, 'transform':transform, 'clip_transform': clip_transform}
        dset = Dataset_fc
    
    elif name.lower() == 'shapes3d':
        data = h5py.File(os.path.join(dset_dir, 'shapes3d', '3dshapes.h5'), 'r')
        images = data["images"][()]
        data = images
        train_kwargs = {'data_tensor':data,'transform':transform, "clip_transform":clip_transform}
        dset = Dataset_fc


    else:
        raise NotImplementedError


    return dset(**train_kwargs)

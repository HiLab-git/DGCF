import os

import numpy as np
import torch

from code_dataset.base_dataset import BaseDataset
from code_dataset.image_folder import get_image_directory_plural, make_datset_plural, make_dataset_2d_from_3d, make_dataset
from code_util.data.read_save import read_image_2d_from_3d,read_image
from code_util.data.prepost_process import Preprocess,Preprocess_class_mask
from code_util.data.read_save import read_dummy


class Aligned2DDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, config):

        BaseDataset.__init__(self, config)
        self.config = config
        self.dim = config["dataset"]["dim"]
        self.transform = Preprocess(self.config)
        self._init_paths()
    
    def _init_paths(self):
        """Initialize the directory paths for the dataset.
        
        phase: train/validation/test
        
        It assumes that the directory '/path/to/data' contains folder {phase}A and {phase}B, paired images must have the same name in both folders, no excessive images are allowed since the order of the images is important.
        
        During test time, only the directory '/path/to/data/testA' is necessary, but to validate the performance of the model, you need to prepare a directory '/path/to/data/testB'.
        
        mask is optional in '/path/to/data/mask/{phase}'
        """
        if self.dim == "3D":
            self.make_dataset = make_dataset_2d_from_3d
            self.read_image = read_image_2d_from_3d
        else:
            self.make_dataset = make_dataset
            self.read_image = read_image
        self.dir_A, self.dir_B, self.dir_mask = get_image_directory_plural(self.config)  # get the image directory
        self.A_paths, self.B_paths, self.Mask_paths = make_datset_plural([self.dir_A, self.dir_B, self.dir_mask],self.make_dataset, self.config)
        if (len(self.A_paths) != len(self.B_paths)) or (len(self.A_paths) != len(self.Mask_paths)):
            print("Warning: The number of images in A, B, and Mask directories are not equal. Please check your dataset.")
            # self.B_paths = [None]*len(self.A_paths)
            # self.Mask_paths = [None]*len(self.A_paths)
            self.B_paths = self.A_paths.copy() 
            self.Mask_paths = self.A_paths.copy()
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int): a random integer for data indexing
        Returns:
            dict: A dictionary containing A, B, and Mask images      
        """
        A,B,Mask = self._get_full_image(index)
        A,B,Mask = self._apply_transform(A,B,Mask)
        return {'A': A, 'B': B, 'Mask': Mask}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.config["phase"] == "test":
            if self.config["dataset"].get("patch_wise",{}).get("use_patch_wise",False) == True:
                return len(self.volume_index_map)
            else:
                return len(self.A_paths)
        else:
            return len(self.A_paths)
    
    """processes for get item"""
    def _get_full_image(self, index):
        """compared to the get_patch function, this function returns the full image."""
        if self.config["dataset"]["dim"] == "3D":
            (A_path, idx_2d, num_slices) = self.A_paths[index]
            (B_path, _, _) = self.B_paths[index] if self.B_paths[index] is not None else (None, None, None)
            (Mask_path, _, _) = self.Mask_paths[index] if self.Mask_paths[index] is not None else (None, None, None)
            A = self.read_image(A_path, idx_2d)
            B = self.read_image(B_path, idx_2d) if B_path is not None else read_dummy()
            Mask = self.read_image(Mask_path, idx_2d) if Mask_path is not None else read_dummy()
        else:
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]
            Mask_path = self.Mask_paths[index] 
            A = self.read_image(A_path) 
            B = self.read_image(B_path) if B_path is not None else read_dummy()
            Mask = self.read_image(Mask_path) if Mask_path is not None else read_dummy()
        return A, B, Mask
    
    def _apply_transform(self,A,B,Mask):
        if self.config["model"]["dim"] == "3D":
            # 对数据增加一个通道维度
            A["data"] = A["data"][np.newaxis, ...]  # Add channel dimension for 3D
            B["data"] = B["data"][np.newaxis, ...] if B["data"] is not None else None
            Mask["data"] = Mask["data"][np.newaxis, ...] if Mask["data"] is not None else None
        A_path = A['params']['path']
        B_path = B['params']['path']
        Mask_path = Mask['params']['path']
        self.transform.init_pipeline()
        A_transform = self.transform('A', A_path)
        if A_path is not None:
            A['data'] = A_transform(A['data'])
        B_transform = self.transform('B', B_path)
        if B_path is not None:
            B['data'] = B_transform(B['data'])
        Mask_transform = self.transform('Mask', Mask_path)
        if Mask_path is not None:
            Mask['data'] = Mask_transform(Mask['data'])
        
        return A, B, Mask

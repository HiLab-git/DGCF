"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import torch.utils.data as data
from abc import ABC, abstractmethod
from code_dataset.image_folder import read_image_3Dto2D,make_dataset_3Dto2D,make_dataset,make_dataset_2d_from_3d
from code_util.data.read_save import read_image

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    """

    def __init__(self, config):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.config = config
        self.root = config["dataset"]["dataroot"]
        if config["dataset"]["dim"] == "3D" and (config["model"].get("dim") == "2D" or config["model"].get("dim") == "25D"):
            self.read_image = read_image_3Dto2D
            self.make_dataset = make_dataset_2d_from_3d
        else:
            self.read_image = read_image
            self.make_dataset = make_dataset
        

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def _get_positions(self, total, patch, stride):
        """
        get patch positions for patch_wise testing
        the last patch position is fixed to total - patch because the last patch may be smaller than patch size.
        Args:
            total (int): total length
            patch (int): patch length
            stride (int): stride length
        Returns:
            list: patch positions
        """
        positions = list(range(0, total - patch + 1, stride))
        if not positions or positions[-1] + patch < total:
            positions.append(total - patch)
        return positions

    





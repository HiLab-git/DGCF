"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import random
import os
from glob import glob
from code_util.data import read_save
from code_util.util import get_file_name,get_full_extension


VALID_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.nii', '.nii.gz', '.mha', '.npy'
]

def choose_image_read_method(data_format):
    if data_format == '.nii.gz' or data_format == '.nii' or data_format == ".mha":
        return read_save.read_medical_image
    else:
        return read_save.read_natural_image

def is_valid_file(filepath):
    return any(filepath.endswith(extension) for extension in VALID_EXTENSIONS) 
    # 这里不能用 os.oath.isfile()检验 .nii.gz文件 是不会检测为文件的

def make_dataset(path, config:dict):
    if path == None or os.path.isdir(path) == False:
        print(f"Warning: {path} is not a valid directory")
        return []
    if "max_size" in config["dataset"].keys():
        max_dataset_size = config["dataset"]["max_size"]
    else:
        max_dataset_size = float('inf')
    # assert os.path.isdir(dir), '%s is not a valid directory' % 
    
    images = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if is_valid_file(filename):
            images.append(filepath)
    images.sort()
    data_len = len(images)
    if config["dataset"]["random_sample"] == True and max_dataset_size < data_len:
        random_seed = config["random_seed"]
        random.seed(random_seed)
        numbers = list(range(data_len))
        sampled_numbers = random.sample(numbers, max_dataset_size)
    else:
        sampled_numbers = range(data_len)
        
    images = [images[i] for i in sampled_numbers if i < data_len]
    images.sort()
    return images

def make_dataset_3Dto2D(path,config:dict):
    if path == None or os.path.isdir(path) == False:
        print(f"Warning: {path} is not a valid directory")
        return []
    if "max_size" in config["dataset"].keys():
        max_dataset_size = config["dataset"]["max_size"]
    else:
        max_dataset_size = float('inf')
    volumes = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if is_valid_file(filename):
            volumes.append(filepath)
    images = []
    for volume_path in volumes:
        folder_path = os.path.dirname(volume_path)
        volume = read_save.read_image(volume_path)  # Read the 3D volume
        z_slices = volume["data"].shape[0]  # Get the number of slices along the z-axis, shape: (d, h, w) or (z, y, x)
        volume_id = get_file_name(volume_path)  # Get the volume ID from the filename
        for z in range(z_slices):
            images.append(os.path.join(folder_path, volume_id + f'_{z}' + get_full_extension(volume_path)))

    data_len = len(images)
    if config["dataset"]["random_sample"] == True and max_dataset_size < data_len:
        random_seed = config["random_seed"]
        random.seed(random_seed)
        numbers = list(range(data_len))
        sampled_numbers = random.sample(numbers, max_dataset_size)
    else:
        sampled_numbers = range(data_len)
    images = [images[i] for i in sampled_numbers if i < data_len]
    images.sort()
    
    return images

def make_dataset_2d_from_3d(volume_dir, config:dict):
    """
    Args:
        volume_dir: 存放 3D 图像的目录，包含多个 .nii/.nii.gz 文件
        mode: 'train', 'val', 'test' 用于划分不同的数据集（可选）

    Returns:
        dataset_list: 一个列表，每个元素是 (volume_path, slice_idx)
    """
    if volume_dir == None or os.path.isdir(volume_dir) == False:
        print(f"Warning: {volume_dir} is not a valid directory")
        return []

    data_format = config["dataset"]["data_format"]
    vol_paths = []
    images = []
    for filename in os.listdir(volume_dir):
        vol_path = os.path.join(volume_dir, filename)
        if is_valid_file(filename):
            vol_paths.append(vol_path)
    for vol_path in vol_paths:
        num_slices = read_save.get_image_params(vol_path)["size"][0]
        for idx in range(num_slices):
            images.append((vol_path, idx, num_slices))  # 每一层作为一张2D图像
    return images

def read_image_3Dto2D(path):
    return read_save.read_image_3Dto2D(path)  # Read the 3D volume

def make_datset_plural(dirs, make_dataset_func,config):
    # modalitys = ["A", "B", "Mask"]
    paths = []
    for i in range(len(dirs)):
        if isinstance(dirs[i], list):
            paths_ = []
            for j, dir_ in enumerate(dirs[i]):
                paths_.extend(make_dataset_func(dir_, config=config))
            paths.append(paths_)
        else:
            paths.append(make_dataset_func(dirs[i], config=config))
    return paths
            
def get_image_directory_plural(config:dict):
    """
    Get the image directory from the config.
    If the config does not contain the image directory, return None.
    """
    modalitys = ["A", "B", "Mask"]
    dirs = []
    for i,modality in enumerate(modalitys):
        dir_config = config["dataset"].get(f"dir_{modality}", None)
        if dir_config is not None:
            dataset_root = config["dataset"]["dataroot"]
            dirs.append(os.path.join(dataset_root, dir_config))
            continue
        if isinstance(config["dataset"]["dataset_position"], list):
            dirs_ = []
            for dataset_position in config["dataset"]["dataset_position"]:
                dirs_.append(get_image_directory(config,dataset_position,modality))
            dirs.append(dirs_)
        else:
            dirs.append(get_image_directory(config, config["dataset"]["dataset_position"],modality))
    return dirs

def get_image_directory(config:dict,dataset_position, modality):
    """
    Get the image directory from the config.
    If the config does not contain the image directory, return None.
    """
    if modality == "Mask":
        dir_ = os.path.join(dataset_position, config["dataset"]["dim"], "mask", config["phase"])
        if not os.path.exists(dir_):
            print(f"Warning: {dir_} does not exist, using validation mask directory instead.")
            dir_ = os.path.join(dataset_position, config["dataset"]["dim"], "mask", "validation")
    else:
        dir_ = os.path.join(dataset_position, config["dataset"]["dim"], config["phase"]+modality)
        if not os.path.exists(dir_):
            print(f"Warning: {dir_} does not exist, using validation directory instead.")
            dir_ = os.path.join(dataset_position, config["dataset"]["dim"], "validation"+ modality)
    if not os.path.exists(dir_):
        print(f"Warning: {dir_} does not exist, using empty directory instead.")
        dir_ = None
    return dir_

class image_folder():
    def __init__(self, config:dict):
        """Initialize the class; save the options in the class

        Parameters:
            config (dict)-- stores all the experiment flags
        """
        self.config = config
        self.dir_A, self.dir_B, self.dir_mask = get_image_directory_plural(self.config)

    def _init_paths(self):
        """Initialize the directory paths for the dataset.
        
        phase: train/validation/test
        
        It assumes that the directory '/path/to/data' contains folder {phase}A and {phase}B, paired images must have the same name in both folders, no excessive images are allowed since the order of the images is important.
        
        During test time, only the directory '/path/to/data/testA' is necessary, but to validate the performance of the model, you need to prepare a directory '/path/to/data/testB'.
        
        mask is optional in '/path/to/data/mask/{phase}'
        """
        if self.config["dataset"]["dim"] == "3D" and (self.config["model"].get("dim") == "2D" or self.config["model"].get("dim") == "25D"):
            self.getdata_2d_form_3d = True
        if self.getdata_2d_form_3d:
            self.read_image = read_image_3Dto2D
            self.make_dataset = make_dataset_2d_from_3d
        else:
            # self.read_image = read_image
            self.make_dataset = make_dataset
       
        self.dir_A, self.dir_B, self.dir_mask = get_image_directory_plural(self.config)  # get the image directory
        self.A_paths, self.B_paths, self.Mask_paths = make_datset_plural([self.dir_A, self.dir_B, self.dir_mask],  self.config)
        if self.B_paths == []:
            self.B_paths = [None]*len(self.A_paths)
        if self.Mask_paths == []:
            self.Mask_paths = [None]*len(self.A_paths)
        assert len(self.A_paths) == len(self.B_paths) and len(self.A_paths) == len(self.Mask_paths), "The number of images in A, B and Mask should be the same."

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int): a random integer for data indexing
        Returns:
            dict: A dictionary containing A, B, and Mask images      
        """
        if self.getdata_2d_form_3d:
            A
        else:
            A = self.read_image(self.A_paths[index])
            if self.B_paths[index] is not None:
                B = self.read_image(self.B_paths[index])
            else:
                B = None
            if self.Mask_paths[index] is not None:
                Mask = self.read_image(self.Mask_paths[index])
            else:
                Mask = None
        
        return {'A': A, 'B': B, 'Mask': Mask}
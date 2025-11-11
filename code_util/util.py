"""This module contains simple helper functions """
import torch
import numpy as np
from collections import OrderedDict
import random
import os
import copy
from pathlib import Path


def tensor2im(input_image, imtype=np.uint8, return_first=True, dynamic_range=None):
    """"Converts a Tensor array into a numpy image array.
    convert the intensity range from -1~1(float) to 0~255(int8)

    Parameters:
        input_image (tensor) -- the input image tensor array
        imtype (type)        -- the desired type of the converted numpy array
        return_first (bool)  -- if True, return only the first image in the batch; otherwise, return all images
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_numpy = input_image.detach().cpu().float().numpy()
        else:
            return input_image
        if return_first:
            image_numpy = image_numpy[0:1]  # select the first image
        processed_images = []
        for img in image_numpy:
            if img.ndim == 2:  # if it is a grayscale image
                print(img.shape)
                img = np.expand_dims(img, axis=0)  # add a channel dimension
            if img.ndim == 4:  # if it is a batch of images
                print(img.shape)
                img = np.squeeze(img, axis=0)
            if img.shape[0] == 1:  # grayscale to RGB
                img = np.tile(img, (3, 1, 1))
            img = np.transpose(img, (1, 2, 0))
            if dynamic_range is not None:
                # -1~1 to 0~255
                # img = (img + 1) / 2.0 * 255.0  # post-processing: transpose and scaling
                img = (img - dynamic_range[0]) / (dynamic_range[1] - dynamic_range[0]) * 255.0
            else:
                # min~max to 0~255
                # img = (img + 1) / 2.0 * 255.0
                img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
            img = np.clip(img, 0, 255).astype(imtype)  # post-processing: transpose and scaling
            processed_images.append(img)
        if return_first:
            return processed_images[0]
        return processed_images
    else:  # if it is a numpy array, do nothing
        return input_image

def tensor2np(input_image, return_first=True):
    """"Converts a Tensor array into a numpy array.

    Parameters:
        input_image (tensor) -- the input image tensor array
        return_first (bool)  -- if True, return only the first image in the batch; otherwise, return all images
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_numpy = input_image.detach().cpu().float().numpy()
        else:
            return input_image
        if return_first:
            image_numpy = image_numpy[0:1]  # select the first image
        processed_images = []
        for img in image_numpy:
            img = np.squeeze(img)  # remove single-dimensional entries
            processed_images.append(img)
        if return_first:
            return processed_images[0]
        return processed_images
    else:  # if it is a numpy array, do nothing
        return input_image


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def merge_dicts_add_values(dict1, dict2):
    merged_dict = OrderedDict()
    for key in dict1:
        if key in dict2:
            merged_dict[key] = dict1[key] + dict2[key]
        else:
            merged_dict[key] = dict1[key]
    for key in dict2:
        if key not in merged_dict:
            merged_dict[key] = dict2[key]
    return merged_dict

def dict_divided_by_number(dictn,number):
    for key in dictn:
        dictn[key] = dictn[key]/number
    return dictn


def deep_update(dict1, dict2, overwrite=False):
    if not overwrite:
        dict1 = copy.deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            if overwrite:
                deep_update(dict1[key], value, overwrite)
            else:
                dict1[key] = deep_update(dict1[key], value, overwrite)
        else:
            dict1[key] = value
    return dict1

def find_latest_experiment(experiment_dir,experiment_root="./file_record"):
    record_dir = os.path.join(experiment_root, experiment_dir)
    experiment_time = sorted(os.listdir(record_dir))[-1]
    experiment_folder = os.path.join(record_dir,experiment_time)
    return experiment_folder

def is_valid_value(value):
    if isinstance(value, float):
        # 检测是否为nan或inf
        return not(np.isinf(value) or np.isnan(value))
    elif isinstance(value, torch.Tensor):
        return not(torch.isinf(value) or torch.isnan(value))
    else:
        # raise TypeError("Input should be either a NumPy array or a PyTorch tensor.")
        return True
    
def get_module_by_name(model,target_name):
    # 打印所有name
    # for name, module in model.named_modules():
    #     print(name)
    target_layer = None
    for name, module in model.named_modules():
        if name == target_name:
            target_layer = module
            break
    return target_layer

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

def set_random_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_full_extension(file_path):
    return ''.join(Path(file_path).suffixes)

def get_file_name(file_path):
    base_filename = os.path.basename(file_path)
    # 分割文件名和扩展名
    file_name, _ = os.path.splitext(base_filename)
    # 如果文件名有多个扩展名，处理它们
    if '.' in file_name:
        file_name = file_name.split('.')[0]
    return file_name

def InMakedirs(dirs, exist_tips = True):
    if isinstance(dirs, str):
        dirs = [dirs]
    for dir_ in dirs:
        if not os.path.exists(dir_):
            print(f"'{dir_}' not exists, create it.")
            os.makedirs(dir_)
        else:
            if exist_tips:
                # print(f"'{dir_}' exists.")
                pass
    return dirs

def dict2str(dict_):
    """Convert a dictionary to a string

    Parameters:
        dict_ (dict) --  the input dictionary
    """
    return ', '.join([f"{k}: {v:.5f}" for k, v in dict_.items()])

def generate_paths_from_list(path_list,prefix="",postfix=""):
    if isinstance(path_list, str):
        path_list = [path_list]
    return [os.path.join(prefix, path, postfix) for path in path_list]
 

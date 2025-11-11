
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import json

class Identity():
    def __call__(self,x):
        return x

class Preprocess():
    def __init__(self, config):
        
        self.config = config
        self.args = {}
        dataset = config["dataset"]
        norm_file_path = os.path.join(dataset["dataroot"], dataset["info"]["name"], "args.json")
        print("norm_file_path:", norm_file_path)
        if os.path.exists(norm_file_path):
            with open(norm_file_path, "r") as f:
                self.args = json.load(f)
        else:
            self.args = {
                "minmax": {"A": {}, "B": {}},
                "meanstd": {"A": {}, "B": {}},
                "99ptile": {"A": {}, "B": {}}
            }
            print("Warning: No normalization args file found. Using real-time calculation.")
        self.init_pipeline()

    def init_pipeline(self):
        config = self.config
        self.transform_list = []
        self.transform_count = 0
        self.transform_list.append(InToTensor())
        self.transform_count += 1
        # clip
        if config["preprocess"].get("clip",False):
            self.clip_pos = self.transform_count
            self.transform_count += 1
            self.clip = Identity()
            self.transform_list.append(self.clip)
        # resize
        if config["preprocess"].get("resize",{}).get("use_resize",False):
            self.resize_pos = self.transform_count
            self.transform_count += 1
            osize = config["preprocess"]["resize"]["resize_size"]
            method = get_resize_method(config)
            self.transform_list.append(resize(osize,method))
        # crop 
        if config["preprocess"].get("crop",{}).get("use_crop",False):
            self.crop_pos = self.transform_count
            self.transform_count += 1
            pos = get_crop_pos(config)
            crop_size = config["preprocess"]["crop"]["crop_size"]
            self.transform_list.append(FixedCrop(pos,crop_size))
        # flip
        if config["preprocess"].get("flip",{}).get("use_flip",False):
            self.flip_pos = self.transform_count
            self.transform_count += 1
            flip_direction = config["preprocess"]["flip"]["flip_direction"]
            flip_prob = config["preprocess"]["flip"].get("flip_prob",1)
            self.transform_list.append(FixedFlip(flip_direction,flip_prob))
        # rotation
        if config["preprocess"].get("rotation",{}).get("use_rotation",False):
            self.rotation_pos = self.transform_count
            self.transform_count += 1
            angle = config["preprocess"]["rotation"]["rotation_angle"]
            prob = config["preprocess"]["rotation"].get("rotation_prob",0.5)
            self.transform_list.append(FixedRotation(angle,prob))
        # transform
        if config["preprocess"].get("transform",{}).get("use_transform",False):
            self.transform_pos = self.transform_count
            self.transform_count += 1
            transform_type = config["preprocess"]["transform"]["transform_type"]
            transform_prob = config["preprocess"]["transform"].get("transform_prob",0.5)
            self.transform = Transform(transform_type,transform_prob)
            self.transform_list.append(self.transform)
        # normalization
        if config["preprocess"].get("norm",False):
            self.norm_pos = self.transform_count
            self.transform_count += 1
            self.norm = Identity()
            self.transform_list.append(self.norm)
        
    def __call__(self,modality,img_path):
        if modality in ['A', 'B']:
            clip = get_clip(self.config, modality, img_path, self.args)
            norm = get_norm(self.config, modality, img_path, self.args)
            self.transform_list[self.clip_pos] = clip
            self.transform_list[self.norm_pos] = norm
        else:
            self.transform_list[self.clip_pos] = Identity()
            self.transform_list[self.norm_pos] = Identity()
        return transforms.Compose(self.transform_list)
    
    def print_transform(self):
        for i in range(len(self.transform_list)):
            print("transform %d: %s" % (i,self.transform_list[i].__class__.__name__))
        
class Postprocess():
    def __init__(self, config, resize_range):
        self.config = config
        self.resize_range = resize_range
        self.transform_list = []
        self.transform_count = 0
        self.contruct_pipeline()
    
    def contruct_pipeline(self):
        config = self.config
        # normalization
        if config["preprocess"].get("norm",None) != None:
            self.norm_pos = self.transform_count
            self.transform_count += 1
            if config["preprocess"]["norm"]["use_norm_A"] == True:
                self.norm_A = get_norm_post(config, "A")
            else:
                self.norm_A = Identity()
            if config["preprocess"]["norm"]["use_norm_B"] == True:
                self.norm_B = get_norm_post(config, "B")
            else:
                self.norm_B = Identity()
            self.norm = Identity()
            self.transform_list.append(self.norm)
        # flip
        if config["preprocess"].get("flip",{}).get("use_flip",False):
            self.flip_pos = self.transform_count
            self.transform_count += 1
            flip_direction = config["preprocess"]["flip"]["flip_direction"]
            self.transform_list.append(FixedFlip(flip_direction)) 
        # resize
        if config["preprocess"].get("resize",{}).get("use_resize",False):
            self.transform_count += 1
            osize = self.resize_range
            method = get_resize_method(config)
            self.transform_list.append(resize(osize,method))
    
    def __call__(self,modality):
        if modality == 'A':
            self.transform_list[self.norm_pos] = self.norm_A
        elif modality == 'B':
            self.transform_list[self.norm_pos] = self.norm_B
        else:
            pass
        return transforms.Compose(self.transform_list)
    
class Preprocess_class_mask():
    def __init__(self,config):
        self.config = config
        self.transform_list = []
        self.transform_list.append(transforms.ToTensor())
        self.contruct_pipeline()

    def contruct_pipeline(self):
        # normalization
        config = self.config
        if config["preprocess"]["resize"] == True:
            osize = [config["preprocess"]["resize_size"], config["preprocess"]["resize_size"]]
            method = transforms.InterpolationMode.NEAREST
            self.transform_list.append(resize(osize,method))
        
    def __call__(self):
        return transforms.Compose(self.transform_list)
    
"""ToTensor"""
class InToTensor():
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy()).float()
        elif isinstance(img, torch.Tensor):
            img = img.float()
        if img.ndim == 2:  # If the image is grayscale
            img = img.unsqueeze(0)  # Add a channel dimension
        elif img.ndim == 3 and img.shape[0] != 1:
            img = img.unsqueeze(0)
        return img

"""CLIP"""

import torch

class ClipTransform:
    def __init__(self, min_value=None, max_value=None, ptile=None):
        self.min_value = min_value
        self.max_value = max_value
        self.ptile = ptile  # 例如 0.99 表示 [1%, 99%] 对称区间

    def __call__(self, img):
        """
        img: torch.Tensor，任意维度（通常是 [C, D, H, W] 或 [H, W]）
        """
        if self.min_value is not None and self.max_value is not None:
            # 如果显式给定 min 和 max，直接用
            return torch.clamp(img, self.min_value, self.max_value)
        
        if self.ptile is not None:
            # 计算 ptile 范围
            min_val = torch.quantile(img, 1 - self.ptile)
            max_val = torch.quantile(img, self.ptile)

            return torch.clamp(img, min_val, max_val)
        
        # 如果都没给，直接返回原图
        return img
   
def get_clip(config, modality, img_path = None, args = {}):
    from code_util.data.read_save import read_medical_image
    from code_dataset import find_3D_form_2D
    from code_util.util import get_file_name
    if config["preprocess"]["clip"]["use_clip_%s" % modality] == False:
        return Identity()
    if config["preprocess"]["clip"]["clip_level_%s" % modality] == "population":
        min_val, max_val = config["preprocess"]["clip"]["clip_range_%s" % modality]
        return ClipTransform(min_val, max_val)
    elif config["preprocess"]["clip"]["clip_level_%s" % modality] == "patient":
        if config["dataset"]["dim"] == "2D":
            img_path = find_3D_form_2D(img_path,config["dataset"]["data_format"])
            img_index = get_file_name(img_path)
            if config["preprocess"]["clip"]["clip_type_%s" % modality] == "99ptile":
                if img_index in args["99ptile"][modality]:  
                    min,max = args["99ptile"][modality][img_index]
                else:
                    image = read_medical_image(img_path)
                    min,max = calculate_ptile(image, ptile = 0.99)
                    args["minmax"][modality].update({img_index: (min,max)})
                clip = ClipTransform(min,max)
        elif config["dataset"]["dim"] == "3D":
            if config["preprocess"]["clip"]["clip_type_%s" % modality] == "99ptile":
                clip = ClipTransform(ptile = config["preprocess"]["clip"]["clip_type_%s" % modality])
    elif config["preprocess"]["clip"]["clip_level_%s" % modality] == "image":
        if config["preprocess"]["clip"]["clip_type_%s" % modality] == "99ptile":
            clip = ClipTransform(ptile = config["preprocess"]["clip"]["clip_type_%s" % modality])
    else:
        raise ValueError("wrong clipalization level")
    return clip


def calculate_ptile(image, mask = None, ptile = 0.99):
    if mask is not None:
        pixels   = image[mask > 0]
        if(len(pixels) > 0):
            min_val = np.percentile(pixels, 100 * (1 - ptile))
            max_val = np.percentile(pixels, 100 * ptile)
        else:
            min_val, max_val = 0.0, 1.0
    else:
        min_val = np.percentile(image, 100 * (1 - ptile))
        max_val = np.percentile(image, 100 * ptile)
    return min_val, max_val

        
"""RESIZE""" 

def resize(size, method):
    return transforms.Resize(size, method,antialias=None)

def get_resize_method(config):
    if config["preprocess"]["resize"]["resize_method"] == 'BILINEAR':
        method = transforms.InterpolationMode.BILINEAR
    elif config["preprocess"]["resize"]["resize_method"] == 'BICUBIC':
        method = transforms.InterpolationMode.BICUBIC
    else: 
        method = transforms.InterpolationMode.NEAREST
    return method

"""CROP"""

class FixedCrop():
    def __init__(self, pos, size):
        self.left, self.top = pos
        self.height,self.width = size

    def __call__(self, img):
        # print("============",self.top, self.left, self.height, self.width)
        return F.crop(img, self.top, self.left, self.height, self.width)
    
def get_crop_pos(config):
    
    crop_size = config["preprocess"]["crop"]["crop_size"]
    resize_size = config["preprocess"]["resize"]["resize_size"]

    # 随机选取一个点作为crop的左上角
    crop_pos = (
        random.randint(0, resize_size[i] - crop_size[i]) for i in range(len(crop_size))
    )

    return crop_pos

"""ROTATION"""
class FixedRotation():
    def __init__(self, angle_range,prob):
        # 在-angle～angle之间随机选择一个角度
        if random.random() >= prob:
            self.angle = 0
        else:
            angle_range = angle_range if isinstance(angle_range, (list, tuple)) else (-angle_range, angle_range)
            self.angle = random.uniform(angle_range[0], angle_range[1])
        
    def __call__(self, img):
        if self.angle == 0:
            return img
        # if img.ndim == 2:
        #     img = img.unsqueeze(0)
        rotated_img = F.rotate(img, self.angle)
        # img = img.squeeze(0) 
        return rotated_img  

"""FLIP"""

class FixedFlip():
    def __init__(self, direction, prob = 0.5):
        self.direction = direction 
        self.flip_list = []
        if random.random() < prob:
            # 在direction中随机选择一个翻转方向
            direction = random.choice(self.direction)
            if direction == 'h':
                self.flip_list.append(transforms.RandomHorizontalFlip(1))
            elif direction == 'v':
                self.flip_list.append(transforms.RandomVerticalFlip(1))
            else:
                raise ValueError("wrong flip direction")
        else:
            self.flip_list.append(Identity())
        self.flip = transforms.Compose(self.flip_list)
        
    def __call__(self,img):
        return self.flip(img)

# def get_flip_direction(config):
#     flip_direction = config["preprocess"]["flip"]["flip_direction"]
#     p = random.random()
#     assert len(flip_direction) <= 2, "wrong configuration of flip direction"
#     if len(flip_direction) == 2:
#         flip_p_v = flip_p_h = 0.25
#     elif len(flip_direction) == 1:
#         if flip_direction == 'v':
#             flip_p_v = 0.5
#             flip_p_h = 0
#         elif flip_direction == 'h':
#             flip_p_v = 0
#             flip_p_h = 0.5
#     else:
#         flip_p_v = 0
#         flip_p_h = 0
#     if p <= flip_p_v:
#         return 'v'
#     elif p <= flip_p_v + flip_p_h:
#         return 'h'
#     else:
#         return None

"""Transform"""

class Transform():
    def __init__(self, transform_type = [],transform_prob = 0.5):
        self.transform_type_ref = ["bezier", "gamma"]
        if isinstance(transform_type, str):
            transform_type = [transform_type]
        self.transform_type = transform_type
        self.transform_type = [t for t in self.transform_type if t in self.transform_type_ref]
        self.transform_prob = transform_prob
        
    def __call__(self, img):
        # 在self.transform_type中随机选择一个变换
        if len(self.transform_type) == 0:
            return img
        if random.random() < self.transform_prob:
            transform_type = random.choice(self.transform_type)
            if transform_type == "bezier":
                img = bezier_transformation(img)
            elif transform_type == "gamma":
                img = gamma_transformation(img)
            else:
                raise ValueError("wrong transform type")
        return img
    
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def bezier_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    # xpoints = [p[0] for p in points]
    # ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def gamma_transformation(x, prob=0.5):
    # 有一定的概率进行伽马变换，否则返回原值
    if random.random() >= prob:
        return x

    # 随机选择伽马值 (类似于贝塞尔中的控制点)
    gamma = random.uniform(0.5, 2.0)  # 随机选择伽马值，范围可以调整
    A = 1  # 常数系数，通常为1

    # 计算伽马变换结果
    transformed_x = A * np.power(x, gamma)
    
    return transformed_x

        
"""NORMALIZATION"""
    
class LinearNormalize():
    def __init__(self, source_range = None, target_range = None):
        self.source_range = source_range
        self.target_range = target_range
        if self.target_range is None:
            self.target_range = (0, 1)
            
    def __call__(self, img_tensor):
        if self.source_range is None:
            self.source_range = (img_tensor.min().item(), img_tensor.max().item())
            if self.source_range[0] == self.source_range[1]:
                # 则映射到target_range的中点
                self.source_range = (self.source_range[0] - 1, self.source_range[1] + 1)
        # print("source_range:", self.source_range)
        source_min, source_max = self.source_range
        target_min, target_max = self.target_range
        return (img_tensor - source_min) / (source_max - source_min) * (target_max - target_min) + target_min
        
class MeanStdNormalize():
    """
    PYMIC
    """
    def __init__(self,chns = None, mean = None, std = None, mask_thrd = None, bg_random = True, inverse = False):
        self.chns = chns
        self.mean = mean
        self.std  = std
        self.mask_thrd = mask_thrd
        self.bg_random = bg_random
        self.inverse = inverse

    def __call__(self, image):
        if(self.chns is None):
            self.chns = range(image.shape[0])
        if(self.mean is None):
            self.mean = [None] * len(self.chns)
            self.std  = [None] * len(self.chns)
        if not isinstance(self.mean,list):
            self.mean = [self.mean]
        if not isinstance(self.std,list):
            self.std = [self.std]
        for i in range(len(self.chns)):
            chn = self.chns[i]
            chn_mean, chn_std = self.mean[i], self.std[i]
            if(chn_mean is None):
                if(self.mask_thrd is not None):
                    pixels   = image[chn][image[chn] > self.mask_thrd]
                    if(len(pixels) > 0):
                        chn_mean, chn_std = pixels.mean(), pixels.std() + 1e-5
                    else:
                        chn_mean, chn_std = 0.0, 1.0
                else:
                    chn_mean, chn_std = image[chn].mean(), image[chn].std() + 1e-5
    
            chn_norm = (image[chn] - chn_mean)/chn_std

            if(self.mask_thrd is not None and self.bg_random):
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[image[chn] <= self.mask_thrd] = chn_random[image[chn] <=self.mask_thrd]
            image[chn] = chn_norm
        return image

class tanhNormalize():
    def __init__(self, a = None, b = None, reverse = False):
        if a == None:
            self.a = 0
        else:
            self.a = a
        if b == None:
            self.b = 1
        else:
            self.b = b
        self.reverse = reverse
            
    def __call__(self, img_tensor):
        if self.reverse:
            # 使用反向tanh归一化
            img_tensor = torch.atanh(img_tensor) * self.b + self.a
        else:
            img_tensor = torch.tanh((img_tensor - self.a)/self.b)
        return img_tensor
    
def get_norm(config, modality, img_path = None, args = {}):
    from code_util.data.read_save import read_medical_image
    from code_dataset import find_3D_form_2D
    from code_util.util import get_file_name
    # if modality == "A":
    #     return tanhNormalize(500,750)
    # elif modality == "B":
    #     # return LinearNormalize(source_range = (-1024,3000), target_range = (-1,1))
    #     return tanhNormalize(500,750)
    if config["preprocess"]["norm"]["use_norm_%s" % modality] == False:
        return Identity()
    if config["preprocess"]["norm"]["norm_level_%s" % modality] == "population":
        if config["preprocess"]["norm"]["norm_type_%s" % modality] == "minmax":
            min_val,max_val = config["preprocess"]["norm"]["minmax_norm_range_%s" % modality]
            norm = LinearNormalize(source_range = (min_val,max_val), target_range = (-1,1))
        elif config["preprocess"]["norm"]["norm_type_%s" % modality] == "meanstd":
            mean,std = config["preprocess"]["norm"]["meanstd_norm_ms_range_%s" % modality]
            norm = MeanStdNormalize(mean=mean,std=std)
        else:
            raise ValueError("wrong normalization type")
    elif config["preprocess"]["norm"]["norm_level_%s" % modality] == "patient":
        if config["dataset"]["dim"] == "2D":
            img_path = find_3D_form_2D(img_path, config["dataset"]["data_format"])
            img_index = get_file_name(img_path)
            if config["preprocess"]["norm"]["norm_type_%s" % modality] == "minmax":
                if img_index in args["minmax"][modality]:  
                    min,max = args["minmax"][modality][img_index]
                else:
                    image = read_medical_image(img_path)
                    min,max = calculate_minmax(image)
                    args["minmax"][modality].update({img_index: (min,max)})
                norm = LinearNormalize(source_range = (min,max), target_range = (-1,1))
            elif config["preprocess"]["norm"]["norm_type_%s" % modality] == "meanstd":
                if img_index in args["meanstd"][modality]:
                    mean,std = args["meanstd"][modality][img_index]
                    if config["preprocess"]["norm"].get("meanstd_disturbance_%s" % modality,False) == True:
                        mean_ratio,std_ratio = config["preprocess"]["norm"]["meanstd_disturbance_ratio_%s" % modality]
                        if config["phase"] == "train": # validation的时候不扰动 这只是临时设置 后续需要改进配置方法
                            mean = np.random.normal(mean,mean*mean_ratio)
                            std = np.random.normal(std,std*std_ratio)
                else:
                    image = read_medical_image(img_path)
                    mean,std = calculate_meanstd(image)
                    args["meanstd"][modality].update({img_index: (mean,std)})
                norm = MeanStdNormalize(mean=mean,std=std)
                # if config["preprocess"]["norm"].get("norm_addition_%s" % modality,False) == True:
                #     # 额外添加一个minmax归一化
                #     norm_minmax = LinearNormalize(target_range=(-1,1))
                #     norm = transforms.Compose([norm, norm_minmax])
            elif config["preprocess"]["norm"]["norm_type_%s" % modality] == "99ptile":
                if img_index in args["99ptile"][modality]:  
                    min,max = args["99ptile"][modality][img_index]
                else:
                    image = read_medical_image(img_path)
                    min,max = calculate_ptile(image, ptile = 0.99)
                    args["99ptile"][modality].update({img_index: (min,max)})
                norm = LinearNormalize(source_range = (min,max), target_range = (-1,1))
            else:
                raise ValueError("wrong normalization type")
        elif config["dataset"]["dim"] == "3D":
            if config["preprocess"]["norm"]["norm_type_%s" % modality] == "minmax":
                norm = LinearNormalize(target_range= (-1,1))
            elif config["preprocess"]["norm"]["norm_type_%s" % modality] == "meanstd":
                norm = MeanStdNormalize()
            else:
                raise ValueError("wrong normalization type")
    elif config["preprocess"]["norm"]["norm_level_%s" % modality] == "image":
        if config["preprocess"]["norm"]["norm_type_%s" % modality] == "minmax":
            norm = LinearNormalize(target_range= (-1,1))
        elif config["preprocess"]["norm"]["norm_type_%s" % modality] == "meanstd":
            norm = MeanStdNormalize()
        else:
            raise ValueError("wrong normalization type")
    else:
        raise ValueError("wrong normalization level")
    return norm

def get_norm_post(config, modality):
    # if modality == 'A':
    #     return tanhNormalize(500,750,reverse=True)
    # elif modality == 'B':
    #     # return LinearNormalize(source_range = (-1,1), target_range = (-1024,3000))
    #     return tanhNormalize(500,750,reverse=True)
    if config["preprocess"]["norm"]["norm_level_%s" % modality] == "population":
        if config["preprocess"]["norm"]["norm_type_%s" % modality] == "minmax":
            min_val,max_val = config["preprocess"]["norm"]["minmax_norm_range_%s" % modality]
            norm = LinearNormalize(source_range = (-1,1), target_range = (min_val,max_val))
        elif config["preprocess"]["norm"]["norm_type_%s" % modality] == "meanstd":
            norm = Identity()
            # mean,std = config["preprocess"]["norm"]["meanstd_norm_ms_range_%s" % modality]
            # norm = MeanStdNormalize(mean=mean,std=std)
        else:
            raise ValueError("wrong normalization type")
    else: 
        norm = Identity()
    return norm

def calculate_meanstd(image, mask = None):
    if mask is not None:
        pixels   = image[mask > 0]
        if(len(pixels) > 0):
            mean, std = pixels.mean(), pixels.std() + 1e-5
        else:
            mean, std = 0.0, 1.0
    else:
        mean, std = image.mean(), image.std() + 1e-5
    return mean, std

def calculate_minmax(image, mask = None):
    if mask is not None:
        pixels   = image[mask > 0]
        if(len(pixels) > 0):
            min_val, max_val = pixels.min(), pixels.max()
        else:
            min_val, max_val = 0.0, 1.0
    else:
        min_val, max_val = image.min(), image.max()
    return min_val, max_val



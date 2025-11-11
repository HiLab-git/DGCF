import SimpleITK as sitk
from PIL import Image
import numpy as np
from code_util.data.prepost_process import Postprocess
from code_util import util
import os
from code_util.util import get_full_extension,get_file_name
import torch
"""
READ

read_medical_image: 读取.nii.gz或.nii格式的医学图像 返回np.ndarray
read_natural_image: 读取.jpg或.png格式的自然图像 返回np.ndarray
get_image_params: 读取.nii.gz或.nii格式的医学图像的参数 返回dict

"""

def read_dummy():
    """
    Read dummy image and return the numpy array of the image.
    """
    return {"data": np.array([],dtype=np.float32), "params": {}}
    
def read_medical_image(path):
    """
    Read nii file and return the numpy array of the image.
    """
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    return image_array.astype(np.float32)

def read_natural_image(path):

    # PIL的Image按照(width,height)的方式读取图像 和cv2正好相反
    return np.array(Image.open(path).convert('RGB'))

def read_image(image_path:str):
    
    data = get_image_data(image_path)
    params = get_image_params(image_path)

    return {"data": data, "params": params}

def read_image_3Dto2D(image_path:str):
    file_name = util.get_file_name(image_path)  # Get the volume ID from the filename
    slice_index = int(file_name.split('_')[-1])  # Extract slice index from the file name
    volume_name = file_name.split('_')[0]
    folder_path = os.path.dirname(image_path)
    data_format = get_full_extension(image_path)
    volume_path = os.path.join(folder_path, volume_name + data_format)

    extension = get_full_extension(volume_path)
    if extension in [".nii.gz", ".nii", ".mha"]:
        volume = sitk.ReadImage(volume_path)
        size = volume.GetSize() # GetSize() returns (X, Y, Z) in SimpleITK
        slice_filter = sitk.ExtractImageFilter()
        slice_filter.SetSize([size[0], size[1], 0]) # Set the size to extract only one slice in Z direction
        slice_filter.SetIndex([0, 0, slice_index]) # Set the index to extract the slice at slice_index in Z direction
        image = slice_filter.Execute(volume)
        data = sitk.GetArrayFromImage(image).astype(np.float32)
        params = {
            'size_3D': np.flip(np.array(volume.GetSize())).copy(), # GetSize() returns (X, Y, Z) in SimpleITK
            'size' : np.flip(np.array(image.GetSize())).copy(),
            'spacing': np.array(image.GetSpacing()),
            'origin': np.array(image.GetOrigin()),
            'direction': np.array(image.GetDirection()),
            'path': image_path
        }
    else:
        image = Image.open(volume_path)
        # 取z轴上的第slice_index层
        image = np.array(image)[:,:,slice_index]
        data = image.astype(np.float32)
        params = {
            'size' : np.array(image.size),
            'path' : image_path
        }
    return {"data": data, "params": params}

def read_image_2d_from_3d(volume_path:str,idx_2d):
    """
    Read 2D image from 3D volume file.
    Args:
        volume_path (str): Path to the 3D volume file.
        idx_2d (int): Index of the 2D slice to read from the 3D volume.
    Returns:
        dict: A dictionary containing the 2D image data and its parameters.
    """
    extension = get_full_extension(volume_path)
    if extension in [".nii.gz", ".nii", ".mha"]:
        volume = sitk.ReadImage(volume_path)
        size = volume.GetSize() # GetSize() returns (X, Y, Z) in SimpleITK
        slice_filter = sitk.ExtractImageFilter()
        slice_filter.SetSize([size[0], size[1], 0]) # Set the size to extract only one slice in Z direction
        slice_filter.SetIndex([0, 0, idx_2d]) # Set the index to extract the slice at idx_2d in Z direction
        image = slice_filter.Execute(volume)
        data = sitk.GetArrayFromImage(image).astype(np.float32)
        params = {
            'size_3D': np.flip(np.array(volume.GetSize())).copy(), # GetSize() returns (X, Y, Z) in SimpleITK
            'size' : np.flip(np.array(image.GetSize())).copy(),
            'spacing': np.array(image.GetSpacing()),
            'origin': np.array(image.GetOrigin()),
            'direction': np.array(image.GetDirection()),
            'idx_2d': idx_2d,
            'path_3D': volume_path,  # 保存3D图像的路径
            'path': get_2d_image_path_from_3d(volume_path, idx_2d)  # 保存2D图像的路径
        }
    else:
        image = Image.open(volume_path)
        # 取z轴上的第idx_2d层
        image = np.array(image)[:,:,idx_2d]
        data = image.astype(np.float32)
        params = {
            'size' : np.array(image.size),
            'idx_2d': idx_2d,
            'path' : volume_path
        }
    return {"data": data, "params": params}

def read_image_25d_from_3d(volume_path:str,idx_2d,z_pad = 0):
    """
    Read 2.5D image from 3D volume file.
    Args:
        volume_path (str): Path to the 3D volume file.
        idx_2d (int): Index of the 2D slice to read from the 3D volume.
        z_pad (int): Number of slices to pad in the Z direction. Default is 0.
    Returns:
        dict: A dictionary containing the 2D image data and its parameters.
    """
    extension = get_full_extension(volume_path)
    if extension in [".nii.gz", ".nii", ".mha"]:
        volume = sitk.ReadImage(volume_path)
        size = volume.GetSize()  # (X, Y, Z)
        z = size[2]
        # 计算要取的切片索引，超出范围的用边界值代替
        slice_indices = [min(max(idx_2d + i, 0), z - 1) for i in range(-z_pad, z_pad + 1)]
        # 依次取出每个切片
        slices = []
        for idx in slice_indices:
            slice_filter = sitk.ExtractImageFilter()
            slice_filter.SetSize([size[0], size[1], 0])
            slice_filter.SetIndex([0, 0, idx])
            image = slice_filter.Execute(volume)
            arr = sitk.GetArrayFromImage(image).astype(np.float32)
            slices.append(arr)
            if idx == idx_2d:
                image_center = image  # 保存中心切片的图像信息
        # 堆叠为 (2*z_pad+1, H, W)
        data = np.stack(slices, axis=0)
        params = {
            'size_3D': np.flip(np.array(volume.GetSize())).copy(),
            'size_25D': np.array(data.shape),
            'size': np.flip(np.array(image_center.GetSize())).copy(),
            'spacing_25D': np.array(volume.GetSpacing()),
            'spacing': np.array(image_center.GetSpacing()),
            'origin_25D': np.array(volume.GetOrigin()),
            'origin': np.array(image_center.GetOrigin()),
            'direction_25D': np.array(volume.GetDirection()),
            'direction': np.array(image_center.GetDirection()),
            'idx_2d': idx_2d,
            'z_pad': z_pad,
            'path_3D': volume_path,
            'path': get_2d_image_path_from_3d(volume_path, idx_2d)  # 保存2D图像的路径
        }
    else:
        image = Image.open(volume_path)
        # 取z轴上的第idx_2d层
        image = np.array(image)[:,:,idx_2d]
        data = image.astype(np.float32)
        params = {
            'size' : np.array(image.size),
            'idx_2d': idx_2d,
            'path' : volume_path
        }
    return {"data": data, "params": params}

def get_2d_image_path_from_3d(volume_path:str, idx_2d:int):
    volume_name = util.get_file_name(volume_path)  # Get the volume ID from the filename
    slice_index = idx_2d  # Use the provided idx_2d as the
    image_name = f"{volume_name}_{slice_index}" + util.get_full_extension(volume_path)
    folder_path = os.path.dirname(volume_path)
    image_path = os.path.join(folder_path, image_name)
    return image_path

def get_image_data(image_path:str):
    """
    Read image file by path and return the numpy array of the image.
    """
    # determine medical or natural image
    extension = get_full_extension(image_path)
    if extension in [".nii.gz", ".nii", ".mha"]:
        image_array = read_medical_image(image_path)
    else: 
        image_array = read_natural_image(image_path)
    return image_array.astype(np.float32)

def get_image_params(image_path:str):
    """
    Read image file by path and return the size of the image.
    """
    # determine medical or natural image
    extension = get_full_extension(image_path)
    if extension in [".nii.gz", ".nii", ".mha"]:
        image = sitk.ReadImage(image_path)
        image_params = dict()
        image_params = {
        # GetSize() returns (X, Y, Z) in SimpleITK, but we need (Z, Y, X) for our model
        # copy is for get contiguous memory for tensor (e.g. torch.from_numpy)
        'size' : np.flip(np.array(image.GetSize())).copy(), 
        'spacing': np.array(image.GetSpacing()),
        'origin': np.array(image.GetOrigin()),
        'direction': np.array(image.GetDirection()),
        'path': image_path
    }
    else: 
        image = Image.open(image_path)
        image_params = {
            'size' : np.array(image.size),
            'path' : image_path
        }
    return image_params

"""
SAVE

save_image_4_final 将输入图像做最后的储存 需要根据参考图像对其进行resize 然后决定储存为医学图像或是自然图像
save_image_4_show 将输入图像做临时储存 无需resize 总是保存为自然图像格式
write_nii 将输入图像保存为医学图像格式

"""

def save_image_4_final(image,img_params,target_path,config,modality):
    """
    Save a numpy image to the disk for final results
    """
    # if config["dataset"].get("patch_wise",{}).get("use_patch_wise",False) != True:
    size = tuple(np.array(img_params['size']))
    transform = Postprocess(config,size)(modality)
    image = transform(image)   
    image = util.tensor2np(image)
    extension = get_full_extension(target_path)
    if extension == ".nii.gz" or extension == ".nii" or extension == ".mha":
        write_nii(image, img_params, target_path) # 在写回nii文件时，SimpleITK接受Z,Y,X的顺序
    else: 
        write_jpg(image, img_params, target_path)

def save_image_4_show(image_numpy, image_path):
    """Save a numpy image to the disk for showing on the html page

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    # print(image_path)
    # print(image_numpy.min(),image_numpy.max())
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_test_image(visuals, params, config, save_list = ["real_A", "real_B", "fake_B"]):
    target_path_base = os.path.join(config["work_dir"],"2D","synthesis")
    os.makedirs(target_path_base ,exist_ok=True)
    # 将测试结果全部保存到本地
    for label, image in visuals.items():
        if label in save_list:
            modality = label.split("_")[-1]
            # 对一个batch中的每一个元素均进行保存
            batch_size = image.shape[0]
            for i in range(batch_size):
                # 对params中的每一个key所对应的value均取出第i个元素
                one_image = image[i:i+1]
                params_one_image = {key: value[i] for key, value in params.items()}
                file_name = get_file_name(params_one_image["path"]) 
                # 规定output image和reference image之间的名称关系
                if config["dataset"].get("patch_wise",{}).get("use_patch_wise",False) == True:
                    dhw_str = params_one_image["dhw"]
                    dhw = eval(dhw_str)
                    patch_pos_str = f"{dhw[0]}_{dhw[1]}_{dhw[2]}"  # d_h_w
                    file_name = file_name + "_" + patch_pos_str
                file_name = file_name + "_" + label
                target_file_name = file_name + config["dataset"]["data_format"]
                target_path = os.path.join(target_path_base,target_file_name)
                save_image_4_final(one_image,params_one_image,target_path,config,modality)

   
def write_nii(image_array, image_params, nii_path):
    """
    Write nii file from numpy array and parameters.
    """
    if isinstance(image_array, np.ndarray) == False:
        image_array = np.array(image_array)

    size = np.array(image_params['size']) 
    # if (image_array.shape != size).any(): 
    #     print("image_array.shape:", image_array.shape)
    #     print("size form the image:", size)
    #     raise ValueError('The size of the image is not the same as the size in the parameters.')
    space = np.squeeze(np.array(image_params['spacing']))
    origin = np.squeeze(np.array(image_params['origin']))
    direction = np.squeeze(np.array(image_params['direction']))
    # print("space:", space)
    # print("origin:", origin)
    # print("direction:", direction)
    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing(space)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    sitk.WriteImage(image, nii_path)

def write_jpg(image_array, image_params, jpg_path):
    if isinstance(image_array, np.ndarray) == False:
        image_array = np.array(image_array)

    size = np.array(image_params['size'])
    if (image_array.shape != size).any():
        print("image_array.shape:", image_array.shape)
        print("size form the image:", size)
        raise ValueError('The size of the image is not the same as the size in the parameters.')
    image_pil = Image.fromarray(image_array)
    image_pil.save(jpg_path)

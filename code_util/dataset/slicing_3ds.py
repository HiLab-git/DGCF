
import os
import SimpleITK as sitk
from code_util.util import InMakedirs
from code_util.util import get_file_name
import re
from code_util.dataset.prepare import generate_paths_from_dict
import numpy as np

def split_3d_to_2d(input_path, output_dir, split_list = [], prefix="", data_format = ".nii.gz"):
    
    InMakedirs(output_dir)
    
    # 读取3D NIfTI文件
    image = sitk.ReadImage(input_path)

    # 获取图像的尺寸
    size = image.GetSize()
    print(f"Image size: {size}")
    # 如果不是3D图像，则输出警告信息并跳过
    output_file_paths = []
    if len(size) != 3:
        print(f"Input image is not 3D. Size: {size}")
        return output_file_paths
    if len(split_list) == 0:
        split_list = range(size[2])
        # 根据切片比例按照顺序选取切片
    # 遍历Z轴，将每个切片保存为单独的2D NIfTI文件
    for z in split_list:
        # 提取Z轴上的切片
        slice_filter = sitk.ExtractImageFilter()
        slice_filter.SetSize([size[0], size[1], 0])
        slice_filter.SetIndex([0, 0, z])
        slice_image = slice_filter.Execute(image)

        # 构造输出文件名
        # input_path的扩展名为输出文件的扩展名
        # file_extension = get_full_extension(input_path)
        file_extension = data_format
        output_file_name = f"{prefix}{z}" + file_extension
        output_file_path = os.path.join(output_dir, output_file_name)

        # 保存切片为2D NIfTI文件
        sitk.WriteImage(slice_image, output_file_path)
        # print(f"Saved slice {z} to {output_file_path}")
        output_file_paths.append(output_file_path)
    return output_file_paths

def split_3d_to_2d_in_folder(input_directory, output_directory, pattern = ".*"):
    """
    遍历输入目录中的所有文件，找到匹配给定正则表达式的 3D NIfTI 文件，
    并调用 split_3d_to_2d 函数将其切片为 2D 图像。
    
    :param input_directory: 包含 3D NIfTI 文件的目录
    :param output_directory: 存放 2D 切片的目录
    :param pattern: 用于匹配 3D NIfTI 文件的正则表达式
    :param list_file: 如果提供，将切片文件路径写入该文件
    """

    # 确保输出目录存在
    if not os.path.exists(output_directory):
        print(f"'{output_directory}' do not exist")
        os.makedirs(output_directory)
        print(f"create '{output_directory}' successfully")
    
  
    full_file_list = []
    # 遍历输入目录中的所有文件（仅第一层）
    for f in os.listdir(input_directory):
        file_path = os.path.join(input_directory, f)
        print(file_path)
        file_name = get_file_name(f)
        # 检查文件是否匹配给定的正则表达式
        if os.path.isfile(file_path) and re.match(pattern, file_name):
            print(f"Found file: {file_path}")
            
            # 调用 split_3d_to_2d 函数处理该文件
            output_file_paths = split_3d_to_2d(file_path, output_directory, prefix=file_name+ "_")
            full_file_list.extend(output_file_paths)

    return full_file_list

def split_3d_to_2d_SynthRAD202X(data_root, mode, dataset_info, pattern = ".*", seg_task = None, modality_folders = None):
    paths = generate_paths_from_dict(dataset_info)
    if modality_folders == None:
        modality_folders = ['A','B']
    for path in paths:
        for modality_folder in modality_folders:
            if seg_task is not None:
                if not isinstance(seg_task, list):
                    seg_task = [seg_task]
                for task in seg_task:
                    input_dir = os.path.join(data_root, path, "3D", "segmentation", task, mode + modality_folder)
                    output_dir = os.path.join(data_root, path,"2D", "segmentation", task, mode + modality_folder)
                    split_3d_to_2d_in_folder(input_dir, output_dir, pattern)
            else:
                input_dir = os.path.join(data_root, path, "3D", mode + modality_folder)
                output_dir = os.path.join(data_root, path,"2D", mode + modality_folder)
                split_3d_to_2d_in_folder(input_dir, output_dir, pattern)

def slice_3d(data_path, position, axis = "z", save_path=None):
    from PIL import Image
    # 如果指定了保存路径，保存切片
    slice_filter = sitk.ExtractImageFilter()
    image = sitk.ReadImage(data_path)
    size = image.GetSize()
    if axis == "z":
        if position < 0 or position >= size[2]:
            raise ValueError(f"Position {position} is out of bounds for axis 'z' with size {size[2]}")
        slice_filter.SetSize([size[0], size[1], 0])
        slice_filter.SetIndex([0, 0, position])
    elif axis == "y":
        if position < 0 or position >= size[1]:
            raise ValueError(f"Position {position} is out of bounds for axis 'y' with size {size[1]}")
        slice_filter.SetSize([size[0], 0, size[2]])
        slice_filter.SetIndex([0, position, 0])
    elif axis == "x":
        if position < 0 or position >= size[0]:
            raise ValueError(f"Position {position} is out of bounds for axis 'x' with size {size[0]}")
        slice_filter.SetSize([0, size[1], size[2]])
        slice_filter.SetIndex([position, 0, 0])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    slice_image = slice_filter.Execute(image)
    if save_path is not None:
        sitk.WriteImage(slice_image, save_path)
        print(f"Saved slice to {save_path}")
    return slice_image


    


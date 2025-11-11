import os
import re
from collections import defaultdict
import SimpleITK as sitk
from code_util.util import get_file_name,generate_paths_from_list
import numpy as np


def recontruct_3D_from_2D(modality, threeD_id, files, output_dir, data_format, ref_folder, delete_2D = False):
    """
    处理每个分组的文件列表。

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Processing group {modality, threeD_id}:")
    slices = []
    # print(files)
    for f in files:
        # print(f" - {f}")
        slice_image = sitk.ReadImage(f)
        slices.append(slice_image)
    
    output_filename = threeD_id + data_format
    stacked_image = sitk.JoinSeries(slices)
    if ref_folder is not None:
        # 判断是否为一个文件名
        ref_image = None
        if isinstance(ref_folder,str) and os.path.isfile(ref_folder):
            ref_image = sitk.ReadImage(ref_folder)
        else:
            ref_image_path = generate_paths_from_list(ref_folder, postfix=output_filename)
            print(ref_image_path)
            for ref_image_path_ in ref_image_path:
                if os.path.exists(ref_image_path_):
                    ref_image = sitk.ReadImage(ref_image_path_)
                    break
        if ref_image is None:
            raise ValueError(f"Reference image not found for {output_filename} in {ref_folder}")
        stacked_image.SetSpacing(ref_image.GetSpacing())
        stacked_image.SetOrigin(ref_image.GetOrigin())
        stacked_image.SetDirection(ref_image.GetDirection())
    # output_filename = f'{threeD_id}_{modality}' + data_format
    output_path = os.path.join(output_dir, output_filename)
    sitk.WriteImage(stacked_image, output_path)
    print(f'Saved {output_path}')
    if delete_2D:
        for f in files:
            os.remove(f)
        print(f"Deleted 2D file of %s" % output_path)
    return output_path

def recontruct_3D_from_2D_4folder(input_dir, output_dir, pattern, data_format, ref_folder):
    """
    查找并处理匹配给定正则表达式模式的文件。
    
    :param directory: 需要查找文件的目录
    :param pattern: 用于匹配文件名的正则表达式模式
    """
    print(f"Recontructing 3D from 2D in {input_dir} to {output_dir} with pattern {pattern}")
    from code_util.util import get_file_name
    # 获取目录下的所有文件
    files = os.listdir(input_dir)
    # 创建3级嵌套字典
    nested_dict = lambda: defaultdict(nested_dict)
    grouped_files = nested_dict()

    # 遍历列表，匹配字符串并存入字典
    for file in files:
        file_name = get_file_name(file)
        match = re.match(pattern, file_name)
        if match:
            threeD_num = match.group(1)  # 3D number
            twoD_num = match.group(2)  # 2D number
            modality = match.group(3)  # modality
            value = file  
            grouped_files[modality][threeD_num][twoD_num] = os.path.join(input_dir, value)

    # 将字典的最后一层转换为列表，并根据2D number进行排序
    for modality, threeDs in grouped_files.items():
        for threeD_id, twoDs in threeDs.items():
            twoD_ids = sorted(twoDs.keys(), key=lambda x: int(x))
            grouped_files[modality][threeD_id] = [twoDs[k] for k in twoD_ids]
    # 对每个分组进行处理
    file_paths = []
    for modality, threeDs in grouped_files.items():
        output_dir_modality = os.path.join(output_dir, modality)
        if not os.path.exists(output_dir_modality):
            os.makedirs(output_dir_modality)
        for threeD_id, twoDs in threeDs.items():
            file_path = recontruct_3D_from_2D(modality, threeD_id, twoDs, output_dir_modality, data_format, ref_folder, delete_2D = True)
            file_paths.append(file_path)
    return file_paths

def reconstruct_3D_from_patch(patches, positions, patch_size):
    """
    patches: list of numpy arrays with shape [D, H, W]
    positions: list of (d, h, w)
    vol_shape: (D, H, W)
    patch_size: (pd, ph, pw)
    """
    # 计算整体体积大小
    max_d = max([pos[0] for pos in positions]) + patch_size[0]
    max_h = max([pos[1] for pos in positions]) + patch_size[1]
    max_w = max([pos[2] for pos in positions]) + patch_size[2]
    vol_shape = (max_d, max_h, max_w)
    C = 1  # Assuming single channel for simplicity
    result = np.zeros((C,) + vol_shape, dtype=np.float32)
    count_map = np.zeros_like(result)
    for patch, (d, h, w) in zip(patches, positions):
        patch_numpy = sitk.ReadImage(patch)
        patch = sitk.GetArrayFromImage(patch_numpy)  # Convert to numpy array
        # read the patch 
        result[:, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += patch[np.newaxis, ...]
        count_map[:, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += 1

    result /= np.maximum(count_map, 1)  # Avoid division by zero
    return result.squeeze(0)  # Remove channel dimension if not needed

def recontruct_3D_from_patch_4folder(input_dir, output_dir, pattern, config, ref_folder=None):
    """
    查找并处理匹配给定正则表达式模式的文件，并根据patch重建3D图像。
    
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    :param pattern: 用于匹配文件名的正则表达式模式
    :param config: 配置字典，包含数据格式和patch大小等信息
    :param save_list: 保存的模态列表
    """
    files = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nested_dict = lambda: defaultdict(nested_dict)
    grouped_files = nested_dict()

    for file in files:
        print(file)
        file_name = get_file_name(file)
        match = re.match(pattern, file_name)
        if match:
            threeD_id = match.group(1)  # 3D number
            d = match.group(2)  # d position
            h = match.group(3)  # h position
            w = match.group(4)  # w position
            modality = match.group(5)  # modality
            value = file
            grouped_files[modality][threeD_id][(int(d), int(h), int(w))] = os.path.join(input_dir, value)
    
    file_paths = []
    for modality, threeDs in grouped_files.items():
        if modality == "fake_B":
            for threeD_id, patches in threeDs.items():
                positions = list(patches.keys())
                patches = list(patches.values())
                threeD_data = reconstruct_3D_from_patch(
                    patches, positions,config["dataset"]["patch_wise"]["patch_size"],
                )
                output_filename = threeD_id + config["dataset"]["data_format"]
                if ref_folder is not None:
                    # 判断是否为一个文件名
                    if os.path.isfile(ref_folder):
                        ref_image = sitk.ReadImage(ref_folder)
                    else:
                        ref_image_path = os.path.join(ref_folder, output_filename)
                        ref_image = sitk.ReadImage(ref_image_path)
                    threeD_image = sitk.GetImageFromArray(threeD_data)
                    threeD_image.SetSpacing(ref_image.GetSpacing())
                    threeD_image.SetOrigin(ref_image.GetOrigin())
                    threeD_image.SetDirection(ref_image.GetDirection())
                output_path = os.path.join(output_dir, output_filename)
                sitk.WriteImage(threeD_image, output_path)
                print(f'Saved {output_path}')
                file_paths.append(output_path)
    return file_paths
    
    


if __name__ == "__main__":
    input_dir = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest/images"
    output_dir = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest/3D"    
    pattern = r"^(.+)_(\d+)_(.+)\.nii\.gz$"
    
    recontruct_3D_from_2D(input_dir, output_dir, pattern)

"""this file provide operations for some specific dataset 

1. MICCAI SynthRAD2023 Task2 brain 
2. MICCAI SynthRAD2025 
"""

import os
import SimpleITK as sitk
import re
import shutil
from code_util.dataset.prepare import split_3d_to_2d
from code_util.dataset.prepare import generate_mask_with_class_by_histogram,generate_mask_with_class_by_range
from code_util.dataset.analysis import count_slices_from_3d
from code_util.dataset.prepare import generate_paths_from_dict
from code_util.util import InMakedirs

def convert_SynthRAD202X_single(data_root, output_dir, pattern, dim, mode, data_format, list_file = None, slice_ratio = 1):
    """
    convert SynthRAD202X paired dataset to 2D slices
    Args:
        data_root: str, the root path of the dataset
        output_dir: str, the output path of the 2D slices
        mode: str, the mode of the dataset, train, test or validation
        pattern: str, the regular expression pattern of the folder name
        list_file: str, the file path to save the list of the 2D slices
        slice_ratio: float, the ratio of the slices to be extracted in a 3D volume
    """
    # slice_num = 1000
    # pattern = "^[1-2](AB|HN|TH)[A-E]\d{3}$" # SynthRad2025
    # pattern = "^[1-2][B|P][A-C]\d{3}$" # SynthRad2023
    # 遍历data_root下所有符合pattern的文件夹
    # data_root = os.path.join(data_root)
    data_root = os.path.join(data_root, mode)
    print("data_root: ", data_root)
    if list_file != None:
        full_file_list = []
    # 如果data_root不存在，直接返回
    if not os.path.exists(data_root):
        print("data_root does not exist: ", data_root)
        return
    for folder_name in os.listdir(data_root):
        if re.match(pattern,folder_name):
            print("process %s ......" % folder_name)
            # 由正则表达式匹配的结果判断task
            task = int(folder_name[0])
            if task == 1:
                source_path = os.path.join(data_root, folder_name, "mr" + data_format)
            elif task == 2:
                source_path = os.path.join(data_root, folder_name, "cbct" + data_format)
            else:
                continue # 如果task不是1或2，跳过当前文件夹

            ct_path = os.path.join(data_root, folder_name, "ct" + data_format)

            # 选择cbct/mri或ct，如果文件存在的话
            if os.path.exists(source_path) or os.path.exists(ct_path):
                pass 
            else:
                continue  # 如果cbct/mri和ct不存在，跳过当前文件夹

            if dim == "2D":
                # 读取cbct或ct的z轴维度
                image = sitk.ReadImage(source_path)
                size = image.GetSize()[2]
            
                split_list = list(range(size))
                if slice_ratio < 1:
                    split_list = split_list[::int(1/slice_ratio)]

                # 将文件夹名作为前缀，分别对cbct和ct调用函数split_3d_to_2d
                
                prefix = folder_name + "_"
                if os.path.exists(source_path):
                    output_file_paths_A = split_3d_to_2d(source_path,os.path.join(output_dir,dim,mode+"A"), split_list, prefix, data_format)
                if os.path.exists(ct_path):
                    output_file_paths_B = split_3d_to_2d(ct_path, os.path.join(output_dir,dim,mode+"B"), split_list, prefix, data_format)
                if list_file != None:
                    for i in range(len(output_file_paths_A)):
                        full_file_list.append([output_file_paths_A[i],output_file_paths_B[i]])
            elif dim == "3D":
                file_name = folder_name
                # 将3D数据直接复制到目标文件夹
                output_folder_path_A = os.path.join(output_dir, dim, mode+"A")
                output_folder_path_B = os.path.join(output_dir, dim, mode+"B")
                InMakedirs([output_folder_path_A,output_folder_path_B], exist_tips=False)
                output_file_path_A = os.path.join(output_folder_path_A, file_name + data_format)
                output_file_path_B = os.path.join(output_folder_path_B, file_name + data_format)
                if os.path.exists(source_path):
                    shutil.copy(source_path, output_file_path_A)
                if os.path.exists(ct_path):
                    shutil.copy(ct_path, output_file_path_B)
                if list_file != None:
                    full_file_list.append([output_file_path_A,output_file_path_B])
    if list_file != None:
        with open(list_file, 'a') as f:
            for item in full_file_list:
                f.write(item[0] + ',' + item[1] + '\n')
    
def convert_SynthRAD202X(data_root, output_dir, dim, mode, data_format, pattern, dataset_info, list_file = None, slice_ratio = 1):
    # pattern = "^[1-2](AB|HN|TH)[A-E]\d{3}$" # SynthRad2025
    # pattern = "^[1-2][B|P][A-C]\d{3}$" # SynthRad2023
    # dataset_info = {
    #     "name": "SynthRAD2025",
    #     "task": ["Task1","Task2"],
    #     "organ": ["AB","HN","TH"]
    # } # SynthRad2025
    # dataset_info = {
    #     "name": "SynthRAD2023",
    #     "task": ["Task2"],
    #     "organ": ["brain","pelvis"]
    # } # SynthRad2023
    paths = generate_paths_from_dict(dataset_info)
    for path in paths:
        data_dir = os.path.join(data_root, path)
        output_dir_temp = os.path.join(output_dir, path)
        convert_SynthRAD202X_single(data_dir, output_dir_temp, pattern, dim, mode, data_format, list_file = list_file, slice_ratio = slice_ratio)
        convert_SynthRAD202X_single_mask(data_dir, output_dir_temp, pattern, dim, mode, data_format, list_file = list_file, slice_ratio = slice_ratio)

def convert_SynthRAD2023_CT(data_root = "../datasets/Task2/brain/", output_dir = "./datasets/SynthRAD2023_cycleGAN/brain", data_fromat = ".nii.gz", list_file = None, slice_ratio = 1):
    """
    convert CT volumes in SynthRAD2023 dataset to 2D slices
    """
    # slice_num = 1000
    pattern = "^\d{1}B[A-C]\d{3}$"
    
    # 遍历data_root下所有以"xB"开头的文件夹
    if list_file != None:
        full_file_list = []
    for folder_name in os.listdir(data_root):
        if re.match(pattern,folder_name):
            # 构造cbct和ct文件的路径
            ct_path = os.path.join(data_root, folder_name, "ct" + data_fromat)

            # 选择cbct或ct，如果文件存在的话
            if os.path.exists(ct_path):
                pass 
            else:
                continue  # 如果cbct和ct不存在，跳过当前文件夹

            # 读取cbct或ct的z轴维度
            image = sitk.ReadImage(ct_path)
            size = image.GetSize()[2]
        
            split_list = list(range(size))
            if slice_ratio < 1:
                split_list = split_list[::int(1/slice_ratio)]
            
            # 将文件夹名作为前缀，分别对cbct和ct调用函数split_3d_to_2d
            print("process %s ......" % folder_name)
            prefix = folder_name + "_"
            output_file_paths = split_3d_to_2d(ct_path, output_dir, split_list, prefix)
            if list_file != None:
                for i in range(len(output_file_paths)):
                    full_file_list.append(output_file_paths[i])
    if list_file != None:
        with open(list_file, 'a') as f:
            for item in full_file_list:
                f.write(item + '\n')

def convert_SynthRAD202X_single_seg_mask(data_root = "../datasets/Task2/brain/", output_dir = "./datasets/SynthRAD2023/brain2",isTrain=True,method = "histogram",class_range = None):
    pattern = "^2B[A-C]\d{3}$"
    os.makedirs(output_dir,exist_ok=True)
    # 遍历data_root下所有以"2B"开头的文件夹
    for folder_name in os.listdir(data_root):
        if re.match(pattern,folder_name):
            # 构造ct文件的路径
            ct_path = os.path.join(data_root, folder_name, "ct.nii.gz")
            # 将文件夹名作为前缀，分别对cbct和ct调用函数split_3d_to_2d
            print("process %s ......" % folder_name)
            if method == "histogram":
                generate_mask_with_class_by_histogram(ct_path,output_dir,2,output_file_name = folder_name)
            elif method == "range":
                generate_mask_with_class_by_range(ct_path,output_dir,class_range=class_range,output_file_name = folder_name)

def convert_SynthRAD202X_single_mask(data_root, output_dir, pattern, dim, mode, data_format, list_file = None, slice_ratio = 1):
    # slice_num = 1000
    # pattern = "^[1-2](AB|HN|TH)[A-E]\d{3}$" # SynthRad2025
    # pattern = "^[1-2][B|P][A-C]\d{3}$" # SynthRad2023
    # 遍历data_root下所有符合pattern的文件夹
    data_root = os.path.join(data_root, mode)
    if list_file != None:
        full_file_list = []
    # 如果data_root不存在，直接返回
    if not os.path.exists(data_root):
        return
    for folder_name in os.listdir(data_root):
        if re.match(pattern,folder_name):
            print("process %s ......" % folder_name)
            mask_path = os.path.join(data_root, folder_name, "mask" + data_format)
            # 确保目标文件夹存在
            if os.path.exists(mask_path):
                pass 
            else:
                continue  # 如果cbct/mri和ct不存在，跳过当前文件夹
            
            if dim == "2D":
                # 读取mask的z轴维度
                image = sitk.ReadImage(mask_path)
                size = image.GetSize()[2]
                split_list = list(range(size))
                if slice_ratio < 1:
                    split_list = split_list[::int(1/slice_ratio)]
                # 将文件夹名作为前缀，对mask调用函数split_3d_to_2d
                prefix = folder_name + "_"
                output_file_paths = split_3d_to_2d(mask_path,os.path.join(output_dir,dim,"mask",mode), split_list, prefix, data_format)
                if list_file != None:
                    for i in range(len(output_file_paths)):
                        full_file_list.append(output_file_paths[i])
            elif dim == "3D":
                # 将3D数据直接复制到目标文件夹
                file_name = folder_name 
                output_folder_path = os.path.join(output_dir, dim, "mask", mode)
                InMakedirs(output_folder_path, exist_tips=False)
                output_file_path = os.path.join(output_folder_path, file_name + data_format)
                shutil.copy(mask_path, output_file_path)
                if list_file != None:
                    full_file_list.append(output_file_path)
    if list_file != None:
        with open(list_file, 'a') as f:
            for item in full_file_list:
                f.write(item + '\n')
            
def count_slices_SynthRAD202X(dir_path, pattern, axis='x'):
    total_slices = 0
    regex = re.compile(pattern)
    
    for foldername in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, foldername)
        if os.path.isdir(folder_path) and regex.match(foldername):
            ct_file_path = os.path.join(folder_path, 'ct.nii.gz')
            if os.path.isfile(ct_file_path):
                total_slices += count_slices_from_3d(ct_file_path, axis)
    
    return total_slices

if __name__ == "__main__":
    
    generate_class_mask_SynthRAD2023_Task_2(data_root= "/home/xdh/data/intelland/code/datasets/Task2/brain/division2/", output_dir= "/home/xdh/data/intelland/code/frameworks/InTransNet/file_dataset/SynthRAD2023/brain2/mask/3D",isTrain=True)

    convert_SynthRAD202X("/home/xdh/data/intelland/dataset","/home/xdh/data/intelland/code/InTransNet/file_dataset")
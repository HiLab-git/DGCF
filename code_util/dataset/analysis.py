"""
this file provides some tools for dataset analysis
"""

import os
import re
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from collections import defaultdict
from code_util.data.read_save import read_medical_image

def get_intensity_range_folder_depth(data_root, depth, data_format):
    """ 根据深度获取指定层级的子文件夹路径列表 """
    subfolders = [data_root]

    for _ in range(depth - 1):
        subfolders = [f.path for folder in subfolders for f in os.scandir(folder) if f.is_dir()]

    # 用字典存储按文件名分类的文件路径列表
    classified_files = defaultdict(list)

    # 遍历每个子文件夹，收集符合条件的文件
    for subfolder in subfolders:
        # if "3D" in subfolder:
        print(f"Processing folder: {subfolder}")
    
        # 获取当前文件夹下所有数据文件的列表
        data_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(data_format)]
        
        # 根据文件名分类
        for data_file in data_files:
            class_key = os.path.basename(data_file).split('_')[0]  # 以文件名前缀作为分类标准
            classified_files[class_key].append(data_file)

    # 遍历每个类别
    for class_key, data_files in classified_files.items():
        # 初始化最大和最小值
        min_of_all = float('inf')
        max_of_all = float('-inf')
        min_file_path = None
        max_file_path = None
        print(f"Processing class: {class_key}")
        
        # 遍历每个.nii文件
        for data_file in data_files:
            # 读取.nii文件
            image = sitk.ReadImage(data_file)

            # 获取体素值的范围
            min_value = sitk.GetArrayViewFromImage(image).min()
            max_value = sitk.GetArrayViewFromImage(image).max()

            # 更新全局最小和最大值
            if min_value < min_of_all:
                min_of_all = min_value
                min_file_path = data_file
            if max_value > max_of_all:
                max_of_all = max_value
                max_file_path = data_file

            # 输出每个文件的范围
            print(f"  {(data_file)} - Value Range: ({min_value}, {max_value})")

        # 输出全局范围和对应文件路径
        print(f"Value Range of All Slices: ({min_of_all}, {max_of_all})")
        print(f"File with min value: {min_file_path}")
        print(f"File with max value: {max_file_path}")

def get_intensity_range_folder(data_root, data_format=".mha"):
    data_files = [f.path for f in os.scandir(data_root) if f.is_file() and f.name.endswith(data_format)]

    ranges = []
    min_of_all = float('inf')
    max_of_all = float('-inf')
    min_file_path = None
    max_file_path = None

    for data_file in data_files:
        image = sitk.ReadImage(data_file)
        img_array = sitk.GetArrayViewFromImage(image)
        min_value = img_array.min()
        max_value = img_array.max()
        ranges.append((min_value, max_value))
        print(f"{data_file} - Value Range: ({min_value}, {max_value})")
        if min_value < min_of_all:
            min_of_all = min_value
            min_file_path = data_file
        if max_value > max_of_all:
            max_of_all = max_value
            max_file_path = data_file

    if ranges:
        avg_range = [(np.mean([r[0] for r in ranges]), np.mean([r[1] for r in ranges]))]
        print(f"Global min: {min_of_all} (file: {min_file_path})")
        print(f"Global max: {max_of_all} (file: {max_file_path})")
        print(f"Average range: {avg_range}")
    else:
        print("No files found matching the format.")

def get_shape_range_folder(data_root, data_format=".mha"):
    """ 获取指定文件夹下所有数据文件的shape范围，并分别统计每个维度的最小、最大、平均值 """
    data_files = [f.path for f in os.scandir(data_root) if f.is_file() and f.name.endswith(data_format)]

    shapes = []
    min_shape = [float('inf')] * 3
    max_shape = [float('-inf')] * 3
    min_file_path = [None] * 3
    max_file_path = [None] * 3

    for data_file in data_files:
        image = sitk.ReadImage(data_file)
        shape = image.GetSize()
        shapes.append(shape)
        print(f"{data_file} - Shape: {shape}")

        for i in range(3):
            if shape[i] < min_shape[i]:
                min_shape[i] = shape[i]
                min_file_path[i] = data_file
            if shape[i] > max_shape[i]:
                max_shape[i] = shape[i]
                max_file_path[i] = data_file

    if shapes:
        avg_shape = [np.mean([s[i] for s in shapes]) for i in range(3)]
        print(f"Min shape per dimension: {min_shape} (files: {min_file_path})")
        print(f"Max shape per dimension: {max_shape} (files: {max_file_path})")
        print(f"Average shape per dimension: {avg_shape}")
    else:
        print("No files found matching the format.")

def get_intensity_histogram(data_root, depth, data_format, bins=100, output_dir="./output_histograms"):
    """ 根据深度获取指定层级的子文件夹路径列表，并计算每个文件的体素值直方图，并保存到本地 """
    # 创建保存输出图像的目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subfolders = [data_root]

    for _ in range(depth - 1):
        subfolders = [f.path for folder in subfolders for f in os.scandir(folder) if f.is_dir()]

    # 用字典存储按文件名分类的文件路径列表
    classified_files = defaultdict(list)

    # 遍历每个子文件夹，收集符合条件的文件
    for subfolder in subfolders:
        print(f"Processing folder: {subfolder}")
        
        # 获取当前文件夹下所有数据文件的列表
        data_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(data_format)]
        
        # 根据文件名分类
        for data_file in data_files:
            class_key = os.path.basename(data_file).split('_')[0]  # 以文件名前缀作为分类标准
            classified_files[class_key].append(data_file)

    # 遍历每个类别
    for class_key, data_files in classified_files.items():
        print(f"Processing class: {class_key}")
        
        # 初始化一个列表来存储所有文件的体素值
        all_intensities = []

        # 遍历每个.nii文件
        for data_file in data_files:
            # 读取.nii文件
            image = sitk.ReadImage(data_file)

            # 将图像转换为 numpy 数组
            img_array = sitk.GetArrayViewFromImage(image)

            # 获取当前文件的体素值并加入列表
            all_intensities.extend(img_array.flatten())

        # 计算总体素值的直方图
        all_intensities = np.array(all_intensities)

        # 创建直方图并保存到文件
        plt.figure(figsize=(8, 6))
        plt.hist(all_intensities, bins=bins, alpha=0.7, color='blue', label=f"{class_key} - All Files")
        plt.title(f"Overall Intensity Histogram for Class: {class_key}")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")

        # 保存图像到指定路径
        output_file = os.path.join(output_dir, f"{class_key}_intensity_histogram.png")
        plt.savefig(output_file)
        plt.close()  # 关闭图表

        print(f"Saved histogram for class {class_key} to {output_file}")

def get_intensity_range_and_histogram(data_root, depth, data_format, bins=100, output_dir="./output_histograms", generate_histogram=False,  lower_percentile=5, upper_percentile=95):
    """
    根据深度获取指定层级的子文件夹路径列表，计算每个文件的体素值范围，并选择性生成直方图。
    
    :param data_root: 数据根目录，包含子文件夹
    :param depth: 子文件夹的层级
    :param data_format: 数据文件的后缀，例如 .mha
    :param bins: 直方图的分箱数，默认为 100
    :param output_dir: 直方图图像的保存目录（如果 generate_histogram 为 True）
    :param generate_histogram: 是否生成直方图图像并保存到本地
    """
    # 创建保存输出图像的目录
    if generate_histogram and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subfolders = [data_root]

    for _ in range(depth - 1):
        subfolders = [f.path for folder in subfolders for f in os.scandir(folder) if f.is_dir()]

    # 用字典存储按文件名分类的文件路径列表
    classified_files = defaultdict(list)

    # 遍历每个子文件夹，收集符合条件的文件
    for subfolder in subfolders:
        print(f"Processing folder: {subfolder}")
        
        # 获取当前文件夹下所有数据文件的列表
        data_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(data_format) and "mask" not in f.name]
        
        # 根据文件名分类（以文件名前缀作为分类标准）
        for data_file in data_files:
            class_key = os.path.basename(data_file).split('_')[0]  # 假设文件名前缀是分类标准
            classified_files[class_key].append(data_file)

    # 遍历每个类别
    for class_key, data_files in classified_files.items():
        if class_key ==  "ct.mha":
            continue
        print(f"Processing class: {class_key}")
        
        # 初始化最大、最小值和文件路径
        min_of_all = float('inf')
        max_of_all = float('-inf')
        min_file_path = None
        max_file_path = None
        
        # 初始化一个列表来存储所有文件的体素值
        all_intensities = []

        # 遍历每个文件
        for data_file in data_files:
            # 读取.nii文件
            image = sitk.ReadImage(data_file)

            # 将图像转换为 numpy 数组
            img_array = sitk.GetArrayViewFromImage(image)

            # 获取当前文件的体素值并更新最大最小值
            min_value = img_array.min()
            max_value = img_array.max()

            if min_value < min_of_all:
                min_of_all = min_value
                min_file_path = data_file
            if max_value > max_of_all:
                max_of_all = max_value
                max_file_path = data_file

            # 将当前文件的体素值添加到总的体素值列表中
            all_intensities.extend(img_array.flatten())

            # 输出每个文件的范围
            print(f"  {data_file} - Value Range: ({min_value}, {max_value})")

        # 输出该类的范围和对应文件路径
        print(f"Value Range of All Slices in class {class_key}: ({min_of_all}, {max_of_all})")
        print(f"File with min value: {min_file_path}")
        print(f"File with max value: {max_file_path}")

        # 计算 95 百分位数
        lower_percentile_value = np.percentile(all_intensities, lower_percentile)
        upper_percentile_value = np.percentile(all_intensities, upper_percentile)
        print(f"{lower_percentile}th Percentile for class {class_key}: {lower_percentile_value}")
        print(f"{upper_percentile}th Percentile for class {class_key}: {upper_percentile_value}")
        
        # 如果需要生成直方图
        if generate_histogram:
            # 计算总体素值的直方图
            all_intensities = np.array(all_intensities)

            # 创建直方图并保存到文件
            plt.figure(figsize=(8, 6))
            plt.hist(all_intensities, bins=bins, alpha=0.7, color='blue', label=f"{class_key} - All Files")
            plt.title(f"Overall Intensity Histogram for Class: {class_key}")
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.legend(loc="upper right")

            # 标出两个百分位数
            plt.axvline(x=lower_percentile_value, color='red', linestyle='--', label=f"{lower_percentile}th Percentile ({lower_percentile_value})")
            plt.axvline(x=upper_percentile_value, color='green', linestyle='--', label=f"{upper_percentile}th Percentile ({upper_percentile_value})")
            plt.legend(loc="upper right")

            # 保存图像到指定路径
            output_file = os.path.join(output_dir, f"{class_key}_intensity_histogram.png")
            plt.savefig(output_file)
            plt.close()  # 关闭图表

            print(f"Saved histogram for class {class_key} to {output_file}")

def get_slice_value_range(nii_file_path):
    # 读取.nii文件
    image = sitk.ReadImage(nii_file_path)

    # 获取.nii文件的数组表示
    image_array = sitk.GetArrayFromImage(image)

    # 获取体素值范围
    min_values = []
    max_values = []

    #
    min_of_all = image_array.min()
    max_of_all = image_array.max()

    # 遍历每个切片
    for slice_index in range(image_array.shape[0]):
        # 获取当前切片的体素值范围
        min_value = image_array[slice_index, :, :].min()
        max_value = image_array[slice_index, :, :].max()

        # 添加到列表中
        min_values.append(min_value)
        max_values.append(max_value)

        # 输出结果
        print(f"Slice {slice_index + 1} - Value Range: ({min_value}, {max_value})")

    # 输出结果
    print(f"Value Range of All Slices: ({min_of_all}, {max_of_all})")
    return min_values, max_values

def plot_file_HU_histograms(nii_file_path):
    # 读取.nii文件
    image = sitk.ReadImage(nii_file_path)

    # 获取.nii文件的数组表示
    image_array = sitk.GetArrayFromImage(image)

    # 扁平化数组以获取所有体素值
    flattened_array = image_array.flatten()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_array, bins=100, color='blue', alpha=0.7)
    plt.title('HU Value Distribution')
    plt.xlabel('HU Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def count_slices_from_3d(path_3d, axis):
    image_array = read_medical_image(path_3d)
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis in axis_map:
        return image_array.shape[axis_map[axis]]
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

def count_slices_from_3d_dir(dir_path, pattern, axis='x'):
    total_slices = 0
    regex = re.compile(pattern)
    for filename in os.listdir(dir_path):
        if regex.match(filename):
            path_3d = os.path.join(dir_path, filename)
            total_slices += count_slices_from_3d(path_3d, axis)
    return total_slices


if __name__ == '__main__':
    # 使用示例
    # file_path = '/home/xdh/data/intelland/datasets/SynthRAD2023/original/Task2/brain/division2/train/2BA002/ct.nii.gz'
    # get_slice_value_range(file_path)
    # plot_file_HU_histograms(file_path)

    # # 使用示例
    folder_path = '/home/xdh/data/intelland/datasets/SynthRAD2023/original/Task2/brain/division2/train'
    # get_nii_value_range(folder_path)    



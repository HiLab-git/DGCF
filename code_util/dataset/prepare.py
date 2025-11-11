"""this file provides functions to contruct the standard dataset required in the framework"""

import os
import SimpleITK as sitk
import re
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
# deepcopy
from copy import deepcopy
from code_util.util import InMakedirs

# from scipy.signal import find_peaks
from code_util.data.read_save import read_medical_image
from code_util.util import get_full_extension,get_file_name

def replace_in_filenames(directory, A:str, B:str):
    # 检查输入目录是否存在
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 构建完整的文件路径
        file_path = os.path.join(directory, filename)
        print(filename)
        # 确保只处理文件
        if os.path.isfile(file_path):
            # 检查文件名中是否包含字符串A
            if A in filename:
                # 创建新的文件名
                new_filename = filename.replace(A, B)
                new_file_path = os.path.join(directory, new_filename)
                
                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} -> {new_file_path}")

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

            
def generate_mask_with_class_by_histogram(file_path, output_dir, peaks_num=2, peak_width_factor=0.08, output_file_name = None):
    """
    读取一个.nii文件，根据其直方图找到指定数量的峰，将峰附近的区域和非峰区域划分为不同类别，
    生成一个.nii.gz格式的分割结果，并可视化结果。

    :file_path: path of a 3D or 2D .nii.gz file
    :peaks_num: number of peaks to find in the histogram of the image or volume 
    :peak_width_factor: fraction of total histogram range to consider as "near" the peak
    """
    # Step 1: Read the medical image using the provided function
    image_array = read_medical_image(file_path)
    
    # Step 2: Compute the histogram of the image array
    hist, bin_edges = np.histogram(image_array, bins='auto')
    # print(hist,bin_edges)
    
    # Step 3: Find the peaks in the histogram
   
    peaks, _ = find_peaks(hist,distance = len(hist)*peak_width_factor*2)
    # peaks = sorted(hist,reverse=True)[:peaks_num]
    if 0 not in peaks:
        peaks = np.append(peaks, 0)
    # print(peaks)
    # Sort peaks by height (histogram value at each peak)
    sorted_peaks = sorted(peaks, key=lambda p: hist[p], reverse=True)

    # Take the top n peaks
    top_peaks = sorted_peaks[:peaks_num]
    sorted_top_peaks = sorted(top_peaks, key=lambda p: bin_edges[p])
    # print(sorted_top_peaks)
    
    # Step 4: Create the mask based on the found peaks
    mask = np.zeros_like(image_array)
    
    # Step 5: Define regions around each peak
    total_range = bin_edges[-1] - bin_edges[0]
    peak_regions = []
    
    for i, peak in enumerate(sorted_top_peaks):
        peak_value = bin_edges[peak]
        # Determine the range around the peak
        lower_bound = max(bin_edges[0], peak_value - peak_width_factor * total_range)
        upper_bound = min(bin_edges[-1], peak_value + peak_width_factor * total_range)
        
        # Store the region for visualization
        peak_regions.append((lower_bound, upper_bound))

    # step 6: assign class for different region: region outside and inside peaks for different peaks,respectively
    class_regions = []
    upper_last = bin_edges[0]
    for i, (lower_bound,upper_bound) in enumerate(peak_regions):
        if lower_bound > upper_last:
            class_regions.append((upper_last,lower_bound))
        class_regions.append((lower_bound,upper_bound))
        upper_last = upper_bound
        if i == len(peak_regions) - 1:
            if upper_bound < bin_edges[-1]:
                class_regions.append((upper_bound,bin_edges[-1]))
    # print(class_regions)

    for i, (lower_bound,upper_bound) in enumerate(class_regions):
        mask[(image_array >= lower_bound) & (image_array <= upper_bound)] = i + 1

    unique, counts = np.unique(mask, return_counts=True)
    
    # 将统计结果转化为字典
    class_counts = dict(zip(unique, counts))
    # print(class_counts)
    # return 

    # Step 7: Save the mask as a .nii.gz file
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(sitk.ReadImage(file_path))
    if output_file_name == None:
        output_file_name = os.path.basename(file_path).split(".")[0]
    output_file_name = output_file_name + ".nii.gz"
    output_path = os.path.join(output_dir,output_file_name)
    # sitk.WriteImage(mask_image, output_path)
    
    # Step 8: Visualization
    def visualize_classification(image_array, mask, hist, bin_edges, peaks, peak_regions):
        """
        Visualize the classification result along with the histogram and peak regions.
        
        For 2D images, it directly shows the image and mask.
        For 3D images, it shows the middle slice of the volume.
        """
        if image_array.ndim == 2:
            # 2D Image
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].set_title("Original Image")
            axes[0].imshow(image_array, cmap='gray')
            axes[1].set_title("Classified Mask")
            axes[1].imshow(mask, cmap='nipy_spectral')

            # Plot the histogram
            axes[2].set_title("Histogram with Peaks")
            axes[2].plot(bin_edges[:-1], hist, color='gray')
            axes[2].scatter(bin_edges[peaks], hist[peaks], color='red', marker='x')
            
            # Highlight peak regions
            for lower_bound, upper_bound in peak_regions:
                axes[2].axvspan(lower_bound, upper_bound, color='yellow', alpha=0.3)
            
            plt.show()
        
        elif image_array.ndim == 3:
            # 3D Image - visualize the middle slice
            middle_slice = image_array.shape[0] // 2
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].set_title("Original Image (Middle Slice)")
            axes[0].imshow(image_array[middle_slice], cmap='gray')
            axes[1].set_title("Classified Mask (Middle Slice)")
            axes[1].imshow(mask[middle_slice], cmap='nipy_spectral')

            # Plot the histogram
            axes[2].set_title("Histogram with Peaks")
            axes[2].plot(bin_edges[:-1], hist, color='gray')
            axes[2].scatter(bin_edges[peaks], hist[peaks], color='red', marker='x')
            
            # Highlight peak regions
            for lower_bound, upper_bound in peak_regions:
                axes[2].axvspan(lower_bound, upper_bound, color='yellow', alpha=0.3)
            
            plt.show()

        else:
            print("Unsupported image dimensions for visualization.")
    visualize_classification(image_array, mask, hist, bin_edges, sorted_top_peaks, peak_regions)
    return output_path

def generate_mask_with_class_by_range(file_path, output_dir, class_range, output_file_name = None):
    """
    读取一个.nii文件，根据给定的阈值范围将像素分为不同类别
    生成一个.nii.gz格式的分割结果，并可视化结果。

    :file_path: path of a 3D or 2D .nii.gz file
    :peaks_num: number of peaks to find in the histogram of the image or volume 
    :peak_width_factor: fraction of total histogram range to consider as "near" the peak
    """
    # Step 1: Read the medical image using the provided function
    image_array = read_medical_image(file_path)
    mask = np.zeros_like(image_array)

    class_regions = class_range 
    
    for i, (lower_bound,upper_bound) in enumerate(class_regions):
        mask[(image_array >= lower_bound) & (image_array <= upper_bound)] = i

    unique, counts = np.unique(mask, return_counts=True)
    
    # 将统计结果转化为字典
    class_counts = dict(zip(unique, counts))
    # print(class_counts)
    # return 

    # Step 7: Save the mask as a .nii.gz file
    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(sitk.ReadImage(file_path))
    if output_file_name == None:
        output_file_name = os.path.basename(file_path).split(".")[0]
    output_file_name = output_file_name + ".nii.gz"
    output_path = os.path.join(output_dir,output_file_name)
    sitk.WriteImage(mask_image, output_path)

    return output_path

def convert_dataset_group(dataset_root,output_root,convert_function,dataset_info, **kwargs):
    levels = dataset_info.keys()
    if len(levels) == 0:
        convert_function(dataset_root,output_root,**kwargs)
    else: 
        for i,level in enumerate(levels):
            nodes = dataset_info[level]
            if not isinstance(nodes,list):
                nodes = [nodes]
            # 删除dataset_info中key为level的元素，但是不改变原始的dataset_info
            dataset_info_copy = deepcopy(dataset_info)
            dataset_info_copy.pop(level)
            for node in nodes:
                # 删除dataset_info中key为level的元素
                convert_dataset_group(os.path.join(dataset_root,node),os.path.join(output_root,node),convert_function,dataset_info_copy,**kwargs)

def generate_paths_from_dict(dataset_info):
    values = [v if isinstance(v, list) else [v] for v in dataset_info.values()]
    paths = ["/".join(items) for items in product(*values)]
    return paths


if __name__ == "__main__":

    """example"""
    # 批量修改文件名
    # replace_in_filenames('/home/xdh/data/intelland/code/frameworks/InTransNet/file_data/SynthRAD2023_cycleGAN/brain/trainB','ct_','')

    # 将单个3D volume切片为2D slice
    # split_3d_to_2d(input_nifti_path, output_directory)

    # 将整个文件夹下匹配的3D volume均切片为2D slice
    # 
    
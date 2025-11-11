import SimpleITK as sitk
import numpy as np
from PIL import Image
import os

def slice_nifti(data_path, axis, position, save_path=None, window_range=None):
    # 读取 .nii.gz 文件
    image = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(image)

    # 根据给定的轴和位置进行切片
    if axis == 'x':
        slice_img = data[position, :, :]
    elif axis == 'y':
        slice_img = data[:, position, :]
    elif axis == 'z':
        slice_img = data[:, :, position]
    else:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'.")

    # 设置窗宽 (window range)
    if window_range is not None:
        min_val, max_val = window_range
        slice_img = np.clip(slice_img, min_val, max_val)

    # 如果指定了保存路径，保存切片
    if save_path:
        file_extension = os.path.splitext(save_path)[-1].lower()
        if file_extension == '.gz' or file_extension == ".mha":
            # 创建新的 SimpleITK 图像并保存
            slice_image = sitk.GetImageFromArray(slice_img)
            sitk.WriteImage(slice_image, save_path)
        elif file_extension == '.png':
            # 归一化图像到 [0, 255] 以保存为 PNG
            slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
            img = Image.fromarray(slice_img)
            img.save(save_path)
        else:
            raise ValueError("Save path must end with either '.nii.gz' or '.png'.")

    return slice_img



import SimpleITK as sitk
import os

def change_spacing(input_folder,output_folder, spacing):
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = sitk.ReadImage(input_path)

            # Set the new spacing
            image.SetSpacing(spacing)

            # Write the transformed image
            sitk.WriteImage(image, output_path)
            print(f"Transformed {filename} with spacing {spacing} and saved to {output_path}")

def resample_image_to_spacing(image, new_spacing, interpolator=sitk.sitkLinear):
    """
    使用 SimpleITK 将图像重采样到指定 spacing。
    
    参数：
        image: SimpleITK.Image 对象
        new_spacing: 目标 spacing (tuple/list)，如 (1.0, 1.0, 1.0)
        interpolator: 重采样插值方式（默认线性），常见的有：
            - sitk.sitkNearestNeighbor（最近邻，适合标签图）
            - sitk.sitkLinear（线性插值，适合CT/MRI等灰度图）
            - sitk.sitkBSpline（更平滑）

    返回：
        重采样后的 SimpleITK.Image
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_direction = image.GetDirection()
    original_origin = image.GetOrigin()

    # 计算新的 size（四舍五入为整数）
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    # 设置重采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetTransform(sitk.Transform())  # 恒等变换
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())  # 处理边界区域

    return resampler.Execute(image)

def resample_image_to_spacing_folder(input_folder, output_folder, spacing, interpolator=sitk.sitkLinear):
    """
    遍历文件夹中的所有 NIfTI 文件，将它们重采样到指定的 spacing。
    
    参数：
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        spacing: 目标 spacing (tuple/list)，如 (1.0, 1.0, 1.0)
        interpolator: 重采样插值方式（默认线性）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图像
            image = sitk.ReadImage(input_path)

            # 重采样图像
            resampled_image = resample_image_to_spacing(image, spacing, interpolator)

            # 保存重采样后的图像
            sitk.WriteImage(resampled_image, output_path)
            print(f"Resampled {filename} to spacing {spacing} and saved to {output_path}")

if __name__ == '__main__':
    input_folder = 'file_dataset/SynthRAD2023/Task1/brain/3D/testB'
    output_folder = 'file_dataset/SynthRAD2023/Task1/brain/3D/testB_spacing_1.0'
    spacing = (1.0, 1.0, 1.0)
    # spacing_transform(input_folder, output_folder, spacing)
    resample_image_to_spacing_folder(input_folder, output_folder, spacing)

    
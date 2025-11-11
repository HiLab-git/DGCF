import numpy as np
from code_util.dataset.prepare import generate_paths_from_dict
import os

# 写一个函数 对CT/MRI做totalseg的分割 得到分割图
def totalseg_segmentation(input, output_path, **kwargs):
    """
    input: Union[str, Path, sitk.Image, nib.Nifti1Image]
        - 如果是文件路径，检查是否为.mha格式，如果是则用sitk读取并转化为nib.Nifti1Image格式
        - 如果是sitk.Image格式，转化为nib.Nifti1Image格式
        - 如果是nib.Nifti1Image格式，直接使用
    output_path: Union[str, Path]
        - 如果是文件路径，检查文件夹是否存在，不存在则创建，并保存分割结果到该路径
    **kwargs: 传递给totalsegmentator的其他参数，参考：
    ml = True, task = "total", device = "gpu:1"
    """
    import SimpleITK as sitk
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    from pathlib import Path

    if isinstance(output_path, (str, Path)):
        output_path = Path(output_path)
        # 创建文件夹（如果不存在）
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("output_path should be a string or Path object.")
    
    if isinstance(input, (str, Path)):
        input = str(input)
        print("Reading input file:", input)
        if input.endswith('.mha'):
            sitk_image = sitk.ReadImage(input)
            sitk.WriteImage(sitk_image, output_path)
            input = nib.load(str(output_path))
        elif input.endswith('.nii') or input.endswith('.nii.gz'):
            input = nib.load(input)
        else:
            raise ValueError("Unsupported file format. Please provide a .mha, .nii, or .nii.gz file.")
    elif isinstance(input, sitk.Image):
        sitk.WriteImage(input, output_path)
        input = nib.load(output_path)
    elif not isinstance(input, nib.Nifti1Image):
        raise ValueError("Unsupported input type. Please provide a file path, sitk.Image, or nib.Nifti1Image.")
    
    totalsegmentator(input = input, output = output_path, **kwargs)

    
# 对一个文件夹下的所有nii.gz文件做totalseg的分割
def totalseg_segmentation_batch(input_folder, output_folder, **kwargs):
    """
    input_folder: Union[str, Path]
        - 文件夹路径，包含多个.nii或.nii.gz文件
    output_folder: Union[str, Path]
        - 文件夹路径，保存分割结果
    """
    from pathlib import Path
    import nibabel as nib
    if isinstance(input_folder, (str, Path)):
        input_folder = Path(input_folder)
        if not input_folder.is_dir():
            raise ValueError("input_folder should be a valid directory.")
    else:
        raise ValueError("input_folder should be a string or Path object.")
    
    if isinstance(output_folder, (str, Path)):
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        print("Output folder is set to:", output_folder)
    else:
        raise ValueError("output_folder should be a string or Path object.")
    
    for file in input_folder.iterdir():
        # 提取没有后缀的文件名
        if file.name.endswith('.mha'):
            output_path = output_folder / (file.stem + '.nii.gz')
            totalseg_segmentation(file, output_path, **kwargs)
        elif file.name.endswith('.nii') or file.name.endswith('.nii.gz'):
            output_path = output_folder / file.name
            totalseg_segmentation(file, output_path, **kwargs)
        else:
            print(f"Skipping unsupported file format: {file.name}")

def totalseg_segmentation_SynthRAD202X(data_root, output_dir, mode, modality, modality_folder, dataset_info, ml = True, task = "total", device = "gpu:3"):
    paths = generate_paths_from_dict(dataset_info)
    path_task = task
    if modality == "ct":
        pass
    elif modality == "mr":
        task = task + "_mr"
    else:
        raise ValueError("modality should be 'ct' or 'mr'")
    for path in paths:
        data_dir = os.path.join(data_root, path,"3D",mode + modality_folder)
        output_dir_temp = os.path.join(output_dir, path, "3D", "segmentation", path_task, mode + modality_folder)
        totalseg_segmentation_batch(data_dir, output_dir_temp, ml = ml, task = task, device = device)

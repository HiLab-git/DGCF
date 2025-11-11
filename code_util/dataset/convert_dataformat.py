import SimpleITK as sitk
import nibabel as nib
import os
import numpy as np
from code_util.util import get_file_name

def convert_dataformat(input_path, source_format, target_format, output_path = None):
    if source_format == ".nii" or source_format == ".nii.gz":
        if target_format == ".mha":
            return convert_nifti_to_mha(input_path, output_path)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    elif source_format == ".mha" or source_format == ".mhd":
        if target_format == ".nii" or target_format == ".nii.gz":
            return convert_mha_to_nifti(input_path, output_path)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    else:
        raise ValueError(f"Unsupported source format: {source_format}")

def convert_nifti_to_mha(nifti_path, mha_path = None):
    if mha_path is None:
        file_name = get_file_name(nifti_path)
        mha_path = os.path.join(os.path.dirname(nifti_path), file_name + ".mha")
    # 使用simpleitk load nii
    img_sitk = sitk.ReadImage(nifti_path)
    # 使用simpleitk save mha
    mha_path = sitk.WriteImage(img_sitk, mha_path)
    # 将原数据删掉
    os.remove(nifti_path)
    return mha_path

def convert_mha_to_nifti(mha_path, nifti_path = None):
    if nifti_path is None:
        file_name = get_file_name(mha_path)
        nifti_path = os.path.join(os.path.dirname(mha_path), file_name + ".nii.gz")
    img_sitk = sitk.ReadImage(mha_path)
    nifti_path = sitk.WriteImage(img_sitk, nifti_path)
    os.remove(mha_path)
    return nifti_path

def convert_dataformat_batch(input_dir, source_format, target_format, output_dir = None):
    if output_dir is None:
        output_dir = input_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Converting files in {input_dir} from {source_format} to {target_format} and saving to {output_dir}")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(source_format):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                file_name = get_file_name(file)
                output_path = os.path.join(output_subdir, file_name + target_format)
                convert_dataformat(input_path, source_format, target_format, output_path)

def convert_dataformat_SynthRAD202X(data_root, dim, mode, source_format, target_format, dataset_info):
    from code_util.dataset.specific_dataset import generate_paths_from_dict
    paths = generate_paths_from_dict(dataset_info)
    for path in paths:
        input_dir_root = os.path.join(data_root, path, dim)
        inpur_dir_mask = os.path.join(input_dir_root, "mask", mode)
        input_dir_image_A = os.path.join(input_dir_root, mode + "A")
        input_dir_image_B = os.path.join(input_dir_root, mode + "B")
        convert_dataformat_batch(inpur_dir_mask, source_format, target_format)
        convert_dataformat_batch(input_dir_image_A, source_format, target_format)
        convert_dataformat_batch(input_dir_image_B, source_format, target_format)


        


        
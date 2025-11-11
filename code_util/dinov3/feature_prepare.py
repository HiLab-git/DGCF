
import torch
from code_network.dinov3.tools.dinov3_adapter import Dinov3Adapter
from code_util.dataset.prepare import generate_paths_from_dict
from code_util.util import get_file_name
from code_dataset import find_3D_form_2D
import os

def dinov3_feature_extraction_SynthRAD202X(data_root, output_dir, mode, modality, modality_folder, dataset_info, device):
    # 加载dinov3模型
    dinov3_adapter = Dinov3Adapter(model_name="dinov3_vitb16",freeze = True)
    # 将其移动到指定设备
    dinov3_adapter.to(device)
    dinov3_adapter.eval()
    torch.set_grad_enabled(False)
    paths = generate_paths_from_dict(dataset_info)
    for path in paths:
        data_dir = os.path.join(data_root, path,"2D", mode + modality_folder)
        output_dir_temp = os.path.join(output_dir, path, "2D", "feature", "dinov3", mode + modality_folder)
        dinov3_feature_extraction_batch(data_dir, output_dir_temp, feature_extracter = dinov3_adapter, modality = modality, modality_folder = modality_folder)
    
def dinov3_feature_extraction_batch(input_dir, output_dir, feature_extracter, modality, modality_folder):
    from pathlib import Path
    import os
    if isinstance(input_dir, (str, Path)):
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise ValueError("input_dir should be a valid directory.")
    else:
        raise ValueError("input_dir should be a string or Path object.")
    
    if isinstance(output_dir, (str, Path)):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Output folder is set to:", output_dir)
    else:
        raise ValueError("output_dir should be a string or Path object.")
    
    for file in input_dir.iterdir():
        if file.name.endswith('.nii') or file.name.endswith('.nii.gz') or file.name.endswith('.mha'):
            # 将file_name中的extension替换为'npy'
            file_name = get_file_name(file.name) + ".npy"
            output_path = output_dir / file_name
            dinov3_feature_extraction(file, output_path, feature_extracter, modality, modality_folder)
    
def dinov3_feature_extraction(input_path, output_path, feature_extracter, modality, modality_folder, resize = (256,256), save = True):
    import SimpleITK as sitk
    import numpy as np
    import os
    import torch

    args = {
        "SynthRAD2023": "file_dataset/SynthRAD2023/args.json",
        "SynthRAD2025": "file_dataset/SynthRAD2025/args.json",
    }

    # 使用SimpleITK读取医学图像
    image = sitk.ReadImage(str(input_path))
    image_array = sitk.GetArrayFromImage(image)  # 二维医学图像

    if modality == "ct":
        # 将图像由-1024~2000归一化到-1~1之间
        image_array = np.clip(image_array, -1024, 2000)
        image_array = (image_array + 1024) / (2000 + 1024) * 2 - 1
        image_array = image_array.astype(np.float32)
    elif modality == "mr":
        # 获取数据集的名字 
        if "SynthRAD2023" in str(input_path):
            args_path = args["SynthRAD2023"]
        elif "SynthRAD2025" in str(input_path):
            args_path = args["SynthRAD2025"]
        else:
            raise ValueError("input_path should contain 'SynthRAD2023' or 'SynthRAD2025' to determine the dataset.")
        import json
        with open(args_path, 'r') as f:
            dataset_args = json.load(f)
        img_path = find_3D_form_2D(str(input_path))
        img_index = get_file_name(img_path)
        min, max = dataset_args["99ptile"][modality_folder][img_index]
        # 将图像由min~max归一化到-1~1之间
        image_array = np.clip(image_array, min, max)
        image_array = (image_array - min) / (max - min) * 2
        image_array = image_array.astype(np.float32)
    else: 
        raise ValueError("modality should be 'ct' or 'mr'")
    # resize到256x256
    import cv2
    # 二维图像resize
    image_array = cv2.resize(image_array, resize, interpolation=cv2.INTER_LINEAR)

    # 将图像转换为PyTorch张量，并添加批次和通道维度
    image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # 形
    # 将图像通道数扩展为3
    image_tensor = image_tensor.repeat(1, 3, 1, 1)
    
    # 将图像张量移动到与模型相同的设备
    device = next(feature_extracter.parameters()).device
    image_tensor = image_tensor.to(device)
    # 使用模型提取特征
    with torch.no_grad():
        features = feature_extracter(image_tensor)
    # 将特征从GPU移动到CPU，并保存为npy文件 注意总特征为元组形式 每个元素分别为3,6,9,12层的特征
    features = np.array([feat.cpu().numpy().squeeze(0) for feat in features])
    if save:
        np.save(output_path, features)
    return features

    


    
    
    
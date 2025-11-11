from code_util.data.read_save import read_medical_image
from code_util.util import is_valid_value, get_file_name,generate_paths_from_list
import pandas as pd
import os
import re
import torch
import numpy as np

def calculate_dynamic_range(image, mask, dynamic_range=None):
    """
    Calculate the dynamic range of an image using a mask.
    """
    if mask is not None and mask.any():
        masked_image = image[mask > 0]
    else:
        return dynamic_range
    min_val = np.min(masked_image)
    max_val = np.max(masked_image)
    return (min_val, max_val)

def calculate(ct_path, sct_path, fun, mask_path=None, class_mask_path=None, device_id = None):
    """
    Calculate the metrics of two images.
    """
    if device_id == None:
        ct = read_medical_image(ct_path)
        sct = read_medical_image(sct_path)
        if mask_path:
            mask = read_medical_image(mask_path)
        else:
            mask = None
        if class_mask_path:
            class_mask = read_medical_image(class_mask_path)
        else:
            class_mask = None
    else:
        if device_id != -1:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{device_id}')
            else:
                print(f'cuda:{device_id} is not avaliable, use cpu')
                device = torch.device('cpu')
        else:
            device = torch.device('cpu') 
        ct = torch.from_numpy(read_medical_image(ct_path)).to(device)
        # 将ct的数据类型转换为float
        ct = ct.float()
        sct = torch.from_numpy(read_medical_image(sct_path)).to(device)
        sct = sct.float()
        if mask_path:
            mask = torch.from_numpy(read_medical_image(mask_path)).to(device)
        else:
            mask = None
        if class_mask_path:
            class_mask = torch.from_numpy(read_medical_image(class_mask_path)).to(device)
        else:
            class_mask = None

    masks = [None]
    if not isinstance(class_mask,type(None)):
        class_masks = []
        if isinstance(class_mask,np.ndarray):
            unique_values = sorted(np.unique(class_mask))
            for val in unique_values:
                class_masks.append((class_mask == val).astype(np.uint8))
        elif isinstance(class_mask,torch.Tensor):
            unique_values = sorted(torch.unique(class_mask))
            for val in unique_values:
                class_masks.append((class_mask == val).int())
        if not isinstance(mask,type(None)):
            class_masks = [class_mask*mask for class_mask in class_masks]
        masks = class_masks
    elif not isinstance(mask,type(None)):
        masks = [mask]
    metrics = []
    if len(masks) == 1:
        Ls = [3024]
    else:
        Ls = [-250-(-1024),250-(-250),3000-250]
    for i,mask in enumerate(masks):
        parameters = {
            "ct": ct,
            "sct": sct,
            "L": Ls[i],
            "window_size": 7,
            "mask": mask
        }
        metric = fun(**parameters)
        if is_valid_value(metric):
            metrics.append(metric)
        else:
            metrics.append(0)
    # print(metrics)
    return metrics

def calculate_folder(synthesis_folder, source_folder, target_folder, pattern, mask_folder = None, class_mask_folder = None, metric_names = ["SSIM","PSNR"], device_id = None,output_folder=None):
    """
    用于计算在InTransNet框架中产生的一个文件夹下的测试结果的metrics
    计算指标所使用的函数自己编写 有使用numpy和troch的两个版本
    """
    if not output_folder:
        output_folder = os.path.dirname(synthesis_folder)
    if device_id == None:
        from code_util.metrics.image_similarity.numpy import MSSIM_3D,MSE_3D,MAE_3D,PSNR_3D,RMSE_3D,SSIM_3D,Med_MSSIM_3D 
        metric_funs = {
        "MSSIM": MSSIM_3D,
        "SSIM": SSIM_3D,
        "PSNR": PSNR_3D,
        "MSE": MSE_3D,
        "MAE": MAE_3D,
        "RMSE": RMSE_3D,
        "Med_MSSIM": Med_MSSIM_3D
    }
    else:
        from code_util.metrics.image_similarity.torch import MSSIM_3D,MSE_3D,MAE_3D,PSNR_3D,RMSE_3D,SSIM_3D
        metric_funs = {
        "MSSIM": MSSIM_3D,
        "SSIM": SSIM_3D,
        "PSNR": PSNR_3D,
        "MSE": MSE_3D,
        "MAE": MAE_3D,
        "RMSE": RMSE_3D,
    }
    # 获取文件夹下所有文件名
    file_names = os.listdir(synthesis_folder)
        
    # 定义字典用于存储每个序号下的文件路径
    file_paths = {}

    # 定义正则表达式来提取序号和类型
    # pattern = re.compile(r'2BA(\d+)_(real_A|fake_B|real_B|mask)\.nii\.gz')
    # pattern = re.compile(r'2B(A|B|C)(\d+)_(real_A|fake_B|real_B)\.nii\.gz')
    # 遍历文件名
    for file_name in file_names:
        # 使用正则表达式提取信息
        match = re.match(pattern,file_name)
        if match:
            # 获取没有后缀的文件名
            key = os.path.splitext(file_name)[0]
            # 根据类型存储文件路径
            file_paths[key] = {}
            file_paths[key]['source'] = os.path.join(source_folder, file_name)
            file_paths[key]['synthesis'] = os.path.join(synthesis_folder, file_name)
            file_paths[key]['target'] = os.path.join(target_folder, file_name)
            if mask_folder != None:
                file_paths[key]['mask'] = os.path.join(mask_folder, file_name)
    print(file_paths)
    pattern_class_mask = re.compile(r'2B(A|B|C)(\d+)\.nii\.gz')
    if class_mask_folder != None:
        class_mask_names = os.listdir(class_mask_folder)
        for class_mask_name in class_mask_names:
            match = pattern_class_mask.match(class_mask_name)
            if match:
                group, seq = match.groups()
                key = int(seq)
                file_paths[key]['class_mask'] = os.path.join(class_mask_folder, class_mask_name)

    metrics = {'Sequence and type': []}
    # 遍历每个序号并计算指标
    for seq, file_paths_dict in file_paths.items():
        source_path = file_paths_dict.get('source')
        target_path = file_paths_dict.get('target')
        synthesis_path = file_paths_dict.get('synthesis')
        mask_path = file_paths_dict.get('mask')
        class_mask_path = file_paths_dict.get('class_mask')

        metrics['Sequence and type'].append("source_target" + "_" + str(seq))
        metrics['Sequence and type'].append("synthesis_target" + "_" + str(seq))
        # 确保ct和sct文件都存在
        if target_path and synthesis_path and source_path:
            print("Processing sequence", seq)
            print(source_path)
            print(target_path)
            print(synthesis_path)
            print(mask_path)
            print(class_mask_path)

            # 计算指标
            for metric_name in metric_names:
               
                if mask_path != None:   
                    full_mask_metric_cbct = calculate(target_path, source_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = None, device_id = device_id)
                    full_mask_metric_sct = calculate(target_path, synthesis_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = None, device_id = device_id)
                    column_name = metric_name + "_" + "all" 
                    if column_name not in metrics.keys():
                        metrics[column_name] = [full_mask_metric_cbct[0]]
                        metrics[column_name].append(full_mask_metric_sct[0])
                    else:
                        metrics[column_name].append(full_mask_metric_cbct[0])
                        metrics[column_name].append(full_mask_metric_sct[0])
                if class_mask_path != None:
                    class_mask_metric_cbct = calculate(target_path, source_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = class_mask_path, device_id = device_id)
                    class_mask_metric_sct = calculate(target_path, synthesis_path, metric_funs[metric_name], mask_path = mask_path, class_mask_path = class_mask_path, device_id = device_id)
                    for (i, class_metric) in enumerate(class_mask_metric_cbct):
                        column_name = metric_name + "_" + str(i)
                        if column_name not in metrics.keys():
                            metrics[column_name] = [class_metric]
                        else:
                            metrics[column_name].append(class_metric)
                    for (i, class_metric) in enumerate(class_mask_metric_sct):
                        column_name = metric_name + "_" + str(i)
                        if column_name not in metrics.keys():
                            metrics[column_name] = [class_metric]
                        else:
                            metrics[column_name].append(class_metric)
    for key in metrics.keys():
        if key == 'Sequence and type':
            metrics[key].append("cbct_ct_mean")
            metrics[key].append("sct_ct_mean")
        else:
            cbct_metrics = metrics[key][::2]
            cbct_mean_metric = sum(cbct_metrics)/len(cbct_metrics)
            sct_metrics = metrics[key][1::2]
            sct_mean_metric = sum(sct_metrics)/len(sct_metrics)
            metrics[key].append(cbct_mean_metric)
            metrics[key].append(sct_mean_metric)

    # print(metrics)
    # Create DataFrame
    results_df = pd.DataFrame(metrics)

    # Print the average of the last two rows
    print(results_df.iloc[-2:])
    
    # 保存结果到CSV文件
    if device_id == None:
        cal_tool = "numpy"
    else:
        cal_tool = "torch"
    if mask_folder == None:
        mask_postfix = "wo_mask"
    else:
        mask_postfix = os.path.normpath(mask_folder).split(os.path.sep)[-3]
    if class_mask_folder == None:
        class_mask_postfix = "wo_class"
    else:
        class_mask_postfix = os.path.normpath(class_mask_folder).split(os.path.sep)[-3]
    metrics_file = 'metrics_results_%s_%s_%s_%s_L.csv' % (metric_names[0],cal_tool,mask_postfix,class_mask_postfix)
    results_df.to_csv(os.path.join(output_folder, metrics_file), index=False)

def calculate_folder_SynthRAD2023(synthesis_folder, source_folder, target_folder, pattern, metric_names, mask_folder = None, class_range = None, dynamic_range = None, output_folder=None, cal_item = ""):
    """
    用于计算在InTransNet框架中产生的一个文件夹下的测试结果的metrics
    其中PSNR和SSIM的计算使用SynthRAD2023官方提供的函数
    """
    from code_util.metrics.image_similarity.SynthRAD2023 import psnr
    from code_util.metrics.image_similarity.SynthRAD2025 import ms_ssim as ssim
    from code_util.data.mask import generateSegMask,segMask2binaryMasks

    # 获取文件夹下所有文件名
    file_names = os.listdir(synthesis_folder)
        
    # 定义字典用于存储每个序号下的文件路径
    file_paths = {}

    # 定义正则表达式来提取序号和类型
    # pattern = re.compile(r'2BA(\d+)_(real_A|fake_B|real_B|mask)\.nii\.gz')
    # pattern = re.compile(r'2B(A|B|C)(\d+)_(real_A|fake_B|real_B)\.nii\.gz')
    # 遍历文件名
    for file_name in file_names:
        # 使用正则表达式提取信息
        base_file_name = get_file_name(file_name)
        match = re.match(pattern,base_file_name)
        if match:
            # 获取没有后缀的文件名
            key = file_name
            # 根据类型存储文件路径
            file_paths[key] = {}
            source_list = generate_paths_from_list(source_folder, postfix=file_name)
            target_list = generate_paths_from_list(target_folder, postfix=file_name)
            mask_list = generate_paths_from_list(mask_folder, postfix=file_name) if mask_folder != None else []
            # 找到list中第一个存在的路径 并解除list
            file_paths[key]['source'] = [path for path in source_list if os.path.exists(path)][0] if source_list else None
            file_paths[key]['synthesis'] = os.path.join(synthesis_folder, file_name)
            file_paths[key]['target'] = [target for target in target_list if os.path.exists(target)][0] if target_list else None
            if mask_folder != None:
                file_paths[key]['mask'] = [mask for mask in mask_list if os.path.exists(mask)][0] if mask_list else None

    metrics = {'Sequence and type': []}
    # 遍历每个序号并计算指标
    for seq, file_paths_dict in file_paths.items():
        source_path = file_paths_dict.get('source')
        target_path = file_paths_dict.get('target')
        synthesis_path = file_paths_dict.get('synthesis')
        mask_path = file_paths_dict.get('mask')

        metrics['Sequence and type'].append("source_target" + "_" + str(seq))
        metrics['Sequence and type'].append("synthesis_target" + "_" + str(seq))
        # 确保ct和sct文件都存在
        if target_path and synthesis_path and source_path:
            print("Processing sequence", seq)
            print(source_path)
            print(target_path)
            print(synthesis_path)
            print(mask_path)

            target = read_medical_image(target_path)
            source = read_medical_image(source_path)
            synthesis = read_medical_image(synthesis_path)
            masks = []
            dynamic_ranges = []
            if class_range != None:
                # calculate with mask indicate by range
                mask = generateSegMask(target,class_range)
                masks.extend(segMask2binaryMasks(mask,class_range = class_range, fuse=False))
                dynamic_ranges.extend(class_range)
            if mask_path != None:
                # calculate with mask in file
                mask = read_medical_image(mask_path)
                masks.extend(segMask2binaryMasks(mask,fuse=False))
                dynamic_ranges.append(calculate_dynamic_range(target, mask))
            masks.append(None)
            dynamic_ranges.append(dynamic_range)
            for i, mask in enumerate(masks):
                # dynamic_range_temp = calculate_dynamic_range(target, mask, dynamic_range)
                dynamic_range_temp = dynamic_ranges[i]
                for metric_name in metric_names:
                    if metric_name == "SSIM":
                        metric_source = ssim(target, source, mask, dynamic_range_temp)
                        metric_synthesis = ssim(target, synthesis, mask, dynamic_range_temp)
                    elif metric_name == "PSNR":
                        metric_source = psnr(target, source, mask, use_population_range=True, dynamic_range = dynamic_range_temp)
                        metric_synthesis = psnr(target, synthesis, mask, use_population_range=True, dynamic_range = dynamic_range_temp)
                    column_name = metric_name + "_" + str(i + 1)
                    if column_name not in metrics.keys():
                        metrics[column_name] = [metric_source]
                        metrics[column_name].append(metric_synthesis)
                    else:
                        metrics[column_name].append(metric_source)
                        metrics[column_name].append(metric_synthesis)
    # 计算平均值
    for key in metrics.keys():
        if key == 'Sequence and type':
            metrics[key].append("source_target_mean")
            metrics[key].append("synthesis_target_mean")
        else:
            source_metrics = metrics[key][::2]
            source_mean_metric = sum(source_metrics)/len(source_metrics)
            synthesis_metrics = metrics[key][1::2]
            synthesis_mean_metric = sum(synthesis_metrics)/len(synthesis_metrics)
            metrics[key].append(source_mean_metric)
            metrics[key].append(synthesis_mean_metric)
    # Create DataFrame
    results_df = pd.DataFrame(metrics)

    # Print the average of the last two rows
    print(results_df.iloc[-2:])
    # 
    if not output_folder:
        output_folder = os.path.dirname(synthesis_folder)
    metrics_file = 'metrics_results_SynthRAD2023_%s_msssim.csv' % (cal_item)
    results_df.to_csv(os.path.join(output_folder, metrics_file), index=False)

if __name__ == '__main__':
    data_folder = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest/3D"
    result_folder = "/home/xdh/data/intelland/code/frameworks/InTransNet/file_result/CBCT2CT_pix2pix/test_latest"
    calculate_folder(data_folder, result_folder)
   

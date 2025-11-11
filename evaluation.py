import os
from code_config.parser import parse
from code_util.util import generate_paths_from_list

def evaluation(status_config = None, common_config = None):
    
    # opt >>>> config
    config,common_config = parse("evaluation",status_config = status_config, common_config=common_config) 
    data_format = config["dataset"]["data_format"]
    result_dir = config["work_dir"]
    config["phase"] = "test"
    
    if config["reconstruction"]["conduct_reconstruction"] == True:
        from code_util.dataset.reconstruct import recontruct_3D_from_2D_4folder
        input_dir = os.path.join(result_dir,"2D/synthesis") # result 2D images
        output_dir = os.path.join(result_dir,"3D/synthesis")
        pattern = config["reconstruction"]["pattern"]
        phase = config["phase"]
        if config["dataset"].get("dir_A",None) is not None:
            ref_folder  = [os.path.join(config["dataset"]["dataroot"],config["dataset"]["dir_A"])]
        else:
            ref_folder = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D/"+config["phase"]+"A") # ref_folder指定了模态 是不灵活的 但是方便 因此暂时这样
        recontruct_3D_from_2D_4folder(input_dir, output_dir, pattern, data_format, ref_folder)

    # calculate metrics
    if config["metrics"]["image_similarity"].get("calculate_metrics",False) == True:
        from code_util.metrics.calculate import calculate_folder,calculate_folder_SynthRAD2023
        cal_list = ["fake_B","fake_A"]
        ref_folder = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D") # reference images
        for cal_item in cal_list:
            result_folder = os.path.join(result_dir,"3D/synthesis",cal_item) # generated images
            if not os.path.exists(result_folder):
                print("Result folder does not exist: {}, skip.".format(result_folder))
                continue
            phase = config["phase"]
            target_modality = cal_item.split("_")[-1]
            source_modality = "A" if target_modality == "B" else "B"
            source_folder = generate_paths_from_list(ref_folder,postfix=phase+source_modality)
            target_folder = generate_paths_from_list(ref_folder,postfix=phase+target_modality)
            mask_folfer = generate_paths_from_list(ref_folder,postfix="mask/"+phase)
            dynamic_range = config["metrics"]["image_similarity"].get("dynamic_range",None)
            metric_names = config["metrics"]["image_similarity"].get("metric_names", None)
            pattern = config["metrics"]["pattern"]
            # calculate_folder(result_folder, source_folder,target_folder, pattern, mask_folder = binary_mask_folfer, metric_names=metric_names, device_id=device_id)
            class_range = config["metrics"]["image_similarity"].get("class_range",None)
            calculate_folder_SynthRAD2023(result_folder, source_folder, target_folder, pattern, metric_names, mask_folder = mask_folfer, class_range=class_range, dynamic_range = dynamic_range, output_folder = result_dir, cal_item = cal_item)

    return common_config

if __name__ == '__main__':
    evaluation()
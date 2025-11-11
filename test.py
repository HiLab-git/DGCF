from collections import defaultdict
import os
import torch
try :
    from tqdm import tqdm
except ImportError:
    pass
from code_dataset import create_dataset
from code_model import create_model
from code_config.parser import parse
try:
    from code_record.visualizer import Visualizer
except ImportError:
    pass
from code_util.data.read_save import save_test_image
from code_util.util import get_file_name,generate_paths_from_list

def test(status_config = None, common_config = None):

    config,common_config = parse("test",status_config = status_config, common_config=common_config) 
    config["record"]["validation"] = False
    
    # dataset
    dataset, _ = create_dataset(config)  # create a dataset given dataset_mode and other configurations

    # model
    model = create_model(config)      # create a model given opt.model and other options
    model.setup(config)               # regular setup: load and print networks; create schedulers

    # create a website
    if config.get("docker",{}).get("use_docker") != True:
        dataset = tqdm(dataset, desc="Testing")
        visualizer = Visualizer(config)    

    # test with eval mode. This only affects layers like batchnorm and dropout.
    model.eval()
    epoch_iter = 0

    use_html = config["record"].get("html",{}).get("use_html",False) 

    # # Save all test results locally
    save_list = config["result"].get("save_list",["fake_B","fake_A"])
    
    # for i, data in enumerate(tqdm(dataset, desc="Testing")):
    for i, data in enumerate(dataset):
        epoch_iter += 1
        model.set_input(data)  # unpack data from data loader
        if config["model"].get("use_ft16"):
            with torch.amp.autocast('cuda',dtype=torch.float16):
                model.test()           # run inference
        else:
            model.test()           # run inference

        # Display results to HTML if needed
        if use_html:
            if epoch_iter % config["record"]["html"]["display_per_iter"] == 0:
                # print('processing (%04d)-th image... %s' % (i, img_paths))
                visualizer.display_on_html(model.get_current_visuals(), data["A"]["params"]["path"], phase = "test")
        if config["record"].get("CAM",{}).get("use_CAM",False):
            if epoch_iter % config["record"]["CAM"]["display_CAM_per_iter"] == 0:
                # visualizer.draw_CAM(model,config,img_paths = img_paths)
                pass
        A_params = data["A"]["params"]
        save_test_image(model.get_current_results(), A_params, config, save_list)

    # if the dataset is not patch_wise, it must be a 2D dataset
    if config.get("reconstruction",{}).get("conduct_reconstruction",False) == True:
        data_format = config["dataset"]["data_format"]
        result_dir = config["work_dir"]
        # reconstruct the whole volume from 2D images
        from code_util.dataset.reconstruct import recontruct_3D_from_2D_4folder
        twoD_dir = os.path.join(result_dir,"2D/synthesis") # result 2D images
        pattern = config["reconstruction"]["pattern_2D"]
        
        threeD_dir = os.path.join(result_dir,"3D/synthesis")
        # ref_pos = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D/"+config["phase"])
        if config["dataset"].get("dir_A",None) is not None:
            ref_folder  = [os.path.join(config["dataset"]["dataroot"],config["dataset"]["dir_A"]).replace("2D","3D")]
        else:
            ref_folder = generate_paths_from_list(config["dataset"]["dataset_position"],postfix="3D/"+config["phase"]+"A") # ref_folder指定了模态 是不灵活的 但是方便 因此暂时这样
        recontruct_3D_from_2D_4folder(twoD_dir, threeD_dir, pattern, data_format, ref_folder)
    
    if config.get("segmentation",{}).get("conduct_segmentation",False) == True:
        from code_util.data.totalseg import totalseg_segmentation_batch
        seg_input_folder_base = os.path.join(result_dir,"3D/synthesis")
        task = config["segmentation"].get("task","total")
        seg_output_folder_base = os.path.join(result_dir,"3D/segmentation",task)
        seg_list = ["fake_B","fake_A"]
        for seg_item in seg_list:
            seg_input_folder = os.path.join(seg_input_folder_base,seg_item)
            if not os.path.exists(seg_input_folder):
                print("Segmentation input folder does not exist: {}, skip.".format(seg_input_folder))
                continue
            seg_output_folder = os.path.join(seg_output_folder_base,seg_item)
            print("Totalseg segmentation starts, input folder: {}, output folder: {}".format(seg_input_folder,seg_output_folder))
            ml = config["segmentation"].get("ml",True)
            gpu = config["model"]["gpu_ids"][0]
            device = "gpu:{}".format(gpu) if gpu >=0 else "cpu"
            totalseg_segmentation_batch(seg_input_folder, seg_output_folder, ml = ml, task = task, device = device)
            print("Totalseg segmentation ends.")
  
    return common_config
    
if __name__ == '__main__':
    test()
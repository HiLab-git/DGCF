
"""this fucntion parser is used to analyze the configuration file to get an opt"""

import os
import json
import time
import argparse
import pprint
from pathlib import Path
from datetime import datetime

from code_util.dataset.prepare import generate_paths_from_dict
from code_util.util import deep_update

config_root = "./file_config"

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    """ dict to string for logger """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse_json_file(json_path):
    json_str = ""
    with open(json_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str)
    return opt

def parse(status = "train", status_config = None, common_config = None, save = True, val = False):

    # 提供使用命令行指定参数的功能
    parser = init_parser(status)
    cmdline_opt = vars(parser.parse_args())
    # base config
    base_config_path = os.path.join(config_root,"base.json")
    base_opt = parse_json_file(base_config_path)
    # train/test/evaludation config
    if status_config is not None:
        status_opt = status_config
    else:
        status_config = os.path.join(config_root,status + ".json")
        status_opt = parse_json_file(status_config)
    if val == True:
        status_opt = deep_update(status_opt,status_opt["validation"])
    
    if status == "train" or status == "validation" or status == "test":
        # experiment config
        experiment_config_path = os.path.join(config_root,"experiments.json")
        experiment_opt = parse_json_file(experiment_config_path)

        if cmdline_opt.get("work"):
            experiment_opt["work"]= cmdline_opt["work"]
        experiment_opt = experiment_opt[experiment_opt["work"]]
        if cmdline_opt["gpu"] is not None:
            if "model" not in experiment_opt["general"]:
                experiment_opt["general"]["model"] = {}
            experiment_opt["general"]["model"]["gpu_ids"] = [cmdline_opt["gpu"]]
        experiment_general_opt = experiment_opt.get("general",{})
        experiment_status_opt = experiment_opt.get(status,{})
        common_opt_temp = deep_update(base_opt,experiment_opt["general"])
    elif status == "evaluation":
        # evaludation dose not require experiment config
        experiment_general_opt = {}
        experiment_status_opt = {}
        common_opt_temp = base_opt
        if cmdline_opt["recons"] == True:
            status_opt["reconstruction"]["conduct_reconstruction"] = True
        if cmdline_opt["metrics"] == True:
            status_opt["metrics"]["calculate_metrics"] = True
        if cmdline_opt["name"] is not None:
            status_opt["name"] = cmdline_opt["name"]
        if cmdline_opt["gpu"] is not None:
            status_opt["gpu"] = cmdline_opt["gpu"]
    else:
        raise ValueError("Invalid status: {}".format(status))
    final_opt = deep_update(base_opt,status_opt)
    final_opt = deep_update(final_opt,experiment_general_opt)
    # common config
    if common_config is not None:
        common_opt = common_config
        common_opt = deep_update(common_opt_temp,common_opt)
        final_opt = deep_update(final_opt,common_opt)
    else:
        common_opt = common_opt_temp
    final_opt = deep_update(final_opt,experiment_status_opt)

    # update config with given config file by command line or fucntion input
    if cmdline_opt["config"] is not None:
        config_path = cmdline_opt["config"]
        # 1. from command line
        # 2. from function input
        config_opt = parse_json_file(config_path)
        final_opt = deep_update(final_opt,config_opt)

    final_opt["work_relative_path"] = construct_work_relative_path(final_opt,status)
    final_opt["dataset"]["dataset_position"] = [os.path.join(final_opt["dataset"]["dataroot"],dataset_relative_path) for dataset_relative_path in generate_paths_from_dict(final_opt["dataset"]["info"])]
    if status == "train":
        current_time = time.localtime()
        formatted_time = time.strftime("%Y%m%d_%H%M%S", current_time)
        final_opt["work_dir"] = os.path.join(final_opt["record"]["record_dir"],final_opt["work_relative_path"],formatted_time)
    elif status == "test" or status == "evaluation":
        final_opt["work_dir"] = os.path.join(final_opt["result"]["result_dir"], final_opt["work_relative_path"], final_opt["result"]["test_epoch"])
    else:
        raise ValueError("Invalid status: {}".format(status))
    os.makedirs(final_opt["work_dir"],exist_ok=True)
    # save configuration 
    if save:
        config_path = os.path.join(final_opt["work_dir"],status + "_config.json")
        with open(config_path, 'w') as json_file:
            json.dump(final_opt, json_file, indent=4)
        common_config_path = os.path.join(final_opt["work_dir"],"common_config.json")
        with open(common_config_path, 'w') as json_file:
            json.dump(common_opt, json_file, indent=4)
        # pprint.pprint(final_opt)
    else:
        common_config_path = None
    
    os.makedirs(final_opt["work_dir"],exist_ok=True)

    return final_opt, common_opt

def init_parser(status):
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--config', type=str, default=None, help='configuration file')
    parser.add_argument('--gpu', type=int, default=None, help='gpu id')
    if status == "train" or status == "test" or status == "validation":
        # train/test
        parser.add_argument('--work', type=str, default=None, help='work name')
        parser.add_argument('--epoch', type=int, default=None, help='epoch number')
    elif status == "evaluation":
        # evaluation
        parser.add_argument('--recons', action='store_true', help='conduct_reconstruction')
        parser.add_argument('--metrics', action='store_true', help='calculate_metrics')
        parser.add_argument('--name', type=str, default=None, help='experiment name')
    else:
        raise ValueError("Invalid status: {}".format(status))
    return parser

def construct_work_relative_path(config,phase = "train"):
    """
    基于dataset中的info将数据集的相对路径构造出来
    """
    dataset_info = config["dataset"]["info"]
    # 将info中的元素按照顺序拼接成相对路径
    dataset_relative_path = ""
    for key in dataset_info:
        value = dataset_info[key]
        if isinstance(value, list):
            value = "_".join(str(v) for v in value)
        dataset_relative_path = os.path.join(dataset_relative_path, value)
    if phase == "train" or phase == "validation" or phase == "evaluation":
        dim = config["model"]["dim"]
    elif phase == "test":
        dim = config["model"]["dim"]
    work_relative_path = os.path.join(dataset_relative_path,dim,config["name"])
    # if config["phase"] == "train":
    #     if config.get("continue", {}).get("continue_train", False):
            
    return work_relative_path

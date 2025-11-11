import importlib
from torch import nn

from code_network.tools.initialization import init_gpus,init_weights
from code_network.modules.general import get_norm_layer

def find_network_using_name(network_file, network_name):
    """Import the module "code_network/[network_file].py".

    """
    # 和code_data中找dataset类一样的逻辑
    network_filename = "code_network." + network_file
    networklib = importlib.import_module(network_filename)
    network = None
    # target_network_name = network_name.replace('_', '')
    target_network_name = network_name
    for name, cls in networklib.__dict__.items():
        if name.lower() == target_network_name.lower() \
           and issubclass(cls,nn.Module):
            network = cls

    if network is None:
        print("In %s.py, there should be a subclass of nn.Module with class name that matches %s in lowercase." % (network_filename, target_network_name))
        exit(0)

    return network

def get_parameters_by_network(config):
    network_name = config["network"]["netG"]
    # Define a mapping from network_name to parameter construction functions
    def default_params(cfg):
        return {
            "input_nc": cfg["dataset"]["image_channel"],
            "output_nc": cfg["dataset"]["image_channel"],
            "ngf": cfg["network"]["ngf"],
            "down_step": cfg["network"].get("down_step", 1),
            "mode": cfg["model"].get("predict"),
            "activate": cfg["network"].get("activate", True),
        }

    def unetplusplus_params(cfg):
        return {
            "input_nc": cfg["dataset"]["image_channel"],
            "output_nc": cfg["dataset"]["image_channel"],
            "deep_supervision": cfg["model"]["deep_supervision"]
        }

    def swinunet_params(cfg):
        return {
            "patch_size": cfg["network"]["patch_size"]
        }

    def meunet_params(cfg):
        params = {
            "input_nc": cfg["dataset"]["image_channel"],
            "output_nc": cfg["dataset"]["image_channel"],
            "ngf": cfg["network"]["ngf"],
            "down_step": cfg["network"]["down_step"],
            "patch_sizes": cfg["network"]["patch_sizes"],
            "mamba_blocks": cfg["network"]["mamba_blocks"],
            "residual": cfg["network"]["residual"],
        }
        return params

    def reunet_params(cfg):
        params = {
            "input_nc": cfg["dataset"]["image_channel"],
            "output_nc": cfg["dataset"]["image_channel"],
            "ngf": cfg["network"]["ngf"],
            "down_step": cfg["network"]["down_step"],
            "res_blocks": cfg["network"]["res_blocks"],
            "residual": cfg["network"]["residual"]
        }
        return params

    def teunet_params(cfg):
        params = {
            "input_nc": cfg["dataset"]["image_channel"],
            "output_nc": cfg["dataset"]["image_channel"],
            "ngf": cfg["network"]["ngf"],
            "down_step": cfg["network"]["down_step"],
            "transformer_depths": cfg["network"]["transformer_depths"],
            "residual": cfg["network"]["residual"],
        }
        return params

    def weunet_params(cfg):
        params = {
            "input_nc": cfg["dataset"]["image_channel"],
            "output_nc": cfg["dataset"]["image_channel"],
            "ngf": cfg["network"]["ngf"],
            "down_step": cfg["network"]["down_step"],
            "wavelet_blocks": cfg["network"]["wavelet_blocks"],
            "wavelet_kss": cfg["network"]["wavelet_kss"],
            "wavelet_levels": cfg["network"]["wavelet_levels"],
            "residual": cfg["network"]["residual"]
        }
        return params

    def discriminator_params(cfg):
        norm = cfg["network"]["norm"]
        return {
            "norm_layer": get_norm_layer(norm),
            "input_nc": int(2 * cfg["dataset"]["image_channel"]),
            "ndf": cfg["network"]["ndf"],
            "n_layers_D": cfg["network"]["n_layers_D"]
        }

    def wtunet_params(cfg):
        return {
            "input_nc": cfg["dataset"]["image_channel"],
            "output_nc": cfg["dataset"]["image_channel"],
            "ngf": cfg["network"]["ngf"],
            "down_step": cfg["network"]["down_step"],
            "wt_level": cfg["network"]["wt_level"],
        }

    def mlla_unet_params(cfg):
        return {
            "img_size": cfg["preprocess"]["resize"]["resize_size"][0],
            "patch_size": cfg["network"]["patch_size"],
            "in_chans": cfg["dataset"]["image_channel"],
            "num_classes": cfg["dataset"]["image_channel"],
        }

    def beziernet_params(cfg):
        return {
            "in_channels": cfg["dataset"]["image_channel"]
        }

    def unetplusplusseg_params(cfg):
        return {
            "in_channels": cfg["dataset"]["image_channel"],
            "classes": cfg["dataset"]["image_channel"],
            "encoder_name": cfg["network"]["encoder"],
        }
    def edm_unet_params(cfg):
        return {
            "img_resolution": cfg["preprocess"]["resize"]["resize_size"][0],
            "in_channels": cfg["dataset"]["image_channel"],
            "out_channels": cfg["dataset"]["image_channel"],
        }
    def unet_song_t_params(cfg):
        if cfg["model"].get("use_source_aid", False):
            input_channels = 2 * cfg["dataset"]["image_channel"]
        else:
            input_channels = cfg["dataset"]["image_channel"]
        return {
            "input_nc": input_channels,
            "output_nc": cfg["dataset"]["image_channel"],
            "ngf": cfg["network"]["ngf"],
            "down_step": cfg["network"].get("down_step", 1),
            "mode": cfg["model"].get("predict"),
            "activate": cfg["network"].get("activate", True),
        }
    
    def dpt_params(cfg):
        return {
            "model_name": cfg["network"].get("model_name", "dinov3_vitb16"),
            "use_ft16": cfg["model"].get("use_ft16", False),
            "features": cfg["network"].get("features", 128),
        }

    # Map network names to their parameter functions
    param_func_dict = {
        "unet": default_params,
        "unet3d": default_params,
        "unet25d": default_params,
        "cascadeunet": default_params,
        "resnet": default_params,
        "unetplusplus": unetplusplus_params,
        "swinunet": default_params,
        "mambaunet": swinunet_params,
        "meunet": meunet_params,
        "reunet": reunet_params,
        "teunet": teunet_params,
        "weunet": weunet_params,
        "discriminator": discriminator_params,
        "wtunet": wtunet_params,
        "mlla_unet": mlla_unet_params,
        "beziernet": beziernet_params,
        "polyct2mrinet": beziernet_params,
        "splinect2mrinet": beziernet_params,
        "unetplusplusseg": unetplusplusseg_params,
        "piecewiselinearnet": beziernet_params,
        "transreconsunet": default_params,
        "songunet": edm_unet_params,
        "unet_t": default_params,
        "unet_song_t": unet_song_t_params,
        "unetplusplus_song_t": unetplusplus_params,
        "DPT": dpt_params,
        "transunet": default_params,
        "resunet": default_params,
    }

    # Get parameters using the mapping, fallback to empty dict if not found
    parameters = param_func_dict.get(network_name, lambda cfg: {})(config)
    return parameters

def define_network(config, net_type = "g"):

    gpu_ids =  config["model"]["gpu_ids"]
    parameters = get_parameters_by_network(config)

    if net_type == "g":
        network_name = config["network"]["netG"]
        network_file = config["network"]["filename"]
    elif net_type.startswith("d"):
        netD = config["network"]["netD"]
        if netD == "basic":
            network_name = "NLayerDiscriminator"
        elif netD == "pixel":
            network_name = "PatchGAN"
        elif netD == "pixel_large":
            network_name = "LargePatchGAN"
        network_file = config["network"]["filename_d"]
        if net_type == "d":
            parameters.update({
                "ndf": config["network"]["ndf"],
                "input_nc": 2*config["dataset"]["image_channel"]
            })
        elif net_type == "d_unc":
            parameters.update({
                "ndf": config["network"]["ndf"],
                "input_nc": config["dataset"]["image_channel"]
            })
        else:
            raise ValueError("net_type for discriminator must be 'd' or 'd_unc'")
    else:
        raise ValueError("net_type must be 'g' or 'd*'")

    ClassNetG = find_network_using_name(network_file, network_name)
    net = ClassNetG(**parameters)
    net = init_gpus(net,gpu_ids)
    if config["phase"] == "train":
        init_weights(net, config["network"]["init_type"], config["network"]["init_gain"])

    return net        

    

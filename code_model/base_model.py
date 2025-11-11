import os
import torch
import numpy as np
from collections import OrderedDict
from abc import ABC, abstractmethod
from code_network.tools.scheduler import get_scheduler
from code_util import util
from code_util.model.load import load_partial_state_dict,set_trainable_params

try:
    from code_util.metrics.image_similarity.SynthRAD2023 import ssim
    # from code_util.metrics.image_similarity.SynthRAD2025 import ms_ssim as ssim

except ImportError:
    pass

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, config):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.config = config
        self.gpu_ids = config["model"]["gpu_ids"]
        self.use_ft16 = config["model"].get("use_ft16", False)
        self.scaler = torch.amp.GradScaler('cuda')
       
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        if config["phase"] == "train":
            self.save_dir = os.path.join(config["work_dir"],"models")  # save all the checkpoints to save_dir
            os.makedirs(self.save_dir,exist_ok=True)
        else:
            if config["result"].get("test_model_dir") == None:
                experiment_folder = util.find_latest_experiment(config["work_relative_path"])
                test_model_dir = os.path.join(experiment_folder,"models")
                self.save_dir = test_model_dir
            else:
                self.save_dir = os.path.join(config["result"]["test_model_dir"])

        # if config["preprocess"]["resize"] == True: 
        #     torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.metric_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def setup(self, config):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if config["phase"] == "train":
            self.schedulers = [get_scheduler(optimizer, config) for optimizer in self.optimizers]
            if config.get("continue", {}).get("continue_train", False):
                load_suffix = config["continue"]["continue_epoch"]
                model_dir = config["continue"].get("continue_model_dir", None)
                self.load_networks(load_suffix,model_dir)
        elif config["phase"] == "test":
            load_suffix = config["result"]["test_epoch"]
            model_dir = config["result"].get("test_model_dir", None)
            self.load_networks(load_suffix,model_dir)
        else:
            raise ValueError("phase must be train or test to setup the model")
        self.print_networks()

    def eval(self):
        self.phase = "val"
        """Make models eval mode during validation or test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        self.phase = "train"
        """Make models eval mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()      

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        if self.config["phase"] == "test":
            if self.config["model"]["dim"] == "25D":
                if self.config["model"].get("predict") == "all":
                    #  real_B.shape: B, C, D, H, W 选出real_B的D维度正中间的一张图像作为real_B
                    self.real_B = self.real_B[:, :, self.real_B.shape[2] // 2, :, :]
                    self.fake_B = self.fake_B[:, :, self.fake_B.shape[2] // 2, :, :]
                    self.real_A = self.real_A[:, :, self.real_A.shape[2] // 2, :, :]  # (B, C, H, W)
                    self.mask = self.mask[:, :, self.mask.shape[2] // 2, :, :]  # (B, C, H, W)
        # # 将visual截断在st的范围之间 得到real_A_st,real_B_st,fake_B_st
        # full_range = [-1024, 2000]
        # full_range_norm = [-1, 1]
        # st_range = [-200, 200]
        # full_min, full_max = full_range
        # norm_min, norm_max = full_range_norm
        # # 计算st_range在full_range归一化到full_range_norm下的值
        # st_range_norm = [
        #     (val - full_min) / (full_max - full_min) * (norm_max - norm_min) + norm_min
        #     for val in st_range
        # ]
        # # print(f"st_range_norm: {st_range_norm}, full_range_norm: {full_range_norm}")
        # self.real_A_st = torch.clamp(self.real_A, min=st_range_norm[0], max=st_range_norm[1])
        # self.real_B_st = torch.clamp(self.real_B, min=st_range_norm[0], max=st_range_norm[1])
        # self.fake_B_st = torch.clamp(self.fake_B, min=st_range_norm[0], max=st_range_norm[1])
        

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.config["model"]["lr_policy"] == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self, only_one = False):
        self.compute_visuals()
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                if not hasattr(self, name):
                    # print(f"Warning: {name} is not found in the model. Please check your visual_names.")
                    continue
                if only_one == True:
                    visual_ret[name] = getattr(self, name)[0:1]
                else:
                    visual_ret[name] = getattr(self, name)
        # # 创建一个dummy visual 应对tensorboard生成最后一个图像有延迟的问题
        # visual_ret["dummy"] = visual_ret[name]
        if self.config["model"]["dim"] == "3D":
            visual_ret = self.visuals_3D_to_2D(visual_ret)
        return visual_ret

    def get_current_results(self, only_one = False):
        self.compute_visuals()
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if not hasattr(self, name):
                # print(f"Warning: {name} is not found in the model. Please check your visual_names.")
                continue
            if isinstance(name, str):
                if only_one == True:
                    visual_ret[name] = getattr(self, name)[0:1]
                else:
                    visual_ret[name] = getattr(self, name)
        # 创建一个dummy visual 应对tensorboard生成最后一个图像有延迟的问题
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                if hasattr(self, 'loss_' + name):
                    errors_ret[name] = float(getattr(self, 'loss_' + name).detach())  # float(...) works for both scalar tensor and float number
                else:
                    errors_ret[name] = float(0) # 说明还没有开始计算
        return errors_ret

    def clear_loss(self):
        # 用于在val之前清除train留下的loss
        for name in self.loss_names:
            if isinstance(name, str):
                setattr(self, 'loss_' + name, float(0))
        
    def get_current_metrics(self):
        """Return traning metrics. train.py will print out these errors on console, and save them to a file"""
        metrics_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                if hasattr(self, 'metric_' + name):
                    metrics_ret[name] = float(getattr(self, 'metric_' + name))
                else:   
                    metrics_ret[name] = float(0)
        return metrics_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if isinstance(self.gpu_ids,list):
                    if len(self.gpu_ids) > 1:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, model_dir=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if model_dir is not None:
                    load_path = os.path.join(model_dir, load_filename)
                else:
                    load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                include = self.config["model"].get("continue", {}).get("include", None)
                exclude = self.config["model"].get("continue", {}).get("exclude", None)
                load_partial_state_dict(net, load_path, device=self.device, include = include, exclude = exclude)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                # state_dict = torch.load(load_path, map_location=str(self.device))
                # if hasattr(state_dict, '_metadata'):
                #     del state_dict._metadata
                # # # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                # net.load_state_dict(state_dict)
                trainable_keys = self.config["continue"].get("trainable_keys", None)
                freeze_keys = self.config["continue"].get("freeze_keys", None)
                set_trainable_params(net, trainable_keys=trainable_keys, freeze_keys=freeze_keys)


    def print_networks(self):
        """record the total number of parameters in the network and network architecture
        """
        net_info = []
        log_lines = []
        log_lines.append('---------- Networks initialized -------------\n')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                net_info.append(str(net) + '\n')
                net_params_info = '[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6)
                print(net_params_info)
                log_lines.extend(net_info)
                log_lines.append(net_params_info)
        log_lines.append('-----------------------------------------------\n')
        
        # Save to file
        os.makedirs(self.config["work_dir"],exist_ok=True)
        with open(os.path.join(self.config["work_dir"],'network_log.txt'), 'w') as f:
            for line in log_lines:
                f.write(line)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.real_A = input['A']["data"].to(self.device)
        self.real_B = input['B']["data"].to(self.device)
        # self.class_mask_matrix = input['class_mask'].to(self.device)
        self.mask = input['Mask']["data"].to(self.device)
        self.image_paths = {'A_path':input['A']["params"].get("path"),
                            'B_path':input['B']["params"].get("path"),
                            'Mask_path':input['Mask']["params"].get("path")}

    def visuals_3D_to_2D(self, visuals_3D: dict):
        """Convert 3D visuals to 2D visuals for visualization"""
        visuals_2D = OrderedDict()
        for key, value in visuals_3D.items():
            # Ensure the input is a 4D tensor (B, C, D, H, W)
            if value.ndim != 5:
                raise ValueError(f"Expected 5D tensor for {key}, but got {value.ndim}D tensor.")

            batch_size, channels, depth, height, width = value.shape

            # Randomly select a slice along each dimension for the entire batch
            slice_d = torch.randint(0, depth, (1,)).item()
            slice_h = torch.randint(0, height, (1,)).item()
            slice_w = torch.randint(0, width, (1,)).item()

            # Extract slices and add to visuals_2D
            visuals_2D[f"{key}_D"] = value[:, :, slice_d, :, :]
            visuals_2D[f"{key}_H"] = value[:, :, :, slice_h, :]
            visuals_2D[f"{key}_W"] = value[:, :, :, :, slice_w]

        return visuals_2D
        
    def calclulate_metric(self):
        if self.config["model"]["dim"] == "25D":
            if self.config["model"].get("predict") == "all":
                #  real_B.shape: B, C, D, H, W 选出real_B的D维度正中间的一张图像作为real_B
                self.real_B = self.real_B[:, :, self.real_B.shape[2] // 2, :, :]  # (B, C, H, W)
                self.fake_B = self.fake_B[:, :, self.fake_B.shape[2] // 2, :, :]
                self.real_A = self.real_A[:, :, self.real_A.shape[2] // 2, :, :]  # (B, C, H, W)
                self.mask = self.mask[:, :, self.mask.shape[2] // 2, :, :]  # (B, C, H, W)  
        with torch.no_grad():
            if "ssim" in self.metric_names:
                fake_B = self.fake_B.to(torch.float).cpu().numpy()
                real_B = self.real_B.to(torch.float).cpu().numpy()
                if hasattr(self, 'mask'):
                    mask = self.mask.to(torch.float).cpu().numpy() 
                else:
                    mask = [None]*fake_B.shape[0]
                # Compute SSIM for all samples and average
                metric_ssims = []
                for i in range(fake_B.shape[0]):
                    metric_ssims.append(ssim(fake_B[i], real_B[i], mask=mask[i], dynamic_range=(-1, 1)))
                self.metric_ssim = np.mean(metric_ssims)

    def optimize_parameters(self):
        # if self.use_ft16 == True:
        if self.use_ft16 == True:
            with torch.amp.autocast('cuda',dtype=torch.float16):
                self.forward()                   # compute fake images: G(A)
                # update G
                self.cal_loss_G()

            self.optimizer_G.zero_grad()  
            self.scaler.scale(self.loss_G_lambda).backward()

            # self.optimizer_G.step()             # update G's weights
            self.scaler.step(self.optimizer_G)             # update G's weights
            self.scaler.update()
        else:
            self.forward()                   # compute fake images: G(A)
            # update G
            self.cal_loss_G()

            self.optimizer_G.zero_grad()  
            self.loss_G_lambda.backward()
            self.optimizer_G.step()             # update G's weights

    def calculate_loss(self):
        if self.use_ft16 == True:
            with torch.amp.autocast('cuda',dtype=torch.float16):
                with torch.no_grad():
                    self.forward()
                    self.cal_loss_G()
        else:
            with torch.no_grad():
                self.forward()
                self.cal_loss_G()
        
        
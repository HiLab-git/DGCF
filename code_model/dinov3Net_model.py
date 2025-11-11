import torch
from .base_model import BaseModel
from code_network import define_network
from code_util.util import get_file_name
import numpy as np
import os

# from code_network.dinov3.dinoplusdecoder import DINOv3MultiStageDecoder
# from code_network.dinov3.dinoplusdecoder import DINOv3MultiScaleDecoder
from code_network.dinov3.dinoparallelencoderplusdecoder import DINOGuidedGenerator
 
class Dinov3NetModel(BaseModel):
    
    def __init__(self, config):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, config)
        self.config = config
        self.model_names = ['G']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'diff']
        self.loss_names = ['G']
        self.metric_names = ['ssim']
        ngf = config["network"].get("ngf", 64)
        use_cnn = config["network"].get("use_cnn", True)
        fusion_type = config["network"].get("fusion_type", "cnn")
        use_dino_layers = config["network"].get("use_dino_layers", [True,True,True,True], )
        self.netG = DINOGuidedGenerator(base_ch = ngf, use_dino_layers= use_dino_layers, use_cnn = use_cnn, fusion_type = fusion_type).to(self.device)
       
        if config["phase"] == "train":
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=config["network"]["lr"], betas=(config["network"]["beta1"], 0.999))
            self.optimizers.append(self.optimizer_G)  
        # 检查网络中哪些部分是冻结的
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            for param_name, param in net.named_parameters():
                if not param.requires_grad:
                    print(f'Parameter {param_name} in net{name} is frozen.')
                else:
                    print(f'Parameter {param_name} in net{name} is trainable.')
        
                """Perceptual Loss"""
        self.use_perceptual_loss = config["model"].get("use_perceptual_loss", False)
        if self.use_perceptual_loss == True:
            from code_network.tools.loss.perceptual import PerceptualLoss
            perceptual_loss_type = config["model"].get("perceptual_loss_type", "l1")
            perceptual_model = config["model"].get("perceptual_model","dinov3_vitb16")
            multi_resolution = config["model"].get("perceptual_multi_resolution", False)
            multi_layer = config["model"].get("perceptual_multi_layers", [False,False,False,False])
            self.criterionPerL = PerceptualLoss(feature_extractor=perceptual_model, loss_type=perceptual_loss_type, multi_layers=multi_layer,multi_resolution= multi_resolution, use_ft16 = self.use_ft16)
            # 将perceptual loss的模型移动到当前设备
            self.criterionPerL.feature_extractor.to(self.device)
            self.loss_names.append("percep")

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.real_A = input['A']["data"].to(self.device)
        self.real_B = input['B']["data"].to(self.device)
        # self.class_mask_matrix = input['class_mask'].to(self.device)
        self.mask = input['Mask']["data"].to(self.device)
        self.dinov3_feature = input['feature'].to(self.device)
        self.image_paths = {'A_path':input['A']["params"].get("path"),
                            'B_path':input['B']["params"].get("path"),
                            'Mask_path':input['Mask']["params"].get("path")}
        
    def forward(self):

        self.fake_B = self.netG(self.real_A, self.dinov3_feature, 256, 256)  # G(A)
        self.diff = torch.abs(self.fake_B - self.real_B)

    def cal_loss_G(self):
        if self.use_perceptual_loss == True:
            self.loss_percep = self.criterionPerL(self.fake_B, self.real_B)
            self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
            self.loss_G = self.loss_percep + self.loss_L1
        else:
            self.loss_G = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_lambda = self.loss_G * self.config["network"]["lambda_L1"]
      
    
    

import torch
from .base_model import BaseModel
from code_network import define_network
from code_network.tools.loss import get_loss_by_name

class vanillaSLModel(BaseModel):
    
    def __init__(self, config):

        BaseModel.__init__(self, config)

        self.config = config
        self.model_names = ['G']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'diff']
        self.loss_names = ['G']
        self.metric_names = ['ssim']
        self.netG = define_network(config, net_type = "g")

        # # 将netG中所有参数设置为可训练
        # for param in self.netG.parameters():
        #     param.requires_grad = True
        
        if config["phase"] == "train":
            self.init_train_configs()
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

    def init_train_configs(self):
        config = self.config
        """Multiple Contrast loss (MCL)"""
        self.use_MCL = self.config.get("MCL",{}).get("use_MCL",False)
        if self.use_MCL == True:
            from code_network.tools.loss.mcl import MCLLoss
            self.class_mask = self.config["MCL"].get("class_mask")    
            self.criterionMCL = MCLLoss(class_mask_range = self.config["MCL"]["class_mask_range"], class_weight = self.config["MCL"]["class_weight"], class_norm = self.config["MCL"]["class_norm"])
            self.loss_names = self.loss_names + ['G_0','G_1','G_2']
            self.visual_names = self.visual_names + ['class_mask_matrix']
        """"""

        """Random Patch Loss (PRL)"""
        self.use_PRL = self.config.get("PRL",{}).get("use_PRL",False)
        if self.use_PRL == True:
            from code_network.tools.loss.image_similarity import RPLoss
            patch_loss_type = self.config["PRL"].get("loss","L1")
            patch_loss = get_loss_by_name(patch_loss_type)
            self.criterionPRL = RPLoss(patch_loss=patch_loss, patch_size=self.config["PRL"].get("patch_size",7), patch_num=self.config["PRL"].get("patch_num",10), norm=self.config["PRL"].get("norm",True))
        """"""

        """Powered L1 Loss (PL)"""
        self.use_PL = self.config.get("PL",{}).get("use_PL",False)
        if self.use_PL == True:
            from code_network.tools.loss.image_similarity import PoweredL1Loss
            # powerd_loss_type = config["PL"].get("loss","L1")
            # powerd_loss = get_loss_by_name(powerd_loss_type)
            self.criterionPL = PoweredL1Loss(power = self.config["PL"].get("power",1))
        """"""

        """Graph Smooth Loss"""
        self.use_graph_smooth = self.config["model"].get("use_graph_smooth",False)
        if self.use_graph_smooth == True:
            from code_network.tools.loss.image_similarity import GraphSmoothLoss
            self.criterionGraphSmooth = GraphSmoothLoss()
            self.loss_names.append("L1")
            self.loss_names.append("graph_smooth")
        """"""
        
        """Use Mask"""
        self.use_mask = config["model"].get("use_mask",False)
        if self.use_mask == True:
            from code_network.tools.loss.image_similarity import MaskedL1Loss
            self.criterionMask = MaskedL1Loss(reduction='mean')
        """"""

        """Gradient Clip"""
        self.use_grad_clip = config.get("grad_clip",{}).get("use_grad_clip",False)
        """"""

        """Contrast Balance"""
        self.use_contrast_balance = config["model"].get("use_contrast_balance",False)
        if self.use_contrast_balance == True:
            pass
        """"""

        """Localvar Balance"""
        self.use_localvarbalance = config["model"].get("use_localvar_balance",False)
        if self.use_localvarbalance == True:
            
            self.loss_names.append("balanced_L1")
            self.loss_names.append("L1")
            self.visual_names.append("balanced_real_B")
            self.visual_names.append("balanced_fake_B")
            self.localvar_balance_weight = config["model"]["localvar_balance_weight"]
            self.L1_weight = config["model"]["L1_weight"]
            self.beta = config["model"].get("beta",0.5)
        """"""

        """Histogram Equalization"""
        self.use_hiseq = config["model"].get("use_hiseq",False)
        if self.use_hiseq == True:
            
            self.loss_names.append("L1_eq")
            self.loss_names.append("L1")
            self.visual_names = self.visual_names + ['fake_B_eq', 'real_B_eq']
            self.hiseq_weight = config["model"].get("hiseq_weight",1.0)
        """"""

        """Frequency Loss"""
        self.use_freq = config["model"].get("use_freq",False)
        if self.use_freq == True:
            from code_network.tools.loss.frequency import EdgeLoss,Wavelet2DLoss
            self.loss_names.append("freq")
            self.loss_names.append("L1")
            self.criterionFreq = EdgeLoss()
            # self.criterionFreq = Wavelet2DLoss()
        """"""

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

            

        # use_MCL/use_PRL 不能同时为True
        assert not ([self.use_MCL, self.use_PRL].count(True) > 1)

    def forward(self):
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.diff = torch.abs(self.fake_B - self.real_B)

    def cal_loss_G(self):
        if self.use_MCL == True:
            if self.class_mask == "prepared":
                pass
            elif self.class_mask == "realtime_man":
                self.loss_G, [self.loss_G_0, self.loss_G_1, self.loss_G_2], self.class_mask_matrix = self.criterionMCL(self.fake_B, self.real_B)
        elif self.use_PRL == True:
            self.loss_G = self.criterionPRL(self.fake_B, self.real_B) + self.criterionL1(self.fake_B, self.real_B)
        elif self.use_PL == True:
            self.loss_G = self.criterionPL(self.fake_B, self.real_B)
        elif self.use_graph_smooth == True:
            if self.use_mask == True:
                self.loss_L1 = 0.9*self.criterionMask(self.fake_B, self.real_B, self.mask) +  0.1*self.criterionL1(self.fake_B, self.real_B)
                self.loss_graph_smooth = self.criterionGraphSmooth(self.fake_B, self.mask) 
            else:
                self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
                self.loss_graph_smooth = self.criterionGraphSmooth(self.fake_B, self.real_B)
            self.loss_G = self.loss_L1 + self.loss_graph_smooth
        elif self.use_contrast_balance == True:
            from code_util.data.transform_ import constrast_balance
            self.fake_B = constrast_balance(self.fake_B)
            self.real_B = constrast_balance(self.real_B)
        elif self.use_localvarbalance == True:
            from code_util.data.transform_ import localvar_balance
            self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
            self.balanced_fake_B = localvar_balance(self.fake_B,mask = self.mask, beta=self.beta)
            self.balanced_real_B = localvar_balance(self.real_B,mask = self.mask, beta=self.beta)
            self.loss_balanced_L1 = self.localvar_balance_weight * self.criterionL1(self.balanced_fake_B, self.balanced_real_B)
            self.loss_G = self.L1_weight*self.loss_L1 + self.loss_balanced_L1
        elif self.use_mask == True:
            self.loss_G = 0.9*self.criterionMask(self.fake_B, self.real_B, self.mask) + 0.1*self.criterionL1(self.fake_B, self.real_B)
        elif self.use_hiseq == True:
            from code_util.data.transform_ import histogram_equalization
            # print(torch.min(self.fake_B), torch.max(self.fake_B))
            # print(torch.min(self.real_B), torch.max(self.real_B))
            self.fake_B_eq = histogram_equalization(self.fake_B,self.mask)
            self.real_B_eq = histogram_equalization(self.real_B,self.mask)
            # print(torch.min(self.fake_B_eq), torch.max(self.fake_B_eq))
            # print(torch.min(self.real_B_eq), torch.max(self.real_B_eq))
            self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
            self.loss_L1_eq = self.criterionMask(self.fake_B_eq, self.real_B_eq, self.mask) * self.hiseq_weight
            # print("loss_L1:", self.loss_L1.item())
            # print("loss_L1_eq:", self.loss_L1_eq.item())
            self.loss_G = self.loss_L1_eq + self.loss_L1
        elif self.use_freq == True:
            self.loss_freq = self.criterionFreq(self.fake_B, self.real_B)
            self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
            self.loss_G = self.loss_freq + self.loss_L1
        elif self.use_perceptual_loss == True:
            self.loss_percep = self.criterionPerL(self.fake_B, self.real_B)
            self.loss_L1 = self.criterionL1(self.fake_B, self.real_B)
            self.loss_G = self.loss_percep + self.loss_L1
        else:
            self.loss_G = self.criterionL1(self.fake_B, self.real_B)
            
        self.loss_G_lambda = self.loss_G * self.config["network"]["lambda_L1"]
        # self.loss_G_lambda = self.loss_G 
     

    
            
            




    
    
    
    

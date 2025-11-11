import torch
from .base_model import BaseModel
from code_network import define_network
 
class UnetPlusPlusModel(BaseModel):
    
    def __init__(self, config):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, config)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        
        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = define_network(config,net_type="g")

        if config["phase"] == "train":
            # define loss functions
            lr = config["network"]["lr"]
            beta1 = config["network"]["beta1"]
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = {'A_path':input['A_path'],'B_path':input["B_path"]}

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Bs = self.netG(self.real_A)
        if not isinstance(self.fake_Bs, list):
            self.fake_Bs = [self.fake_Bs]
        self.fake_B = self.fake_Bs[-1]  # G(A)

    def cal_loss_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = 0
        self.loss_G_L1s = []
        for fake_B in self.fake_Bs:
            self.loss_G_L1s.append(self.criterionL1(fake_B, self.real_B) )
        self.loss_G_L1 = self.loss_G_L1s[-1]
        self.loss_G = sum(self.loss_G_L1s)

    # def backward_G(self):
      
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()  
        self.cal_loss_G()
        self.loss_G.backward()
        self.optimizer_G.step()             # update G's weights

    def calculate_loss(self):
        with torch.no_grad():
            self.forward()
            self.cal_loss_G()
    
    

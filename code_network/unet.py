import torch
import torch.nn as nn

from code_network.modules.general import ContinusParalleConv
from code_network.modules.unet import Up,Down,DoubleConv,OutConv
from code_network.modules.unet import Up3d,Down3d,DoubleConv3d,OutConv3d

""" Parts of the U-Net model """

class Unet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, output = True, activate = True, bilinear=False, **kwargs):
        super(Unet, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step
        
        self.inc = (DoubleConv(input_nc, ngf))
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1
        for i in range(down_step):
            self.downs.append(Down(ngf*2**i, ngf*2**(i+1)))
            self.ups.append(Up(ngf*2**(down_step-i), ngf*2**(down_step-i-1) // factor, bilinear))
        if output:
            self.outc = [OutConv(ngf, output_nc)]
            if activate:
                self.outc.append(nn.Tanh())
        else:
            self.outc = [nn.Identity()]
        self.outc = nn.Sequential(*self.outc)

    def forward(self, x):
        x = self.inc(x)
        x_skips = []
        for i in range(self.down_step):
            x_skips.append(x)
            x = self.downs[i](x)
        for i in range(self.down_step):
            x = self.ups[i](x,x_skips[self.down_step-i-1])
        result = self.outc(x)
        return result

class UnetPlusPlus(nn.Module):
    def __init__(self, input_nc, output_nc, deep_supervision=False, norm = "batch", **kwargs):
        super(UnetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]
        
        self.CONV3_1 = ContinusParalleConv(512*2, 512, pre_Norm = True, norm = norm)
 
        self.CONV2_2 = ContinusParalleConv(256*3, 256, pre_Norm = True, norm = norm)
        self.CONV2_1 = ContinusParalleConv(256*2, 256, pre_Norm = True, norm = norm)
 
        self.CONV1_1 = ContinusParalleConv(128*2, 128, pre_Norm = True, norm = norm)
        self.CONV1_2 = ContinusParalleConv(128*3, 128, pre_Norm = True, norm = norm)
        self.CONV1_3 = ContinusParalleConv(128*4, 128, pre_Norm = True, norm = norm)
 
        self.CONV0_1 = ContinusParalleConv(64*2, 64, pre_Norm = True, norm = norm)
        self.CONV0_2 = ContinusParalleConv(64*3, 64, pre_Norm = True, norm = norm)
        self.CONV0_3 = ContinusParalleConv(64*4, 64, pre_Norm = True, norm = norm)
        self.CONV0_4 = ContinusParalleConv(64*5, 64, pre_Norm = True, norm = norm)
 
 
        self.stage_0 = ContinusParalleConv(input_nc, 64, pre_Norm = False, norm = norm)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Norm = False, norm = norm)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Norm = False, norm = norm)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Norm = False, norm = norm)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Norm = False, norm = norm)
 
        self.pool = nn.MaxPool2d(2)
    
        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
 
        
        # 分割头
        self.final_super_0_1 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
        self.final_super_0_2 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
        self.final_super_0_3 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
        self.final_super_0_4 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_nc, 3, padding=1),
          nn.Tanh()
        )        
 
        
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))
        
        x_0_1 = torch.cat([self.upsample_0_1(x_1_0) , x_0_0], 1)
        x_0_1 =  self.CONV0_1(x_0_1)
        
        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)
        
        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)
        
        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)
 
        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)
        
        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)
        
        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)
 
        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)
        
        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)
        
        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            out_put4 = self.final_super_0_4(x_0_4)
            return out_put4
    

class WtUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, wt_level = 1, bilinear=False, **kwargs):
        from code_network.modules.wtconv import DoubleWtConv
        from code_network.modules.wtconv import WtDown,WtUp
        super(WtUnet, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step
        
        self.inc = (DoubleConv(input_nc, ngf))
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1
        for i in range(down_step):
            self.downs.append(WtDown(ngf*2**i, ngf*2**(i+1), wt_level))
            self.ups.append(WtUp(ngf*2**(down_step-i), ngf*2**(down_step-i-1) // factor, wt_level, bilinear))
        self.outc = nn.Sequential(OutConv(ngf, output_nc), nn.Tanh())

    def forward(self, x):
        x = self.inc(x)
        x_skips = []
        for i in range(self.down_step):
            x_skips.append(x)
            x = self.downs[i](x)
        for i in range(self.down_step):
            x = self.ups[i](x,x_skips[self.down_step-i-1])
        result = self.outc(x)
        return result


class UNet3d(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, down_step=4, bilinear=False):
        super(UNet3d, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step

        self.inc = DoubleConv3d(input_nc, ngf)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1

        for i in range(down_step):
            self.downs.append(Down3d(ngf * 2 ** i, ngf * 2 ** (i + 1)))
            self.ups.append(Up3d(ngf * 2 ** (down_step - i), ngf * 2 ** (down_step - i - 1) // factor, bilinear))

        self.outc = nn.Sequential(OutConv3d(ngf, output_nc), nn.Tanh())

    def forward(self, x):
        x = self.inc(x)
        x_skips = []
        for i in range(self.down_step):
            x_skips.append(x)
            x = self.downs[i](x)
        for i in range(self.down_step):
            x = self.ups[i](x, x_skips[self.down_step - i - 1])
        return self.outc(x)

class CascadeUnet(nn.Module):
    def __init__(self, input_nc, output_nc, cascade_step=2, ngf=64, down_step=4, activation = True, bilinear=False, **kwargs):
        super(CascadeUnet, self).__init__()
        self.cascade_step = cascade_step
        self.unets = nn.ModuleList()
        for i in range(cascade_step):
            self.unets.append(Unet(input_nc, output_nc, ngf, down_step, activate=False, bilinear=bilinear))
        if activation:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        for i in range(self.cascade_step):
            if i == 0:
                x = self.unets[i](x)
            else:
                x = self.unets[i](x) + x
        return self.activation(x)


class Unet25D(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, down_step=4, mode='center', D=5, bilinear=False):
        """
        mode: 'center' -> 预测中心切片；'all' -> 预测所有切片
        D: 输入切片数（如3表示用3张切片作为上下文）
        """
        super(Unet25D, self).__init__()
        self.mode = mode
        self.D = D
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear
        self.down_step = down_step

        if mode == 'center':
            input_channels = input_nc * D      # 融合通道信息
            output_channels = output_nc        # 输出1张切片
        elif mode == 'all':
            input_channels = input_nc * D       
            output_channels = output_nc * D
        else:
            raise ValueError("mode must be 'center' or 'all'")

        self.inc = DoubleConv(input_channels, ngf)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        factor = 2 if bilinear else 1
        for i in range(down_step):
            self.downs.append(Down(ngf * 2**i, ngf * 2**(i+1)))
            self.ups.append(Up(ngf * 2**(down_step - i), ngf * 2**(down_step - i - 1) // factor, bilinear))

        self.outc = nn.Sequential(OutConv(ngf, output_channels), nn.Tanh())

    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        if self.mode == 'center':
            x = x.view(B, C * D, H, W)  # → (B, C*D, H, W)
            x = self._unet_forward(x)  # → (B, C, H, W)
            return x.unsqueeze(2)      # → (B, C, 1, H, W)

        elif self.mode == 'all':
            x = x.view(B, C * D, H, W)  # → (B, C*D, H, W)
            x = self._unet_forward(x)  # → (B, C*D, H, W)
            return x.view(B, self.output_nc, D, H, W)  # → (B, C, D, H, W)
        else:
            raise ValueError("mode must be 'center' or 'all'")

    def _unet_forward(self, x):
        x = self.inc(x)
        x_skips = []
        for down in self.downs:
            x_skips.append(x)
            x = down(x)
        for i, up in enumerate(self.ups):
            x = up(x, x_skips[-(i+1)])
        return self.outc(x)

class TransReconsUnet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, down_step=4, bilinear=False, **kwargs):
        super(TransReconsUnet, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step
        
        self.trans_net = Unet(input_nc, output_nc, ngf, down_step, activate=True, bilinear=bilinear)
        self.recons_net = Unet(input_nc, output_nc, ngf, down_step, activate=False, bilinear=bilinear)
        self.trans_act = nn.Tanh()

    def forward(self, x):
        trans_x = self.trans_net(x)
        recons_x = self.recons_net(trans_x)
        return trans_x, recons_x
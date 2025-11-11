import torch
import torch.nn as nn

from code_network.modules.transformer import PatchEmbed, RebuildImage
from code_network.modules.unet import Up,Down,DoubleConv,OutConv
from code_network.modules.general import ResidualBlock
from code_network.mambaunet import VSSBlock
from code_network.modules.general import Identity
import torch.nn.functional as F

class MambaLayer(nn.Module):
    def __init__(self, input_channel, block_num, patch_size, residual = True, embed_dim = 96, d_state = 16):
        super(MambaLayer, self).__init__()
        depth = block_num
        embed = PatchEmbed(patch_size = patch_size, in_chans = input_channel, embed_dim = embed_dim, norm_layer=nn.LayerNorm)
        blocks = nn.ModuleList([
        VSSBlock(
            hidden_dim = embed_dim,
            norm_layer=nn.LayerNorm,
            d_state=d_state
        )
        for _ in range(depth)])
        rebuild = RebuildImage(patch_size = patch_size, in_chans = input_channel, embed_dim=embed_dim)
        self.model = nn.Sequential(
            embed,
            *blocks,
            rebuild
        )
        if residual == True:
            self.model = ResidualBlock(self.model)

    
    def forward(self,x):
        return self.model(x)

class MfUnet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, mamba_blocks = [32,16,8,4], patch_sizes = [8,4,2,1], bilinear=False, **kwargs):
        super(MfUnet, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step
        self.group = mamba_blocks[down_step-1]

        self.inc = (DoubleConv(input_nc, ngf))
        self.embeds = nn.ModuleList()
        self.rebuilds = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1
        for i in range(down_step):
            if mamba_blocks[i] != 0:
                self.skips.append(MambaMixLayer(ngf*2**i, mamba_blocks[i], group_num = self.group,  patch_size=patch_sizes[i]))
            else:
                self.skips.append(Identity())
            self.downs.append(Down(ngf*2**i, ngf*2**(i+1)))
            self.ups.append(Up(ngf*2**(down_step-i), ngf*2**(down_step-i-1) // factor, bilinear))
        self.outc = nn.Sequential(OutConv(ngf, output_nc), nn.Tanh())
        

    def forward(self, x):
        x = self.inc(x)
        x_skips = []
        for i in range(self.down_step):
            x_skips.append(self.skips[i][0](x))
            x = self.downs[i](x)
        for i in range(self.group):
            for j in range(self.down_step):
                x_skips[j] = self.skips[j][2*i+1](x_skips[j])
            x_skips_mix = torch.cat(x_skips, dim=-1).permute(0,3,1,2)
            for j in range(self.down_step):
                x_skips[j] = self.skips[j][2*i+2](x_skips_mix).permute(0,2,3,1)
        for i in range(self.down_step):
            x_skips[i] = self.skips[i][-1](x_skips[i])
        for i in range(self.down_step):
            x = self.ups[i](x,x_skips[self.down_step-i-1])
        result = self.outc(x)
        return result

class MambaMixLayer(nn.ModuleList):
    def __init__(self, input_channel, block_num, group_num, patch_size, embed_dim = 96, d_state = 16):
        super(MambaMixLayer, self).__init__()
        embed = PatchEmbed2D(patch_size = patch_size, in_chans = input_channel, embed_dim = embed_dim, norm_layer=nn.LayerNorm)
        blocks = nn.ModuleList()
        for _ in range(group_num):
            blocks.append(
                nn.Sequential(*[
                VSSBlock(
                    hidden_dim = embed_dim,
                    norm_layer=nn.LayerNorm,
                    d_state=d_state
                ) for __ in range(int(block_num/group_num))
                ])
            )
            blocks.append(nn.Conv2d(in_channels=embed_dim*2,out_channels=embed_dim,kernel_size=1,bias=False))
        rebuild = RebuildImage2D(patch_size = patch_size, in_chans = input_channel, embed_dim=embed_dim)
        self.model = nn.ModuleList(
            [embed,
            *blocks,
            rebuild]
        )
    
    def __getitem__(self,n):
        return self.model[n]

class Mr2Unet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf = 64, down_step = 4, mamba_blocks = [32,16,8,4], patch_sizes = [8,4,2,1], bilinear=False, **kwargs):
        super(Mr2Unet, self).__init__()
        self.n_channels = input_nc
        self.n_classes = output_nc
        self.bilinear = bilinear
        self.down_step = down_step
        
        self.inc = (DoubleConv(input_nc, ngf))
        self.skips = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        factor = 2 if bilinear else 1
        for i in range(down_step):
            if mamba_blocks[i] != 0:
                self.skips.append(MambaLayer(ngf*2**i, mamba_blocks[i], patch_size=patch_sizes[i]))
            else:
                self.skips.append(Identity())
            self.downs.append(MambaHybridDown(ngf*2**i, ngf*2**(i+1), mamba_blocks = 4, patch_size = patch_sizes[i]))
            self.ups.append(MambaHybridUp(ngf*2**(down_step-i), ngf*2**(down_step-i-1) // factor, mamba_blocks = 4, patch_size = patch_sizes[down_step-i-1], bilinear = bilinear))
        self.outc = nn.Sequential(OutConv(ngf, output_nc), nn.Tanh())

    def forward(self, x):
        x = self.inc(x)
        x_skips = []
        for i in range(self.down_step):
            x_skips.append(self.skips[i](x))
            x = self.downs[i](x)
        for i in range(self.down_step):
            x = self.ups[i](x,x_skips[self.down_step-i-1])
        result = self.outc(x)
        return result

class MambaHybridBlock(nn.Module):

    def __init__(self, in_channels, out_channels, mamba_blocks, patch_size, residual=True):
        super().__init__()
        self.conv = DoubleConv(in_channels, in_channels)
        self.vss = MambaLayer(in_channels, mamba_blocks, patch_size)
        self.mix = nn.Conv2d(in_channels=in_channels*2,out_channels=out_channels,kernel_size=1,bias=False)

        self.residual = residual

        
    def forward(self, x):
        y = torch.cat([self.conv(x),self.vss(x)], dim=1)

        y = self.mix(y)
        # if self.residual == True:
        #     return y+x
        # else:
        #     return y
        return y

class MambaHybridDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mamba_blocks, patch_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MambaHybridBlock(in_channels, out_channels, mamba_blocks, patch_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class MambaHybridUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mamba_blocks, patch_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = MambaHybridBlock(in_channels, out_channels, mamba_blocks, patch_size)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = MambaHybridBlock(in_channels, out_channels, mamba_blocks, patch_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
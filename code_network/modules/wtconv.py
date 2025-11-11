import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt
import pywt.data


class WTConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, stride=1, bias=True, wt_level=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        self.in_channels = in_channels
        self.wt_level = wt_level
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_level)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_level)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        # 初始化记录
        # print("Input shape:", x.shape)

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        # 小波分解阶段
        for i in range(self.wt_level):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            # print(f"Wavelet Decomposition Level {i+1} - Input shape:", curr_shape)

            # 边界填充
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
                # print(f"Padded shape at level {i+1}:", curr_x_ll.shape)

            # 执行小波变换
            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            # print(f"Low-frequency component shape at level {i+1}:", curr_x_ll.shape)

            # 转换形状，准备卷积
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            # print(f"Post-conv shape at level {i+1}:", curr_x_tag.shape)

            # 记录低频和高频分量
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # 小波逆变换阶段
        next_x_ll = 0

        for i in range(self.wt_level - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            
            # print(f"Inverse Wavelet Level {self.wt_level - i} - Low-frequency shape:", curr_x_ll.shape)
            # print(f"High-frequency shape at level {self.wt_level - i}:", curr_x_h.shape)

            # 低频加上上一层结果
            curr_x_ll = curr_x_ll + next_x_ll

            # 拼接并进行逆小波变换
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            # 去除边界填充
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
            # print(f"Shape after inverse transform at level {self.wt_level - i}:", next_x_ll.shape)

        x_tag = next_x_ll
        # print("Reconstructed shape after all levels:", x_tag.shape)

        # 基础卷积和缩放
        x = self.base_scale(self.base_conv(x))
        # print("Base conv output shape:", x.shape)

        # 和小波重建的特征相加
        x = x + x_tag
        # print("Final combined shape before stride:", x.shape)

        # 步长下采样（如果有）
        if self.do_stride is not None:
            x = self.do_stride(x)
            # print("Final output shape after stride:", x.shape)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class DoubleWtConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, wt_level = 1):
        super().__init__()
        self.double_conv = nn.Sequential(
            WTConv2d(in_channels, in_channels, wt_level),
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class WtDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, wt_level = 1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleWtConv(in_channels, out_channels,wt_level)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class WtUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, wt_level, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleWtConv(in_channels, out_channels, wt_level)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleWtConv(in_channels, out_channels, wt_level)

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

if __name__ == "__main__":

    # 定义输入参数
    in_channels = 3  # 输入通道数
    out_channels = 3  # 输出通道数（需与输入通道一致）
    kernel_size = 5
    stride = 1
    wt_level = 2  # 小波分解层数
    wt_type = 'db1'  # 小波类型

    # 初始化 WTConv2d 模型
    model = WTConv2d(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=kernel_size, stride=stride, 
                    wt_level=wt_level, wt_type=wt_type)

    # 打印模型结构
    # print("WTConv2d model structure:")
    print(model)

    # 生成随机输入张量（假设为批大小为 1，3 通道，64x64 大小）
    input_tensor = torch.randn(1, in_channels, 64, 64)

    # 前向传播
    output_tensor = model(input_tensor)

    # 打印输入和输出形状
    # print(f"Input shape: {input_tensor.shape}")
    # print(f"Output shape: {output_tensor.shape}")

    # 检查输出张量的数值范围
    # print(f"Output tensor min: {output_tensor.min().item()}, max: {output_tensor.max().item()}")


class WTConv2dLayer(nn.Module):
    def __init__(self, in_channels, block_num, kernel_size = 5, wt_level=1, wt_type='db1', residual=True):
        super(WTConv2dLayer, self).__init__()
        # print(wt_level)
        self.wtconvs = nn.Sequential(*[WTConv2d(in_channels, kernel_size, wt_level=wt_level, wt_type=wt_type) for _ in range(block_num)])
        # self.wtconvs = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1) for _ in range(block_num)])
        self.residual = residual

    def forward(self, x):
        if self.residual:
            return x + self.wtconvs(x)
            # return self.wtconvs(x)
        else:
            return self.wtconvs(x)

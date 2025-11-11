import torch
import torch.nn as nn
import torch.nn.functional as F

from code_network.dinov3.tools.dinov3_adapter import Dinov3Adapter

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor = "vgg", loss_type = "l1", multi_layers=[True,True,True,True], multi_resolution = False, **kwargs):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.multi_layers = multi_layers
        self.multi_resolution = multi_resolution
        if loss_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif loss_type == "cosine":
            # cosine similarity loss
            self.criterion = CosineSimilarityLoss()
        elif loss_type == "gram":
            self.criterion = GramLoss()
        elif loss_type == "dcsa":
            self.criterion = DCSALoss()
        elif loss_type == "high_freq":
            self.criterion = HighFreqLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        self.load_feature_extractor(**kwargs)
        
    def load_feature_extractor(self, **kwargs):
        if self.feature_extractor.startswith("dinov3"):
            self.feature_extractor = Dinov3Adapter(model_name=self.feature_extractor, **kwargs)
        elif self.feature_extractor == "vgg":
            from torchvision import models
            vgg_model = models.vgg19(pretrained=True).features.eval()
            for param in vgg_model.parameters():
                param.requires_grad = False
            self.feature_extractor = vgg_model
        else:
            raise ValueError(f"Unsupported feature extractor: {self.feature_extractor}")

    def forward(self, input, target):
        # 将input和target扩展为3通道
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        if self.multi_resolution: 
            # scales = [2, 1, 0.5]
            scales = [1, 0.5, 0.25]
            loss = 0
            for scale in scales:
                if scale != 1.0:
                    input_scaled = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
                    target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    input_scaled = input
                    target_scaled = target
                # multi_layers 是一个布尔列表，例如 [True, False, True, True]
                input_features = self.feature_extractor(input_scaled)
                target_features = self.feature_extractor(target_scaled)
                valid_losses = []
                for use, inp_feat, tgt_feat in zip(self.multi_layers, input_features, target_features):
                    if use:  # 只在 True 的层计算损失
                        valid_losses.append(self.criterion(inp_feat, tgt_feat))

                if len(valid_losses) > 0:
                    loss += sum(valid_losses) / len(valid_losses)
                else:
                    # 若全为 False，则默认不加任何感知项
                    loss += 0.0
        else:
            loss = 0
            # multi_layers 是一个布尔列表，例如 [True, False, True, True]
            input_features = self.feature_extractor(input)
            target_features = self.feature_extractor(target)
            valid_losses = []
            for use, inp_feat, tgt_feat in zip(self.multi_layers, input_features, target_features):
                if use:  # 只在 True 的层计算损失
                    valid_losses.append(self.criterion(inp_feat, tgt_feat))
            if len(valid_losses) > 0:
                loss += sum(valid_losses) / len(valid_losses)
            else:
                # 若全为 False，则默认不加任何感知项
                loss += 0.0
        return loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, input, target):
        if input.dim() == 3:
            B,N,C = input.shape
            # 在每个位置上计算余弦相似度 再取平均
            input = input.permute(0,2,1).contiguous().view(-1, C)  # (B*N, C)
            target = target.permute(0,2,1).contiguous().view(-1, C)  # (B*N, C)
        # Create a target tensor filled with 1s (indicating similarity)
        target_tensor = torch.ones(input.size(0)).to(input.device)
        loss = self.cosine_loss(input, target, target_tensor)
        return loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 工具：统一 reshape 成 (B,C,L) ----------
def _flatten(feat):
    """(B,N,C) or (B,C,H,W) -> (B,C,L) ,  L=H*W or N"""
    if feat.dim() == 3:                      # (B,N,C)
        # 交换通道维到中间
        B, N, C = feat.shape
        return feat.permute(0, 2, 1).contiguous()  #
    elif feat.dim() == 4:                    # (B,C,H,W)
        B, C, H, W = feat.shape
        return feat.view(B, C, H * W)

# =====================================================
# 1. 通道-空间双重归一化 L1  (DCSA)
# =====================================================
class DCSALoss(nn.Module):
    def __init__(self):
        super(DCSALoss, self).__init__()

    def forward(self, input, target):
        inp = _flatten(input)                  # (B,C,L)
        tgt = _flatten(target)
        # 通道维单位化
        inp = F.normalize(inp, dim=1, eps=1e-8)
        tgt = F.normalize(tgt, dim=1, eps=1e-8)
        # 由于单位化后损失的数值缩小，乘以通道数进行放大
        inp = inp * inp.shape[1]
        tgt = tgt * tgt.shape[1]
        return F.l1_loss(inp, tgt)


# =====================================================
# 2. Gram 矩阵 L1
# =====================================================
class GramLoss(nn.Module):
    def __init__(self):
        super(GramLoss, self).__init__()

    def forward(self, input, target):
        inp = _flatten(input)                  # (B,C,L)
        tgt = _flatten(target)
        B, C, L = inp.shape
        # Gram 矩阵 & 归一化
        G_in = torch.bmm(inp, inp.transpose(1, 2)) / (C * L)   # (B,C,C)
        G_tg = torch.bmm(tgt, tgt.transpose(1, 2)) / (C * L)
        return F.l1_loss(G_in, G_tg)


# =====================================================
# 3. 高通 FFT L1
# =====================================================
class HighFreqLoss(nn.Module):
    def __init__(self, radius=0.15):
        super(HighFreqLoss, self).__init__()
        self.radius = radius                   # 圆环半径比例

    def forward(self, input, target):
        # 先确保 4-D
        if input.dim() == 3:
            input = _flatten(input)
        B, C, L = input.shape
        H = W = int(L ** 0.5)
        input = input.view(B ,C, H, W).contiguous().view(B * C, H, W)
        target = target.view(B ,C, H, W).contiguous().view(B * C, H, W)
        # 2D FFT
        fft_in = torch.fft.fft2(input, dim=(-2, -1))
        fft_tg = torch.fft.fft2(target, dim=(-2, -1))
        fft_in = torch.fft.fftshift(fft_in, dim=(-2, -1))
        fft_tg = torch.fft.fftshift(fft_tg, dim=(-2, -1))
        # 圆形高通掩码
        y, x = torch.meshgrid(torch.arange(H, device=input.device),
                              torch.arange(W, device=input.device), indexing='ij')
        mask = ((x - W // 2) ** 2 + (y - H // 2) ** 2) > (min(H, W) * self.radius) ** 2
        loss = F.l1_loss(fft_in.real * mask, fft_tg.real * mask) + \
               F.l1_loss(fft_in.imag * mask, fft_tg.imag * mask)
        return loss
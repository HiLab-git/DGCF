import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================
# 基础模块
# ======================

class ConvBNGELU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c)
        )
    def forward(self, x):
        return x + self.block(x)


# ======================
# CrossFusion 模块
# ======================

class CrossFusion(nn.Module):
    def __init__(self, c, fusion_type="cnn", heads=4):
        """
        fusion_type:
            - "cnn"  : 原始卷积融合 (默认)
            - "se"   : 通道注意力融合 (SE-style)
            - "attn" : 空间交叉注意力融合 (Cross Attention)
        """
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.heads = heads

        # 通用投影
        self.proj_dino = ConvBNGELU(c, c)
        self.proj_img  = ConvBNGELU(c, c)

        if self.fusion_type == "cnn":
            self.merge = nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.GELU(),
                ResidualBlock(c)
            )
        elif self.fusion_type == "gated":  
            self.gate = nn.Sequential(
                nn.Conv2d(c * 2, c, 1),
                nn.BatchNorm2d(c),
                nn.Sigmoid()
            )
            self.out_proj = ConvBNGELU(c, c)
        elif self.fusion_type == "se":
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c * 2, c // 4, 1),
                nn.GELU(),
                nn.Conv2d(c // 4, c * 2, 1),
                nn.Sigmoid()
            )
        elif self.fusion_type == "attn":
            self.q_proj = nn.Conv2d(c, c, 1)
            self.k_proj = nn.Conv2d(c, c, 1)
            self.v_proj = nn.Conv2d(c, c, 1)
            self.out_proj = nn.Conv2d(c, c, 1)
            self.scale = (c // heads) ** -0.5
        elif self.fusion_type == "none":
            # 不进行融合，直接相加
            pass
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, img_feat, dino_feat):
        if self.fusion_type == "cnn":
            img_feat  = self.proj_img(img_feat)
            dino_feat = self.proj_dino(dino_feat)
            x = torch.cat([img_feat, dino_feat], dim=1)
            return self.merge(x)

        elif self.fusion_type == "se":
            img_feat  = self.proj_img(img_feat)
            dino_feat = self.proj_dino(dino_feat)
            cat = torch.cat([img_feat, dino_feat], dim=1)
            w = self.fc(cat)
            w_img, w_dino = w.chunk(2, dim=1)
            return img_feat * w_img + dino_feat * w_dino
        elif self.fusion_type == "gated":  
            img_feat  = self.proj_img(img_feat)
            dino_feat = self.proj_dino(dino_feat)
            gate = self.gate(torch.cat([img_feat, dino_feat], dim=1))
            out = gate * img_feat + (1 - gate) * dino_feat
            return self.out_proj(out)

        elif self.fusion_type == "attn":
            img_feat  = self.proj_img(img_feat)
            dino_feat = self.proj_dino(dino_feat)
        elif self.fusion_type == "none":
            # 直接相加
            out = dino_feat
            return out + img_feat  # 残差连接
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

            
# ======================
# Encoder / Decoder
# ======================

class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNGELU(in_ch, base_ch), ResidualBlock(base_ch))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBNGELU(base_ch, base_ch * 2), ResidualBlock(base_ch * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBNGELU(base_ch * 2, base_ch * 4), ResidualBlock(base_ch * 4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ConvBNGELU(base_ch * 4, base_ch * 8), ResidualBlock(base_ch * 8))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return [e1, e2, e3, e4]


class Decoder(nn.Module):
    def __init__(self, base_ch=64, out_ch=3):
        super().__init__()
        self.up1 = self._up(base_ch * 8, base_ch * 4)
        self.up2 = self._up(base_ch * 4, base_ch * 2)
        self.up3 = self._up(base_ch * 2, base_ch)
        self.out = nn.Sequential(
            ConvBNGELU(base_ch, base_ch // 2),
            nn.Conv2d(base_ch // 2, out_ch, 3, padding=1),
            nn.Tanh()
        )

    def _up(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNGELU(in_c, out_c),
            ResidualBlock(out_c)
        )

    def forward(self, e4, e3, e2, e1):
        d3 = self.up1(e4) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up3(d2) + e1
        return self.out(d1)


# ======================
# DINO-Guided 主体网络
# ======================

class DINOGuidedGenerator(nn.Module):
    def __init__(self,
                 in_ch=1,
                 embed_dim=768,
                 base_ch=64,
                 out_ch=1,
                 use_dino_layers=[True, True, True, True],
                 use_cnn=True,
                 fusion_type="cnn"):
        """
        Args:
            use_dino_layers (list[bool]): 长度为4，控制使用哪些 DINO 层
            use_cnn (bool): 是否启用 CNN 编码器
            fusion_type (str): "cnn" | "se" | "attn"
        """
        super().__init__()

        assert len(use_dino_layers) == 4
        self.use_dino_layers = use_dino_layers
        self.use_cnn = use_cnn

        # CNN encoder
        self.encoder = Encoder(in_ch, base_ch) if use_cnn else None

        # DINO 投影层
        self.proj_f3  = ConvBNGELU(embed_dim, base_ch)      if use_dino_layers[0] else nn.Identity()
        self.proj_f6  = ConvBNGELU(embed_dim, base_ch * 2)  if use_dino_layers[1] else nn.Identity()
        self.proj_f9  = ConvBNGELU(embed_dim, base_ch * 4)  if use_dino_layers[2] else nn.Identity()
        self.proj_f12 = ConvBNGELU(embed_dim, base_ch * 8)  if use_dino_layers[3] else nn.Identity()

        # 融合模块
        self.fuse1 = CrossFusion(base_ch, fusion_type)       if use_dino_layers[0] else nn.Identity()
        self.fuse2 = CrossFusion(base_ch * 2, fusion_type)   if use_dino_layers[1] else nn.Identity()
        self.fuse3 = CrossFusion(base_ch * 4, fusion_type)   if use_dino_layers[2] else nn.Identity()
        self.fuse4 = CrossFusion(base_ch * 8, fusion_type)   if use_dino_layers[3] else nn.Identity()

        # Decoder
        self.decoder = Decoder(base_ch, out_ch)

    def tokens_to_grid(self, tkns, H, W):
        """(B, N, C) -> (B, C, H/16, W/16)"""
        B, N, C = tkns.shape
        h0, w0 = H // 16, W // 16
        return tkns.permute(0, 2, 1).contiguous().view(B, C, h0, w0)

    def forward(self, img, dino_feats=None, H=None, W=None):
        # CNN 编码
        e1 = e2 = e3 = e4 = None
        if self.use_cnn:
            e1, e2, e3, e4 = self.encoder(img)

        # DINO 特征
        f_maps = [None, None, None, None]
        if dino_feats is not None:
            if self.use_dino_layers[0]:
                f3 = self.tokens_to_grid(dino_feats[:, 0, :], H, W)
                f_maps[0] = F.interpolate(self.proj_f3(f3), size=(H, W), mode='bilinear', align_corners=False)
            if self.use_dino_layers[1]:
                f6 = self.tokens_to_grid(dino_feats[:, 1, :], H, W)
                f_maps[1] = F.interpolate(self.proj_f6(f6), size=(H // 2, W // 2), mode='bilinear', align_corners=False)
            if self.use_dino_layers[2]:
                f9 = self.tokens_to_grid(dino_feats[:, 2, :], H, W)
                f_maps[2] = F.interpolate(self.proj_f9(f9), size=(H // 4, W // 4), mode='bilinear', align_corners=False)
            if self.use_dino_layers[3]:
                f12 = self.tokens_to_grid(dino_feats[:, 3, :], H, W)
                f_maps[3] = F.interpolate(self.proj_f12(f12), size=(H // 8, W // 8), mode='bilinear', align_corners=False)

        # 若不使用 CNN，则直接使用 DINO 特征进行解码
        if not self.use_cnn:
            # 确保所有 DINO 特征都已提供
            assert all(f is not None for f in f_maps), "All DINO features must be provided when CNN encoder is disabled."
            e1 = f_maps[0] 
            e2 = f_maps[1] 
            e3 = f_maps[2]
            e4 = f_maps[3] 
            return self.decoder(e4, e3, e2, e1)

        # 否则进行 CrossFusion 融合
        if self.use_dino_layers[0] and f_maps[0] is not None:
            e1 = self.fuse1(e1, f_maps[0])
        if self.use_dino_layers[1] and f_maps[1] is not None:
            e2 = self.fuse2(e2, f_maps[1])
        if self.use_dino_layers[2] and f_maps[2] is not None:
            e3 = self.fuse3(e3, f_maps[2])
        if self.use_dino_layers[3] and f_maps[3] is not None:
            e4 = self.fuse4(e4, f_maps[3])

        # Decoder
        return self.decoder(e4, e3, e2, e1)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ---- helpers ----
# class ConvBNGELU(nn.Module):
#     def __init__(self, in_c, out_c, k=3, padding=1):
#         super().__init__()
#         self.conv = nn.Conv2d(in_c, out_c, k, padding=padding)
#         self.bn = nn.BatchNorm2d(out_c)
#         self.act = nn.GELU()
#     def forward(self,x): return self.act(self.bn(self.conv(x)))

# class DWConv(nn.Module):
#     def __init__(self, c, k=3, padding=1):
#         super().__init__()
#         self.dw = nn.Conv2d(c, c, k, padding=padding, groups=c)
#         self.pw = nn.Conv2d(c, c, 1)
#         self.act = nn.GELU()
#         self.norm = nn.BatchNorm2d(c)
#     def forward(self,x): return self.act(self.norm(self.pw(self.dw(x))))

# # ---- Cross Attention Block ----
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim, heads=8, head_dim=16):
#         super().__init__()
#         self.num_heads = heads
#         self.scale = head_dim ** -0.5
#         mh_dim = heads * head_dim
#         self.to_q = nn.Conv2d(dim, mh_dim, 1, bias=False)
#         self.to_k = nn.Conv2d(dim, mh_dim, 1, bias=False)
#         self.to_v = nn.Conv2d(dim, mh_dim, 1, bias=False)
#         self.proj = nn.Conv2d(mh_dim, dim, 1)
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, q_feat, kv_feat):
#         # q_feat, kv_feat: (B, C, H, W) where C == dim
#         B, C, H, W = q_feat.shape
#         q = self.to_q(q_feat).reshape(B, -1, H*W)  # (B, mh_dim, HW)
#         k = self.to_k(kv_feat).reshape(B, -1, H*W)
#         v = self.to_v(kv_feat).reshape(B, -1, H*W)
#         # reshape into heads
#         def split_heads(x):
#             B, M, L = x.shape
#             x = x.view(B, self.num_heads, M//self.num_heads, L)  # (B, heads, head_dim, L)
#             return x
#         qh = split_heads(q)
#         kh = split_heads(k)
#         vh = split_heads(v)
#         # attention: (B, heads, head_dim, Lq) x (B, heads, head_dim, Lk) --> attn (B, heads, Lq, Lk)
#         # compute q^T k
#         qh_t = qh.permute(0,1,3,2)  # (B, heads, Lq, head_dim)
#         kh_t = kh  # (B, heads, head_dim, Lk)
#         attn = torch.matmul(qh_t, kh_t) * self.scale  # (B, heads, Lq, Lk)
#         attn = torch.softmax(attn, dim=-1)
#         out = torch.matmul(attn, vh.permute(0,1,3,2))  # (B, heads, Lq, head_dim)
#         out = out.permute(0,1,3,2).contiguous()  # (B, heads, head_dim, Lq)
#         out = out.view(B, -1, H*W)
#         out = out.view(B, -1, H, W)
#         out = self.proj(out)
#         # residual
#         return out + q_feat

# import torch
# import torch.nn as nn

# class CrossAttnToken(nn.Module):
#     """
#     Token-level cross-attention between decoder tokens (query) and DINO tokens (key/value).
#     Query: (B, Nq, C)
#     Key/Value: (B, Nk, C)
#     Output: (B, Nq, C)
#     """
#     def __init__(self, dim, num_heads=8, dropout=0.0):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5

#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.proj_out = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, q, kv):
#         # q: (B, Nq, C), kv: (B, Nk, C)
#         B, Nq, C = q.shape
#         Nk = kv.shape[1]

#         q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, Nq, head_dim)
#         k = self.k_proj(kv).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(kv).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

#         attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, Nq, Nk)
#         attn = attn.softmax(dim=-1)
#         attn = self.dropout(attn)

#         out = (attn @ v)  # (B, heads, Nq, head_dim)
#         out = out.transpose(1, 2).reshape(B, Nq, C)
#         out = self.proj_out(out)
#         return out

# # ---- UpBlock ----
# class UpBlock(nn.Module):
#     def __init__(self, dim, dim_out, use_attention=True):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#         self.conv1 = ConvBNGELU(dim, dim_out)
#         self.attn = CrossAttentionBlock(dim_out) if use_attention else nn.Identity()
#         self.attn_token = CrossAttnToken(dim_out) if use_attention else nn.Identity()
#         self.refine = nn.Sequential(DWConv(dim_out), DWConv(dim_out))
#     def forward(self, x, kv=None):
#         x = self.up(x)
#         x = self.conv1(x)
#         if kv is not None:
#             B, C, H, W = x.shape
#             q = x.flatten(2).transpose(1,2)   # (B, HW, C)
#             kv = kv.flatten(2).transpose(1,2) # (B, N, C), same N=(H/16)*(W/16)
#             x = self.attn_token(q, kv)  # 全局 cross attention
#             x = x.transpose(1,2).view(B,C,H,W)
#         x = self.refine(x)
#         return x
#         # inside UpBlock.forward


# # ---- Full Decoder ----
# class DINOv3MultiStageDecoder(nn.Module):
#     def __init__(self, in_c, D=128):
#         super().__init__()
#         self.use_attn = True
#         # in_c: DINO token channel C
#         self.proj12 = nn.Conv2d(in_c, D, 1)
#         self.proj9  = nn.Conv2d(in_c, D, 1)
#         self.proj6  = nn.Conv2d(in_c, D, 1)
#         self.proj3  = nn.Conv2d(in_c, D, 1)
#         # initial refinement
#         self.init_conv = ConvBNGELU(D, D)
#         # progressive up blocks
#         self.up1 = UpBlock(D, D, use_attention=True)  # H/8 fuse F9
#         self.up2 = UpBlock(D, D, use_attention=True)  # H/4 fuse F6
#         self.up3 = UpBlock(D, D, use_attention=True)  # H/2 fuse F3
#         self.up4 = UpBlock(D, D, use_attention=False) # H full, final refine
#         # output head
#         self.head = nn.Sequential(
#             ConvBNGELU(D, D//2),
#             nn.Conv2d(D//2, 1, 1)
#         )

#     def tokens_to_grid(self, tkns, H, W):
#         B, N, C = tkns.shape
#         h0, w0 = H//16, W//16
#         x = tkns.permute(0,2,1).contiguous().view(B, C, h0, w0)
#         return x

#     def forward(self, dino_feats, H, W):
#         # dino_feats: dict with keys 'f3','f6','f9','f12' each (B,N,C)
#         f3 = self.tokens_to_grid(dino_feats[:,0,:], H, W)
#         f6 = self.tokens_to_grid(dino_feats[:,1,:], H, W)
#         f9 = self.tokens_to_grid(dino_feats[:,2,:], H, W)
#         f12 = self.tokens_to_grid(dino_feats[:,3,:], H, W)
#         # project
#         p12 = self.proj12(f12)
#         p9  = self.proj9(f9)
#         p6  = self.proj6(f6)
#         p3  = self.proj3(f3)

#         x = self.init_conv(p12)  # start from most semanti
#         x = self.up1(x, kv=p9)   # H/8
#         x = self.up2(x, kv=p6)   # H/4
#         x = self.up3(x, kv=p3)   # H/2
#         x = self.up4(x, kv=None) # H
#         out = self.head(x)
#         out = torch.nn.Tanh()(out)  # [-1,1]
      
#         return out
    
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


class ResidualRefine(nn.Module):
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
# 显式空间对齐模块
# ======================

def align_and_reshape(feature_tokens, B, H, W, C):
    """
    将 DINO token 序列 reshape 为 (B, C, H/16, W/16)
    """
    feat = feature_tokens.permute(0, 2, 1).contiguous()  # (B, C, N)
    feat = feat.view(B, C, H // 16, W // 16)             # (B, C, H/16, W/16)
    return feat


# ======================
# 上采样块（单层级）
# ======================

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBNGELU(in_c, out_c)
        self.refine = ResidualRefine(out_c)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.refine(x)
        return x


# ======================
# 多层级 Decoder 主体
# ======================

class DINOv3MultiScaleDecoder(nn.Module):
    def __init__(self, embed_dim=768, out_ch=1):
        super().__init__()

        # 通道压缩层（对齐多层特征）
        self.reduce_p3 = ConvBNGELU(embed_dim, 256)
        self.reduce_p6 = ConvBNGELU(embed_dim, 256)
        self.reduce_p9 = ConvBNGELU(embed_dim, 256)
        self.reduce_p12 = ConvBNGELU(embed_dim, 256)

        # 融合层
        self.fuse = ConvBNGELU(256 * 4, 512)

        # Decoder：逐层上采样生成图像
        self.up1 = UpBlock(512, 256)   # H/8
        self.up2 = UpBlock(256, 128)   # H/4
        self.up3 = UpBlock(128, 64)    # H/2
        self.up4 = UpBlock(64, 32)     # H

        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(32, out_ch, 3, padding=1),
            nn.Tanh()  # 输出范围 [-1,1]
        )

    def forward(self, features, H, W):
        """
        features: list of [p3, p6, p9, p12], 每个为 (B, N, C)
        orig_hw: 原始图像 (H, W)
        """
        B, _, C = features[-1].shape

        # 显式空间对齐
        p3, p6, p9, p12 = [align_and_reshape(f, B, H, W, C) for f in features]

        # 通道压缩
        p3 = self.reduce_p3(p3)
        p6 = self.reduce_p6(p6)
        p9 = self.reduce_p9(p9)
        p12 = self.reduce_p12(p12)

        # 显式重采样特征到同一分辨率（以 H/16 为基准）
        target_size = p12.shape[-2:]
        p3 = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p6 = F.interpolate(p6, size=target_size, mode='bilinear', align_corners=False)
        p9 = F.interpolate(p9, size=target_size, mode='bilinear', align_corners=False)

        # 融合
        fused = torch.cat([p3, p6, p9, p12], dim=1)
        x = self.fuse(fused)

        # 多层级上采样生成图像
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        out = self.out(x)

        return out

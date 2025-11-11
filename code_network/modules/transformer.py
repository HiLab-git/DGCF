import torch.nn as nn
import torch

from einops import rearrange
from einops.layers.torch import Rearrange

"""
transformer 
"""

"vision transformer"

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, mode='2d'):
        super().__init__()
        self.mode = mode
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.mode == '1d':
            x = rearrange(x, 'b h w c -> b (h w) c')
        if self.norm is not None:
            x = self.norm(x) 
        return x

class RebuildImage(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, mode='2d',w = None, h = None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mode = mode
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)
        self.hw = (h,w) if h is not None and w is not None else (32, 32)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.mode == '1d':
            x = rearrange(x, 'b (h w) c -> b w h c', h=self.hw[0], w=self.hw[1])
        if self.norm is not None:
            x = self.norm(x) 
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
        x = self.proj(x)  # inverse projection
        return x

class PatchEmbedVanilla(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_dim = patch_size * patch_size * in_chans

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        return self.to_patch_embedding(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class TransformerLayer(nn.Module):
    def __init__(self, input_channel, block_num, image_size, patch_size, residual = True, embed_dim = 96):
        super().__init__()

        self.residual = residual
        self.embed = PatchEmbed(patch_size = patch_size, in_chans = input_channel, embed_dim = embed_dim, norm_layer= nn.LayerNorm, mode='1d')
        self.rebuild = RebuildImage(patch_size = patch_size, in_chans = input_channel, embed_dim=embed_dim, norm_layer= nn.LayerNorm, mode='1d')

        num_patches = int((image_size/patch_size)**2)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.block = Transformer(
            dim = embed_dim,
            depth = block_num,
            heads = 8,
            dim_head = 64,
            mlp_dim = embed_dim*4
            )

        
    def forward(self, img):
    
        x = self.embed(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.block(x)
        x = self.rebuild(x)
        if self.residual == True:
            return x + img
        else:
            return x


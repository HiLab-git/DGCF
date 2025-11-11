import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import uniform_filter

def normalize(img, range_before=None, range_after=None):
    if range_before is None:
        if isinstance(img, torch.Tensor):
            range_before = torch.min(img), torch.max(img)
        else:
            range_before = np.min(img), np.max(img)
    if range_after is None:
        range_after = (0, 1)
    
    img = (img - range_before[0]) / (range_before[1] - range_before[0])
    img = img * (range_after[1] - range_after[0]) + range_after[0]
    return img

def constrast_balance(img, alpha=1, use_torch=False):
    img = normalize(img, range_before=(-1, 1), range_after=(0, 1))
    contrast = local_contrast_map(img, use_torch=use_torch)
    img = img / (1 + alpha * contrast)
    img = normalize(img, range_before=(0, 1), range_after=(-1, 1))
    return img

def local_contrast_map(img, kernel_size=3, use_torch=False):
    if use_torch:
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).unsqueeze(0)
        pad = kernel_size // 2
        mean = F.avg_pool2d(img, kernel_size=kernel_size, stride=1, padding=pad)
        contrast = torch.abs(img - mean)
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min())
        return contrast.squeeze(0).cpu().numpy() if isinstance(img, torch.Tensor) else contrast
    else:
        mean = uniform_filter(img, size=kernel_size)
        contrast = np.abs(img - mean)
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min())
        return contrast
    
def local_contrast_map_torch(image, kernel_size=3):
    """纯 PyTorch 实现的局部对比度计算，支持 GPU Tensor"""
    padding = kernel_size // 2
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    if image.ndim == 3:
        image = image.unsqueeze(0) # [C, H, W] -> [1, C, H, W]
    
    mean = F.avg_pool2d(image, kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(image ** 2, kernel_size, stride=1, padding=padding)
    var = torch.clamp(mean_sq - mean ** 2, min=1e-8)
    std = torch.sqrt(var)

    return std.squeeze(0)  # 返回 [C, H, W] 或 [1, H, W]

def localvar_balance(image, alpha=1.0, beta=0.3, kernel_size=3, use_log=False, mask=None):
    is_torch = isinstance(image, torch.Tensor)
    if not is_torch:
        image = torch.tensor(image, dtype=torch.float32)

    dtype, device = image.dtype, image.device

    # Normalize the image to range [0, 1]
    norm_img = normalize(image, range_before=(-1, 1), range_after=(0, 1))

    # Compute local contrast
    contrast = local_contrast_map_torch(norm_img, kernel_size)
    contrast = contrast / (contrast.mean() + 1e-8)

    # Apply mask if provided
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
        contrast = contrast * mask

    # Apply transformation
    if use_log:
        transformed = torch.log1p(norm_img / (1 + alpha * contrast))* (1 - beta) + norm_img * beta
    else:
        transformed = norm_img / (1 + alpha * contrast) * (1 - beta) + norm_img * beta


    # Normalize back to range [-1, 1]
    result = normalize(transformed, range_before=(0, 1), range_after=(-1, 1))
    return result.to(dtype=dtype, device=device)

def histogram_equalization(img_tensor, mask=None, num_bins=512):
    """
    对范围为 [-1, 1] 的 batch 图像做直方图均衡化，支持更大灰度范围。
    
    参数：
        img_tensor: [B, C, H, W]，值域为 [-1, 1]
        num_bins: 灰度等级数量，默认  512

    返回：
        eq_img: [B, C, H, W]，值域仍为 [-1, 1]
    """
    B, C, H, W = img_tensor.shape

    # 将 [-1, 1] 映射到 [0, num_bins - 1]
    img_scaled = ((img_tensor + 1) * 0.5 * (num_bins - 1)).clamp(0, num_bins - 1).to(torch.long)

    eq_img_scaled = img_scaled.clone()

    if mask is not None:
        mask = mask.to(img_tensor.device)
        if mask.dtype != torch.bool:
            mask = mask > 0

    for b in range(B):
        for c in range(C):
            channel = img_scaled[b, c]  # [H, W]
            if mask is not None:
                mask_bc = mask[b, c] if mask.dim() == 4 else mask[b] if mask.dim() == 3 else mask
                masked_pixels = channel[mask_bc]
                if masked_pixels.numel() == 0:
                    continue
                hist = torch.bincount(masked_pixels.view(-1), minlength=num_bins).float()
                cdf = hist.cumsum(0)
                cdf_min = cdf[cdf > 0][0]
                cdf_normalized = (cdf - cdf_min) * (num_bins - 1) / (cdf[-1] - cdf_min)
                cdf_normalized = cdf_normalized.clamp(0, num_bins - 1).long()
                eq_channel = channel.clone()
                eq_channel[mask_bc] = cdf_normalized[masked_pixels]
                eq_img_scaled[b, c] = eq_channel
            else:
                hist = torch.bincount(channel.view(-1), minlength=num_bins).float()
                cdf = hist.cumsum(0)
                cdf_min = cdf[cdf > 0][0]
                cdf_normalized = (cdf - cdf_min) * (num_bins - 1) / (cdf[-1] - cdf_min)
                cdf_normalized = cdf_normalized.clamp(0, num_bins - 1).long()
                eq_img_scaled[b, c] = cdf_normalized[channel.view(-1)].view(channel.shape)

    # 映射回 [-1, 1]
    eq_img = eq_img_scaled.float() / (num_bins - 1) * 2 - 1.0
    eq_img = eq_img.clamp(-1.0, 1.0)
    return eq_img

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import uniform_filter
from code_util.data.transform.general import normalize


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

def localvar_balance(image, alpha=1.0, beta=0.3, kernel_size=3, use_log=False, range_inout = None, mask=None):
    if range_inout is None:
        range_inout = (-1, 1)
    is_torch = isinstance(image, torch.Tensor)
    if not is_torch:
        image = torch.tensor(image, dtype=torch.float32)

    dtype, device = image.dtype, image.device

    # Normalize the image to range [0, 1]
    norm_img = normalize(image, range_before=range_inout, range_after=(0, 1))

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
    result = normalize(transformed, range_before=(0, 1), range_after=range_inout)
    return result.to(dtype=dtype, device=device)


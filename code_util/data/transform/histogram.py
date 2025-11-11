import torch

from code_util.data.transform.general import normalize

def histogram_equalization(img_tensor, mask=None, num_bins=512, range_before = None):
    """
    对范围为 [0, 1] 的 batch 图像做直方图均衡化，支持更大灰度范围。
    
    参数：
        img_tensor: [B, C, H, W]，值域为 [0, 1]
        num_bins: 灰度等级数量，默认  512

    返回：
        eq_img: [B, C, H, W]，值域仍为 [0, 1]
    """
    if range_before is None:
        range_before = (-1,1)
    B, C, H, W = img_tensor.shape

    # 将 [0, 1] 映射到 [0, num_bins - 1]
    img_scaled = normalize(img_tensor, range_before=range_before, range_after=(0, num_bins - 1))
    img_scaled = img_scaled.clamp(0, num_bins - 1).long()  # 确保值在 [0, num_bins - 1] 范围内

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

    # 映射回 [0, 1]
    eq_img = normalize(eq_img_scaled, range_before=(0, num_bins - 1), range_after=range_before)
    eq_img = eq_img.clamp(range_before[0], range_before[1])  # 确保值域在原始范围内
    return eq_img
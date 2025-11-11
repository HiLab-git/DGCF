import numpy as np
from code_util.data.transform.general import normalize

def bilateral_weights(image: np.ndarray, sigma_s: float, sigma_r: float, window_size: int = 5) -> np.ndarray:
    """计算双边加权核：结合空间和像素强度相似性。"""
    pad = window_size // 2
    padded_img = np.pad(image, pad, mode='reflect')
    H, W = image.shape
    weights = np.zeros((H, W, window_size, window_size), dtype=np.float32)

    for i in range(window_size):
        for j in range(window_size):
            dy = i - pad
            dx = j - pad
            shifted = padded_img[pad + dy : pad + dy + H, pad + dx : pad + dx + W]
            spatial = np.exp(-(dx**2 + dy**2) / (2 * sigma_s**2))
            range_weight = np.exp(-((image - shifted) ** 2) / (2 * sigma_r**2))
            weights[:, :, i, j] = spatial * range_weight
    return weights

def local_std_with_bilateral_weight(image: np.ndarray, sigma_s: float, sigma_r: float, window_size: int = 5) -> np.ndarray:
    """计算结构保持的局部标准差。"""
    H, W = image.shape
    weights = bilateral_weights(image, sigma_s, sigma_r, window_size)
    pad = window_size // 2
    padded_img = np.pad(image, pad, mode='reflect')

    mean = np.zeros((H, W), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            shifted = padded_img[i : i + H, j : j + W]
            mean += weights[:, :, i, j] * shifted
    norm = np.sum(weights, axis=(2, 3)) + 1e-8
    mean /= norm

    variance = np.zeros((H, W), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            shifted = padded_img[i : i + H, j : j + W]
            variance += weights[:, :, i, j] * (shifted - mean) ** 2
    variance /= norm
    return np.sqrt(variance)

def structure_preserving_contrast_balance(image: np.ndarray, alpha: float = 15, sigma_s: float = 2.0, sigma_r: float = 1, window_size: int = 5) -> np.ndarray:
    """结构保持的强度均衡变换"""
    image = normalize(image, range_before=(-1024, 3000), range_after=(0, 1))  # 归一化到 [0, 1]
    if image.ndim == 2:
        local_std = local_std_with_bilateral_weight(image, sigma_s, sigma_r, window_size)
        transformed = np.log1p((image - np.min(image)) / (1 + alpha * local_std))
        return normalize(transformed, range_before=(0, 1), range_after=(-1024,3000))  # 恢复到 [-1, 1]
    elif image.ndim == 3:
        return np.stack([
            structure_preserving_contrast_balance(image[z], alpha, sigma_s, sigma_r, window_size)
            for z in range(image.shape[0])
        ])
    else:
        raise ValueError("只支持2D或3D图像输入")
    
# def structure_preserving_contrast_balance(
#     image: np.ndarray,
#     alpha: float = 1.5,
#     sigma_s: float = 2.0,
#     sigma_r: float = 0.1,  # 注意，输入归一化后，range变小，sigma_r也应变小
#     window_size: int = 5,
#     norm_range: str = "auto",  # "auto", "0-1", or "-1-1"
# ) -> np.ndarray:
#     """结构保持的强度均衡变换，支持输入范围自动适配"""
#     orig_min, orig_max = np.min(image), np.max(image)

#     # 自动归一化
#     if norm_range == "0-1" or (norm_range == "auto" and orig_min >= 0):
#         image_norm = (image - orig_min) / (orig_max - orig_min + 1e-8)
#         recover = lambda x: x * (orig_max - orig_min) + orig_min
#     elif norm_range == "-1-1" or norm_range == "auto":
#         image_norm = 2 * (image - orig_min) / (orig_max - orig_min + 1e-8) - 1
#         recover = lambda x: (x + 1) * 0.5 * (orig_max - orig_min) + orig_min
#     else:
#         raise ValueError("Unsupported norm_range")

#     def _process(img2d: np.ndarray) -> np.ndarray:
#         local_std = local_std_with_bilateral_weight(img2d, sigma_s, sigma_r, window_size)
#         out = np.log1p((img2d - np.min(img2d)) / (1 + alpha * local_std))
#         return out

#     if image.ndim == 2:
#         transformed = _process(image_norm)
#         return recover(transformed)
#     elif image.ndim == 3:
#         transformed_stack = np.stack([
#             _process(image_norm[z]) for z in range(image_norm.shape[0])
#         ])
#         return recover(transformed_stack)
#     else:
#         raise ValueError("只支持2D或3D图像输入")


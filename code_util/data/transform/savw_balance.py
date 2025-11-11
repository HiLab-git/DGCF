import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from scipy.ndimage import binary_dilation
from code_util.data.transform.general import normalize

def compute_local_variance(image, window_size=5):
    mean = uniform_filter(image, window_size)
    mean_sq = uniform_filter(image**2, window_size)
    return mean_sq - mean**2

def compute_edge_mask(image, threshold=20):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return (magnitude > threshold).astype(np.float32)

def compute_weight_map(image, window_size=5, epsilon=1e-3, edge_threshold=20, dilation_iter=2):
    """
    计算结构保留的局部方差加权图，只保留边缘交界内区域
    """
    image = normalize(image, range_before=(-1024, 3000), range_after=(0, 1))  # 归一化到 [0, 1]
    # Step 1: 局部方差 -> 基础反比例权重
    var_map = compute_local_variance(image, window_size)
    weight_map = 1.0 / (var_map + epsilon)

    # Step 2: Sobel 边缘检测
    edge_mask = compute_edge_mask(image, threshold=edge_threshold)

    # Step 3: 边缘膨胀 -> 得到内侧区域 mask
    structure = np.ones((3, 3), dtype=bool)  # 8邻域
    interior_mask = binary_dilation(edge_mask, structure=structure, iterations=dilation_iter)

    # Step 4: 屏蔽掉边缘外的部分，只保留“交界内侧”
    weight_map = weight_map * interior_mask

    return weight_map
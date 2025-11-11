import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop

def psnr(gt: np.ndarray, 
         pred: np.ndarray,
         mask: Optional[np.ndarray] = None,
         use_population_range: Optional[bool] = False,
         dynamic_range: Optional[tuple] = None) -> float:
    """
    Compute Peak Signal to Noise Ratio metric (PSNR)
    
    Parameters
    ----------
    gt : np.ndarray
        Ground truth
    pred : np.ndarray
        Prediction
    mask : np.ndarray, optional
        Mask for voxels to include. The default is None (including all voxels).
    use_population_range : bool, optional
        When a predefined population wide dynamic range should be used.
        gt and pred will also be clipped to these values.
    dynamic_range : tuple, optional
        Predefined dynamic range (min, max) for the data. Required if
        use_population_range is True.

    Returns
    -------
    psnr : float
        Peak signal to noise ratio.
    """
    if mask is None:
        mask = np.ones(gt.shape)
    else:
        # Binarize mask
        mask = np.where(mask > 0, 1., 0.)
        
    if use_population_range:
        if dynamic_range is None:
            raise ValueError("Dynamic range must be provided when use_population_range is True.")
        range_min, range_max = dynamic_range
        dynamic_range_value = range_max - range_min
        
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, range_min, range_max)
        pred = np.clip(pred, range_min, range_max)
    else:
        dynamic_range_value = gt.max() - gt.min()
        
    # Apply mask
    gt = gt[mask == 1]
    pred = pred[mask == 1]
    psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range_value)
    return float(psnr_value)
    
    
def ssim(gt: np.ndarray, 
         pred: np.ndarray,
         mask: Optional[np.ndarray] = None,
         dynamic_range: Optional[tuple] = None) -> float:
    """
    Compute Structural Similarity Index Metric (SSIM)

    Parameters
    ----------
    gt : np.ndarray
        Ground truth
    pred : np.ndarray
        Prediction
    mask : np.ndarray, optional
        Mask for voxels to include. The default is None (including all voxels).
    dynamic_range : tuple, optional
        Predefined dynamic range (min, max) for the data. Required for clipping.

    Returns
    -------
    ssim : float
        Structural similarity index metric.
    """
    # 将输入数据的无效维度去掉
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    if mask is not None:
        mask = np.squeeze(mask)
    if dynamic_range is None:
        raise ValueError("Dynamic range must be provided.")
    range_min, range_max = dynamic_range

    # Clip gt and pred to the dynamic range
    # print(gt, range_min, range_max)
    gt = np.clip(gt, range_min, range_max)
    pred = np.clip(pred, range_min, range_max)

    if mask is not None:
        # Binarize mask
        mask = np.where(mask > 0, 1., 0.)
        
        # Mask gt and pred
        gt = np.where(mask == 0, range_min, gt)
        pred = np.where(mask == 0, range_min, pred)
    
    # if mask is all-zero, return 1
    if np.sum(mask) == 0:
        return 1

    # Make values non-negative
    if range_min < 0:
        gt = gt - range_min
        pred = pred - range_min

    # Set dynamic range for ssim calculation and calculate ssim_map per pixel
    dynamic_range_value = range_max - range_min
    # print("shape:", gt.shape, pred.shape, mask.shape if mask is not None else None)
    ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range_value, full=True)

    if mask is not None:
        # Follow skimage implementation of calculating the mean value:  
        # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
        pad = 3
        cropped_ssim = crop(ssim_map, pad)
        cropped_mask = crop(mask, pad).astype(bool)
        if np.any(cropped_mask):
            ssim_value_masked = cropped_ssim[cropped_mask].mean(dtype=np.float64)
        else:
            ssim_value_masked = 1
        return ssim_value_masked
    else:
        return ssim_value_full
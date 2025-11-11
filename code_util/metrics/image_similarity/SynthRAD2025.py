import numpy as np
from skimage.util.arraycrop import crop
from scipy.ndimage import uniform_filter
from scipy.signal import fftconvolve
from skimage.filters import gaussian
from typing import Optional


# Compute the luminance, contrast and structure components of the SSIM between two images    
def structural_similarity_at_scale(im1,
    im2,
    *,
    luminance_weight=1,
    contrast_weight=1,
    structure_weight=1,
    win_size=None,
    gradient=False,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    full=False,
    **kwargs,):
        K1 = kwargs.pop('K1', 0.01)
        K2 = kwargs.pop('K2', 0.03)
        sigma = kwargs.pop('sigma', 1.5)
        if K1 < 0:
            raise ValueError("K1 must be positive")
        if K2 < 0:
            raise ValueError("K2 must be positive")
        if sigma < 0:
            raise ValueError("sigma must be positive")
        use_sample_covariance = kwargs.pop('use_sample_covariance', True)
        if gaussian_weights:
            # Set to give an 11-tap filter with the default sigma of 1.5 to match
            # Wang et. al. 2004.
            truncate = 3.5

        if win_size is None:
            if gaussian_weights:
                # set win_size used by crop to match the filter size
                r = int(truncate * sigma + 0.5)  # radius as in ndimage
                win_size = 2 * r + 1
            else:
                win_size = 7  # backwards compatibility
        if gaussian_weights:
            filter_func = gaussian
            filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
        else:
            filter_func = uniform_filter
            filter_args = {'size': win_size}

        ndim = im1.ndim
        NP = win_size**ndim

        # filter has already normalized by NP
        if use_sample_covariance:
            cov_norm = NP / (NP - 1)  # sample covariance
        else:
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004
        # compute (weighted) means
        ux = filter_func(im1, **filter_args)
        uy = filter_func(im2, **filter_args)


        # compute (weighted) variances and covariances
        uxx = filter_func(im1 * im1, **filter_args)
        uyy = filter_func(im2 * im2, **filter_args)
        uxy = filter_func(im1 * im2, **filter_args)
        vx = cov_norm * (uxx - ux * ux)
        vxsqrt = np.clip(vx, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
        vy = cov_norm * (uyy - uy * uy)
        vysqrt = np.clip(vy, a_min=0, a_max=None) ** 0.5 # TODO: this is very ugly
        vxy = cov_norm * (uxy - ux * uy)

        R = data_range
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2
        C3 = C2 / 2

        L = np.clip((2 * ux * uy + C1) / (ux * ux + uy * uy + C1), a_min=0, a_max=None) # TODO is this clipping necessary or do we increase K1 and K2?

        C = np.clip((2 * vxsqrt * vysqrt + C2) / (vx + vy + C2), a_min=0, a_max=None)
        S = np.clip((vxy + C3) / (vxsqrt * vysqrt + C3), a_min=0, a_max=None)

        result = (L ** luminance_weight) * (C ** contrast_weight) * (S ** structure_weight)
        # to avoid edge effects will ignore filter radius strip around edges
        pad = (win_size - 1) // 2

        # compute (weighted) mean of ssim. Use float64 for accuracy.
        mssim = crop(result, pad).mean(dtype=np.float64)

        if full:
            return mssim, result
        return mssim

def ms_ssim(gt: np.ndarray, pred: np.ndarray, mask: Optional[np.ndarray] = None, dynamic_range: Optional[tuple] = None, scale_weights: Optional[np.ndarray] = None) -> float:
        # Clip gt and pred to the dynamic range
        gt = np.clip(gt, min(dynamic_range), max(dynamic_range))
        pred = np.clip(pred, min(dynamic_range), max(dynamic_range))
        
        if mask is not None:
            #binarize mask
            mask = np.where(mask>0, 1., 0.)
            
            # Mask gt and pred
            gt = np.where(mask==0, min(dynamic_range), gt)
            pred = np.where(mask==0, min(dynamic_range), pred)
        else:
            # if no mask is provided, create an all-one mask
            mask = np.ones_like(gt)

        # Make values non-negative
        if min(dynamic_range) < 0:
            gt = gt - min(dynamic_range)
            pred = pred - min(dynamic_range)

        # Set dynamic range for ssim calculation and calculate ssim_map per pixel
        dynamic_range = dynamic_range[1] - dynamic_range[0]


        # see Eq. 7 in https://live.ece.utexas.edu/publications/2003/zw_asil2003_msssim.pdf
        # Also, the final sentence of section 3.2 (Results)
        scale_weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]) if scale_weights is None else scale_weights
        luminance_weights = np.array([0, 0, 0, 0, 0, 0.1333]) if scale_weights is None else scale_weights 
        levels = len(scale_weights)

        downsample_filter = np.ones((2, 2, 2)) / 8

        gtx, gty, gtz = gt.shape

        # Due to the downsampling in the MS-SSIM, the minimum matrix size must be 97 in every dimension
        target_size = 97

        pad_values = [
            (np.clip((target_size - dim)//2, a_min=0, a_max=None), 
            np.clip(target_size - dim - (target_size - dim)//2, a_min=0, a_max=None)) 
            for dim in [gtx, gty, gtz]]

        gt   = np.pad(gt,   pad_values, mode='edge')
        pred = np.pad(pred, pad_values, mode='edge')
        mask = np.pad(mask, pad_values, mode='edge')

        min_size = (downsample_filter.shape[-1] - 1) * 2 ** (levels - 1) + 1

        ms_ssim_vals, ms_ssim_maps = [], []
        for level in range(levels):
            ssim_value_full, ssim_map = structural_similarity_at_scale(gt, pred, 
                luminance_weight=luminance_weights[level], 
                contrast_weight=scale_weights[level],
                structure_weight=scale_weights[level],
                data_range=dynamic_range, full=True)
            pad = 3
            # at every level, we get the ssim_value_full, which is just mean SSIM at a level L, and the 
            # SSIM map. The masked SSIM is the mean SSIM within this mask
            ssim_value_masked  = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)

            ms_ssim_vals.append(ssim_value_full)
            ms_ssim_maps.append(ssim_value_masked)

            # The images are cleverly downsampled using an uniform filter
            # the mask is just downsampled by selecting every other line in every dimension
            filtered = [fftconvolve(im, downsample_filter, mode='same')
                for im in [gt, pred]]
            gt, pred, mask = [x[::2, ::2, ::2] for x in [*filtered, mask]]

        ms_ssim_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_vals])
        ms_ssim_mask_val = np.prod([np.clip(x, a_min=0, a_max=1) for x in ms_ssim_maps])

        return float(ms_ssim_mask_val)

import torch.nn.functional as F
import torch
from math import exp

def MSSIM_3D(ct: torch.tensor, sct: torch.tensor,  window_size, L = 4024, mask: torch.tensor = None, **kwargs):

    ct = ct + 1024
    sct = sct + 1024
    ct = torch.unsqueeze(ct, 0) 
    ct = torch.unsqueeze(ct, 0)
    sct = torch.unsqueeze(sct, 0)
    sct = torch.unsqueeze(sct, 0)

    window = get_window_3D(window_size,filter = "average").to(ct.device)
    mu1 = F.conv3d(ct , window, padding = window_size//2)
    mu2 = F.conv3d(sct , window, padding = window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(ct*ct, window, padding = window_size//2) - mu1_sq
    sigma2_sq = F.conv3d(sct*sct, window, padding = window_size//2) - mu2_sq
    sigma12 = F.conv3d(ct*sct, window, padding = window_size//2) - mu1_mu2

    K1 = 0.01
    K2 = 0.03
    C1 = K1*L**2
    C2 = K2*L**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    ssim_map = torch.squeeze(torch.squeeze(ssim_map))
    masked_ssim_map = ssim_map[mask>0.5]

    return masked_ssim_map.mean().item()
   
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def get_window_3D(window_size, filter="average"):
    if filter == "average":
        # Creating a 3D average filter
        window = torch.ones((1, 1, window_size, window_size, window_size)) / (window_size ** 3)
    elif filter == "gaussian":
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
        window = _3D_window
    else:
        raise ValueError("Unknown filter type")
    return window


def SSIM_3D(ct: torch.tensor, sct: torch.tensor, L: float = 4024.0, mask = None, **kwargs):
    """
    Calculate the SSIM of two 3D images.
    """
    K1 = 0.01
    K2 = 0.03
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3 and ct.ndim != 2:
        raise ValueError('The dimension of the images is not 3 or 2.')
    if mask is not None:
        ct = ct[mask > 0.5] + 1024
        sct = sct[mask > 0.5] + 1024
    
    # Compute means
    ct_mean = torch.mean(ct)
    sct_mean = torch.mean(sct)
    
    # Compute standard deviations
    ct_std = torch.std(ct)
    sct_std = torch.std(sct)
    
    # Compute covariance
    ct_sct_cov = torch.mean((ct - ct_mean) * (sct - sct_mean))
    
    # Constants
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    
    # SSIM calculation
    ssim_numerator = (2 * ct_mean * sct_mean + c1) * (2 * ct_sct_cov + c2)
    ssim_denominator = (ct_mean ** 2 + sct_mean ** 2 + c1) * (ct_std ** 2 + sct_std ** 2 + c2)
    ssim = ssim_numerator / ssim_denominator
    
    return ssim.item()  # Convert tensor to Python float

def PSNR_3D(ct: torch.Tensor, sct: torch.Tensor, L: float = 4024.0, mask: torch.Tensor = None, **kwargs) -> float:
    """
    Calculate the PSNR of two 3D images using PyTorch.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')

    ct = ct.float()
    sct = sct.float()
    
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    
    mse = torch.mean((ct - sct) ** 2)
    psnr = 10 * torch.log10(L ** 2 / mse)
    return psnr.item()

def MSE_3D(ct: torch.Tensor, sct: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> float:
    """
    Calculate the MSE of two 3D images using PyTorch.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')

    ct = ct.float()
    sct = sct.float()
    
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    
    mse = torch.mean((ct - sct) ** 2)
    return mse.item()

def MAE_3D(ct: torch.Tensor, sct: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> float:
    """
    Calculate the MAE of two 3D images using PyTorch.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')

    ct = ct.float()
    sct = sct.float()
    
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    
    mae = torch.mean(torch.abs(ct - sct))
    return mae.item()

def RMSE_3D(ct: torch.Tensor, sct: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> float:
    """
    Calculate the RMSE of two 3D images using PyTorch.
    """
    if ct.shape != sct.shape:
        raise ValueError('The shapes of the images are not the same.')
    if ct.ndim != 3:
        raise ValueError('The dimension of the images is not 3.')

    ct = ct.float()
    sct = sct.float()
    
    if mask is not None:
        ct = ct[mask > 0.5]
        sct = sct[mask > 0.5]
    # print(ct.shape)
    
    rmse = torch.sqrt(torch.mean((ct - sct) ** 2))
    return rmse.item()

if __name__ == "__main__":
    # Example usage
    window_size = 3
    average_window = get_window_3D(window_size, filter="average")
    gaussian_window = get_window_3D(window_size, filter="gaussian")
    print("Average Window:\n", average_window)
    print("Gaussian Window:\n", gaussian_window)
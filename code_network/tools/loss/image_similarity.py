import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
    
class RPLoss(nn.Module):
    """Random Patch Loss
    """

    def __init__(self, patch_loss = torch.nn.L1Loss(), patch_size = 64, patch_num = 100, norm = False):
        super(RPLoss, self).__init__()
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.criterionLoss = patch_loss
        self.norm = norm

    def forward(self, pre_img, real_img):
        
        for i in range(self.patch_num):
            pre_patch = self.get_patch(pre_img)
            real_patch = self.get_patch(real_img)
            if self.norm == True:
                lower_bound,upper_bound = real_img.min(), real_img.max()
                pre_patch = 2 * (pre_patch - lower_bound) / (upper_bound - lower_bound) - 1
                real_patch = 2 * (real_patch - lower_bound) / (upper_bound - lower_bound) - 1
            if i == 0:
                loss = self.criterionLoss(pre_patch,real_patch)
            else:
                loss += self.criterionLoss(pre_patch,real_patch)
        return loss / self.patch_num

    def get_patch(self, img):
        img_size = img.size()
        x = torch.randint(0, img_size[2] - self.patch_size, (1,))[0]
        y = torch.randint(0, img_size[3] - self.patch_size, (1,))[0]
        return img[:,:,x:x+self.patch_size,y:y+self.patch_size]
    

class MaskedL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: 'mean' or 'sum' or 'none'
        """
        super(MaskedL1Loss, self).__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction

    def forward(self, pred, target, mask):
        """
        Args:
            pred:   predicted tensor, shape (N, *)
            target: target tensor, same shape as pred
            mask:   binary mask tensor (same shape or broadcastable to pred)
        Returns:
            masked L1 loss value
        """
        diff = torch.abs(pred - target)
        masked_diff = diff * mask

        if self.reduction == 'sum':
            return masked_diff.sum()
        elif self.reduction == 'mean':
            return masked_diff.sum() / (mask.sum() + 1e-8)
        else:  # 'none'
            return masked_diff

class PoweredL1Loss(nn.Module):

    def __init__(self, power=1):
        super(PoweredL1Loss, self).__init__()
        self.power = power
        self.L1 = nn.L1Loss(reduction='none')

    def forward(self, pre_img, real_img):
        # 计算 |pre_img - real_img| 并加上一个小的常数 1e-6，防止梯度爆炸
        loss = self.L1(pre_img, real_img) + 1e-6
        # 计算 loss 的 power 次方
        powered_loss = loss ** self.power
        # 返回最终的损失值，求和或平均
        return powered_loss.mean()  # 这里可以改成 .sum() 以得到总和损失

class GraphSmoothLoss(nn.Module):
    """
    Graph Smooth Loss for image smoothing with mask support.
    """

    def __init__(self, sigma=1.0):
        """
        Args:
            sigma: Standard deviation for the Gaussian weight.
        """
        super(GraphSmoothLoss, self).__init__()
        self.sigma = sigma

    def forward(self, img, mask=None):
        """
        Args:
            img: (B, 1, H, W) Image tensor (e.g., HU or network output).
            mask: (B, 1, H, W) Binary mask tensor indicating the region of interest. If None, the entire image is used.
        Returns:
            Graph smooth loss within the masked region.
        """
        B, C, H, W = img.shape

        # If mask is None, use the entire image
        if mask is None:
            mask = torch.ones_like(img, dtype=torch.float32)

        # Define 4-neighborhood offsets
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        loss = 0.0
        count = 0

        for dx, dy in shifts:
            # Shift the image and mask
            shifted_img = torch.roll(img, shifts=(dx, dy), dims=(2, 3))
            shifted_mask = torch.roll(mask, shifts=(dx, dy), dims=(2, 3))

            # Compute spatial distance (all are 1 pixel)
            dist2 = dx**2 + dy**2
            w = np.exp(-dist2 / (2 * self.sigma ** 2))  # Weight is constant

            # Squared difference within the mask
            diff = (img - shifted_img) ** 2
            masked_diff = diff * mask * shifted_mask
            loss += w * masked_diff.mean()
            count += 1

        return loss / count

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class MCLLoss(nn.Module):
    """Multiple Contrast Loss
    """

    def __init__(self, class_mask_range = None, class_weight = None, class_norm = False):
        super(MCLLoss, self).__init__()
        self.class_mask_range = class_mask_range
        # 将class_mask_range从min,max归一化到-1到1
        min_value, max_value = class_mask_range[0] # 默认第一个class的range就是min,max
        for i, class_range in enumerate(class_mask_range):
            lower_bound,upper_bound = class_range
            # 从min_value~max_value归一化到-1到1
            class_mask_range[i] = [2 * (lower_bound - min_value) / (max_value - min_value) - 1, 2 * (upper_bound - min_value) / (max_value - min_value) - 1]
        self.class_weight = class_weight
        # 将class_weight归一化到0到1
        class_weight_sum = sum(class_weight)
        self.class_weight = [weight / class_weight_sum for weight in class_weight]

        self.class_norm = class_norm
        self.loss = MaskedL1Loss()
        self.L1_sum = nn.L1Loss(reduction='sum')
    
    def forward(self, pre_img, real_img, class_mask = None):
        if class_mask != None:
            # 使用临时传递的class_mask计算loss
            single_class_masks = []
            unique_values = sorted(torch.unique(class_mask))
            for val in unique_values:
                single_class_masks.append((class_mask == val).int())
        else:
            # class mask 用于可视化
            class_mask = torch.zeros_like(real_img, dtype=torch.int)
            for i, class_range in enumerate(self.class_mask_range):
                    lower_bound,upper_bound = class_range
                    class_mask[(real_img >= lower_bound) & (real_img <= upper_bound)] = i
            single_class_masks = [(real_img >= lower_bound) & (real_img <= upper_bound).int() for (lower_bound,upper_bound) in self.class_mask_range]
        single_class_losses = []
        for i, single_class_mask in enumerate(single_class_masks):
            if self.class_norm == True:
                lower_bound,upper_bound = self.class_mask_range[i]
                pre_img_temp = 2 * (pre_img - lower_bound) / (upper_bound - lower_bound) - 1
                real_img_temp = 2 * (real_img - lower_bound) / (upper_bound - lower_bound) - 1
            loss_temp = self.loss(pre_img, real_img, single_class_mask)
            # loss_temp = self.loss(pre_img_temp, real_img_temp, single_class_mask)
            single_class_losses.append(loss_temp*self.class_weight[i])
        loss = sum(single_class_losses)
        # 将class_mask归一化到-1到1
        class_mask = 2 * (class_mask.float() - 0) / (len(self.class_mask_range) - 1) - 1
        return loss, single_class_losses, class_mask
    
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
    
    

def get_loss_by_name(loss_name:str) -> nn.Module:
    if loss_name == "L1":
        return nn.L1Loss()
    elif loss_name == "L1_sum":
        return nn.L1Loss(reduction='sum')

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

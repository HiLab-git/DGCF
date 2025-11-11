
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from code_network.tools.loss.image_similarity import MaskedL1Loss

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
            loss_temp = self.loss(pre_img_temp, real_img_temp, single_class_mask)
            # loss_temp = self.loss(pre_img_temp, real_img_temp, single_class_mask)
            single_class_losses.append(loss_temp*self.class_weight[i])
        loss = sum(single_class_losses)
        # 将class_mask归一化到-1到1
        class_mask = 2 * (class_mask.float() - 0) / (len(self.class_mask_range) - 1) - 1
        return loss, single_class_losses, class_mask
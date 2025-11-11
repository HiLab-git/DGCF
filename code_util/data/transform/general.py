import torch
import numpy as np

def normalize(img, range_before=None, range_after=None, ref_image=None):
    # Handle range_before (same as before)
    if range_before is None:
        if isinstance(img, torch.Tensor):
            if img.dim() > 1:  # Batch processing
                img_flatten = img.view(img.size(0), -1)
                mins = img_flatten.min(dim=1)[0]
                maxs = img_flatten.max(dim=1)[0]
                shape = [img.size(0)] + [1] * (img.dim() - 1)
                mins = mins.view(shape)
                maxs = maxs.view(shape)
                range_before = (mins, maxs)
            else:
                range_before = (torch.min(img), torch.max(img))
        else:
            range_before = (img.min(), img.max())
    
    # Handle range_after with ref_image support
    if range_after is None:
        if ref_image is not None:
            # Compute range_after from ref_image (with batch support)
            if isinstance(ref_image, torch.Tensor):
                # 用一个没有梯度的复制
                ref_image = ref_image.detach()
                if ref_image.dim() > 1:  # Batch processing
                    ref_flatten = ref_image.view(ref_image.size(0), -1)
                    ref_mins = ref_flatten.min(dim=1)[0]
                    ref_maxs = ref_flatten.max(dim=1)[0]
                    shape = [ref_image.size(0)] + [1] * (ref_image.dim() - 1)
                    ref_mins = ref_mins.view(shape)
                    ref_maxs = ref_maxs.view(shape)
                    range_after = (ref_mins, ref_maxs)
                else:
                    range_after = (torch.min(ref_image), torch.max(ref_image))
            else:
                range_after = (ref_image.min(), ref_image.max())
        else:
            range_after = (0, 1)
    
    # Normalization with broadcasting support
    img = (img - range_before[0]) / (range_before[1] - range_before[0] + 1e-12)  # Add epsilon to avoid division by zero
    img = img * (range_after[1] - range_after[0]) + range_after[0]
    
    return img

def normalize_nobatch(img, range_before=None, range_after=None):
    if range_before is None:
        if isinstance(img, torch.Tensor):
            range_before = torch.min(img), torch.max(img)
        else:
            range_before = np.min(img), np.max(img)
    if range_after is None:
        range_after = (0, 1)
    
    img = (img - range_before[0]) / (range_before[1] - range_before[0])
    img = img * (range_after[1] - range_after[0]) + range_after[0]
    return img
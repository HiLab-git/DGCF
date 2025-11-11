import torch.nn as nn

def get_loss_by_name(loss_name:str) -> nn.Module:
    if loss_name == "L1":
        return nn.L1Loss()
    elif loss_name == "L1_sum":
        return nn.L1Loss(reduction='sum')

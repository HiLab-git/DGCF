
import torch
try: 
    import ptwt
except ImportError:
    ptwt = None  # Placeholder for wavelet transform library, e.g., PyWavelets or PyTorch Wavelets 
import torch.nn as nn
import torch.nn.functional as F

class Wavelet3DLoss(torch.nn.Module):
    def __init__(self, wavelet='haar', low_weight=1.0, high_weight=0.5):
        super().__init__()
        self.wavelet = wavelet
        self.low_weight  = low_weight
        self.high_weight = high_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true, y_pred: [B, 1, D, H, W]
        coeffs_true = ptwt.wavedec3(y_true.squeeze(1), self.wavelet, level=1)
        coeffs_pred = ptwt.wavedec3(y_pred.squeeze(1), self.wavelet, level=1)

        # coeffs 的结构: [low, dict_of_highs]
        low_true, highs_true = coeffs_true[0], coeffs_true[1]
        low_pred, highs_pred = coeffs_pred[0], coeffs_pred[1]

        low_loss = torch.mean((low_true - low_pred) ** 2)
        #print(low_loss)

        high_loss = 0.0
        for key in highs_true.keys():          # ('aad', 'ada', ... , 'ddd')
            high_loss += torch.mean(torch.abs(highs_true[key] - highs_pred[key]))

        loss = self.low_weight * low_loss + self.high_weight * high_loss
        return loss

class Wavelet2DLoss(torch.nn.Module):
    def __init__(self, wavelet='haar', low_weight=1.0, high_weight=0.5):
        super().__init__()
        self.wavelet = wavelet
        self.low_weight = low_weight
        self.high_weight = high_weight

    def forward(self,y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true, y_pred: [B, C, H, W] (2D inputs)
        coeffs_true = ptwt.wavedec2(y_true.squeeze(1), self.wavelet, level=1)
        coeffs_pred = ptwt.wavedec2(y_pred.squeeze(1), self.wavelet, level=1)

        # coeffs structure: [low, (high_h, high_v, high_d)]
        low_true, highs_true = coeffs_true[0], coeffs_true[1]
        low_pred, highs_pred = coeffs_pred[0], coeffs_pred[1]

        # low_loss = torch.mean((low_true - low_pred) ** 2)
        
        high_loss = 0.0
        for h_true, h_pred in zip(highs_true, highs_pred):
            high_loss += torch.mean(torch.abs(h_true - h_pred))
            
        # loss = self.low_weight * low_loss + self.high_weight * high_loss
        loss = high_loss
        return loss
    


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel filter for edge detection (fixed kernel, not learnable)
        sobel_x = torch.tensor([[[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]])  # Horizontal edge
        sobel_y = torch.tensor([[[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]])  # Vertical edge

        # Register as buffers to ensure moved to same device with model
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0))  # shape: [1,1,3,3]
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0))  # shape: [1,1,3,3]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred and y_true: [B, C, H, W], typically C = 1 (grayscale)
        if y_pred.shape[1] > 1:
            y_pred = y_pred.mean(dim=1, keepdim=True)  # Convert to 1 channel
            y_true = y_true.mean(dim=1, keepdim=True)
        self.sobel_x = self.sobel_x.to(y_pred.device)
        self.sobel_y = self.sobel_y.to(y_pred.device)
        # Compute edge maps
        edge_true_x = F.conv2d(y_true, self.sobel_x, padding=1)
        edge_true_y = F.conv2d(y_true, self.sobel_y, padding=1)
        edge_pred_x = F.conv2d(y_pred, self.sobel_x, padding=1)
        edge_pred_y = F.conv2d(y_pred, self.sobel_y, padding=1)

        edge_true = torch.sqrt(edge_true_x ** 2 + edge_true_y ** 2 + 1e-6)
        edge_pred = torch.sqrt(edge_pred_x ** 2 + edge_pred_y ** 2 + 1e-6)

        # Edge loss (L1 or L2)
        loss = F.l1_loss(edge_pred, edge_true)
        return loss

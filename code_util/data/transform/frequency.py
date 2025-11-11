class Wavelet3DLoss(_Loss):
    def __init__(self, wavelet='haar', low_weight=1.0, high_weight=0.5):
        super().__init__()
        self.wavelet = wavelet
        self.low_weight  = low_weight
        self.high_weight = high_weight

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
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
        return self.low_weight * low_loss + self.high_weight * high_loss
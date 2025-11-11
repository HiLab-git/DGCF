import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你已经定义好了 DINOv3_Adapter 类，并且导入了需要的依赖
from code_network.dinov3.tools.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vit7b16

DINOv3_MODEL_FACTORIES = {
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vit7b16": dinov3_vit7b16,
}

DINOv3_INTERACTION_INDEXES = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}

DINOv3_MODEL_INFO = {
    "dinov3_vits16": {"embed_dim": 384, "depth": 12, "num_heads": 6, "params": "~22M"},
    "dinov3_vitb16": {"embed_dim": 768, "depth": 12, "num_heads": 12, "params": "~86M"},
    "dinov3_vitl16": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "params": "~300M"},
    "dinov3_vit7b16": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "params": "~7B"},
}

DINOv3_MODEL_PATH = {
    "dinov3_vits16": "./file_dataset/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "./file_dataset/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "./file_dataset/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vit7b16": "./file_dataset/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
}

def load_dinov3_model(model_name: str, pretrained_path: str = None, freeze = True, load_vanilla_dino = True, use_ft16 = False):
    """Load DINOv3 model with pretrained weights"""
    if pretrained_path and os.path.exists(pretrained_path):
        pass
    else:
        pretrained_path = DINOv3_MODEL_PATH.get(model_name, None)
        if pretrained_path is None or not os.path.exists(pretrained_path):
            raise ValueError(f"Pretrained path {pretrained_path} does not exist.")
    print(f"Loading custom pretrained weights from {pretrained_path}")
    if load_vanilla_dino == True:
        model = torch.hub.load("../dino/dinov3-main", model_name, source='local', weights=pretrained_path)
    else:
        if model_name not in DINOv3_MODEL_FACTORIES:
            supported_models = list(DINOv3_MODEL_FACTORIES.keys())
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {supported_models}")
        model_factory = DINOv3_MODEL_FACTORIES[model_name]
        model = model_factory(pretrained=False)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        state_dict = state_dict["teacher"] if "teacher" in state_dict else state_dict
        state_dict = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if 'ibot' not in k and 'dino_head' not in k
        }
        model.load_state_dict(state_dict, strict=True)
    print("Successfully loaded custom pretrained weights")
    # cast the model to half 
    if use_ft16:
        model.to(torch.float16)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model

class Dinov3Adapter(nn.Module):
    def __init__(self, model_name: str, pretrained_path: str = None ,layer_index : list = None, freeze = True, load_vanilla_dino = True, use_ft16 = False):
        super(Dinov3Adapter, self).__init__()
        self.backbone = load_dinov3_model(model_name, pretrained_path, freeze, load_vanilla_dino, use_ft16)
        self.layer_index = layer_index
        if self.layer_index is None:
            self.layer_index = DINOv3_INTERACTION_INDEXES.get(model_name, [2,5,8,11])
    
    def forward(self, x):
        features = self.backbone.get_intermediate_layers(
            x, n = self.layer_index
        )
        return features # 这是一个tuple，当存在B维度时 B维度默认在tuple维度的第一维 而不是第0维 第0维是layer维度
import torch
import torch

def load_partial_state_dict(model, state_dict_path, device='cpu', include=None, exclude=None):
    """
    从 checkpoint 加载部分参数到模型中。
    
    参数:
        model (nn.Module): 当前模型。
        state_dict_path (str): checkpoint 路径。
        device (str): 加载设备。
        include (str or None): 只加载包含该字符串的参数（大小写不敏感）。
        exclude (str or None): 不加载包含该字符串的参数（大小写不敏感），优先级高于 include。
    """
    print(f"📂 Loading checkpoint from: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location=device)

    model_dict = model.state_dict()

    loaded_params = []
    missing_in_checkpoint = []
    missing_in_model = []

    def match_key(key):
        key_lower = key.lower()
        if exclude and exclude.lower() in key_lower:
            return False
        if include and include.lower() not in key_lower:
            return False
        return True

    filtered_dict = {}
    for k, v in state_dict.items():
        if not match_key(k):
            continue
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
                loaded_params.append(k)
            else:
                print(f"[Shape mismatch] {k}: checkpoint {v.shape} != model {model_dict[k].shape}")
                missing_in_checkpoint.append(k)
        else:
            missing_in_model.append(k)

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print("\n✅ Loaded parameters:")
    for k in loaded_params:
        print(f"  - {k}")

    print("\n⚠️ Not loaded (due to shape mismatch or not in checkpoint):")
    for k in model_dict.keys():
        if k not in filtered_dict:
            print(f"  - {k}")

    print("\n❗ Extra parameters in checkpoint not found in model:")
    for k in missing_in_model:
        print(f"  - {k}")


def set_trainable_params(model, trainable_keys=None, freeze_keys=None):
    """
    根据关键词设置模型参数是否可训练。
    
    参数：
        model (nn.Module): 目标模型。
        trainable_keys (str or List[str] or None): 参数名中包含这些关键词的设置为可训练（requires_grad=True）
        freeze_keys (str or List[str] or None): 参数名中包含这些关键词的设置为冻结（requires_grad=False），优先级高于 trainable_keys。
    """
    def to_lower_list(x):
        if x is None:
            return []
        if isinstance(x, str):
            return [x.lower()]
        return [s.lower() for s in x]

    trainable_keys = to_lower_list(trainable_keys)
    freeze_keys = to_lower_list(freeze_keys)

    for name, param in model.named_parameters():
        name_lower = name.lower()

        if any(key in name_lower for key in freeze_keys):
            param.requires_grad = False
            print(f"🔒 Freezing: {name}")
        elif any(key in name_lower for key in trainable_keys):
            param.requires_grad = True
            print(f"✅ Trainable: {name}")
        elif trainable_keys:
            # 如果指定了 trainable_keys 但当前参数不匹配任何一个，则默认冻结
            param.requires_grad = False
            print(f"❌ Not trainable (filtered out): {name}")
        else:
            # 如果都没指定，保持原状
            pass


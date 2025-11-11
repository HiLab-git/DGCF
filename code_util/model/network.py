import torch

def get_clip_grad_value(parameters, mode='median'):
    # 提取所有梯度值，并拼接到一个张量中
    all_grads = []
    for param in parameters:
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))  # 将梯度展平成一维
    
    if len(all_grads) == 0:
        return  # 如果没有梯度则返回
    
    all_grads = torch.cat(all_grads)  # 拼接成一个大向量
    
    # 根据指定的模式计算裁剪值
    if mode == 'median':
        clip_value = all_grads.median()
    elif mode == 'mean':
        clip_value = all_grads.mean()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return clip_value

def clip_grad(model,mode='median',layer = "all"):
    
    if layer == "all":
        all_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                all_grads.append(param.grad.view(-1))
        if len(all_grads) == 0:
            return
        all_grads = torch.cat(all_grads)
        if mode == 'median':
            clip_value = all_grads.median()
        elif mode == 'mean':
            clip_value = all_grads.mean()
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        # 在整个模型上应用梯度裁剪
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.data.clamp_(max=clip_value)
    elif layer == "each":
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 展平成一维张量并计算中位数
                grad_flat = param.grad.view(-1)
                if mode == 'median':
                    clip_value = grad_flat.median()
                elif mode == 'mean':
                    clip_value = grad_flat.mean()
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                # 将该层的梯度裁剪到中位数 
                param.grad.data.clamp_(max=clip_value)
    elif layer == "final":
        # 对最后输出层做梯度截断
        for name, param in model.named_parameters():
            if param.grad is not None:
                if "outc" in name :
                    grad_flat = param.grad.view(-1)
                    if mode == 'median':
                        clip_value = grad_flat.median()
                    elif mode == 'mean':
                        clip_value = grad_flat.mean()
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")
                    param.grad.data.clamp_(max=clip_value)
    elif layer == "test":
        # 输出每一层以及对应的梯度的中位数
        for name, param in model.named_parameters():
            print(f"{name}: {param.grad.view(-1).median()}")
    else:
        pass 
        
            

import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def percentile_normalize(img, lower=5, upper=95, eps=1e-8):
    """
    对 B×C×H×W 图像做分位数归一化，将 [5%, 95%] 范围归一化到 [0, 1]
    """
    B, C, H, W = img.shape
    img_flat = img.view(B, C, -1)  # (B, C, H*W)
    
    # 计算分位数
    lower_val = torch.quantile(img_flat, lower / 100.0, dim=-1, keepdim=True)  # (B, C, 1)
    upper_val = torch.quantile(img_flat, upper / 100.0, dim=-1, keepdim=True)  # (B, C, 1)
    
    img_flat = (img_flat - lower_val) / (upper_val - lower_val + eps)
    img_flat = img_flat.clamp(0, 1)
    return img_flat.view(B, C, H, W)

def mean_std_normalize(img, eps=1e-8):
    """
    对 B×C×H×W 图像做每通道 mean-std 标准化
    """
    B, C, H, W = img.shape
    img_flat = img.view(B, C, -1)  # (B, C, H*W)

    mean = img_flat.mean(dim=-1, keepdim=True)  # (B, C, 1)
    std = img_flat.std(dim=-1, keepdim=True)    # (B, C, 1)

    img_flat = (img_flat - mean) / (std + eps)
    return img_flat.view(B, C, H, W)


min_normed, max_normed = 0, 1

def hu_to_normed(hu,min_hu = None,max_hu=None):
    if min_hu is None:
        min_hu = np.min(hu)
    if max_hu is None:
        max_hu = np.max(hu)
    hu = np.asarray(hu)
    return (hu - min_hu) / (max_hu - min_hu) * (max_normed - min_normed) + min_normed

def piecewise_linear_interp(t, ctrl_pts_x, ctrl_pts_y):
    """
    分段线性插值（支持 batch）
    t: (B, N) - 需要插值的位置
    ctrl_pts_x: (B, M) - 控制点 x 坐标（升序排列）
    ctrl_pts_y: (B, M) - 控制点 y 坐标
    return: (B, N) - 插值结果
    """
    B, N = t.shape
    M = ctrl_pts_x.shape[1]

    # 找到每个 t 属于哪个区间 [x_i, x_{i+1}]
    idx = torch.sum(t.unsqueeze(-1) >= ctrl_pts_x.unsqueeze(1), dim=-1) - 1  # (B, N)
    idx = idx.clamp(0, M - 2)  # 保证索引不越界

    x0 = torch.gather(ctrl_pts_x, 1, idx)        # (B, N)
    x1 = torch.gather(ctrl_pts_x, 1, idx + 1)    # (B, N)
    y0 = torch.gather(ctrl_pts_y, 1, idx)        # (B, N)
    y1 = torch.gather(ctrl_pts_y, 1, idx + 1)    # (B, N)

    # 插值计算：线性插值公式
    ratio = (t - x0) / (x1 - x0 + 1e-8)
    y = y0 + ratio * (y1 - y0)
    return y  # (B, N)

def piecewise_linear_interp_img(x, ctrl_pts_x, ctrl_pts_y):
    """
    分段线性插值图像版本
    x: (B, C, H, W) - 输入图像
    ctrl_pts_x: (B, M) - 控制点 x 坐标（升序排列）
    ctrl_pts_y: (B, M) - 控制点 y 坐标
    return: (B, C, H, W) - 插值结果
    """
    B, C, H, W = x.shape
    x = x.view(B, -1)  # 展平为 (B, N) 形式
    
    y = piecewise_linear_interp(x, ctrl_pts_x, ctrl_pts_y)  # (B, W)
    
    return y.view(B, C, H, W)  # 恢复为图像形状

if __name__ == "__main__":

    import SimpleITK as sitk
    from code_util.data.transform.general import normalize

    ct_path = 'file_dataset/SynthRAD2025/Task1/HN/3D/validationB/1HNA060.mha'
    mri_path = 'file_dataset/SynthRAD2025/Task1/HN/3D/validationA/1HNA060.mha'
    ct = sitk.ReadImage(ct_path)
    mri = sitk.ReadImage(mri_path)

    # 转为numpy数组
    ct_np = sitk.GetArrayFromImage(ct)  # shape: [z, y, x]
    mri_np = sitk.GetArrayFromImage(mri)

    # 取中间切片
    mid_slice = ct_np.shape[0] // 2
    ct_slice = ct_np[mid_slice]
    mri_slice = mri_np[mid_slice]

    # 转为torch tensor
    ct_tensor = torch.from_numpy(ct_slice).float().unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
    mri_tensor = torch.from_numpy(mri_slice).float().unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    mri_tensor = percentile_normalize(mri_tensor, lower=2, upper=98)  # 分位数归一化
    mri_tensor = mean_std_normalize(mri_tensor)  # 均值方差标准化

    min_ct,max_ct = -1024,2000
    # min_mri, max_mri = 0,2000
    min_normed, max_normed = 0, 1

    # ct = [-1024,-100,0,10,30,700,1000,2000]
    # mri = [0,1300,250,300,1100,700,150,0]
    ct = [-1024,-100,75,250,400,600,885,1500,2000]
    mri = [-0.34,4.12,2,4.12,3.3,2.28,0.4,0.01,-0.34]
    ct_normed = hu_to_normed(ct, min_ct, max_ct)
    mri_normed = hu_to_normed(mri)

    ctrl_pts_x = torch.tensor([ct_normed])
    ctrl_pts_y = torch.tensor([mri_normed])

    ct_tensor = normalize(ct_tensor, range_before=[-1024, 2000], range_after=[0, 1])  # 归一化处理

    sMRI_tensor = piecewise_linear_interp_img(ct_tensor, ctrl_pts_x, ctrl_pts_y)  # 使用Lagrange插值

    # 转回numpy
    sMRI_slice = sMRI_tensor.squeeze().cpu().numpy()
    mri_slice = mri_tensor.squeeze().cpu().numpy()

    # sMRI_slice = normalize(sMRI_slice, range_before=[0, 1], range_after=[min(mri),max(mri)])  # 归一化处理

    # # 将ct_slice截断在[-250,250]的范围内
    # ct_slice = np.clip(ct_slice, -50, 50)
    # # 将mri_slice截断在[-250,250]的范围内
    # sMRI_slice = np.clip(sMRI_slice, -150, 150)
    
    # 展示三者
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title('CT')
    plt.imshow(ct_slice, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('MRI')
    plt.imshow(mri_slice, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('sMRI (Transformed CT)')
    plt.imshow(sMRI_slice, cmap='gray')
    plt.axis('off')
    # 绘制线性插值曲线
    plt.subplot(1, 4, 4)
    plt.title('Linear Interpolation')
    t = torch.linspace(0, 1, 200)
    y_interp = piecewise_linear_interp(t.reshape(1, -1), ctrl_pts_x, ctrl_pts_y).squeeze().numpy()
    plt.plot(t, y_interp, label='Interpolated Curve')
    plt.plot(ctrl_pts_x.squeeze().numpy(), ctrl_pts_y.squeeze().numpy(), 'ro--', label='Control Points')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()

    plt.show()

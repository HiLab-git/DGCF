import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import comb

from code_network.mathtrans import normalize

def generate_bernstein(n_points = 4, n_times = 20, device=None):
    t = torch.linspace(0.0, 1.0, n_times)
    bernstein = torch.stack([bernstein_poly(i, n_points - 1, t) for i in range(n_points)], dim=0)
    if device is not None:
        bernstein = bernstein.to(device)
    return bernstein  # (N, T)

def bernstein_poly(i, n, t):
    coeff = comb(n, i)
    return coeff * (t ** (n - i)) * ((1 - t) ** i)

def bezier_transform_torch(x, points, bernstein = None):
    """
    Given control points, compute the Bezier curve in PyTorch.
    :param points: list of control points, each (x, y)
    :return: xvals, yvals of the curve
    """
    n_points = points.shape[1]  # number of control points
    x_points = points[:,:, 0]
    y_points = points[:,:, 1]
    sorted_x_points, sort_idx = torch.sort(x_points, dim=1)  # sort x_points
    sorted_y_points = torch.gather(y_points, dim=1, index=sort_idx)  # 用 index 重排 y_points
    print(sorted_x_points, sorted_y_points)

    if bernstein is None:
        bernstein = generate_bernstein(n_points, device = x_points.device)  # (N, T)
    # xvals = torch.einsum('bn,nt->bt', x_points, bernstein)
    # yvals = torch.einsum('bn,nt->bt', y_points, bernstein)
    xvals = torch.einsum('bn,nt->bt', sorted_x_points, bernstein)
    yvals = torch.einsum('bn,nt->bt', sorted_y_points, bernstein)

    sorted_xvals, sort_idx = torch.sort(xvals, dim=1)         # sort_idx: (B, T)
    sorted_yvals = torch.gather(yvals, dim=1, index=sort_idx) # 用 index 重排 yvals

    print("sorted_xvals:", sorted_xvals[0])
    print("sorted_yvals:", sorted_yvals[0])

    return torch_interp_image(x, sorted_xvals, sorted_yvals)

def torch_interp_image(xs, xps, fps):
    """
    Apply 1D piecewise linear Bezier transformation on image pixel values.
    xs:   (B, C, H, W)      - input image tensor, values in [-1,1]
    xps:  (B, T)            - sorted x coordinates of bezier curve points, in [-1,1]
    fps:  (B, T)            - sorted y coordinates of bezier curve points
    
    Output: transformed image of shape (B, C, H, W)
    """

    B, C, H, W = xs.shape
    T = xps.shape[1]
    
    # flatten image to (B, N), where N = C*H*W
    xs_flat = xs.view(B, -1)  # (B, N)
    
    # Clamp to Bezier x-range
    xs_clamped = xs_flat.clamp(min=xps[:, 0:1], max=xps[:, -1:])  # (B, N)

    # Interpolation indices
    # xps_cpu = xps.cpu()  # Ensure xps is on CPU for searchsorted
    # xs_clamped_cpu = xs_clamped.cpu()  # Ensure xs_clamped is on
    # inds = torch.searchsorted(xps_cpu, xs_clamped_cpu, right=True).to(xs.device)  # (B, N)
    inds = torch.searchsorted(xps, xs_clamped, right=True)

    inds = inds.clamp(min=1, max=T - 1)
    idx0 = inds - 1
    idx1 = inds

    # Helper: gather by batch
    def gather_by_batch(values, idx):
        B_idx = torch.arange(B, device=values.device).unsqueeze(1)
        return values[B_idx, idx]

    x0 = gather_by_batch(xps, idx0)
    x1 = gather_by_batch(xps, idx1)
    y0 = gather_by_batch(fps, idx0)
    y1 = gather_by_batch(fps, idx1)
 
    slope = (y1 - y0) / (x1 - x0 + 1e-6)

    ys = y0 + slope * (xs_clamped - x0)

    # reshape back to (B, C, H, W)
    ys_img = ys.view(B, C, H, W)
    return ys_img

if __name__ == "__main__":
    # 读取3D CT和MRI
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

    # 在torch上做transform
    ctrl_points = [
        [0.0, 0.0],
        [0.5, 1.0],
        [1.0, 0.0]
    ]
    ctrl_pts_x = ctrl_points[0::2]  # x坐标
    ctrl_pts_y = ctrl_points[1::2]  # y坐标
    ctrl_points = torch.tensor(ctrl_points, dtype=torch.float32).unsqueeze(0)  # shape: [1, 3, 2]
    ct_tensor = normalize(ct_tensor, range_before=[-1024,1000], range_after=[0,1])  # 归一化处理
    # bernstein = generate_bernstein(len(ctrl_points))  # 生成Bernstein基函数

    # 假设 bezier_transform_torch 支持 BC 维度输入
    B, C, H, W = ct_tensor.shape
    sMRI_tensor = bezier_transform_torch(ct_tensor, ctrl_points)

    sMRI_tensor = normalize(sMRI_tensor, range_before=[0, 1], range_after=[0, 2000])  # 转换回[-1024, 2000]范围

    # 转回numpy
    sMRI_slice = sMRI_tensor.squeeze().cpu().numpy()

    # # 将ct_slice截断在[-250,250]的范围内
    ct_slice = np.clip(ct_slice, -50, 50)
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

    # 绘制Bezier曲线
    plt.subplot(1, 4, 4)
    plt.title('Bezier Curve')
    ctrl_np = ctrl_pts_x.squeeze().numpy()  # 转为numpy数组
    ctrl_np = np.column_stack((ctrl_np, ctrl_pts_y.squeeze().numpy()))  #
    plt.plot(ctrl_np[:, 0], ctrl_np[:, 1], 'ro--', label='Control Points')

    # 生成Bezier曲线点
    t = np.linspace(0, 1, 100)
    curve = np.zeros((100, 2))
    n = len(ctrl_np) - 1
    for i in range(n + 1):
        bern = bernstein_poly(i, n, t)
        curve += np.outer(bern, ctrl_np[i])
    plt.plot(curve[:, 0], curve[:, 1], 'b-', label='Bezier Curve')
    plt.legend()
    plt.axis('equal')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
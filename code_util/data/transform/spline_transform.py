import torch

def uniform_knot_vector(num_ctrl_pts, degree):
    """生成 clamped 均匀 knot 向量"""
    num_knots = num_ctrl_pts + degree + 1
    knots = torch.zeros(num_knots)
    knots[degree:num_knots - degree] = torch.linspace(0, 1, num_knots - 2 * degree)
    knots[num_knots - degree:] = 1.0
    print(knots.shape)
    return knots

def bspline_basis(t, degree, knots):
    """
    Compute B-spline basis (batch version)
    t: (B, N) - B 个样本，每个 N 个位置
    knots: (K,) - 公共 knot 向量
    degree: 样条次数
    return: (B, N, M) - M 是 basis 函数个数 = len(knots) - degree - 1
    """
    B, N = t.shape
    K = knots.shape[0]
    M = K - degree - 1

    t = t.unsqueeze(-1)  # (B, N, 1)
    knots = knots.to(t.device)

    basis = ((t >= knots[:-1]) & (t < knots[1:])).float()  # (B, N, M+degree)

    for d in range(1, degree + 1):
        left_num = t - knots[:-d - 1]
        left_den = knots[d: -1] - knots[:-d - 1] + 1e-8
        left = left_num / left_den
        left = left * basis[..., :-1]

        right_num = knots[d + 1:] - t
        right_den = knots[d + 1:] - knots[1:-d] + 1e-8
        right = right_num / right_den
        right = right * basis[..., 1:]

        basis = left + right  # (B, N, M - d)

    return basis  # (B, N, M)

def bspline_interp(t, ctrl_pts_y, degree):
    """
    B-spline interpolation with batch input
    t: (B, N)
    ctrl_pts_y: (B, M)
    degree: 样条次数
    return: (B, N)
    """
    B, M = ctrl_pts_y.shape
    knots = uniform_knot_vector(M, degree).to(ctrl_pts_y.device)
    basis = bspline_basis(t, degree, knots)  # (B, N, M)
    y = torch.bmm(basis, ctrl_pts_y.unsqueeze(-1)).squeeze(-1)  # (B, N)
    return y

def bspline_interp_img(x, ctrl_pts_y, degree):
    B, C, H, W = x.shape
    M = ctrl_pts_y.shape[1]  # 控制点数量
    knots = uniform_knot_vector(M, degree).to(x.device)
    basis = bspline_basis(x.view(B, -1), degree, knots)  # (B, N, M)

    y = torch.bmm(basis, ctrl_pts_y.unsqueeze(-1)).squeeze(-1)  # (B, N)
    return y.view(B, C, H, W)  # 恢复为图像形状

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# 假设 degree 为 3（三次 B 样条）
degree = 3

# 控制点 y 值（B=1, M=6）
ctrl_pts_y = torch.tensor([[0.0, 0.3, -0.5, 0.7, 0.2, 0.0]])  # shape: (1, 6)

# 插值位置 t（B=1, N=200）
t = torch.linspace(0, 1, steps=200).unsqueeze(0)  # shape: (1, 200)

# ====== 测试 1：1D 插值 ======
y_interp = bspline_interp(t, ctrl_pts_y, degree)  # shape: (1, 200)

# 可视化插值曲线
plt.figure(figsize=(6, 4))
plt.plot(t[0].numpy(), y_interp[0].detach().numpy(), label="Interpolated B-spline")
plt.plot(torch.linspace(0, 1, ctrl_pts_y.shape[1]).numpy(), ctrl_pts_y[0].numpy(), 'ro--', label="Control Points")
plt.title("B-spline Interpolation (1D)")
plt.show()

import torch

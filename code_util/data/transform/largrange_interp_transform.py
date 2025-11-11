import torch

def lagrange_basis(t, ctrl_pts_x):
    """
    Lagrange 多项式基函数
    t: (B, N)
    ctrl_pts_x: (B, M)
    return: (B, N, M)
    """
    B, N = t.shape
    M = ctrl_pts_x.shape[1]

    t = t.unsqueeze(-1)  # (B, N, 1)
    L = torch.ones(B, N, M, device=t.device)

    for j in range(M):
        xj = ctrl_pts_x[:, j].unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        
        others = []
        for m in range(M):
            if m == j:
                continue
            xm = ctrl_pts_x[:, m].unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
            num = t - xm  # (B, N, 1)
            den = xj - xm  # (B, 1, 1)
            L[:, :, j] *= (num / (den + 1e-8)).squeeze(-1)  # (B,N)

    return L  # (B, N, M)


def lagrange_interp(t, ctrl_pts_x, ctrl_pts_y):
    """
    Lagrange 插值主函数
    t: (B, N)
    ctrl_pts_x: (B, M)
    ctrl_pts_y: (B, M)
    return: (B, N)
    """
    basis = lagrange_basis(t, ctrl_pts_x)  # (B, N, M)
    y = torch.bmm(basis, ctrl_pts_y.unsqueeze(-1)).squeeze(-1)  # (B, N)
    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 控制点（非均匀 x）
    ctrl_pts_x = torch.tensor([[0.0, 0.1, 0.4, 0.5, 0.8, 1.0]])
    ctrl_pts_y = torch.tensor([[0.0, 0.5, -0.3, 0.7, 0.2, 0.0]])

    # 插值点
    t = torch.linspace(0, 1, 200).unsqueeze(0)  # shape (1, 200)

    # Lagrange 插值
    y_interp = lagrange_interp(t, ctrl_pts_x, ctrl_pts_y)

    # 可视化
    plt.figure(figsize=(6, 4))
    plt.plot(t[0].numpy(), y_interp[0].numpy(), label="Lagrange Interpolation")
    plt.plot(ctrl_pts_x[0].numpy(), ctrl_pts_y[0].numpy(), 'ro--', label="Control Points")
    plt.title("Lagrange Polynomial Interpolation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

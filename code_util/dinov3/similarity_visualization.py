import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import SimpleITK as sitk
import os

from code_network.dinov3.tools.dinov3_adapter import Dinov3Adapter  # 你的模型加载模块


# =========================
# 功能函数定义
# =========================

def load_image(image_path):
    """
    读取图像（支持彩色图像和医学图像）
    返回：PIL.Image
    """
    ext = os.path.splitext(image_path)[-1].lower()
    medical_ext = [".nii", ".nii.gz", ".mha", ".mhd", ".dcm", ".gz"]
    if ext in medical_ext:
        img = sitk.ReadImage(image_path)
        arr = sitk.GetArrayFromImage(img)  # [D,H,W] 或 [H,W]
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]  # 取中间层
        # arr = np.clip(arr, -1024, 2000)
        # arr = (arr + 1024) / (2000 + 1024)
        # arr = (arr * 255).astype(np.uint8)
        # 99 percentile normalization
        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip(arr, p1, p99)
        arr = (arr - p1) / (p99 - p1)
        arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    return img


def get_dinov3_features(model, image, device, H_img, W_img):
    """
    提取 DINOv3 的 patch 特征（去掉 cls token）
    """
    preprocess = transforms.Compose([
        transforms.Resize((H_img, W_img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])
    img_tensor = preprocess(image).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        feats = model(img_tensor)
        feats = feats[0][0]  # [N, C]
    return feats


def compute_similarity_map(feats, H, W, anchor_y_ratio=0.5, anchor_x_ratio=0.5):
    """
    计算余弦相似度图
    """
    feats = torch.nn.functional.normalize(feats, dim=-1)
    feats_map = feats.view(H, W, -1)
    anchor_y, anchor_x = int(H * anchor_y_ratio), int(W * anchor_x_ratio)
    anchor = feats_map[anchor_y, anchor_x, :]
    sim_map = torch.matmul(feats, anchor.T).squeeze(-1)
    sim_img = sim_map.view(H, W).cpu().numpy()
    return sim_img, (anchor_x, anchor_y)


def visualize_similarity_maps(images, sim_maps, anchors, H_img, W_img):
    """
    绘制多图像两行布局：
    第一行：特征相似度
    第二行：原图
    """
    n = len(images)
    plt.figure(figsize=(4 * n, 8))

    for i in range(n):
        sim_img = sim_maps[i]
        img = images[i].resize((W_img, H_img))
        anchor_x, anchor_y = anchors[i]

        # 第一行：相似度热图
        plt.subplot(2, n, i + 1)
        plt.imshow(sim_img, cmap='viridis')
        plt.scatter([anchor_x], [anchor_y], c='red', s=20, marker='o', label='Anchor')
        plt.title(f"Sim Map {i+1}")
        plt.axis('off')

        # 第二行：原图
        plt.subplot(2, n, n + i + 1)
        plt.imshow(img)
        plt.title(f"Original {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# =========================
# 主执行函数
# =========================

def run_similarity_visualization(
    image_paths,
    model_name="dinov3_vitb16",
    layer_index=11,
    H_img=2048,
    W_img=2048,
    anchor_y_ratio=0.5,
    anchor_x_ratio=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = Dinov3Adapter(model_name=model_name, use_ft16=False, freeze=True).to(device)

    H, W = H_img // 16, W_img // 16
    images, sim_maps, anchors = [], [], []

    for image_path in image_paths:
        img = load_image(image_path)
        feats = get_dinov3_features(model, img, device, H_img, W_img)
        sim_img, anchor_xy = compute_similarity_map(feats, H, W, anchor_y_ratio, anchor_x_ratio)
        images.append(img)
        sim_maps.append(sim_img)
        anchors.append(anchor_xy)

    visualize_similarity_maps(images, sim_maps, anchors, H_img, W_img)


# =========================
# 示例使用
# =========================

if __name__ == "__main__":
    image_list = [
        "./file_dataset/SynthRAD2023/Task1/pelvis/3D/testA/1PA018.nii.gz",
        "./file_dataset/SynthRAD2023/Task1/pelvis/3D/testB/1PA018.nii.gz",
    ]
    run_similarity_visualization(
        image_list,
        layer_index=11,
        H_img=2048,
        W_img=2048,
        anchor_y_ratio=0.5,
        anchor_x_ratio=0.5
    )

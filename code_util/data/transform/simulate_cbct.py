import SimpleITK as sitk
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

import warnings
warnings.filterwarnings("ignore")


def apply_motion_blur(img, kernel_size=15, angle=0):
    """
    img: 2D NumPy 图像，范围[0,1]
    kernel_size: 模糊核大小（越大拖尾越长）
    angle: 模糊方向，单位为度（0是水平，90是垂直）
    """
    # 创建线性核
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel /= kernel_size

    # 旋转核
    center = (int(kernel_size // 2), int(kernel_size // 2))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel_rot = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))

    # 应用滤波
    blurred = cv2.filter2D(img, -1, kernel_rot, borderType=cv2.BORDER_REFLECT)
    return blurred

def add_motion_ghosting(img, num_ghosts=2, shift_pixels=3, alpha=0.2, **kwargs):
    """
    在图像中添加多个偏移副本，模拟重影运动伪影
    """
    ghosts = [img.copy()]
    for i in range(1, num_ghosts+1):
        shifted = np.roll(img, shift_pixels*i, axis=1)
        ghosts.append(shifted * (alpha**i))
    result = np.sum(ghosts, axis=0)
    result = np.clip(result, 0, 1)
    return result

import SimpleITK as sitk

def simulate_motion_blur_volume(sitk_img, mode='blur', **kwargs):
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    arr = normalize(arr, range_before=(-1024, 3000), range_after=(0, 1))  # 假设CT值范围为[-1024, 3000]

    result = []
    for slice_ in arr:
        if mode == 'blur':
            res = apply_motion_blur(slice_, **kwargs)
        elif mode == 'ghost':
            res = add_motion_ghosting(slice_, **kwargs)
        else:
            raise ValueError("mode should be 'blur' or 'ghost'")
        result.append(res)

    result = np.stack(result)
    result = normalize(result, range_before=(0, 1), range_after=(-1024, 3000))  # 恢复到CT值范围
    new_img = sitk.GetImageFromArray(result)
    new_img.CopyInformation(sitk_img)
    return new_img


def add_motion_artifact_periodic(img_array, max_shift=5, freq=0.1):
    """
    对3D图像按层添加周期性平移，模拟呼吸运动
    - max_shift: 最大平移像素
    - freq: 频率，控制周期长度
    """
    moved = np.zeros_like(img_array)
    D, H, W = img_array.shape
    for i in range(D):
        shift = int(max_shift * np.sin(2 * np.pi * freq * i))  # 正弦偏移
        moved[i] = np.roll(img_array[i], shift, axis=1)  # 横向平移
    return moved

def add_motion_artifact_random(img_array, prob=0.2, max_shift=10):
    """
    随机层跳动伪影：每一层有概率平移
    - prob: 某层被扰动的概率
    """
    moved = img_array.copy()
    for i in range(img_array.shape[0]):
        if np.random.rand() < prob:
            dx = np.random.randint(-max_shift, max_shift)
            moved[i] = np.roll(img_array[i], dx, axis=1)
    return moved

from scipy.ndimage import gaussian_filter

def add_local_motion_blur(img_array, region=(slice(20,40), slice(50,150)), sigma=3):
    """
    只在图像的某个局部区域加模糊，模拟局部模糊/运动
    """
    motioned = img_array.copy()
    for i in range(img_array.shape[0]):
        patch = img_array[i][region]
        patch_blur = gaussian_filter(patch, sigma=sigma)
        motioned[i][region] = patch_blur
    return motioned

import cv2

def add_affine_motion(img_array, angle_range=5, shift_range=5):
    """
    对每层图像添加小角度旋转和平移
    """
    D, H, W = img_array.shape
    result = np.zeros_like(img_array)
    center = (W // 2, H // 2)

    for i in range(D):
        angle = np.random.uniform(-angle_range, angle_range)
        dx = np.random.uniform(-shift_range, shift_range)
        dy = np.random.uniform(-shift_range, shift_range)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[:, 2] += (dx, dy)

        result[i] = cv2.warpAffine(img_array[i], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return result

import SimpleITK as sitk

def simulate_motion_artifact_sitk(image_sitk, method="periodic"):
    img_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    img_array = normalize(img_array, range_before=(-1024, 3000), range_after=(0, 1))  # 假设CT值范围为[-1024, 3000]

    if method == "periodic":
        motioned = add_motion_artifact_periodic(img_array, max_shift=10, freq=0.05)
    elif method == "random":
        motioned = add_motion_artifact_random(img_array, prob=0.2)
    elif method == "affine":
        motioned = add_affine_motion(img_array)
    elif method == "local_blur":
        motioned = add_local_motion_blur(img_array)
    else:
        raise ValueError("Unknown method")

    # 归一化到CT值范围
    motioned = normalize(motioned, range_before=(0, 1), range_after=(-1024, 3000))  # 恢复到CT值范围
    result_img = sitk.GetImageFromArray(motioned)
    result_img.CopyInformation(image_sitk)
    return result_img


def add_low_freq_artifact(slice_img, strength=0.2, radius=30):
    """
    slice_img: 2D numpy image (float32, normalized to [0,1])
    strength: 扰动强度，0~1
    radius: 低频扰动半径（中心多少个频域像素受影响）
    """
    f = np.fft.fftshift(np.fft.fft2(slice_img))
    mag, phase = np.abs(f), np.angle(f)
    
    # 中心区域扰动
    h, w = f.shape
    cy, cx = h // 2, w // 2

    # 构造扰动模板
    noise = np.random.normal(0, strength * np.mean(mag), (h, w))
    mask = np.zeros_like(mag)
    for y in range(h):
        for x in range(w):
            if (x - cx)**2 + (y - cy)**2 < radius**2:
                mask[y, x] = 1

    mag_perturbed = mag + noise * mask  # 仅扰动低频
    f_perturbed = mag_perturbed * np.exp(1j * phase)
    image_perturbed = np.fft.ifft2(np.fft.ifftshift(f_perturbed)).real

    # 归一化
    image_perturbed = np.clip(image_perturbed, 0, 1)
    return image_perturbed

import SimpleITK as sitk
import os
import sys

def simulate_fourier_scatter(image_sitk, strength=0.3, radius=40):
    """对SimpleITK图像添加傅里叶低频伪影"""
    img_array = sitk.GetArrayFromImage(image_sitk).astype(np.float32)
    # img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    img_array = normalize(img_array, range_before=(-1024, 3000), range_after=(0, 1))  # 假设CT值范围为[-1024, 3000]
    perturbed_slices = []
    for slice_img in img_array:
        perturbed = add_low_freq_artifact(slice_img, strength=strength, radius=radius)
        perturbed_slices.append(perturbed)

    result_array = np.stack(perturbed_slices, axis=0)
    result_img = normalize(result_array, range_before=(0, 1), range_after=(-1024, 3000))  # 恢复到CT值范围
    result_img = sitk.GetImageFromArray(result_img)
    result_img.CopyInformation(image_sitk)

    return result_img


def normalize(img, range_before=None, range_after=None):
    if range_before is None:
        # range_before = torch.min(img), torch.max(img)
        range_before = (np.min(img), np.max(img))
    if range_after is None:
        range_after = (0, 1)
    
    img = (img - range_before[0]) / (range_before[1] - range_before[0])
    img = img * (range_after[1] - range_after[0]) + range_after[0]
    return img

def simulate_scatter_blur(image, sigma=3.0):
    """模拟散射伪影（高斯模糊）"""
    return sitk.DiscreteGaussian(image, sigma)

def simulate_truncation(image, crop_ratio=0.7):
    """模拟截断伪影（裁剪视野）"""
    size = image.GetSize()
    start = [int(size[i] * (1 - crop_ratio) / 2) for i in range(3)]
    end = [int(size[i] * crop_ratio) for i in range(3)]
    roi = sitk.RegionOfInterest(image, end, start)
    
    # 将ROI重新放到原图大小中，空区域填-1024（假设CT值范围为[-1024, 3000]）
    truncated = sitk.Image(image.GetSize(), image.GetPixelID())
    truncated.CopyInformation(image)
    truncated = sitk.Paste(truncated, roi, roi.GetSize(), destinationIndex=start)
    return truncated

def add_cbct_noise(image, noise_std=0.05):
    """添加噪声（CBCT常见的泊松 + 高斯噪声）"""
    img_array = sitk.GetArrayFromImage(image).astype(np.float32)
    img_array = normalize(img_array,range_before=(-1024,3000), range_after=(0, 1))  # 假设CT值范围为[-1024, 3000]

    # 模拟噪声
    scale_factor = 4095  # 控制噪声强度的关键参数
    poisson = np.random.poisson(img_array * scale_factor) / scale_factor

    gaussian = np.random.normal(0, noise_std, img_array.shape)
    noisy = np.clip(poisson + gaussian, 0, 1)
    noisy = normalize(noisy, range_before=(0, 1), range_after=(-1024,3000))  
    # 转换为SimpleITK格式
    noisy_image = sitk.GetImageFromArray(noisy)
    noisy_image.CopyInformation(image)
    return noisy_image

import numpy as np
from skimage.transform import radon, iradon

import cv2

def radon_transform(image, intensity=1.0, num_directions=1):
    """
    对图像施加射线状伪影（模拟 Radon 空间中遮挡投影角度）。

    参数:
        image (ndarray): 输入图像，2D numpy array。
        intensity (float): 伪影强度，范围 0~1（0 表示完全遮挡，1 表示无影响）。
        num_directions (int): 随机选择的伪影方向数量。

    返回:
        reconstruction (ndarray): 带射线状伪影的重建图像。
    """

    image_resized = cv2.resize(image, (128, 128))  # 例如原图 512x512，降至 128x128

    num_angles = 128  # Radon 变换的角度数量
    # theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    theta = np.linspace(0., 180., num_angles, endpoint=False)
    # sinogram = radon(image, theta=theta)
    sinogram = radon(image_resized, theta=theta)

    # ✅ 随机选择伪影方向的角度索引
    total_angles = len(theta)
    direction_indices = np.random.choice(total_angles, size=num_directions, replace=False)

    for idx in direction_indices:
        if intensity == 0.0:
            sinogram[:, idx] = 0  # 完全遮挡
        else:
            sinogram[:, idx] *= (1.0 - intensity)  # 衰减模拟遮挡强度

    reconstruction = iradon(sinogram, theta=theta, circle=True)

    reconstruction = cv2.resize(reconstruction, (image.shape[1], image.shape[0]))  # 恢复到原图大小

    return reconstruction


def generate_streak_artifacts(image, num_streaks=20, intensity=0.3):
   
    H, W = image.shape
    center = (W // 2, H // 2)
    mask = np.zeros_like(image, dtype=np.float32)

    angles = np.linspace(0, 180, num_streaks, endpoint=False)

    for angle in angles:
        rad = np.deg2rad(angle)
        x = int(np.cos(rad) * W)
        y = int(np.sin(rad) * H)
        cv2.line(mask, center, (center[0] + x, center[1] + y), color=1.0, thickness=1)

    # 模糊使伪影更真实
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3)

    # 将伪影叠加到图像中
    image_with_artifact = image + intensity * mask
    image_with_artifact = np.clip(image_with_artifact, 0, 1)
    return image_with_artifact


def simulate_cbct_from_ct(ct_image):
    cbct_img = ct_image.copy()
    # ct_img_normed = normalize(ct_image, range_before=(-1024, 2000), range_after=(0, 1))  # 假设CT值范围为[-1024, 3000]
    # cbct_img = apply_motion_blur(ct_image, kernel_size=11, angle=40)
    # # cbct = add_motion_ghosting(cbct, num_ghosts=3, shift_pixels=5, alpha=0.3)
    # cbct_img = add_low_freq_artifact(cbct_img, strength=3, radius=40)
    # cbct_img = normalize(cbct_img, range_before=(0, 1), range_after=(-1024, 2000))  #
    cbct_img = radon_transform(cbct_img, intensity = 5, num_directions=6)
    # cbct_img = generate_streak_artifacts(cbct_img, num_streaks=30, intensity=0.3)

    return cbct_img

def simulate_cbct_from_ct_random(ct_image):
    cbct_img = ct_image.copy()

    kernel_sizes = [3, 5, 7, 9, 11]
    angles = [10, 20, 30, 40, 50]
    kernel_size = np.random.choice(kernel_sizes)
    angle = np.random.choice(angles)
    # print(f"Using kernel_size={kernel_size}, angle={angle} for motion blur")
    cbct_img = apply_motion_blur(cbct_img, kernel_size=kernel_size, angle=angle)

    strength = np.random.randint(1, 3)  # 随机强度
    radius = np.random.randint(10, 40)  # 随机半径
    cbct_img = add_low_freq_artifact(cbct_img, strength=strength, radius=radius)

    intensity = np.random.randint(1, 8)  # 随机强度
    num_directions = np.random.randint(2, 10)  # 随机
    cbct_img = radon_transform(cbct_img, intensity=intensity, num_directions=num_directions)

    return cbct_img

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 加载2D医学图像（假设为.nii或.dcm等格式）
    # 这里只做演示，实际请替换为你的文件路径
    ct_path = "file_dataset/SynthRAD2025/Task2/HN/2D/validationB/2HNA021_27.mha"
    cbct_path = "file_dataset/SynthRAD2025/Task2/HN/2D/validationA/2HNA021_27.mha"
    img_sitk = sitk.ReadImage(ct_path)
    img_array = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    slice_img = img_array

    # 归一化到[0,1]
    slice_img_norm = normalize(slice_img, range_before=(-1024, 2000), range_after=(0, 1))

    # 测试运动模糊
    blurred = apply_motion_blur(slice_img_norm, kernel_size=11, angle=30)

    # 测试重影
    ghosted = add_motion_ghosting(slice_img_norm, num_ghosts=3, shift_pixels=5, alpha=0.3)

    # 测试低频傅里叶伪影
    fourier_artifact = add_low_freq_artifact(slice_img_norm, strength=3, radius=40)

    # 全部使用
    cbct = simulate_cbct_from_ct(slice_img_norm)

    # 显示结果
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 6, 1)
    plt.title("Original")
    plt.imshow(slice_img_norm, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.title("Motion Blur")
    plt.imshow(blurred, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 3)
    plt.title("Ghosting")
    plt.imshow(ghosted, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 4)
    plt.title("Fourier Artifact")
    plt.imshow(fourier_artifact, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.title("CBCT Simulation")
    plt.imshow(cbct, cmap="gray")
    plt.axis("off")
    plt.suptitle("Simulated CBCT Artifacts")

    plt.subplot(1, 6, 6)
    plt.title("CBCT Image")
    cbct_img_sitk = sitk.ReadImage(cbct_path)
    cbct_img_array = sitk.GetArrayFromImage(cbct_img_sitk)
    cbct_img_array = normalize(cbct_img_array, range_before=(-1024, 2000), range_after=(0, 1))
    plt.imshow(cbct_img_array, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
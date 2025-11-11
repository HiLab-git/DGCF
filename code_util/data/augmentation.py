import numpy as np
import random
import copy
from code_util.data.transform.simulate_cbct import simulate_cbct_from_ct_random
from code_util.data.transform.general import normalize_nobatch

try:
    from scipy.special import comb
except:
    from scipy.misc import comb

def apply_augmentation(x, config):

    """
    根据配置选择数据增强方法
    :param config: 配置字典，包含数据增强相关的设置
    :return: 选择的数据增强方法
    """
    x = normalize_nobatch(x,range_before=(-1024,2000),range_after=(0,1))  # 假设CT值范围为[-1024, 2000]
    pretext_tasks = config["model"]["pretext_tasks"]
    task_names = pretext_tasks["task_names"]
    for task_name in task_names:
        if task_name == "MAE":
            x = mask_image(x, mask_ratio=pretext_tasks[task_name]["mask_ratio"])
        elif task_name == "bezier":
            x = bezier_transformation(x, prob=pretext_tasks[task_name]["prob"]) 
        elif task_name == "ModelGenesis":
            x = modelGenesis(x, 
                            # local_rate=pretext_tasks[task_name]["local_rate"], 
                            # nonlinear_rate=pretext_tasks[task_name]["nonlinear_rate"], 
                            # paint_rate=pretext_tasks[task_name]["paint_rate"], 
                            # inpaint_rate=pretext_tasks[task_name]["inpaint_rate"]
                            )
        elif task_name == "CBCT":
            x = simulate_cbct_from_ct_random(x)
        else:
            raise ValueError(f"Unsupported pretext task: {task_name}")
    
    x = normalize_nobatch(x, range_before=(0, 1), range_after=(-1024, 2000))  # 假设CT值范围为[-1024, 2000]

    return x

def mask_image(image_array,mask_ratio):
        """对图像进行随机遮挡"""
        height, width = image_array.shape
        num_patches = height * width
        num_mask = int(mask_ratio * num_patches)

        # 随机选择遮挡的patch索引
        mask = np.zeros(num_patches, dtype=np.bool_)
        mask_indices = np.random.choice(num_patches, num_mask, replace=False)
        mask[mask_indices] = True

        # 将mask应用到图像
        mask = mask.reshape(height, width)
        masked_image = image_array.copy()
        masked_image[mask] = 0  # 将遮挡区域设为0
        return masked_image

def bezier_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def modelGenesis(x,local_rate=0.5,nonlinear_rate=0.9,paint_rate=0.9,inpaint_rate=0.2):

    x = normalize_nobatch(x,range_before=(-1024,2000),range_after=(0,1))
    # print(x.shape)
    dim = x.ndim
    if dim == 2:
        # 在最后加上一个通道维度
        x = np.expand_dims(x, axis=0)
    # print(x.shape)

    # x = np.transpose(x, (2, 0, 1))
    
    # # Flip
    # x, y = data_augmentation(x, y, flip_rate)

    # Local Shuffle Pixel
    x = local_pixel_shuffling(x, prob=local_rate)
    
    # Apply non-Linear transformation with an assigned probability
    x = bezier_transformation(x, nonlinear_rate)
    
    # Inpainting & Outpainting
    if random.random() < paint_rate:
        if random.random() < inpaint_rate:
            # Inpainting
            x = image_in_painting(x)
        else:
            # Outpainting
            x = image_out_painting(x)
    
    # x = np.transpose(x, (1, 2, 0))
    #
    if dim == 2:
        # 删除通道维度
        x = np.squeeze(x, axis=0)

    x = normalize_nobatch(x, range_after=(-1024, 2000), range_before=(0, 1))

    return x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols = x.shape  # 只考虑高度和宽度
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, block_noise_size_y))
        
        image_temp[0, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    _, img_rows, img_cols = x.shape  # 只考虑高度和宽度
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        
        # 在图像上填充随机噪声
        x[:, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = np.random.rand(block_noise_size_x, block_noise_size_y) * 1.0
        cnt -= 1
    return x


def image_out_painting(x):
    _, img_rows, img_cols = x.shape  # 只考虑高度和宽度
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2]) * 1.0  # 删除深度维度
    
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    
    # 将原始图像的部分区域填充到新图像中
    x[:, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = image_temp[:, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y]
    
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        
        x[:, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y] = image_temp[:, noise_x:noise_x+block_noise_size_x, noise_y:noise_y+block_noise_size_y]
        cnt -= 1
    return x
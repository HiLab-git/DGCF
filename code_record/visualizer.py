import logging
import numpy as np
import os
import sys
from subprocess import Popen, PIPE

from code_util import util
from code_util.data import read_save
from code_util.util import tensor2im, get_module_by_name


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, config):
        """Initialize the Visualizer class
        Parameters:    
        Step 1: Cache the training/test options
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        对train和test 有不同的处理
        """

        self.config = config  # cache the option
        self.name = config["name"]
        self.work_dir = config["work_dir"]

        self.init_log()

        # html
        if self.config["record"].get("html",{}).get("use_html") == True:
            self.use_html = True
            self.init_html()
        # tensorboardX
        if self.config["record"].get("tensorboard", {}).get("use_tensorboard") == True:
            self.use_tensorboard = True
            self.init_tensorboard()
        
    def init_log(self):
        format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if self.config["phase"] == "train":
            self.logger_train = logging.getLogger("logger_train")
            self.log_train_file = os.path.join(self.work_dir, 'train_log.txt')
            self.handler_train = logging.FileHandler(self.log_train_file)
            self.handler_train.setFormatter(format)
            self.logger_train.setLevel(logging.INFO)
            self.logger_train.addHandler(self.handler_train)
            print(f"Training log initialized: {self.log_train_file}")
            self.logger_train.info(f"Training started")
            self.logger_val = logging.getLogger("logger_val")
            self.log_val_file = os.path.join(self.work_dir, 'val_log.txt')
            self.handler_val = logging.FileHandler(self.log_val_file)
            self.handler_val.setFormatter(format)
            self.logger_val.setLevel(logging.INFO)
            self.logger_val.addHandler(self.handler_val)
            print(f"Training log initialized: {self.log_val_file}")
            self.logger_val.info(f"Validation started")
        elif self.config["phase"] == "test":
            self.logger_test = logging.getLogger("logger_test")
            self.log_test_file = os.path.join(self.work_dir, 'test_log.txt')
            self.handler_test = logging.FileHandler(self.log_test_file)
            self.handler_test.setFormatter(format)
            self.logger_test.setLevel(logging.INFO)
            self.logger_test.addHandler(self.handler_test)
            print(f"Test log initialized: {self.log_test_file}")
            self.logger_test.info(f"Test started")
        else:
            raise ValueError("Invalid phase. Expected 'train' or 'test', got: %s" % self.config["phase"])

    def init_html(self):
        from . import html
        self.web_dir = os.path.join(self.work_dir,'web')
        print("create web directory:", self.web_dir)
        os.makedirs(self.web_dir,exist_ok=True)
        self.win_size = self.config["record"]["html"]["display_size"]
        if self.config["phase"] == "train":
            img_train_dir = os.path.join(self.web_dir, 'train')
            os.makedirs(img_train_dir,exist_ok=True)
            title = '%s | train' % self.work_dir
            self.webpage_train = html.HTML(self.web_dir, title, filename = "train", refresh=0)
    
            img_val_dir = os.path.join(self.web_dir, 'val')
            os.makedirs(img_val_dir,exist_ok=True)
            title = '%s | val' % self.work_dir
            self.webpage_val = html.HTML(self.web_dir, title, filename = "val", refresh=0)
            
            self.img_dir = [img_train_dir, img_val_dir]
        else: 
            img_test_dir = os.path.join(self.web_dir, 'test')
            os.makedirs(img_test_dir,exist_ok=True)
            title = 'Experiment = %s, Epoch = %s' % (self.name, self.config["result"]["test_epoch"])
            self.webpage_test = html.HTML(self.web_dir, title, filename = "test", refresh=0)
            self.img_dir = [img_test_dir]

    def init_tensorboard(self):
        """Initialize TensorBoardX writer."""
        from tensorboardX import SummaryWriter  # 添加 tensorboardX 的导入
        tensorboard_dir = os.path.join(self.work_dir, 'tensorboard_logs')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard logs initialized: {tensorboard_dir}")
        # # 启动tensorboard 会和tqdm的显示冲突 暂时注释掉
        # from tensorboard import program
        # tb = program.TensorBoard()
        # tb.configure(argv=[None, '--logdir', tensorboard_dir])
        # url = tb.launch()
        # print(f"TensorBoard 启动成功，访问链接: {url}")
    
    def init_CAM(self):
        import cv2
        from code_util.cam.grad_cam import GradCAM, show_cam_on_image

    def display_on_html(self, visuals, img_paths, phase = 'train', epoch = 0, iter = 0, ):
        """
        save current results to an HTML file.
        """
        file_name = util.get_file_name(img_paths[0])
        if phase == "train":
            time_info = f"epoch {epoch} iter {iter}, {file_name}"
            save_img_path_clip = '%s_%s_%s' % (epoch, iter, file_name)
            img_dir = self.img_dir[0]
            webpage = self.webpage_train
        elif phase == "val":
            time_info = f"epoch {epoch}, {file_name}"
            save_img_path_clip = '%s_%s' % (epoch, file_name)
            img_dir = self.img_dir[1]
            webpage = self.webpage_val
        else: # test
            test_epoch = self.config["result"]["test_epoch"]
            time_info = f"test epoch {test_epoch}, {file_name}"
            save_img_path_clip = file_name
            img_dir = self.img_dir[0]
            webpage = self.webpage_test
        # save images to the disk
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image, return_first=True)
            save_img_path = os.path.join(img_dir, save_img_path_clip + '_' + label + '.png')
            read_save.save_image_4_show(image_numpy, save_img_path)

        # update website
        webpage.add_header(time_info)
        ims, txts, links = [], [], []

        for label, image_numpy in visuals.items():
            save_image_name = save_img_path_clip + '_' + label + '.png'
            ims.append(save_image_name)
            txts.append(label)
            links.append(save_image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
        webpage.save()

    def display_on_tensorboard(self, visuals, step, phase='train'):
        """
        Log images to TensorBoard.

        Parameters:
            visuals (dict) -- dictionary of images to log
            epoch (int)    -- current epoch
            phase (str)    -- 'train' or 'val'
            iter (int)     -- current iteration (used for 'train' phase)
        """
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)  # Convert tensor to numpy image(0~255)
            if phase == 'train':
                self.writer.add_image(f"{phase}/{label}", image_numpy, global_step=step, dataformats='HWC')
            elif phase == 'val':
                self.writer.add_image(f"{phase}/{label}", image_numpy, global_step=step, dataformats='HWC')

    def plot_scalars_on_tensorboard(self, scalars, epoch, phase='train'):
        """
        Log scalars to TensorBoard.

        Parameters:
            scalars (dict) -- dictionary of scalars to log
            epoch (int)   -- current epoch
            phase (str)   -- 'train' or 'val'
        """
        for scalar_name, scalar_value in scalars.items():
            self.writer.add_scalar(f"{phase}/{scalar_name}", scalar_value, global_step=epoch)

    def close_tensorboard(self):
        """Close the TensorBoard writer."""
        if hasattr(self, 'writer'):
            self.writer.close()
            
    # losses: same format as |losses| of plot_current_losses
    def record_log(self, info, phase):
        if phase == "train":
            self.logger_train.info(info)
        elif phase == "val":
            self.logger_val.info(info)
        elif phase == "test":
            self.logger_test.info(info)
        else:
            raise ValueError("Invalid phase. Expected 'train', 'val' or 'test', got: %s" % phase)

    def draw_CAM(self, model, config, epoch=None, img_paths=None):
        """Draw and save CAM images."""
        if img_paths is None:
            img_paths = model.get_image_paths()
        target_layer = get_module_by_name(model.netG, config["record"]["CAM"]["CAM_layer"])
        grad_cam = GradCAM(model.netG, target_layers=[target_layer], move2cuda=False)
        grayscale_cam = grad_cam(input_tensor=model.real_A, target=model.real_B)
        real_B = tensor2im(model.real_B)

        # Save CAM images
        save_path = os.path.join(config["work_dir"], "CAM")
        os.makedirs(save_path, exist_ok=True)
        A_path = img_paths["A_path"]
        if isinstance(A_path, list):
            A_path = A_path[0]
        save_name = A_path.split("/")[-1].split(".")[0]
        if epoch is not None:
            save_name = save_name + "_epoch_" + str(epoch)
        save_name = save_name + "_CAM.jpg"
        save_path = os.path.join(save_path, save_name)

        # Save as grayscale image
        cam_real = show_cam_on_image(real_B / 255, grayscale_cam[0, :, :], use_rgb=True)
        cv2.imwrite(save_path, cam_real)
        self.logger_train.info(f"CAM image saved at: {save_path}")
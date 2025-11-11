
## run

for paper "DINOv3-Guided Cross Fusion Framework for Semantic-aware CT generation from MRI and CBCT"

set the correct configuration in './file_config' and run:

```bash
python train.py --gpu 0

python test.py --gpu 0

python evaluation.py --gpu 0
```

or run

```bash
python main.py --gpu 0
```


## Introduction

This is a general-purpose deep learning framework designed for medical image translation.

The framework is organized in a modular fashion and controlled through configuration files.
For each custom model, you need to provide your own configuration files (for training and testing), model files, and network files.
Alternatively, you may directly call existing predefined models and networks through the configuration system.

---

## Dataset

### Training and Testing Data

All datasets are stored by default in `./file_dataset`, although the actual dataset path is specified through `.dataset.dataroot`.

The expected directory structure is:

```
/dataset
  ├── trainA
  │   ├── file1
  │   ├── file2
  │   └── ...
  ├── trainB
  │   ├── file1
  │   ├── file2
  │   └── ...
  ├── validationA
  │   ├── file3
  │   ├── file4
  │   └── ...
  ├── validationB
  │   ├── file3
  │   ├── file4
  │   └── ...
  ├── testA
  │   ├── file5
  │   ├── file6
  │   └── ...
  ├── testB
  │   ├── file5
  │   ├── file6
  │   └── ...
```

Training data is required.
The B-domain of validation and test sets can be optionally provided depending on the task.

For paired training, files under `trainA` and `trainB` must correspond exactly by filename.
If a file exists only in one modality, training will fail.
For unpaired training, there is no constraint on the relationship between the two modality datasets.

Files may be natural image formats such as `.png` or `.jpg`, or medical image formats such as `.nii` and `.nii.gz`.

---

## Mask

Masks may be used during training, testing, or metric computation.
By default, masks are stored in `./file_dataset`.

---

## Model

A *model* defines the full deep learning pipeline, including initialization, loss functions, optimizers, and visualization outputs.
The model is selected using `.model.model_name`.

Model files are stored under `./code_model`.
All custom model classes must inherit from `BaseModel` in `base_model.py`, and must be placed in a file named `modelname_model.py`.

The following models are already implemented:

* `pix2pix_model`
* `cycleGAN_model`
* `UnetPlusPlus_model`
* `vanillaSL_model`: a general-purpose supervised model suitable for paired image generation tasks without any special loss design.

---

## Network

Backbone architectures are defined in the `./code_network` directory.
A single network file may contain multiple architectures, so both `.network.filename` and `.network.netG` must be specified to select the desired network.

---

## Reference

Framework: [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
Configuration file style: [https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

Model references:

* MambaUNet
* SwinUNet
* UNet++
* Grad-CAM
* WTConv: [https://github.com/bgu-cs-vil/wtconv](https://github.com/bgu-cs-vil/wtconv)

---

## Dependency

### Common

```
pip install dominate==2.6.0  
pip install pandas einops timm simpleitk
```

### Medical

```
simpleitk==2.3.1
torch==1.8.1
torchvision==0.9.1
```

### Mamba Environment

```
conda create -n mamba python==3.10 -y
pip install torch==2.0.1 torchvision==0.15.2
pip install packaging
pip install mamba-ssm==1.2.0.post1
pip install causal-conv1d==1.2.0.post2
pip install fvcore
```

Additional references:
[https://github.com/ziyangwang007/Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet)
[https://github.com/wjh892521292/LKM-UNet/issues/3](https://github.com/wjh892521292/LKM-UNet/issues/3)
[https://blog.csdn.net/qq_57433916/article/details/138139534](https://blog.csdn.net/qq_57433916/article/details/138139534)

And:

```
pip install dominate==2.6.0 simpleitk==2.3.1 torch torchvision
```


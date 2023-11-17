- [1. 论文解读](#1-论文解读)
- [2. 下载数据](#2-下载数据)
- [3. 安装](#3-安装)
- [4. 修改](#4-修改)
- [5. 使用](#5-使用)
  - [5.1. 重建](#51-重建)
  - [5.2. 供其他项目使用](#52-供其他项目使用)


---
## 1. 论文解读

[Learning an Animatable Detailed 3D Face Model from In-The-Wild Images](https://arxiv.org/abs/2012.04012)

![Alt text](images/image.png)

$E_c$ 就是代码中的 `self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)`, 然后将 resnet 输出维度为 `n_param` 划分为不同含义。

## 2. 下载数据

```bash
bash fetch_data.sh
```

其实就是下载
- `data/FLAME2020.zip`(FLAME模型的权重文件)，并且把`data/FLAME2020/generic_model.pkl`移动出来`data/generic_model.pkl`
  
  链接：https://pan.baidu.com/s/1ejoL6dezMNoj_vPX4Dd-nA?pwd=ngw8 
  
  提取码：ngw8

- `data/deca_model.tar`(DECA的权重文件)
  
  链接：https://pan.baidu.com/s/1vlc6cYfeZ2-WEXqNkcMr0A?pwd=7fgl 
  
  提取码：7fgl
```
data
├── FLAME2020                   ###
│   ├── Readme.pdf              ###
│   ├── female_model.pkl        ###
│   └── male_model.pkl          ###
├── deca_model.tar              ###
├── fixed_displacement_256.npy
├── generic_model.pkl           ###
├── head_template.obj
├── landmark_embedding.npy- [1. 论文解读](#1-论文解读)
- [2. 下载数据](#2-下载数据)
- [安装](#安装)
- [3. 修改](#3-修改)
- [4. 使用](#4-使用)
  - [4.1. 重建](#41-重建)
  - [4.2. 供其他项目使用](#42-供其他项目使用)

├── mean_texture.jpg
├── texture_data_256.npy
├── uv_face_eye_mask.png
└── uv_face_mask.png
```

## 3. 安装


- python 3.10 才行，3.11不行
- 涉及到 chumpy 模块，这个模块使用 inspect 模块。
  ```python
  want_out = 'out' in inspect.getargspec(func).args
                              ^^^^^^^^^^^^^^^^^^
  
  AttributeError: module 'inspect' has no attribute 'getargspec'. Did you mean: 'getargs'?
  ```
  https://stackoverflow.com/questions/74585622/pyfirmata-gives-error-module-inspect-has-no-attribute-getargspec

- pytorch3d 
  
  https://github.com/sword4869/pytorch3d/blob/main/README2.md

  不用 `pytorch3d` 也可以: `deca_cfg.rasterizer_type = 'pytorch3d'` 改为 `standard` 就行。

# pip install -r requirements.txt
# 老的chumpy 会 from numpy import bool, int, float, complex, object, unicode, str, nan, inf

## 4. 修改
```bash
(pytorch3d) lab@eleven:~/project/DECA$ python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True
Traceback (most recent call last):
  File "/home/lab/project/DECA/demos/demo_reconstruct.py", line 131, in <module>
    main(parser.parse_args())
  File "/home/lab/project/DECA/demos/demo_reconstruct.py", line 40, in main
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/project/DECA/decalib/datasets/datasets.py", line 71, in __init__
    self.face_detector = detectors.FAN()
                         ^^^^^^^^^^^^^^^
  File "/home/lab/project/DECA/decalib/datasets/detectors.py", line 22, in __init__
    self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/miniconda3/envs/pytorch3d/lib/python3.11/enum.py", line 783, in __getattr__
    raise AttributeError(name) from None
AttributeError: _2D
```
```python
self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
```

## 5. 使用

### 5.1. 重建
```python
(ldm) lab@eleven:~/project/DECA$ python demos/demo_reconstruct.py -h
usage: demo_reconstruct.py [-h] [-i INPUTPATH] [-s SAVEFOLDER] [--device DEVICE] [--iscrop ISCROP] [--sample_step SAMPLE_STEP] [--detector DETECTOR] [--rasterizer_type RASTERIZER_TYPE]
                           [--render_orig RENDER_ORIG] [--useTex USETEX] [--extractTex EXTRACTTEX] [--saveVis SAVEVIS] [--saveKpt SAVEKPT] [--saveDepth SAVEDEPTH] [--saveObj SAVEOBJ] [--saveMat SAVEMAT]
                           [--saveImages SAVEIMAGES]

DECA: Detailed Expression Capture and Animation

options:
  -h, --help            show this help message and exit
  -i INPUTPATH, --inputpath INPUTPATH
                        path to the test data, can be image folder, image path, image list, video
  -s SAVEFOLDER, --savefolder SAVEFOLDER
                        path to the output directory, where results(obj, txt files) will be stored.
  --device DEVICE       set device, cpu for using cpu
  --iscrop ISCROP       whether to crop input image, set false only when the test image are well cropped
  --sample_step SAMPLE_STEP
                        sample images from video data for every step
  --detector DETECTOR   detector for cropping face, check decalib/detectors.py for details
  --rasterizer_type RASTERIZER_TYPE
                        rasterizer type: pytorch3d or standard
  --render_orig RENDER_ORIG
                        whether to render results in original image size, currently only works when rasterizer_type=standard
  --useTex USETEX       whether to use FLAME texture model to generate uv texture map, set it to True only if you downloaded texture model
  --extractTex EXTRACTTEX
                        whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode
  --saveVis SAVEVIS     whether to save visualization of output
  --saveKpt SAVEKPT     whether to save 2D and 3D keypoints
  --saveDepth SAVEDEPTH
                        whether to save depth image
  --saveObj SAVEOBJ     whether to save outputs as .obj, detail mesh will end with _detail.obj. Note that saving objs could be slow
  --saveMat SAVEMAT     whether to save outputs as .mat
  --saveImages SAVEIMAGES
                        whether to save visualization output as seperate images
```
```bash
python demos/demo_reconstruct.py -i TestSamples/examples --saveDepth True --saveObj True
```

### 5.2. 供其他项目使用

[my_get3dmm.py](./demos/my_get3dmm.py)

使用其中间过程的 latent code，包含了3dmm(FLAME)的参数。

```python
deca_code_shape = codedict['shape']     # [B, 100], FLAME parameters (shape 𝜷)
deca_code_exp = codedict['exp']         # [B, 50], FLAME parameters (expression 𝝍)
deca_code_pose = codedict['pose']       # [B, 6], FLAME parameters (pose 𝝍)
```
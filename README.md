# LiteSegFormer3D：一种高效轻量的三维医学图像分割模型

If you would like to view the English version, click [here](README_EN.md)

## 1. 概述
<img src="./img/LiteSegFormer3D.svg" />
我们基于SegFormer3D设计了一种轻量高效的医学图像分割网络**LiteSegFormer3D**，其在分割精度有所提升的同时，训练时间在三个数据集上缩短了1/3到1/2左右。我们在传统Transformer架构中采用一种新型前馈处理方式加速训练，同时在注意力中使用高斯核函数有效减少可训练参数规模，同时更好的拟合空间纹理特征，同时使用非对称卷积提取浅层特征和动态归一化加速收敛，实现了一种高效且精确的分割算法。

## 2. 安装



## 3. 数据集
  您可以参照[nnFormer](https://github.com/282857341/nnFormer)中数据的预处理方式，或者您也可以直接使用预处理好的数据，您可以从[Google Drive]()下载
## 4. 训练

## 5. 评估

## 6. 致谢

我们的实现基于[PyTorch](https://github.com/pytorch/pytorch)框架，数据集预处理参考了[nnFormer](https://github.com/282857341/nnFormer)和[UNETR++](https://github.com/Amshaker/unetr_plus_plus)的工作，代码构建参考了[nnUNet](https://github.com/MIC-DKFZ/nnUNet)、[FastKAN](https://github.com/ZiyaoLi/fast-kan)、[ACNet](https://github.com/DingXiaoH/ACNet)和[Mona](https://github.com/LeiyiHU/mona)的工作，基线模型实现参考了[SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D)的工作。
此外，我们编写Readme文件时参考了[TiM4Rec](https://github.com/AlwaysFHao/TiM4Rec)的工作。

## 7. 引用

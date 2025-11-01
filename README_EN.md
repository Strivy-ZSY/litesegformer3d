# LiteSegFormer3D: An Efficient and Lightweight Segmentation Model for 3D Medical Images

如果您想阅读中文版本，点击[这里](README.md)

## 1. Overview
<img src="./img/LiteSegFormer3D.svg" />

Based on SegFormer3D, we designed a lightweight and efficient medical image segmentation network called **LiteSegFormer3D**, which improves segmentation accuracy while reducing training time by one-third to one-half across three datasets. By introducing a novel feedforward processing method to accelerate training, employing an attention mechanism with Gaussian kernel functions to reduce parameter size and enhance spatial texture fitting, and combining asymmetric convolutions for shallow feature extraction with dynamic normalization to expedite convergence, we have achieved an efficient and precise segmentation algorithm.

## 2. Installation

```cmd
git clone https://github.com/Strivy-ZSY/litesegformer3d.git
cd litesegformer3d
```
### 2.1 Environmental Requirements
1.This code runs in an environment with CUDA 11.8 and Python 3.8.20.
```cmd
conda create --name lsf3d python=3.8
conda activate lsf3d
```

2.PyTorch installation (CUDA 11.8 is compatible with 11.3, so it can be installed normally):
```cmd
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

3.Other dependencies:
```cmd
pip install -r requirements.txt
```

## 3. Dataset

The datasets we are using are [BraTs](https://www.med.upenn.edu/sbia/brats2017/data.html), [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), and [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).  

You can refer to the data preprocessing method in [nnFormer](https://github.com/282857341/nnFormer), or you can directly download the preprocessed data for [Synapse](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EbHDhSjkQW5Ak9SMPnGCyb8BOID98wdg3uUvQ0eNvTZ8RA?e=YVhfdg), [ACDC](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EY9qieTkT3JFrhCJQiwZXdsB1hJ4ebVAtNdBNOs2HAo3CQ?e=VwfFHC), and [BRaTs](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EaQOxpD2yE5Btl-UEBAbQa0BYFBCL4J2Ph-VF_sqZlBPSQ?e=DFY41h). (These preprocessed datasets are from [UNETR++](https://github.com/Amshaker/unetr_plus_plus), but you need to rename the corresponding files from `unetr_pp` to `litesegformer3d`. Also, please remember to check whether the naming in the `Task003_tumor` folder under the `litesegformer3d_raw_data` directory has been modified. We recommend doing this, as it can save a lot of time 🙂. Once again, thanks to `UNETR++` ❤️.)

Store your downloaded results in the `litesegformer3d` folder. Taking the `BraTS` data as an example, the renamed folder structure is divided as follows:

```
./DATASET_Tumor/
  ├── litesegformer3d_raw/
      ├── litesegformer3d_raw_data/
           ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task003_tumor
       ├── litesegformer3d_cropped_data/
           ├── Task003_tumor
```

## 4. Training

To run the scripts in the `run_scripts` folder, simply comment out the `train` command in the script and enable the `inference` command for subsequent reasoning.

## 5. Evaluation

The `lsf3d/inferencedata` folder contains our evaluation results and the corresponding inference results on the test set. If you wish to perform inference and obtain the corresponding test results yourself, please modify the test path in the `inference_xxx.py` file under the `lsf3d` folder to the path where your test set for inference is stored. Additionally, for the specific division of the test set, you can refer to the division method of [nnFormer](https://github.com/282857341/nnFormer/blob/main/nnformer/dataset_json/). You will also need to modify the value of `splits[self.fold]['val']` in the corresponding `litesegformer3d_trainer_xxx.py` file in the [training code](https://github.com/Strivy-ZSY/litesegformer3d/tree/main/lsf3d/training/network_training) to match the list of your test set.

### 5.1 BraTS 2017
BraTS 2017 is an MRI dataset of brain tumors
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./img/BraTS.svg" alt="Wide Image" width="400" style="display: block;">
    <img src="https://github.com/user-attachments/assets/1eb0ca52-af34-426d-b693-cba5d0a3dcdf" alt="Narrow Image" width="400" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
  </div>
</p>

### 5.2 ACDC
ACDC is a cardiac organ segmentation MRI dataset
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./img/ACDC.svg" alt="Wide Image" width="400" style="display: block;">
    <img src="https://github.com/user-attachments/assets/028f9314-7344-4f61-bec8-9592a1064d1f" alt="Narrow Image" width="400" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
  </div>
</p>

### 5.3 Synapse
Synapse is a CT dataset for multi-organ segmentation in the abdomen
<p align="center">
  <div style="position: relative; display: inline-block;">
    <img src="./img/Synapse.svg" alt="Wide Image" width="400" style="display: block;">
    <img src="https://github.com/user-attachments/assets/2e276e44-c1a8-4498-b299-d22da4b32ce3" alt="Narrow Image" width="400" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
  </div>
</p>

## 6. Acknowledgments

Our implementation is based on the [PyTorch](https://github.com/pytorch/pytorch) framework. The dataset preprocessing refers to the work of [nnFormer](https://github.com/282857341/nnFormer) and [UNETR++](https://github.com/Amshaker/unetr_plus_plus). The code construction draws on the work of [nnUNet](https://github.com/MIC-DKFZ/nnUNet), [FastKAN](https://github.com/ZiyaoLi/fast-kan), [ACNet](https://github.com/DingXiaoH/ACNet), and [Mona](https://github.com/LeiyiHU/mona). The baseline model implementation refers to the work of [SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D). Additionally, we adopted the writing style of [TiM4Rec](https://github.com/AlwaysFHao/TiM4Rec) for composing the Readme documentation.

## 7. Citation
Under expert review, will provide updates later 🙂

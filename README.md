# DPGA-Former: Detail-Preserving and Geometry-Aware Semantic Segmentation for VHR Remote Sensing Imagery

## Introduction

Semantic segmentation of Very High Resolution (VHR) remote sensing imagery presents challenges in preserving fine spatial details, modeling long-range dependencies, and accurately delineating complex object boundaries.

To address these challenges, we propose **DPGA-Former**, a detail-preserving and geometry-aware semantic segmentation framework for VHR remote sensing imagery.

The proposed framework is evaluated on three benchmark datasets: ISPRS Potsdam, LandCover.ai (LCAI), and GID.

## Overview Framework of DPGA-Former

DPGA-Former incorporates complementary feature extraction and geometry-aware representation mechanisms to improve semantic segmentation of remote sensing images.

The framework integrates HiLoStem, AxialLite attention, and a boundary-aware decoder to enhance spatial detail preservation, contextual feature representation, and boundary delineation.

**Overview framework:**

[See the framework figures in the Figures directory](Figures/Fig_2.pdf)

## Numerical Results

We evaluate DPGA-Former on three public remote sensing semantic segmentation benchmarks.

| Method | Dataset | mIoU (%) |
|:---:|:---:|---:|
| DPGA-Former | ISPRS Potsdam | **73.39** |
| DPGA-Former | LCAI | **87.10** |
| DPGA-Former | GID | **80.08** |

## Datasets

We evaluate our method on the following three public benchmark datasets:

- **ISPRS Potsdam:** [Dataset Link](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx)
- **LandCover.ai (LCAI):** [Dataset Link](https://landcover.ai.linuxpolska.com)
- **GID (Gaofen Image Dataset):** [Dataset Reference](https://doi.org/10.1109/IGARSS.2018.8518389)

### Dataset Splits

The train/validation/test splits used in our experiments are available on IEEE DataPort.

[![DOI](https://img.shields.io/badge/DOI-10.21227/16vt--b551-blue)](https://doi.org/10.21227/16vt-b551)

**DOI:** [10.21227/16vt-b551](https://doi.org/10.21227/16vt-b551)

The package includes:

- **ISPRS Potsdam:** Official image partitions (17 training, 7 validation, and 14 test images).
- **LCAI:** Random 5:2:3 splits generated using a fixed random seed.
- **GID:** Scene-level 5:2:3 splits.

### Dataset Directory Structure

Organize the datasets as follows:

```text
data/
├── potsdam/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
├── LCAI/
│   ├── images/
│   ├── labels/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
└── GID/
    ├── images/
    ├── labels/
    ├── train_5pd.txt
    ├── val_5pd.txt
    └── test_5pd.txt
```

**Note:** The current Potsdam experimental configuration uses `data/potsdam/test/` as the validation dataset and for final testing. This arrangement is intentional and should be preserved when reproducing the reported experiments.

## Installation

**Step 0: Create a Conda environment.**

```bash
conda create -n DPGAFormer python=3.8.20 -y
conda activate DPGAFormer
```

**Step 1: Install PyTorch and Torchvision with CUDA 11.8.**

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
--index-url https://download.pytorch.org/whl/cu118
```

**Step 2: Install project requirements.**

```bash
pip install -r requirements.txt
```

**Step 3: Install additional dependencies.**

```bash
pip install catalyst==20.9 addict==2.4.0 tensorboardX==2.6.2.2
```

## Pretrained Backbone

DPGA-Former uses pretrained STSeg-Base backbone weights.

Download the pretrained checkpoint:

**[STSeg-Base Pretrained Weights]**((https://drive.google.com/file/d/1o3kUOJul5GxsPwMbMsIgeGeWP93BBWay/view?usp=drive_link))

Place the downloaded checkpoint in:

```text
pretrain_weights/stseg_base.pth
```

## Training

The `-c` argument specifies the configuration file. Use the corresponding configuration to train DPGA-Former on each dataset.

**ISPRS Potsdam**

```bash
CUDA_VISIBLE_DEVICES=1 python train_supervision_potsdam.py \
-c config/potsdam/dpgaformer.py
```

**LCAI**

```bash
CUDA_VISIBLE_DEVICES=1 python train_supervision_lcai.py \
-c config/lcai/dpgaformer.py
```

**GID**

```bash
CUDA_VISIBLE_DEVICES=1 python train_supervision_gid.py \
-c config/gid/dpgaformer.py
```

## Testing

The testing scripts load the corresponding trained checkpoint from `model_weights/`.

The `-c` argument specifies the configuration file.

The `-o` argument specifies the prediction output directory.

The `-g` argument specifies the GPU index visible to the process.

The `--rgb` option enables RGB-colored prediction masks.

**ISPRS Potsdam**

```bash
CUDA_VISIBLE_DEVICES=1 python test_potsdam.py \
-c config/potsdam/dpgaformer.py -g 0
```

**LCAI**

```bash
CUDA_VISIBLE_DEVICES=1 python test_lcai.py \
-c config/lcai/dpgaformer.py -g 0
```

**GID**

```bash
CUDA_VISIBLE_DEVICES=1 python test_gid.py \
-c config/gid/dpgaformer.py -g 0
```

By default, testing is performed without test-time augmentation.

## Evaluation

After generating the segmentation predictions, use the corresponding evaluation script to calculate the segmentation metrics.

**ISPRS Potsdam**

```bash
python get_result_potsdam.py \
--pred_dir fig_results/potsdam/dpgaformer/pre_p \
--gt_dir data/potsdam/test/masks/
```

**LCAI**

```bash
python get_result_lcai.py \
--pred_dir fig_results/lcai/dpgaformer/pre_p \
--gt_dir data/LCAI/labels/
```

**GID**

```bash
python get_result_gid.py \
--pred_dir fig_results/gid/dpgaformer/pre_p \
--gt_dir data/GID/labels/
```

The evaluation scripts calculate mIoU, mF1, mPA, fwIoU, pixel accuracy, boundary F-score, and trimap IoU.

## Pretrained Models

Due to GitHub file-size limitations, the trained DPGA-Former checkpoints are available through Google Drive.

| Dataset | mIoU | Download |
|:---:|---:|:---:|
| ISPRS Potsdam | 73.39% | [Download](https://drive.google.com/drive/folders/1QjOdKAHMNw4yy02c3W85P67oDwmmP3ET?usp=sharing) |
| LCAI | 87.10% | [Download](https://drive.google.com/drive/folders/1cQcF8AC5sDvh1CDroVEze6J-q5cqBXOP?usp=sharing) |
| GID | 80.08% | [Download](https://drive.google.com/drive/folders/1j_iVx9PXBMnjB9AEAWfSi6kdwt4Ew9aw?usp=drive_link) |

Place the downloaded `.ckpt` files in their respective directories:

```text
model_weights/
├── potsdam/
│   └── dpgaformer-512crop-ms-e200/
│       └── dpgaformer-512crop-ms-e200.ckpt
├── lcai/
│   └── dpgaformer-512crop-lcai-e200/
│       └── dpgaformer-512crop-lcai-e200.ckpt
└── gid/
    └── dpgaformer-512crop-gid-e200/
        └── dpgaformer-512crop-gid-e200.ckpt
```

## Acknowledgement

Our implementation builds upon the GeoSeg framework and related open-source research in remote sensing semantic segmentation.

We acknowledge the authors and maintainers of the ISPRS Potsdam, LandCover.ai, and GID datasets.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{DPGAFormer,
  title={Detail-Preserving and Geometry-Aware Semantic Segmentation for VHR Remote Sensing Imagery},
  author={Insert Authors},
  journal={Insert Journal Name},
  year={Insert Year}
}
```

DPGA-Former

Detail-Preserving and Geometry-Aware Semantic Segmentation for VHR Remote Sensing Imagery

DPGA-Former is a detail-preserving and geometry-aware semantic segmentation framework designed for Very High Resolution (VHR) remote sensing imagery.

The framework is evaluated on three benchmark datasets: ISPRS Potsdam, LandCover.ai (LCAI), and GID.

This repository provides the implementation, training configurations, testing scripts, evaluation tools, and dataset split files required to reproduce the experiments.

1. Datasets

We evaluate DPGA-Former on three remote sensing semantic segmentation benchmarks.

Dataset

Description

Number of Classes

ISPRS Potsdam

5 cm GSD aerial imagery

6

LCAI (LandCover.ai)

Aerial imagery

5

GID

Gaofen-2 satellite imagery

6

Dataset Sources

ISPRS Potsdam: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx

LandCover.ai: https://landcover.ai.linuxpolska.com

GID: https://doi.org/10.1109/IGARSS.2018.8518389

Please obtain the original datasets from their respective sources and organize them according to the directory structures described below.

2. Dataset Splits

The dataset splits used in our experiments are available through IEEE DataPort.



DOI: https://doi.org/10.21227/16vt-b551

The dataset split package includes:

ISPRS Potsdam: Official train, validation, and test image partitions (17, 7, and 14 original images, respectively).

LCAI: Random 5:2:3 train/validation/test splits generated using a fixed random seed.

GID: Scene-level 5:2:3 train/validation/test splits.

The corresponding split files are:

data/
├── LCAI/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
└── GID/
    ├── train_5pd.txt
    ├── val_5pd.txt
    └── test_5pd.txt

For Potsdam, the image partitions are organized into separate directories.

Important Potsdam implementation detail: In the current verified configuration, data/potsdam/test/ is assigned to the validation dataset and used for inference during testing. This is intentional in the released experimental setup. The separate data/potsdam/val/ directory is not automatically selected by the current configuration. Preserve this arrangement when reproducing the existing experiments.

3. Repository Structure

The repository is organized as follows:

DPGA-Former/
│
├── config/
│   ├── potsdam/
│   │   └── dpgaformer.py
│   ├── lcai/
│   │   └── dpgaformer.py
│   └── gid/
│       └── dpgaformer.py
│
├── data/
│   ├── potsdam/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── test/
│   │       ├── images/
│   │       └── masks/
│   │
│   ├── LCAI/
│   │   ├── images/
│   │   ├── labels/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   │
│   └── GID/
│       ├── images/
│       ├── labels/
│       ├── train_5pd.txt
│       ├── val_5pd.txt
│       └── test_5pd.txt
│
├── geoseg/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   └── ...
│
├── tools/
│   ├── cfg.py
│   ├── metric.py
│   └── potsdam_patch_split.py
│
├── pretrain_weights/
│   └── stseg_base.pth
│
├── model_weights/
│   ├── potsdam/
│   ├── lcai/
│   └── gid/
│
├── Figures/
│
├── train_supervision_potsdam.py
├── train_supervision_lcai.py
├── train_supervision_gid.py
│
├── test_potsdam.py
├── test_lcai.py
├── test_gid.py
│
├── get_result_potsdam.py
├── get_result_lcai.py
├── get_result_gid.py
│
├── requirements.txt
├── .gitignore
└── README.md

Dataset files, pretrained weights, and trained checkpoints are distributed separately because of their large storage requirements.

4. Environment Setup

The experiments were developed and verified using the following environment.

Component

Version

Operating System

Ubuntu Linux

Python

3.8.20

PyTorch

2.4.1

Torchvision

0.19.1

CUDA Runtime

11.8

PyTorch Lightning

2.4.0

Albumentations

1.4.18

Catalyst

20.09

timm

1.0.15

NumPy

1.24.4

OpenCV

4.11.0

TorchMetrics

1.5.2

einops

0.8.1

The experiments were conducted using NVIDIA GeForce RTX 3090 GPUs.

4.1 Create the Conda Environment

conda create -n DPGAFormer python=3.8.20 -y

conda activate DPGAFormer

4.2 Install PyTorch with CUDA 11.8

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
--index-url https://download.pytorch.org/whl/cu118

Verify the installation:

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

4.3 Install Project Dependencies

pip install -r requirements.txt

Install the additional dependencies used by the verified environment:

pip install catalyst==20.9 addict==2.4.0 tensorboardX==2.6.2.2

Check the installed packages:

pip check

Note: The dependency versions above reflect the verified experimental environment. CUDA availability also depends on a compatible NVIDIA driver.

5. Pretrained Backbone

DPGA-Former uses pretrained STSeg-Base backbone weights for initialization.

The required checkpoint is:

pretrain_weights/
└── stseg_base.pth

Checkpoint size: Approximately 464 MB.

Download the pretrained backbone:

STSeg-Base Pretrained Weights – Google Drive

After downloading, place the checkpoint inside:

DPGA-Former/pretrain_weights/stseg_base.pth

The training configurations already reference this relative path:

pretrained=True,
weight_path='pretrain_weights/stseg_base.pth'

Make sure the pretrained checkpoint is available before starting training.

6. Dataset Preparation

Download the three benchmark datasets and the corresponding split files.

Organize them as follows.

6.1 ISPRS Potsdam

data/potsdam/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/

6.2 LandCover.ai (LCAI)

data/LCAI/
├── images/
├── labels/
├── train.txt
├── val.txt
└── test.txt

6.3 GID

data/GID/
├── images/
├── labels/
├── train_5pd.txt
├── val_5pd.txt
└── test_5pd.txt

Ensure that image and annotation filenames correspond to the identifiers specified in the respective split files.

7. Training

All training configurations are provided in the config/ directory.

The configurations specify the model architecture, optimization settings, loss functions, data loaders, and checkpoint locations.

The verified training setup uses:

Maximum epochs: 200

Training batch size: 8

Validation batch size: 8

Optimizer: AdamW with Lookahead

Initial learning rate: 1e-4

Backbone learning rate: 3e-5

Weight decay: 5e-4

Learning rate scheduler: CosineAnnealingWarmRestarts

Automatic Mixed Precision (16-bit)

7.1 Train on ISPRS Potsdam

CUDA_VISIBLE_DEVICES=0 python train_supervision_potsdam.py \
-c config/potsdam/dpgaformer.py

7.2 Train on LCAI

CUDA_VISIBLE_DEVICES=0 python train_supervision_lcai.py \
-c config/lcai/dpgaformer.py

7.3 Train on GID

CUDA_VISIBLE_DEVICES=0 python train_supervision_gid.py \
-c config/gid/dpgaformer.py

To use a different GPU, change CUDA_VISIBLE_DEVICES accordingly.

For example, CUDA_VISIBLE_DEVICES=1 selects the second physical GPU. The process then accesses that GPU as cuda:0.

Checkpoint Locations

The training scripts save checkpoints in the following directories:

model_weights/
├── potsdam/
│   └── dpgaformer-512crop-ms-e200/
├── lcai/
│   └── dpgaformer-512crop-lcai-e200/
└── gid/
    └── dpgaformer-512crop-gid-e200/

8. Testing

Pretrained DPGA-Former checkpoints are provided for direct evaluation without retraining.

Download the appropriate checkpoint and place it in the corresponding directory under model_weights/.

The testing scripts automatically load the checkpoint specified by the dataset configuration.

8.1 Test on ISPRS Potsdam

CUDA_VISIBLE_DEVICES=0 python test_potsdam.py \
-c config/potsdam/dpgaformer.py \
-g 0

8.2 Test on LCAI

CUDA_VISIBLE_DEVICES=0 python test_lcai.py \
-c config/lcai/dpgaformer.py \
-g 0

8.3 Test on GID

CUDA_VISIBLE_DEVICES=0 python test_gid.py \
-c config/gid/dpgaformer.py \
-g 0

By default, testing is performed without test-time augmentation.

The scripts save predicted segmentation masks in:

fig_results/
├── potsdam/
│   └── dpgaformer/
│       └── pre_p/
├── lcai/
│   └── dpgaformer/
│       └── pre_p/
└── gid/
    └── dpgaformer/
        └── pre_p/

The --rgb option can be used to save RGB-colored segmentation predictions instead of indexed prediction masks.

Additional testing options are available through:

python test_potsdam.py --help
python test_lcai.py --help
python test_gid.py --help

9. Evaluation

After generating the predicted segmentation masks, use the corresponding evaluation script to calculate the segmentation metrics.

The evaluation includes:

Mean Intersection over Union (mIoU)

Mean F1-score (mF1)

Mean Pixel Accuracy (mPA)

Frequency-weighted IoU (fwIoU)

Pixel Accuracy (pAcc)

Boundary F-score (BFScore)

Trimap IoU

9.1 Evaluate ISPRS Potsdam

python get_result_potsdam.py \
--pred_dir fig_results/potsdam/dpgaformer/pre_p \
--gt_dir data/potsdam/test/masks/

9.2 Evaluate LCAI

python get_result_lcai.py \
--pred_dir fig_results/lcai/dpgaformer/pre_p \
--gt_dir data/LCAI/labels/

9.3 Evaluate GID

python get_result_gid.py \
--pred_dir fig_results/gid/dpgaformer/pre_p \
--gt_dir data/GID/labels/

The evaluation scripts also support optional boundary visualizations.

For example:

python get_result_potsdam.py \
--pred_dir fig_results/potsdam/dpgaformer/pre_p \
--gt_dir data/potsdam/test/masks/ \
--vis_id IMAGE_ID \
--vis_out_dir fig_results/potsdam/boundary_visualizations

Use --help for the complete set of evaluation options.

10. Experimental Results

The following results correspond to the reported experimental performance of DPGA-Former.

Dataset

mIoU

ISPRS Potsdam

73.39%

LCAI

87.10%

GID

80.08%

Potsdam Reproduction Verification

The released Potsdam checkpoint was independently evaluated using the provided testing and evaluation scripts.

Metric

Reproduced Result

mIoU

73.40%

mF1

83.28%

mPA

81.94%

fwIoU

78.88%

pAcc

88.02%

mBFScore @ 1px

39.20%

mBFScore @ 2px

49.63%

mBFScore @ 3px

55.79%

The reproduced Potsdam mIoU differs from the reported manuscript value by 0.01 percentage point.

The LCAI and GID checkpoints have also been verified using their corresponding testing and evaluation scripts.

11. Pretrained DPGA-Former Checkpoints

Due to GitHub file-size limitations, trained checkpoints are distributed through Google Drive.

Dataset

Reported mIoU

Checkpoint Download

ISPRS Potsdam

73.39%

Download

LCAI

87.10%

Download

GID

80.08%

Download

Place the downloaded checkpoint files in the corresponding directories:

model_weights/
├── potsdam/
│   └── dpgaformer-512crop-ms-e200/
│       └── dpgaformer-512crop-ms-e200.ckpt
│
├── lcai/
│   └── dpgaformer-512crop-lcai-e200/
│       └── dpgaformer-512crop-lcai-e200.ckpt
│
└── gid/
    └── dpgaformer-512crop-gid-e200/
        └── dpgaformer-512crop-gid-e200.ckpt

The directory names and checkpoint filenames must match the corresponding configuration files.

12. Reproducibility

To reproduce the experiments:

Set up the Python environment and install the required dependencies.

Download and prepare the three datasets.

Obtain the dataset split files from IEEE DataPort.

Download the STSeg-Base pretrained backbone.

Run the corresponding training script.

Alternatively, download the released DPGA-Former checkpoint.

Generate segmentation predictions using the testing scripts.

Calculate the evaluation metrics using the provided evaluation scripts.

All dataset paths and checkpoint paths in the released configurations are relative to the repository root.

13. Citation

If you find this work useful in your research, please cite our paper:

[Insert the final published manuscript citation here.]

@article{DPGAFormer,
  title  = {Detail-Preserving and Geometry-Aware Semantic Segmentation for VHR Remote Sensing Imagery},
  author = {Insert Authors},
  journal = {Insert Journal or Conference},
  year   = {Insert Year}
}

14. Acknowledgements

We acknowledge the authors and maintainers of the ISPRS Potsdam, LandCover.ai, and GID datasets, as well as the open-source research projects and libraries used in this work.

Repository: DPGA-Former
Code: Training, testing, and evaluation implementations are provided in this repository.

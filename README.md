# DPGA-Former: Detail-Preserving and Geometry-Aware Semantic Segmentation for VHR Remote Sensing Imagery

DPGA-Former is a detail-preserving and geometry-aware semantic segmentation framework designed for Very High Resolution (VHR) remote sensing imagery. 

The framework is evaluated on three benchmark datasets: ISPRS Potsdam, LandCover.ai (LCAI), and GID.

This repository provides the implementation, training configurations, testing scripts, evaluation tools, and dataset split files required to reproduce the experiments.

---

## 1. Datasets

We evaluate DPGA-Former on three remote sensing semantic segmentation benchmarks.

| Dataset | Description | Number of Classes |
| :--- | :--- | :--- |
| **ISPRS Potsdam** | 5 cm GSD aerial imagery | 6 |
| **LCAI (LandCover.ai)** | Aerial imagery | 5 |
| **GID** | Gaofen-2 satellite imagery | 6 |

**Dataset Sources:**
*   **ISPRS Potsdam**: [https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx)
*   **LandCover.ai**: [https://landcover.ai.linuxpolska.com](https://landcover.ai.linuxpolska.com)
*   **GID**: [https://doi.org/10.1109/IGARSS.2018.8518389](https://doi.org/10.1109/IGARSS.2018.8518389)

Please obtain the original datasets from their respective sources and organize them according to the directory structures described below.

---

## 2. Dataset Splits

The dataset splits used in our experiments are available through IEEE DataPort.

**DOI**: [https://doi.org/10.21227/16vt-b551](https://doi.org/10.21227/16vt-b551)

The dataset split package includes:
*   **ISPRS Potsdam**: Official train, validation, and test image partitions (17, 7, and 14 original images, respectively).
*   **LCAI**: Random 5:2:3 train/validation/test splits generated using a fixed random seed.
*   **GID**: Scene-level 5:2:3 train/validation/test splits.

The corresponding split files are organized as follows:

```text
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

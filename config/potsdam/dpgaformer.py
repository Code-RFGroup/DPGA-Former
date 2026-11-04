from torch.utils.data import DataLoader
import torch
from catalyst.contrib.nn import Lookahead
from catalyst import utils

from geoseg.datasets.potsdam_dataset import *   # CLASSES, train_dataset, val_dataset
from geoseg.models.DPGAFormer import dpga_former

# -------- Training hparams --------
max_epoch = 200
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8

# Optim / LR
lr = 1e-4
weight_decay = 5e-4
backbone_lr = 3e-5
backbone_weight_decay = 5e-4

num_classes = len(CLASSES)
classes = CLASSES

# -------- Logging / ckpt --------
weights_name = "dpgaformer-512crop-ms-e200"
weights_path = f"model_weights/potsdam/{weights_name}"
test_weights_name = weights_name
log_name = f"potsdam/{weights_name}"

monitor = "val_OA"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = [0]

# -------- Model --------
USE_HILO = True
USE_AXIAL_AT = "b3"
RETURN_AUX = True

net = dpga_former(
    num_classes=num_classes,
    decoder_channels=256,
    use_hilo=USE_HILO,
    use_axial_at=USE_AXIAL_AT,
    return_aux=RETURN_AUX,
    pretrained=True,
    weight_path='pretrain_weights/stseg_base.pth'
)

# -------- Loss --------
from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0, 1.0
)

use_aux_loss = True        
lambda_bd   = 0.2          # <-- add: boundary BCE weight
lambda_dist = 0.1          # <-- add: distance SmoothL1 weight

# -------- Dataloaders --------
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

# -------- Optim / Sched --------
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

import os
import random
from pathlib import Path
import argparse
import time  # <-- added

import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from tools.cfg import py2cfg
from tools.metric import Evaluator


# -----------------------------
# Repro
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (turn off bench to avoid nondeterministic algo choice)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# CLI
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path",
        default="config/lcai/dpgaformer.py",
        type=Path,
        help="Path to the config.")
    parser.add_argument("-s", "--strategy", default="auto", type=str)
    return parser.parse_args()


# -----------------------------
# Lightning Module
# -----------------------------
class Supervision_Train(pl.LightningModule):
    """
    Trainer that understands optional BD Head:
      - If model returns just logits: trains as usual.
      - If model returns (seg_logits, b_map, d_map) and config.use_aux_loss=True:
          adds λ_bd * BCE(b_map, boundary_target) + λ_dist * L1(d_map, distance_target).
    Metrics (mIoU/F1/OA) are always computed from the main segmentation logits only.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

        # Aux-loss control (safe defaults if not in config)
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))
        self.lambda_bd = float(getattr(config, "lambda_bd", 0.0))
        self.lambda_dist = float(getattr(config, "lambda_dist", 0.0))

        # Aux loss functions
        self.bd_loss_fn = nn.BCEWithLogitsLoss()
        self.dist_loss_fn = nn.SmoothL1Loss()

    # -------- helpers --------
    @staticmethod
    def _seg_logits_from(prediction):
        """Return the main segmentation logits tensor regardless of RETURN_AUX."""
        return prediction[0] if isinstance(prediction, (tuple, list)) else prediction

    @staticmethod
    def _boundary_from_labels(lbl: torch.Tensor) -> torch.Tensor:
        """
        Build 0/1 boundary targets from class labels.
        lbl: (B,H,W) int64 -> (B,1,H,W) float32 in {0,1}
        """
        up = (lbl[:, 1:, :] != lbl[:, :-1, :])
        dn = (lbl[:, :-1, :] != lbl[:, 1:, :])
        lf = (lbl[:, :, 1:] != lbl[:, :, :-1])
        rt = (lbl[:, :, :-1] != lbl[:, :, 1:])

        B, H, W = lbl.shape
        edge = torch.zeros((B, H, W), device=lbl.device, dtype=torch.bool)
        edge[:, 1:, :] |= up
        edge[:, :-1, :] |= dn
        edge[:, :, 1:] |= lf
        edge[:, :, :-1] |= rt
        return edge.float().unsqueeze(1)  # (B,1,H,W)

    @staticmethod
    def _distance_from_labels(lbl: torch.Tensor) -> torch.Tensor:
        """
        Unsigned distance to boundary, normalized to [0,1].
        lbl: (B,H,W) int64 -> (B,1,H,W) float32
        """
        b, h, w = lbl.shape
        out = []
        bou = Supervision_Train._boundary_from_labels(lbl).squeeze(1)  # (B,H,W)
        for i in range(b):
            bou_np = bou[i].detach().cpu().numpy().astype(np.uint8)
            inv = (1 - bou_np).astype(np.uint8)
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
            if dist.max() > 0:
                dist /= (dist.max() + 1e-6)
            out.append(torch.from_numpy(dist))
        out = torch.stack(out, 0).unsqueeze(1).to(lbl.device)
        return out  # (B,1,H,W)

    def _compute_full_loss(self, prediction, mask):
        """
        Main seg loss is ALWAYS computed on the primary segmentation logits tensor.
        If aux is enabled and maps are present, add BD/DIST losses.
        """
        # 1) base seg loss on logits only (avoid passing a tuple to JointLoss)
        seg_logits = self._seg_logits_from(prediction)  # (B,C,H,W)
        total = self.loss(seg_logits, mask)

        # 2) optional aux terms
        if self.use_aux_loss and isinstance(prediction, (tuple, list)):
            b_map = prediction[1] if len(prediction) > 1 else None
            d_map = prediction[2] if len(prediction) > 2 else None

            if (b_map is not None) and (self.lambda_bd > 0):
                bd_t = self._boundary_from_labels(mask)  # (B,1,H,W)
                total = total + self.lambda_bd * self.bd_loss_fn(b_map, bd_t)

            if (d_map is not None) and (self.lambda_dist > 0):
                dist_t = self._distance_from_labels(mask)  # (B,1,H,W)
                total = total + self.lambda_dist * self.dist_loss_fn(d_map, dist_t)

        return total

    # -------- Lightning plumbing --------
    def forward(self, x):
        # Only net is used in the prediction/inference
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img = batch["img"].to(self.device)
        mask = batch["gt_semantic_seg"].to(self.device)

        prediction = self.net(img)
        loss = self._compute_full_loss(prediction, mask)

        # metrics from main seg logits only
        seg_logits = self._seg_logits_from(prediction)
        pre_mask = torch.softmax(seg_logits, dim=1).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss": loss}

    def on_train_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())

        print("train:", {"mIoU": mIoU, "F1": F1, "OA": OA})
        self.metrics_train.reset()
        self.log_dict({"train_mIoU": mIoU, "train_F1": F1, "train_OA": OA}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img = batch["img"].to(self.device)
        mask = batch["gt_semantic_seg"].to(self.device)

        prediction = self.forward(img)
        loss_val = self._compute_full_loss(prediction, mask)

        seg_logits = self._seg_logits_from(prediction)
        pre_mask = torch.softmax(seg_logits, dim=1).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if "vaihingen" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "potsdam" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1())
        elif "whubuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "massbuilding" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif "cropland" in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())

        print("val:", {"mIoU": mIoU, "F1": F1, "OA": OA})
        self.metrics_val.reset()
        self.log_dict({"val_mIoU": mIoU, "val_F1": F1, "val_OA": OA}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


# -----------------------------
# Extra: Params, GFLOPs, FPS
# -----------------------------
def count_params_millions(module: torch.nn.Module, trainable_only: bool = True) -> float:
    if trainable_only:
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in module.parameters())
    return n / 1e6


class _SegOnlyWrapper(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
    def forward(self, x):
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def make_seg_only(module: nn.Module) -> nn.Module:
    return _SegOnlyWrapper(module)


def compute_gflops(module: torch.nn.Module, input_shape=(1, 3, 512, 512)):
    try:
        from thop import profile
    except Exception:
        return None, input_shape  # thop not installed

    device = torch.device("cpu")
    safe = make_seg_only(module).to(device).eval()
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        macs, _ = profile(safe, inputs=(dummy,), verbose=False)
    gflops = (macs / 1e9) * 2.0
    return float(gflops), input_shape


def compute_fps(module: torch.nn.Module,
                input_shape=(1, 3, 512, 512),
                warmup_iters: int = 10,
                measure_iters: int = 50,
                use_amp: bool = True) -> float:
    safe = make_seg_only(module).eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        safe = safe.to(device)
        dummy = torch.zeros(input_shape, device=device)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            for _ in range(warmup_iters):
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = safe(dummy)
                else:
                    _ = safe(dummy)
        torch.cuda.synchronize()

        elapsed_ms = 0.0
        with torch.inference_mode():
            for _ in range(measure_iters):
                starter.record()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = safe(dummy)
                else:
                    _ = safe(dummy)
                ender.record()
                torch.cuda.synchronize()
                elapsed_ms += starter.elapsed_time(ender)

        avg_ms = elapsed_ms / measure_iters
        return 0.0 if avg_ms <= 0 else 1000.0 / avg_ms
    else:
        device = torch.device("cpu")
        safe = safe.to(device)
        dummy = torch.zeros(input_shape, device=device)
        with torch.inference_mode():
            for _ in range(warmup_iters):
                _ = safe(dummy)
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(measure_iters):
                _ = safe(dummy)
        end = time.perf_counter()
        total_s = end - start
        return 0.0 if total_s <= 0 else measure_iters / total_s


# -----------------------------
# training
# -----------------------------
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    # (Optional) better Tensor Core utilization on Ampere+
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        save_last=config.save_last,
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name
    )

    early_stop_callback = EarlyStopping(
        monitor=config.monitor,
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode="max"
    )
    logger = CSVLogger("lightning_logs", name=config.log_name)

    model = Supervision_Train(config)
    if getattr(config, "pretrained_ckpt_path", None):
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    # ---- Added: Params, GFLOPs, FPS prints ----
    total_M = count_params_millions(model.net, trainable_only=True)
    print(f"[PARAM] Total trainable params (M): {total_M:.2f}")

    try:
        gflops_256, _ = compute_gflops(model.net, input_shape=(1, 3, 256, 256))
        if gflops_256 is not None:
            print(f"[GFLOPs] per image @ 256x256: {gflops_256:.2f}")
        else:
            print("[GFLOPs] thop not installed; run `pip install thop`.")
    except Exception as e:
        print(f"[GFLOPs] Warning @256x256: {e}")

    try:
        gflops_512, _ = compute_gflops(model.net, input_shape=(1, 3, 512, 512))
        if gflops_512 is not None:
            print(f"[GFLOPs] per image @ 512x512: {gflops_512:.2f}")
        else:
            print("[GFLOPs] thop not installed; run `pip install thop`.")
    except Exception as e:
        print(f"[GFLOPs] Warning @512x512: {e}")

    try:
        fps_256 = compute_fps(model.net, input_shape=(1, 3, 256, 256), warmup_iters=10, measure_iters=50, use_amp=True)
        print(f"[FPS] @ 256x256: {fps_256:.2f}")
    except Exception as e:
        print(f"[FPS] Warning @256x256: {e}")

    try:
        fps_512 = compute_fps(model.net, input_shape=(1, 3, 512, 512), warmup_iters=10, measure_iters=50, use_amp=True)
        print(f"[FPS] @ 512x512: {fps_512:.2f}")
    except Exception as e:
        print(f"[FPS] Warning @512x512: {e}")
    # -------------------------------------------

    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator="auto",
        precision="16-mixed",
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
        strategy=args.strategy,
        logger=logger
    )
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    main()

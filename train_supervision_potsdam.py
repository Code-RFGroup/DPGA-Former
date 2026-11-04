import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from tools.cfg import py2cfg
from tools.metric import Evaluator


# =========================
# Utility: count parameters
# =========================
def count_params_millions(module: torch.nn.Module, trainable_only: bool = True) -> float:
    """
    Count parameters in millions.
    If trainable_only=True, counts only requires_grad parameters.
    """
    if trainable_only:
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        n = sum(p.numel() for p in module.parameters())
    return n / 1e6


# ==========================================
# Utility: wrap nets that may return tuples
# ==========================================
class _SegOnlyWrapper(nn.Module):
    """
    Ensures forward() returns a single tensor of logits (B,C,H,W),
    even if the original model returns (seg, boundary, distance).
    """
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


# ==========================
# GFLOPs (via thop profile)
# ==========================
def compute_gflops(module: torch.nn.Module, input_shape=(1, 3, 512, 512)):
    """
    Returns GFLOPs for a single image at the given shape.
    Uses thop to get MACs and converts: GFLOPs ≈ 2 * GMacs.
    If thop is unavailable, returns (None, shape).
    """
    try:
        from thop import profile
    except Exception:
        return None, input_shape  # thop not installed

    device = torch.device("cpu")  # keep FLOPs on CPU for consistency
    safe = make_seg_only(module).to(device).eval()
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        macs, _ = profile(safe, inputs=(dummy,), verbose=False)
    gflops = (macs / 1e9) * 2.0  # MACs -> FLOPs
    return float(gflops), input_shape


# ==========================================
# FPS Benchmark (GPU if available; else CPU)
# ==========================================
def compute_fps(module: torch.nn.Module,
                input_shape=(1, 3, 512, 512),
                warmup_iters: int = 10,
                measure_iters: int = 50,
                use_amp: bool = True) -> float:
    """
    Measures FPS (images/second) for forward() at a fixed input shape.
    - Uses CUDA events if CUDA is available; otherwise uses time.perf_counter()
    - AMP is enabled on CUDA by default (use_amp=True) to reflect mixed precision inference
    Returns: FPS (float)
    """
    safe = make_seg_only(module).eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        safe = safe.to(device)
        dummy = torch.zeros(input_shape, device=device)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # warm-up
        with torch.inference_mode():
            for _ in range(warmup_iters):
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = safe(dummy)
                else:
                    _ = safe(dummy)
        torch.cuda.synchronize()

        # measure
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
        if avg_ms <= 0:
            return 0.0
        return 1000.0 / avg_ms  # images per second

    else:
        # CPU timing
        device = torch.device("cpu")
        safe = safe.to(device)
        dummy = torch.zeros(input_shape, device=device)

        # warm-up
        with torch.inference_mode():
            for _ in range(warmup_iters):
                _ = safe(dummy)

        # measure
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(measure_iters):
                _ = safe(dummy)
        end = time.perf_counter()

        total_s = end - start
        if total_s <= 0:
            return 0.0
        return measure_iters / total_s


# ==========================================
# Memory Benchmark (GPU only)
# ==========================================
def measure_memory_gb(module: torch.nn.Module,
                      input_shape=(1, 3, 512, 512),
                      use_amp: bool = True) -> float:
    """
    Measures peak GPU memory usage (GB) during a single forward pass.
    Returns: Peak memory in GB.
    """
    if not torch.cuda.is_available():
        print("[Memory] CUDA not available; skipping memory measurement.")
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    safe = make_seg_only(module).to("cuda").eval()
    dummy = torch.zeros(input_shape, device="cuda")

    with torch.inference_mode():
        if use_amp:
            with torch.cuda.amp.autocast():
                _ = safe(dummy)
        else:
            _ = safe(dummy)

    torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)  # Convert to GB
    return peak_memory_gb


# ==========================================
# Time Benchmark (GPU if available; else CPU)
# ==========================================
def measure_time_ms(module: torch.nn.Module,
                    input_shape=(1, 3, 512, 512),
                    warmup_iters: int = 10,
                    measure_iters: int = 50,
                    use_amp: bool = True) -> float:
    """
    Measures average inference time per image (ms) for a single forward pass.
    Returns: Average time in milliseconds.
    """
    safe = make_seg_only(module).eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        safe = safe.to(device)
        dummy = torch.zeros(input_shape, device=device)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # warm-up
        with torch.inference_mode():
            for _ in range(warmup_iters):
                if use_amp:
                    with torch.cuda.amp.autocast():
                        _ = safe(dummy)
                else:
                    _ = safe(dummy)
        torch.cuda.synchronize()

        # measure
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
        return avg_ms

    else:
        # CPU timing
        device = torch.device("cpu")
        safe = safe.to(device)
        dummy = torch.zeros(input_shape, device=device)

        # warm-up
        with torch.inference_mode():
            for _ in range(warmup_iters):
                _ = safe(dummy)

        # measure
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(measure_iters):
                _ = safe(dummy)
        end = time.perf_counter()

        total_s = end - start
        avg_s = total_s / measure_iters
        return avg_s * 1000.0  # Convert to ms


def seed_everything(seed: int):
    # Python & NumPy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # disable dynamic algo selection (determinism)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", default='config/potsdam/dpgaformer.py',
        type=Path, help="Path to the config.")
    parser.add_argument("-s", "--strategy", default='auto', type=str)
    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

        # Aux-loss control (safe defaults if not in config)
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))
        self.lambda_bd    = float(getattr(config, "lambda_bd", 0.0))
        self.lambda_dist  = float(getattr(config, "lambda_dist", 0.0))

        # Aux loss functions
        self.bd_loss_fn   = nn.BCEWithLogitsLoss()
        self.dist_loss_fn = nn.SmoothL1Loss()

    # --- helpers ---
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
        edge[:, 1:, :]  |= up
        edge[:, :-1, :] |= dn
        edge[:, :, 1:]  |= lf
        edge[:, :, :-1] |= rt

        return edge.float().unsqueeze(1)

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
        return out

    def _compute_full_loss(self, prediction, mask):
        """
        Main seg loss is ALWAYS computed on the primary segmentation logits tensor.
        If aux is enabled and maps are present, add BD/DIST losses.
        """
        seg_logits = self._seg_logits_from(prediction)   # (B,C,H,W)
        total = self.loss(seg_logits, mask)

        if self.use_aux_loss and isinstance(prediction, (tuple, list)):
            b_map = prediction[1] if len(prediction) > 1 else None
            d_map = prediction[2] if len(prediction) > 2 else None

            if (b_map is not None) and (self.lambda_bd > 0):
                bd_t = self._boundary_from_labels(mask)          # (B,1,H,W)
                total = total + self.lambda_bd * self.bd_loss_fn(b_map, bd_t)

            if (d_map is not None) and (self.lambda_dist > 0):
                dist_t = self._distance_from_labels(mask)        # (B,1,H,W)
                total = total + self.lambda_dist * self.dist_loss_fn(d_map, dist_t)

        return total

    # --- Lightning plumbing ---
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img = batch['img'].to(self.device)
        mask = batch['gt_semantic_seg'].to(self.device)

        prediction = self.net(img)
        loss = self._compute_full_loss(prediction, mask)

        seg_logits = self._seg_logits_from(prediction)
        pre_mask = torch.softmax(seg_logits, dim=1).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss": loss}

    def on_train_epoch_end(self):
        if 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1   = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())

        print('train:', {'mIoU': mIoU, 'F1': F1, 'OA': OA})
        self.metrics_train.reset()
        self.log_dict({'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img = batch['img'].to(self.device)
        mask = batch['gt_semantic_seg'].to(self.device)

        prediction = self.forward(img)
        loss_val = self._compute_full_loss(prediction, mask)

        seg_logits = self._seg_logits_from(prediction)
        pre_mask = torch.softmax(seg_logits, dim=1).argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        if 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_val.F1())
        elif 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1   = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1   = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())

        print('val:', {'mIoU': mIoU, 'F1': F1, 'OA': OA})
        self.metrics_val.reset()
        self.log_dict({'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


def main():
    args = get_args()

    # Determinism before building datasets/loaders
    seed_everything(42)

    # Use tensor cores effectively on Ampere+ (e.g., RTX 3090)
    torch.set_float32_matmul_precision('high')

    # build config (creates datasets/loaders with deterministic RNG state)
    config = py2cfg(args.config_path)

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
        mode='max'
    )

    logger = CSVLogger('lightning_logs', name=config.log_name)

    model = Supervision_Train(config)
    if getattr(config, 'pretrained_ckpt_path', None):
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    # --- PARAMS: printed up front ---
    total_M = count_params_millions(model.net, trainable_only=True)
    print(f"[PARAM] Total trainable params (M): {total_M:.2f}")

    # --- GFLOPs: print BOTH 256x256 and 512x512 (per image) ---
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

    # --- FPS: print BOTH 256x256 and 512x512 ---
    # Note: FPS depends on your hardware and precision; we use AMP on CUDA by default.
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

    # --- MEMORY: print GPU memory for BOTH 256x256 and 512x512 ---
    # <-- NEW: Memory measurement
    try:
        mem_256 = measure_memory_gb(model.net, input_shape=(1, 3, 256, 256))
        print(f"[MEMORY] GPU peak memory @ 256x256: {mem_256:.2f} GB")
    except Exception as e:
        print(f"[MEMORY] Warning @256x256: {e}")

    try:
        mem_512 = measure_memory_gb(model.net, input_shape=(1, 3, 512, 512))
        print(f"[MEMORY] GPU peak memory @ 512x512: {mem_512:.2f} GB")
    except Exception as e:
        print(f"[MEMORY] Warning @512x512: {e}")

    # --- TIME: print per-image inference time for BOTH 256x256 and 512x512 ---
    # <-- NEW: Time measurement
    try:
        time_256 = measure_time_ms(model.net, input_shape=(1, 3, 256, 256), warmup_iters=10, measure_iters=50, use_amp=True)
        print(f"[TIME] per image @ 256x256: {time_256:.2f} ms")
    except Exception as e:
        print(f"[TIME] Warning @256x256: {e}")

    try:
        time_512 = measure_time_ms(model.net, input_shape=(1, 3, 512, 512), warmup_iters=10, measure_iters=50, use_amp=True)
        print(f"[TIME] per image @ 512x512: {time_512:.2f} ms")
    except Exception as e:
        print(f"[TIME] Warning @512x512: {e}")

    # ---------------------------------------------------------------
    trainer = pl.Trainer(
        devices=config.gpus,
        max_epochs=config.max_epoch,
        accelerator='auto',
        precision='16-mixed',  # modern AMP flag
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
        strategy=args.strategy,
        logger=logger
    )

    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
    main()

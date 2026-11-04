import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import ttach as tta

from train_supervision_potsdam import py2cfg, Supervision_Train


def img_writer(inp):
    (mask, output_path, mask_name, rgb, palette, label2rgb_fn) = inp
    if rgb:
        mask_RGB_name_png = os.path.join(output_path, 'pre_rgb', mask_name + '.png')
        mask_rgb_png = label2rgb_fn(mask)  # returns RGB ndarray (H,W,3)
        mask_rgb_png = cv2.cvtColor(mask_rgb_png, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_RGB_name_png, mask_rgb_png)
    else:
        mask_P_name_png = os.path.join(output_path, 'pre_p', mask_name + '.png')
        mask_p_png = mask.astype(np.uint8)
        mask_p_png = Image.fromarray(mask_p_png, 'P')
        mask_p_png.putpalette(palette)
        mask_p_png.save(mask_P_name_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", default='config/potsdam/dpgaformer.py', type=Path, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Where to save resulting masks")
    arg("-t", "--tta", help="Test time augmentation", default="d4", choices=[None, "d4", "lr"])
    arg("-g", "--gpu_index", default=0, type=int)
    arg("--rgb", default=False, action='store_true', help="whether to output RGB masks")
    arg("--batch_size", default=2, type=int)
    arg("--num_workers", default=4, type=int)
    return parser.parse_args()


def makedirs(output_path, rgb):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'pre_rgb' if rgb else 'pre_p'), exist_ok=True)


class SegHeadOnly(nn.Module):
    """Wrapper that forces models with RETURN_AUX=True to return only the main seg logits (N,C,H,W)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]  # main segmentation head
        return out


def make_label2rgb_from_palette(palette):
    """
    Build a label2rgb(mask)->RGB function using a flat PIL palette (len == 3*K).
    Works when the config doesn’t provide label2rgb.
    """
    # Ensure palette is a flat list of ints length 3*K; PIL requires 768, but will pad automatically.
    def _f(mask: np.ndarray) -> np.ndarray:
        imgP = Image.fromarray(mask.astype(np.uint8), 'P')
        imgP.putpalette(palette)
        imgRGB = imgP.convert('RGB')
        return np.array(imgRGB)
    return _f


if __name__ == "__main__":
    args = get_args()
    config = py2cfg(args.config_path)

    # Default output path if not provided
    if args.output_path is None:
        args.output_path = Path('fig_results') / args.config_path.parent.name / args.config_path.stem
    args.output_path.mkdir(parents=True, exist_ok=True)

    device = f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu'

    # Load LightningModule checkpoint with strict=False
    lit = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'),
        config=config,
        strict=False  # ← ADD THIS
    )
    lit.eval()
    lit.to(device)

    # Wrap to ensure tensor output (N,C,H,W) for TTA even when RETURN_AUX=True
    base_model = SegHeadOnly(lit).eval().to(device)

    # Optional TTA
    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
        ])
        model = tta.SegmentationTTAWrapper(base_model, transforms, merge_mode='mean')
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Scale(scales=[0.75, 1.0, 1.25, 1.5],
                      interpolation='bicubic',
                      align_corners=False),
        ])
        model = tta.SegmentationTTAWrapper(base_model, transforms, merge_mode='mean')
    else:
        model = base_model  # no TTA

    test_dataset = config.test_dataset
    makedirs(str(args.output_path), args.rgb)

    # --- palette & label2rgb fallback handling ---
    palette = getattr(config, 'palette', None)
    if palette is None:
        raise AttributeError("Your config must define 'palette' (flat list of 3*K ints).")

    # Prefer config.label2rgb if present; otherwise build one from the palette
    if hasattr(config, 'label2rgb'):
        label2rgb_fn = config.label2rgb
    else:
        label2rgb_fn = make_label2rgb_from_palette(palette)

    results = []
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        for batch in tqdm(test_loader):
            imgs = batch['img'].to(device)
            image_ids = batch['img_id']

            # Logits tensor thanks to SegHeadOnly
            logits = model(imgs)                    # (N, C, H, W)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)             # (N, H, W)

            for i in range(preds.shape[0]):
                mask = preds[i].cpu().numpy()
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path), mask_name, args.rgb, palette, label2rgb_fn))

    # Save masks (simple loop avoids multiprocessing pickling headaches across envs)
    for r in tqdm(results, desc="Writing"):
        img_writer(r)

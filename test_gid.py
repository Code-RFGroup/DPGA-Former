import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
import multiprocessing
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from train_supervision_gid import py2cfg, Supervision_Train


# -------- utils (picklable) --------
def mask_to_rgb_with_palette(mask: np.ndarray, palette) -> np.ndarray:
    """
    Top-level (picklable) converter: label mask -> RGB using a flat PIL palette (len == 3*K).
    """
    imgP = Image.fromarray(mask.astype(np.uint8), 'P')
    imgP.putpalette(palette)
    imgRGB = imgP.convert('RGB')
    return np.array(imgRGB)


def img_writer(inp):
    """
    Save mask as RGB (using palette) or as P-mode with palette.
    inp = (mask, output_path, mask_name, rgb, palette)
    """
    (mask, output_path, mask_name, rgb, palette) = inp
    if rgb:
        rgb_path = os.path.join(output_path, 'pre_rgb', mask_name + '.png')
        mask_rgb = mask_to_rgb_with_palette(mask, palette)  # RGB ndarray (H,W,3)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_path, mask_bgr)
    else:
        p_path = os.path.join(output_path, 'pre_p', mask_name + '.png')
        mask_p = mask.astype(np.uint8)
        imgP = Image.fromarray(mask_p, 'P')
        imgP.putpalette(palette)
        imgP.save(p_path)


def makedirs(output_path, rgb):
    os.makedirs(output_path, exist_ok=True)
    if rgb:
        os.makedirs(os.path.join(output_path, 'pre_rgb'), exist_ok=True)
    else:
        os.makedirs(os.path.join(output_path, 'pre_p'), exist_ok=True)


class SegHeadOnly(nn.Module):
    """Ensure the wrapped module returns ONLY logits (N,C,H,W) even if it outputs a tuple."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]  # keep only main segmentation logits
        return out


def multipool(function, args, processes=40):
    pool = multiprocessing.Pool(processes=processes)
    with tqdm(total=len(args)) as pbar:
        for _ in pool.imap_unordered(function, args):
            pbar.update(1)
    pool.close()
    pool.join()


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", default='config/gid/dpgaformer.py',
        type=Path, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks")
    arg("-t", "--tta", help="Test time augmentation.", default="d4", choices=[None, "d4", "lr"])
    arg("-g", "--gpu_index", default=0, type=int)
    arg("--rgb", default=False, action='store_true', help="whether output rgb masks")
    arg("--batch_size", default=2, type=int)
    arg("--num_workers", default=4, type=int)
    arg("--save_workers", default=40, type=int, help="processes for saving images")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = py2cfg(args.config_path)

    # Default output path if not provided
    if args.output_path is None:
        args.output_path = Path('fig_results') / args.config_path.parent.name / args.config_path.stem
    args.output_path.mkdir(parents=True, exist_ok=True)

    device = f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------
    # Load LightningModule checkpoint BUT drop old BD-head weights
    # (net.decoder.bd_head.*) to avoid size mismatch, just like test_lcai.py.
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state = ckpt["state_dict"]
    # filter out any keys belonging to the old BD head
    filtered_state = {
        k: v for k, v in state.items()
        if not k.startswith("net.decoder.bd_head.")
    }

    # instantiate model with current code
    lit = Supervision_Train(config)
    # load filtered state_dict; strict=False to allow missing bd_head and other buffers
    msg = lit.load_state_dict(filtered_state, strict=False)
    print("State dict load result:", msg)

    lit.to(device)
    lit.eval()

    # Wrap so TTA always receives logits tensor (N,C,H,W) even if RETURN_AUX=True
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

    test_dataset = config.test_dataset  # Use dataset from config
    makedirs(str(args.output_path), args.rgb)

    # Palette is required for saving
    palette = getattr(config, 'palette', None)
    if palette is None:
        raise AttributeError("Your config must define 'palette' (flat list of 3*K ints).")

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

            logits = model(imgs)                    # (N, C, H, W)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)            # (N, H, W)

            for i in range(preds.shape[0]):
                mask = preds[i].cpu().numpy()
                mask_name = image_ids[i]
                # Only basic types in the tuple -> safe for multiprocessing pickling
                results.append((mask, str(args.output_path), mask_name, args.rgb, palette))

    multipool(img_writer, results, processes=args.save_workers)

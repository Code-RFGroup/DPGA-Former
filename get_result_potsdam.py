import csv
import os
from os.path import join
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threshold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threshold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def fast_hist(a, b, n):
    """
    Build a confusion matrix using only valid semantic GT pixels.

    Ground-truth values outside [0, n-1], such as Potsdam boundary/ignore
    pixels encoded with label 6, are excluded rather than clipped into a
    semantic class.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    valid = (a >= 0) & (a < n)
    a = a[valid].astype(np.int64)
    b = b[valid].astype(np.int64)

    pred_valid = (b >= 0) & (b < n)
    a = a[pred_valid]
    b = b[pred_valid]

    return np.bincount(
        n * a + b,
        minlength=n ** 2
    ).reshape(n, n)


def get_tp_fp_tn_fn(hist):
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - np.diag(hist)
    fn = hist.sum(axis=1) - np.diag(hist)
    tn = np.diag(hist).sum() - np.diag(hist)
    return tp, fp, tn, fn


def get_Precision(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    precision = tp / (tp + fp)
    return precision


def get_Recall(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    recall = tp / (tp + fn)
    return recall


def get_F1(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1 = (2.0 * Precision * Recall) / (Precision + Recall)
    return F1


def get_Intersection_over_Union(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    IoU = tp / (tp + fn + fp)
    return IoU


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


# ----------------------------- Pure-NumPy Boundary Utilities (copied from lcai) -----------------------------
def four_neighbor_boundary(lbl, valid_mask=None):
    """
    Boolean 4-neighborhood boundary map.

    A transition is counted only when both neighboring pixels are valid
    evaluation pixels. This prevents ignored/padded labels from creating
    artificial semantic boundaries.
    """
    H, W = lbl.shape
    b = np.zeros((H, W), dtype=bool)

    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)

    # Vertical neighboring pairs
    pair_valid = valid_mask[:-1, :] & valid_mask[1:, :]
    diff = (lbl[:-1, :] != lbl[1:, :]) & pair_valid
    b[:-1, :] |= diff
    b[1:, :] |= diff

    # Horizontal neighboring pairs
    pair_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
    diff = (lbl[:, :-1] != lbl[:, 1:]) & pair_valid
    b[:, :-1] |= diff
    b[:, 1:] |= diff

    return b

def binary_dilate(mask, radius):
    """
    Fast binary dilation with a square (2r+1)x(2r+1) structuring element using only NumPy.
    """
    if radius <= 0:
        return mask.copy()
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    pad = radius
    padded = np.pad(mask.astype(bool), pad, mode='constant', constant_values=False)
    for dy in range(-radius, radius + 1):
        srow = pad + dy
        erow = srow + H
        for dx in range(-radius, radius + 1):
            scol = pad + dx
            ecol = scol + W
            out |= padded[srow:erow, scol:ecol]
    return out


# ----------------------------- BFScore & Trimap IoU (per-class accumulators) -----------------------------
class BoundaryStats:
    """Accumulate counts for BFScore across many images, per class."""
    def __init__(self, num_classes, radius):
        self.num_classes = int(num_classes)
        self.radius = int(radius)
        self.prec_num = np.zeros(num_classes, dtype=np.float64)
        self.prec_den = np.zeros(num_classes, dtype=np.float64)
        self.rec_num  = np.zeros(num_classes, dtype=np.float64)
        self.rec_den  = np.zeros(num_classes, dtype=np.float64)

    def update(self, gt_lbl, pr_lbl):
        r = self.radius

        # Only semantic GT pixels are valid for evaluation.
        valid = (gt_lbl >= 0) & (gt_lbl < self.num_classes)

        for c in range(self.num_classes):
            gt_bin = (gt_lbl == c)
            pr_bin = (pr_lbl == c)

            # Do not count transitions involving ignored/padded GT pixels.
            gt_b = four_neighbor_boundary(
                gt_bin.astype(np.uint8),
                valid_mask=valid
            )
            pr_b = four_neighbor_boundary(
                pr_bin.astype(np.uint8),
                valid_mask=valid
            )

            if r > 0:
                gt_b_d = binary_dilate(gt_b, r)
                pr_b_d = binary_dilate(pr_b, r)
            else:
                gt_b_d, pr_b_d = gt_b, pr_b

            # Precision: predicted boundary pixels matched by GT boundary.
            match_prec = np.logical_and(pr_b, gt_b_d).sum()
            self.prec_num[c] += match_prec
            self.prec_den[c] += pr_b.sum()

            # Recall: GT boundary pixels matched by predicted boundary.
            match_rec = np.logical_and(gt_b, pr_b_d).sum()
            self.rec_num[c] += match_rec
            self.rec_den[c] += gt_b.sum()

    def compute(self, start, end):
        P = np.divide(self.prec_num[start:end], self.prec_den[start:end],
                      out=np.zeros(end - start, dtype=float),
                      where=self.prec_den[start:end] != 0)
        R = np.divide(self.rec_num[start:end], self.rec_den[start:end],
                      out=np.zeros(end - start, dtype=float),
                      where=self.rec_den[start:end] != 0)
        F = np.divide(2 * P * R, (P + R),
                      out=np.zeros_like(P),
                      where=(P + R) != 0)
        return P, R, F


class TrimapIoUStats:
    """Accumulate intersection/union for IoU within a trimap band around GT boundaries, per class."""
    def __init__(self, num_classes, radius):
        self.num_classes = int(num_classes)
        self.radius = int(radius)
        self.intersection = np.zeros(num_classes, dtype=np.float64)
        self.union = np.zeros(num_classes, dtype=np.float64)

    def update(self, gt_lbl, pr_lbl):
        r = self.radius
        for c in range(self.num_classes):
            gt_bin = (gt_lbl == c)

            # Class boundary from GT and its r-pixel band
            gt_b = four_neighbor_boundary(gt_bin.astype(np.uint8))
            tri = binary_dilate(gt_b, r) if r > 0 else gt_b

            if not tri.any():
                continue

            # Restrict evaluation to trimap band
            pr_c = (pr_lbl == c)
            gt_c = gt_bin

            inter = np.logical_and(pr_c, gt_c) & tri
            uni   = (pr_c | gt_c) & tri

            self.intersection[c] += inter.sum()
            self.union[c] += uni.sum()

    def compute(self, start, end):
        tri_iou = np.divide(self.intersection[start:end], self.union[start:end],
                            out=np.zeros(end - start, dtype=float),
                            where=self.union[start:end] != 0)
        return tri_iou


# ----------------------------- Visualization helpers (optional) -----------------------------
def save_rgb(path, arr, scale=1):
    """Save RGB array as PNG, with optional nearest-neighbor upscaling."""
    img = Image.fromarray(arr.astype(np.uint8))
    if scale != 1:
        w, h = img.size
        img = img.resize((w * scale, h * scale), resample=Image.NEAREST)
    img.save(path)


def _potsdam_palette():
    return np.array([
        [255, 255, 255],  # ImSurf
        [0,   0, 255],    # Building
        [0, 255, 255],    # LowVeg
        [0, 255,   0],    # Tree
        [255, 204, 0],    # Car
        [255,   0, 0],    # Clutter
    ], dtype=np.uint8)


def _make_rgb_from_labels(lbl, name_classes, focus_class_idx=None):
    num_classes = len(name_classes)
    lut = _potsdam_palette() if num_classes == 6 else np.zeros((num_classes, 3), dtype=np.uint8)
    # For safety, if classes don't match, use generic colors
    if num_classes != 6:
        base_colors = [
            (0,0,0), (255,0,0), (0,255,0), (0,0,255),
            (255,255,0), (255,0,255), (0,255,255)
        ]
        for c in range(num_classes):
            lut[c] = base_colors[c % len(base_colors)]

    lbl_clipped = np.clip(lbl, 0, num_classes - 1).astype(int)
    rgb = lut[lbl_clipped]

    if focus_class_idx is not None and 0 <= focus_class_idx < num_classes:
        mask_focus = (lbl_clipped == focus_class_idx)
        rgb_dim = (rgb * 0.2).astype(np.uint8)
        rgb = rgb_dim
        focus_color = lut[focus_class_idx]
        rgb[mask_focus] = focus_color

    return rgb


def make_boundary_visuals(gt_lbl, pr_lbl, name_classes, boundary_radius=2,
                          trimap_radii=(3,5,10), out_dir="vis_boundary", basename="example",
                          focus_class_idx=None, vis_scale=1):
    os.makedirs(out_dir, exist_ok=True)
    H, W = gt_lbl.shape

    # GT and Pred masks
    gt_rgb = _make_rgb_from_labels(gt_lbl, name_classes, focus_class_idx)
    pr_rgb = _make_rgb_from_labels(pr_lbl, name_classes, focus_class_idx)
    save_rgb(os.path.join(out_dir, f"{basename}_01_gt_mask.png"), gt_rgb, scale=vis_scale)
    save_rgb(os.path.join(out_dir, f"{basename}_01b_pred_mask.png"), pr_rgb, scale=vis_scale)

    if focus_class_idx is not None:
        gt_src = (gt_lbl == focus_class_idx).astype(np.uint8)
        pr_src = (pr_lbl == focus_class_idx).astype(np.uint8)
    else:
        gt_src = gt_lbl.astype(np.uint8)
        pr_src = pr_lbl.astype(np.uint8)

    B_gt = four_neighbor_boundary(gt_src)
    gt_b_rgb = np.zeros((H,W,3), dtype=np.uint8)
    gt_b_rgb[B_gt] = (255,255,255)
    save_rgb(os.path.join(out_dir, f"{basename}_02_gt_boundary.png"), gt_b_rgb, scale=vis_scale)

    B_pr = four_neighbor_boundary(pr_src)
    pr_b_rgb = np.zeros((H,W,3), dtype=np.uint8)
    pr_b_rgb[B_pr] = (255,255,255)
    save_rgb(os.path.join(out_dir, f"{basename}_02b_pred_boundary.png"), pr_b_rgb, scale=vis_scale)

    r = int(boundary_radius)
    B_gt_d = binary_dilate(B_gt, r)
    gt_dil_rgb = np.zeros((H,W,3), dtype=np.uint8)
    band_only_gt = np.logical_and(B_gt_d, ~B_gt)
    gt_dil_rgb[band_only_gt] = (255,128,128)
    gt_dil_rgb[B_gt] = (255,0,0)
    save_rgb(os.path.join(out_dir, f"{basename}_03_gt_boundary_dilated_r{r}.png"), gt_dil_rgb, scale=vis_scale)

    B_pr_d = binary_dilate(B_pr, r)
    pr_dil_rgb = np.zeros((H,W,3), dtype=np.uint8)
    band_only_pr = np.logical_and(B_pr_d, ~B_pr)
    pr_dil_rgb[band_only_pr] = (128,128,255)
    pr_dil_rgb[B_pr] = (0,0,255)
    save_rgb(os.path.join(out_dir, f"{basename}_03b_pred_boundary_dilated_r{r}.png"), pr_dil_rgb, scale=vis_scale)

    for tr in trimap_radii:
        tr = int(tr)
        tri = binary_dilate(B_gt, tr)
        tri_rgb = np.zeros((H,W,3), dtype=np.uint8)
        tri_rgb[:,:] = (40,40,40)
        tri_rgb[tri] = (255,255,0)
        save_rgb(os.path.join(out_dir, f"{basename}_04_trimap_r{tr}.png"), tri_rgb, scale=vis_scale)

    overlap = np.logical_and(B_gt_d, B_pr_d)
    gt_only = np.logical_and(B_gt_d, ~B_pr_d)
    pr_only = np.logical_and(B_pr_d, ~B_gt_d)
    overlap_rgb = np.zeros((H,W,3), dtype=np.uint8)
    overlap_rgb[gt_only] = (255,128,128)
    overlap_rgb[pr_only] = (128,128,255)
    overlap_rgb[overlap] = (255,255,0)
    save_rgb(os.path.join(out_dir, f"{basename}_05_boundary_overlap_r{r}.png"), overlap_rgb, scale=vis_scale)


def contrast_metric(gt_dir, pred_dir, sam_p_dir, png, start, end, num_classes, name_classes):
    hist_pred = np.zeros((num_classes, num_classes))
    hist_sam = np.zeros((num_classes, num_classes))
    label = np.array(Image.open(os.path.join(gt_dir, png)))
    pred = np.array(Image.open(os.path.join(pred_dir, png)))
    sam_p = np.array(Image.open(os.path.join(sam_p_dir, png)))

    hist_pred += fast_hist(label.flatten(), pred.flatten(), num_classes)
    hist_sam += fast_hist(label.flatten(), sam_p.flatten(), num_classes)

    IoUs_pred = per_class_iu(hist_pred)[start:end]
    Pa_Recall_pred = per_class_PA_Recall(hist_pred)[start:end]
    Precision_pred = per_class_Precision(hist_pred)[start:end]

    IoUs_sam = per_class_iu(hist_sam)[start:end]
    Pa_Recall_sam = per_class_PA_Recall(hist_sam)[start:end]
    Precision_sam = per_class_Precision(hist_sam)[start:end]

    name_classes = name_classes[start:end]
    for ind_class in range(start, end):
        print(f'===>{name_classes[ind_class]}:\tIou:{round(IoUs_pred[ind_class] * 100, 2)},{round(IoUs_sam[ind_class] * 100, 2)},{round(IoUs_pred[ind_class] * 100 - IoUs_sam[ind_class] * 100, 2)};\
               Recall (equal to the PA):{round(Pa_Recall_pred[ind_class] * 100, 2)},{round(Pa_Recall_sam[ind_class] * 100, 2)},{round(Pa_Recall_pred[ind_class] * 100 - Pa_Recall_sam[ind_class] * 100, 2)}; \
               Precision:{round(Precision_pred[ind_class] * 100, 2)},{round(Precision_sam[ind_class] * 100, 2)},{round(Precision_pred[ind_class] * 100 - Precision_sam[ind_class] * 100, 2)}')

    print(f'===> mIoU:{round(np.nanmean(IoUs_pred) * 100, 2)},{round(np.nanmean(IoUs_sam) * 100, 2)},{round(np.nanmean(IoUs_pred) * 100 - np.nanmean(IoUs_sam) * 100, 2)}; \
        mPA_Recall:{round(np.nanmean(Pa_Recall_pred) * 100, 2)},{round(np.nanmean(Pa_Recall_sam) * 100, 2)},{round(np.nanmean(Pa_Recall_pred) * 100 - np.nanmean(Pa_Recall_sam) * 100, 2)};\
        Accuracy:{round(per_Accuracy(hist_pred) * 100, 2)},{round(per_Accuracy(hist_sam) * 100, 2)},{round(per_Accuracy(hist_pred) * 100 - per_Accuracy(hist_sam) * 100, 2)}\nGet miou done.\n\n')


def contrast_metric2(gt_dir, sam_p_dir, sam_p_dir2, png, start, end, num_classes, name_classes):
    hist_sam = np.zeros((num_classes, num_classes))
    hist_sam2 = np.zeros((num_classes, num_classes))
    label = np.array(Image.open(os.path.join(gt_dir, png)))
    sam_p = np.array(Image.open(os.path.join(sam_p_dir, png)))
    sam_p2 = np.array(Image.open(os.path.join(sam_p_dir2, png)))

    hist_sam += fast_hist(label.flatten(), sam_p.flatten(), num_classes)
    hist_sam2 += fast_hist(label.flatten(), sam_p2.flatten(), num_classes)

    hist_sam = hist_sam[start:end, start:end]
    hist_sam2 = hist_sam2[start:end, start:end]

    IoUs_sam = per_class_iu(hist_sam)
    Pa_Recall_sam = per_class_PA_Recall(hist_sam)
    Precision_sam = per_class_Precision(hist_sam)

    IoUs_sam2 = per_class_iu(hist_sam2)
    Pa_Recall_sam2 = per_class_PA_Recall(hist_sam2)
    Precision_sam2 = per_class_Precision(hist_sam2)

    name_classes = name_classes[start:end]
    for ind_class in range(start, end):
        print(f'===>{name_classes[ind_class]}:\tIou:{round(IoUs_sam[ind_class] * 100, 2)},{round(IoUs_sam2[ind_class] * 100, 2)},{round(IoUs_sam[ind_class] * 100 - IoUs_sam2[ind_class] * 100, 2)};\
               Recall (equal to the PA):{round(Pa_Recall_sam[ind_class] * 100, 2)},{round(Pa_Recall_sam2[ind_class] * 100, 2)},{round(Pa_Recall_sam[ind_class] * 100 - Pa_Recall_sam2[ind_class] * 100, 2)}; \
               Precision:{round(Precision_sam[ind_class] * 100, 2)},{round(Precision_sam2[ind_class] * 100, 2)},{round(Precision_sam[ind_class] * 100 - Precision_sam2[ind_class] * 100, 2)}')

    print(f'===> mIoU:{round(np.nanmean(IoUs_sam) * 100, 2)},{round(np.nanmean(IoUs_sam2) * 100, 2)},{round(np.nanmean(IoUs_sam) * 100 - np.nanmean(IoUs_sam2) * 100, 2)}; \
        mPA_Recall:{round(np.nanmean(Pa_Recall_sam) * 100, 2)},{round(np.nanmean(Pa_Recall_sam2) * 100, 2)},{round(np.nanmean(Pa_Recall_sam) * 100 - np.nanmean(Pa_Recall_sam2) * 100, 2)};\
        Accuracy:{round(per_Accuracy(hist_sam) * 100, 2)},{round(per_Accuracy(hist_sam2) * 100, 2)},{round(per_Accuracy(hist_sam) * 100 - per_Accuracy(hist_sam2) * 100, 2)}\nGet miou done.\n\n')


def compute_mIoU_all(gt_dir, pred_dir, png_name_list, num_classes, name_classes, start, end):
    hist = np.zeros((num_classes, num_classes))

    # --- BFScore and Trimap accumulators for radii 1,2,3 ---
    radii_bf = [1, 2, 3]  # as per paper sensitivity
    trimap_radii = [3, 5, 10]  # optional, keep for consistency
    bf_stats = {r: BoundaryStats(num_classes, r) for r in radii_bf}
    tri_stats = {r: TrimapIoUStats(num_classes, r) for r in trimap_radii}
    # -------------------------------------------------------

    gt_imgs = [join(gt_dir, x.rsplit('.png', 1)[0] + '.png') for x in png_name_list]
    pred_imgs = [join(pred_dir, x.rsplit('.png', 1)[0] + '.png') for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # Debug: Print unique values to identify outliers
        print(f"Unique values in pred {pred_imgs[ind]}: {np.unique(pred)}")
        print(f"Unique values in label {gt_imgs[ind]}: {np.unique(label)}")

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # --- update BFScore for all radii ---
        for r in radii_bf:
            bf_stats[r].update(label, pred)
        for r in trimap_radii:
            tri_stats[r].update(label, pred)
        # ----------------------------------

    IoUs = get_Intersection_over_Union(hist)[start:end]
    PA_Recall = get_Recall(hist)[start:end]
    Precision = get_Precision(hist)[start:end]
    F1 = get_F1(hist)[start:end]
    name_classes = name_classes[start:end]

    total_pixels = np.sum(hist, axis=1)
    total_pixels_all = np.sum(total_pixels)
    fwIoU = np.sum((total_pixels[start:end] / total_pixels_all) * IoUs) if total_pixels_all > 0 else 0
    pAcc = per_Accuracy(hist)

    # --- compute BFScore per radius ---
    bf_results = {}
    for r in radii_bf:
        P, R, F = bf_stats[r].compute(start, end)
        bf_results[r] = {'P': P, 'R': R, 'F': F}

    tri_results = {}
    for r in trimap_radii:
        tri_results[r] = tri_stats[r].compute(start, end)
    # ----------------------------------

    for ind_class in range(len(name_classes)):
        cls = name_classes[ind_class]
        tri_parts = " ".join([f"TriIoU@{r}px {tri_results[r][ind_class]*100:.2f}" for r in trimap_radii])
        bf_parts = " ".join([f"BFScore@{r}px {bf_results[r]['F'][ind_class]*100:.2f}" for r in radii_bf])
        print(
            f"===> {cls}: "
            f"IoU {IoUs[ind_class]*100:.2f}; "
            f"F1 {F1[ind_class]*100:.2f}; "
            f"Recall {PA_Recall[ind_class]*100:.2f}; "
            f"Precision {Precision[ind_class]*100:.2f}; "
            f"{bf_parts}; "
            f"{tri_parts}"
        )

    # Print per-radius mBFScore
    for r in radii_bf:
        mBF = np.nanmean(bf_results[r]['F']) * 100.0
        print(f"mBFScore @ r={r}: {mBF:.2f}")

    mean_tri = {r: np.nanmean(tri_results[r]) * 100.0 for r in trimap_radii}
    tri_str = " ".join([f"mTrimapIoU@{r}px: {mean_tri[r]:.2f}" for r in trimap_radii])

    print(
        f"===> mIoU: {np.nanmean(IoUs)*100:.2f}; "
        f"mF1: {np.nanmean(F1)*100:.2f}; "
        f"mPA: {np.nanmean(PA_Recall)*100:.2f}; "
        f"fwIoU: {fwIoU*100:.2f}; "
        f"pAcc: {pAcc*100:.2f}; "
        f"{tri_str}; "
        f"mBFScore@{radii_bf[0]}px: {np.nanmean(bf_results[radii_bf[0]]['F'])*100:.2f}; "
        f"mBFScore@{radii_bf[1]}px: {np.nanmean(bf_results[radii_bf[1]]['F'])*100:.2f}; "
        f"mBFScore@{radii_bf[2]}px: {np.nanmean(bf_results[radii_bf[2]]['F'])*100:.2f}"
    )


def print_hist(gt_dir, pred_dir, png_name_list, name_classes):
    num_classes = len(name_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + '.png') for x in png_name_list]
    pred_imgs = [join(pred_dir, x + '.png') for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    pred_name = pred_dir.split('/')[-2]
    with open(os.path.join('csv_demo', f'{pred_name}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)


def get_miou_all_main(pred_p_dir, gt_dir, name_classes, image_ids, start, end):
    num_classes = len(name_classes)
    print("Get miou.")
    compute_mIoU_all(gt_dir, pred_p_dir, image_ids, num_classes, name_classes, start, end)
    print("Get miou done.")
    print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mIoU for segmentation predictions (Potsdam)")
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory containing predicted masks")
    parser.add_argument('--gt_dir', type=str, required=True, help="Directory containing ground truth masks")

    # Visualization arguments (optional, same as lcai)
    parser.add_argument('--vis_id', type=str, default=None,
                        help="Image id (as in test.txt) to visualize boundaries")
    parser.add_argument('--vis_focus_class', type=int, default=None,
                        help="Class index to focus on in visualization (optional)")
    parser.add_argument('--vis_out_dir', type=str, default='vis_boundary',
                        help="Output directory for boundary visualizations")
    parser.add_argument('--vis_scale', type=int, default=1,
                        help="Integer upscaling factor for visualization images")

    args = parser.parse_args()

    gt_dir = os.path.abspath(args.gt_dir)
    pred_dir = os.path.abspath(args.pred_dir)

    # Potsdam classes (excluding background? Actually Potsdam has 6 classes without explicit background)
    name_classes = ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter']
    start = 0
    end = len(name_classes)

    if not os.path.exists(pred_dir):
        print(f"Error: Prediction directory {pred_dir} does not exist.")
        exit()

    image_ids = [f.split('.png')[0] for f in os.listdir(pred_dir) if f.endswith('.png')]
    if not image_ids:
        print("No prediction files found in", pred_dir)
        exit()

    get_miou_all_main(pred_dir, gt_dir, name_classes, image_ids, start, end)

    # Optional visualization
    if args.vis_id is not None:
        base = args.vis_id
        gt_path = os.path.join(gt_dir, base + '.png')
        pr_path = os.path.join(pred_dir, base + '.png')
        if not (os.path.exists(gt_path) and os.path.exists(pr_path)):
            print(f"[VIS] Skipping: could not find GT or Pred for id '{base}'")
        else:
            gt_lbl = np.array(Image.open(gt_path))
            pr_lbl = np.array(Image.open(pr_path))
            if gt_lbl.shape != pr_lbl.shape:
                print(f"[VIS] Skipping: shape mismatch for id '{base}'")
            else:
                make_boundary_visuals(
                    gt_lbl=gt_lbl,
                    pr_lbl=pr_lbl,
                    name_classes=name_classes,
                    boundary_radius=2,
                    trimap_radii=(3,5,10),
                    out_dir=args.vis_out_dir,
                    basename=base,
                    focus_class_idx=args.vis_focus_class,
                    vis_scale=args.vis_scale
                )
                print(f"[VIS] Saved visualizations for '{base}' to {args.vis_out_dir}")

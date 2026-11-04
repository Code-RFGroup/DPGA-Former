# geoseg/datasets/lcai_dataset.py

import torch  # Added missing import
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np
from PIL import Image
import random
import albumentations as albu
from .transform import *

CLASSES = ('background', 'building', 'woodland', 'water', 'road')

# === UPDATED TO MATCH GROUND-TRUTH PALETTE ===
# Ground truth palette (from PIL): [0,0,0, 255,0,0, 0,255,0, 0,0,255, 255,255,255]
PALETTE = [
    [0, 0, 0],        # class 0
    [255, 0, 0],      # class 1
    [0, 255, 0],      # class 2
    [0, 0, 255],      # class 3
    [255, 255, 255],  # class 4
]
palette = [
    0, 0, 0,          # class 0
    255, 0, 0,        # class 1
    0, 255, 0,        # class 2
    0, 0, 255,        # class 3
    255, 255, 255,    # class 4
]
# ============================================

ORIGIN_IMG_SIZE = (512, 512)  # Updated to 512x512 for LCAI
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)

def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    aug = Compose([RandomHorizontalFlip(prob=0.5), RandomVerticalFlip(prob=0.5), RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                  SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class LCAIDataset(Dataset):
    def __init__(self, root_dir='data/LCAI', mode='val', img_dir='images', mask_dir='labels',
                 img_suffix='.jpg', mask_suffix='_m.png', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE, split_file=None):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.split_file = split_file
        
        if split_file:
            with open(split_file, 'r') as f:
                self.image_files = [line.strip() for line in f]
            self.filtered_img_ids = self.image_files  # No filtering, use all entries
            print(f"Loaded {len(self.filtered_img_ids)} images in mode {self.mode}")
        else:
            self.img_ids = self.get_img_ids(self.root_dir, self.img_dir, self.mask_dir)
            self.filtered_img_ids = self.img_ids

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.filtered_img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.filtered_img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0].replace('_m', '')) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.filtered_img_ids[index]
        img_name = osp.join(self.root_dir, self.img_dir, img_id if img_id.endswith(self.img_suffix) else img_id + self.img_suffix)
        mask_name = osp.join(self.root_dir, self.mask_dir, img_id.replace(self.img_suffix, self.mask_suffix) if img_id.endswith(self.img_suffix) else (img_id + self.mask_suffix))
        try:
            img = Image.open(img_name).convert('RGB')
            mask = Image.open(mask_name)
            return img, mask
        except FileNotFoundError as e:
            print(f"Error loading files for {img_id}: {str(e)}. Check if files exist at {img_name} and {mask_name}.")
            raise  # Re-raise to let DataLoader handle the error
        except Exception as e:
            print(f"Unexpected error loading files for {img_id}: {str(e)}.")
            raise

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.filtered_img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(img_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask

# Instantiate datasets with split files
train_dataset = LCAIDataset(root_dir='data/LCAI', mode='train', 
                           transform=train_aug, split_file='data/LCAI/train.txt')

val_dataset = LCAIDataset(root_dir='data/LCAI', mode='val', 
                         transform=val_aug, split_file='data/LCAI/val.txt')

test_dataset = LCAIDataset(root_dir='data/LCAI', mode='test', 
                          transform=val_aug, split_file='data/LCAI/test.txt')

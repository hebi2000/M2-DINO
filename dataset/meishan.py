import os
import random
from glob import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2

class MeiShan(Dataset):
    def __init__(self, data_path, mode='train', img_size=1024, use_dynamic_split=False, split_ratio=0.9):
        self.mode = mode
        self.img_size = img_size
        self.data_path = data_path  # Store data_path for later use in __getitem__ if needed (though we use absolute paths now)

        # Determine source directories
        # If using dynamic split, we merge 'train' and 'val' folders to create a larger pool
        if use_dynamic_split and mode in ['train', 'val']:
            source_modes = ['train', 'val']
        else:
            source_modes = [mode]

        self.image_files = []

        # Iterate over all source directories (e.g., train/images, val/images)
        for src_mode in source_modes:
            current_image_dir = os.path.join(data_path, src_mode, 'images')
            # current_sar_dir = os.path.join(data_path, src_mode, 'sar_opt') # Removed SAR

            # Get all image files in this directory
            current_files = sorted(glob(os.path.join(current_image_dir, '*.tif')))

            for img_path in current_files:
                # base_name = os.path.basename(img_path)
                # base_name_without_ext, ext = os.path.splitext(base_name)

                # Construct expected SAR filename
                # sar_filename = f"{base_name_without_ext}{ext}"
                # sar_path = os.path.join(current_sar_dir, sar_filename)

                # if os.path.exists(sar_path): # Removed SAR check
                # We store the full path, so we can mix files from different folders
                self.image_files.append(img_path)

        # --- Dynamic Split Logic ---
        if use_dynamic_split and mode in ['train', 'val']:
            # Fix seed to ensure consistent split between train and val runs
            random.seed(42)
            random.shuffle(self.image_files)

            split_idx = int(len(self.image_files) * split_ratio)

            if mode == 'train':
                self.image_files = self.image_files[:split_idx]
                print(
                    f"Dynamic Split (Train): Merged 'train'+'val'. Using {len(self.image_files)} images (First {split_ratio * 100}%)")
            else:
                self.image_files = self.image_files[split_idx:]
                print(
                    f"Dynamic Split (Val): Merged 'train'+'val'. Using {len(self.image_files)} images (Last {(1 - split_ratio) * 100:.1f}%)")

            # Reset random seed
            random.seed()
        else:
            print(f"Initialized {mode} dataset with {len(self.image_files)} valid images.")

        self.label_mapping = {i: i for i in range(10)}

        # Define transforms
        if mode == 'train':
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # 将 'mode' 改为 'border_mode'
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.0625, 0.0625), rotate=(-45, 45), p=0.5,
                         border_mode=cv2.BORDER_REFLECT_101),
                A.GridDistortion(p=0.2),
            ]) # Removed additional_targets={'image0': 'image'}

            self.optical_color_aug = None

            # self.sar_aug = A.Compose([ # Removed SAR aug
            #     A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            # ])

            print(f"Enhanced Data Augmentation enabled for {mode}.")
        else:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
            ]) # Removed additional_targets={'image0': 'image'}
            self.optical_color_aug = None
            # self.sar_aug = None
            print(f"No data augmentation for {mode}.")

    def __len__(self):
        return len(self.image_files)

    def remap_labels(self, mask):
        remapped_mask = np.zeros_like(mask, dtype=np.uint8)
        for original_label, new_label in self.label_mapping.items():
            remapped_mask[mask == original_label] = new_label
        return remapped_mask

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]

            # Need to determine which folder this image came from to find the correct label/sar path
            # Since we stored absolute paths, we can parse the parent directory
            # Structure: .../train/images/xxx.tif OR .../val/images/xxx.tif

            parent_dir = os.path.dirname(img_path)  # .../train/images
            mode_dir = os.path.dirname(parent_dir)  # .../train

            # Reconstruct paths based on the file's actual location
            # sar_dir = os.path.join(mode_dir, 'sar_opt') # Removed SAR
            label_dir = os.path.join(mode_dir, 'labels') if self.mode != 'predict' else None

            base_name = os.path.basename(img_path)
            base_name_without_ext, ext = os.path.splitext(base_name)

            with rasterio.open(img_path) as src:
                img = src.read([1, 2, 3]).transpose(1, 2, 0)

            # Construct SAR filename
            # sar_filename = f"{base_name_without_ext}{ext}"
            # sar_path = os.path.join(sar_dir, sar_filename)

            # if not os.path.exists(sar_path):
            #     raise FileNotFoundError(f"SAR file not found: {sar_path}")

            # with rasterio.open(sar_path) as src:
            #     sar = src.read(1)

            # Load label
            if self.mode != 'predict':
                label_name = base_name_without_ext + '.png'
                label_path = os.path.join(label_dir, label_name)

                if not os.path.exists(label_path):
                    label_path_tif = os.path.splitext(label_path)[0] + '.tif'
                    if os.path.exists(label_path_tif):
                        label_path = label_path_tif

                with rasterio.open(label_path) as src:
                    mask = src.read(1)

                mask = self.remap_labels(mask)
            else:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # --- 1. Apply Geometric Augmentations (Shared) ---
            # transformed = self.transforms(image=img, image0=sar, mask=mask)
            transformed = self.transforms(image=img, mask=mask)
            img_np = transformed['image']
            # sar_np = transformed['image0']
            mask_np = transformed['mask']

            # --- 2. Apply Specific Pixel-level Augmentations ---
            if self.mode == 'train':
                if self.optical_color_aug:
                    img_np = self.optical_color_aug(image=img_np)['image']
                # if self.sar_aug:
                #     sar_np = self.sar_aug(image=sar_np)['image']

            # --- 3. Normalization & Preprocessing ---

            # Optical: [-1, 1]
            img_normalized = (img_np / 255.0 - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])

            # SAR: Log Transform + MinMax Scaling
            # sar_np = sar_np.astype(np.float32)
            # sar_np = np.log1p(sar_np)

            # s_min = sar_np.min()
            # s_max = sar_np.max()
            # if s_max > s_min:
            #     sar_normalized = (sar_np - s_min) / (s_max - s_min)
            # else:
            #     sar_normalized = np.zeros_like(sar_np)

            # --- 4. To Tensor ---
            img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1)
            # sar_tensor = torch.tensor(sar_normalized, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(mask_np, dtype=torch.long)

            # return img_tensor, sar_tensor, mask_tensor
            return img_tensor, mask_tensor
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}. Skipping.")
            new_idx = random.randint(0, len(self.image_files) - 1)
            return self.__getitem__(new_idx)

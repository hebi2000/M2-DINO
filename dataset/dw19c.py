import os
import random
from glob import glob
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import warnings
import json
from rasterio.errors import NotGeoreferencedWarning

# Ignore NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

class DW19C(Dataset):
    def __init__(self, data_path, mode='train', img_size=1024, val_villages=None, config_path=None):
        self.mode = mode
        self.img_size = img_size
        self.data_path = data_path
        
        if val_villages is None:
            val_villages = [
                '刘营镇三道河村',
                '新桥乡团结村',
                '明阳镇凤梁村',
                '长宁镇马村村',
                '九尺镇升平社区'
            ]

        # Get all village directories
        # Assuming data_path contains village directories directly
        if os.path.exists(data_path):
            all_villages = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        else:
            all_villages = []
            print(f"Warning: data_path {data_path} does not exist.")
        
        if mode == 'val':
            self.villages = [v for v in all_villages if v in val_villages]
        elif mode == 'train':
            self.villages = [v for v in all_villages if v not in val_villages]
        else:
            # For predict or other modes, use all available
            self.villages = all_villages

        self.image_files = []
        for village in self.villages:
            img_dir = os.path.join(data_path, village, 'img')
            if os.path.exists(img_dir):
                files = sorted(glob(os.path.join(img_dir, '*.tif')))
                self.image_files.extend(files)
            else:
                # Only warn if we expected to find images here
                if mode != 'predict':
                    print(f"Warning: img directory not found for village {village} at {img_dir}")

        print(f"Initialized {mode} dataset with {len(self.image_files)} images from {len(self.villages)} villages.")

        # Update label mapping for 19 classes
        self.label_mapping = {i: i for i in range(19)}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                if 'label_mapping' in config:
                    # Convert keys to int if they are strings
                    self.label_mapping = {int(k): int(v) for k, v in config['label_mapping'].items()}
                    print(f"Loaded label mapping from {config_path}")
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}")

        # Define transforms
        additional_targets = {'aux': 'image'}
        if mode == 'train':
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Use border_mode for newer albumentations versions
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.0625, 0.0625), rotate=(-45, 45), p=0.5,
                         border_mode=cv2.BORDER_REFLECT_101),
                A.GridDistortion(p=0.2),
            ], additional_targets=additional_targets)

            self.optical_color_aug = None
            print(f"Enhanced Data Augmentation enabled for {mode}.")
        else:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
            ], additional_targets=additional_targets)
            self.optical_color_aug = None
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

            # Structure: data_path/village/img/image.tif
            parent_dir = os.path.dirname(img_path) # .../village/img
            village_dir = os.path.dirname(parent_dir) # .../village
            
            if self.mode != 'predict':
                label_dir = os.path.join(village_dir, 'label')
            else:
                label_dir = None

            base_name = os.path.basename(img_path)
            base_name_without_ext, ext = os.path.splitext(base_name)

            # Read 4 channels
            with rasterio.open(img_path) as src:
                if src.count >= 4:
                    # Read first 4 channels
                    data = src.read([1, 2, 3, 4]).transpose(1, 2, 0)
                    img = data[:, :, :3]
                    aux = data[:, :, 3]
                elif src.count >= 3:
                    # Fallback if only 3 channels exist
                    img = src.read([1, 2, 3]).transpose(1, 2, 0)
                    aux = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
                else:
                    # Handle cases with fewer than 3 channels if necessary
                    # This is a fallback, might need adjustment based on actual data
                    img = src.read([1, 2, 3]).transpose(1, 2, 0) # This might fail if count < 3
                    aux = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

            # Load label
            if self.mode != 'predict':
                # Try finding label with same name but different extension or same extension
                # Assuming label filename matches image filename but in label directory
                
                # Check for .png first
                label_path = os.path.join(label_dir, base_name_without_ext + '.png')
                if not os.path.exists(label_path):
                    # Check for .tif
                    label_path = os.path.join(label_dir, base_name_without_ext + '.tif')
                
                if not os.path.exists(label_path):
                     # Fallback or error handling if label doesn't exist
                     # For now, let's assume it exists or raise the error that rasterio will catch
                     pass

                with rasterio.open(label_path) as src:
                    mask = src.read(1)

                mask = self.remap_labels(mask)
            else:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # --- 1. Apply Geometric Augmentations ---
            # Pass aux as 'aux' target
            transformed = self.transforms(image=img, mask=mask, aux=aux)
            img_np = transformed['image']
            mask_np = transformed['mask']
            aux_np = transformed['aux']

            # --- 2. Apply Specific Pixel-level Augmentations ---
            if self.mode == 'train':
                if self.optical_color_aug:
                    img_np = self.optical_color_aug(image=img_np)['image']

            # --- 3. Normalization & Preprocessing ---
            # Optical: [-1, 1]
            # Normalize to [0, 1] first then subtract mean and divide by std
            # (x / 255.0 - 0.5) / 0.5  ==> x/127.5 - 1
            img_normalized = (img_np / 255.0 - np.array([0.5, 0.5, 0.5])) / np.array([0.5, 0.5, 0.5])
            
            # Aux normalization
            # Assuming aux is 0-255, we normalize to [0, 1]
            aux_np = aux_np.astype(np.float32)
            aux_normalized = aux_np / 255.0

            # --- 4. To Tensor ---
            img_tensor = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1)
            
            # Aux tensor: add channel dimension if needed
            if aux_normalized.ndim == 2:
                aux_normalized = aux_normalized[:, :, np.newaxis]
            aux_tensor = torch.tensor(aux_normalized, dtype=torch.float32).permute(2, 0, 1)
            
            mask_tensor = torch.tensor(mask_np, dtype=torch.long)

            return img_tensor, aux_tensor, mask_tensor
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}. Skipping.")
            # Avoid infinite recursion if all files are bad, but simple retry is okay for now
            new_idx = random.randint(0, len(self.image_files) - 1)
            return self.__getitem__(new_idx)

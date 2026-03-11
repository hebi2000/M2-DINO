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

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


class MultiModalDataset(Dataset):
    def __init__(self, data_path, mode='train', img_size=512, config_path=None):
        self.mode = mode
        self.img_size = img_size
        self.data_path = data_path

        self.aux_modes = []
        self.color_mapping = []

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.color_mapping = config.get('color_mapping', [])
            self.aux_modes = config.get('aux_inputs', [])
            print(f"Loaded color mapping with {len(self.color_mapping)} classes.")
            print(f"Loaded aux inputs from config: {self.aux_modes}")
        else:
            raise ValueError("Config file not found or is empty.")

        self.valid_exts = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        self.image_files = []

        current_image_dir = os.path.join(data_path, mode, 'images')
        if not os.path.exists(current_image_dir):
            current_image_dir = os.path.join(data_path, 'images')

        if os.path.exists(current_image_dir):
            for ext in self.valid_exts:
                self.image_files.extend(sorted(glob(os.path.join(current_image_dir, f'*{ext}'))))
        else:
            print(f"Warning: Image directory not found: {current_image_dir}")

        print(f"Initialized MultiModalDataset ({mode}) with {len(self.image_files)} images.")

        additional_targets = {'aux': 'image'}

        if mode == 'train':
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], additional_targets=additional_targets)
            self.optical_color_aug = A.Compose([A.ColorJitter(p=0.5)])
        else:
            self.transforms = A.Compose([A.Resize(img_size, img_size)], additional_targets=additional_targets)
            self.optical_color_aug = None

    def __len__(self):
        return len(self.image_files)

    def rgb_to_mask(self, rgb_label):
        mask = np.zeros((rgb_label.shape[0], rgb_label.shape[1]), dtype=np.uint8)
        for i, color in enumerate(self.color_mapping):
            # Find all pixels that match the color and assign the class index
            condition = np.all(rgb_label == color, axis=-1)
            mask[condition] = i
        return mask

    def load_file(self, path, is_label=False):
        if os.path.exists(path):
            with rasterio.open(path) as src:
                if is_label:
                    return src.read().transpose(1, 2, 0)  # Read as H, W, C
                else:
                    count = src.count
                    if count >= 3:
                        return src.read([1, 2, 3]).transpose(1, 2, 0)
                    else:
                        return src.read(1)
        return None

    @staticmethod
    def preprocess_aux(aux_data, mode_name):
        aux_data = aux_data.astype(np.float32)
        if 'dsm' in mode_name.lower() or 'dem' in mode_name.lower():
            p2 = np.percentile(aux_data, 2)
            p98 = np.percentile(aux_data, 98)
            if p98 > p2:
                aux_data_clipped = np.clip(aux_data, p2, p98)
                aux_normalized = (aux_data_clipped - p2) / (p98 - p2)
            else:
                aux_normalized = np.zeros_like(aux_data)
            return aux_normalized
        else:
            if aux_data.max() > 1.0:
                return aux_data / 255.0
            return np.clip(aux_data, 0, 1)

    def __getitem__(self, idx):
        try:
            img_path = self.image_files[idx]
            parent_dir = os.path.dirname(img_path)
            mode_dir = os.path.dirname(parent_dir)
            base_name = os.path.basename(img_path)

            img = self.load_file(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            aux_list = []
            for aux_mode in self.aux_modes:
                aux_path = os.path.join(mode_dir, aux_mode, base_name)
                aux_data = self.load_file(aux_path)
                if aux_data is None:
                    aux_data = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

                aux_norm = self.preprocess_aux(aux_data, aux_mode)
                aux_list.append(aux_norm)

            if len(aux_list) > 0:
                aux_stack = np.stack(aux_list, axis=-1)
            else:
                aux_stack = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

            label_path = os.path.join(mode_dir, 'labels', base_name)
            label_rgb = self.load_file(label_path, is_label=True)
            if label_rgb is None:
                raise FileNotFoundError(f"Label not found: {label_path}")
            mask = self.rgb_to_mask(label_rgb)

            transformed = self.transforms(image=img, aux=aux_stack, mask=mask)
            img_aug = transformed['image']
            aux_aug = transformed['aux']
            mask_aug = transformed['mask']

            if self.mode == 'train' and self.optical_color_aug:
                img_aug = self.optical_color_aug(image=img_aug)['image']

            img_norm = (img_aug / 255.0 - 0.5) / 0.5

            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()

            if aux_aug.ndim == 2:
                aux_aug = aux_aug[:, :, np.newaxis]
            aux_tensor = torch.from_numpy(aux_aug).permute(2, 0, 1).float()

            mask_tensor = torch.from_numpy(mask_aug).long()

            return img_tensor, aux_tensor, mask_tensor

        except Exception as e:
            print(f"Warning: Error loading {self.image_files[idx]}: {e}. Skipping this sample.")
            return None

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
    def __init__(self, data_path, mode='train', img_size=512, val_villages=None, config_path=None):
        self.mode = mode
        self.img_size = img_size
        self.data_path = data_path
        
        if val_villages is None:
            val_villages = [
                 '刘营镇三道河村', '新桥乡团结村', '明阳镇凤梁村', '长宁镇马村', '九尺镇升平社区', '龙泉驿区洪安镇洪福村'
            ]

        # Get all village directories
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
            self.villages = all_villages

        self.images = []
        self.labels = []
        for village in self.villages:
            img_dir = os.path.join(data_path, village, 'img')
            label_dir = os.path.join(data_path, village, 'label')
            if not os.path.exists(img_dir):
                continue
            
            image_paths = sorted(glob(os.path.join(img_dir, '*.tif')))
            for img_path in image_paths:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                label_path = None
                if os.path.exists(label_dir):
                    label_path_png = os.path.join(label_dir, base_name + '.png')
                    label_path_tif = os.path.join(label_dir, base_name + '.tif')
                    if os.path.exists(label_path_png):
                        label_path = label_path_png
                    elif os.path.exists(label_path_tif):
                        label_path = label_path_tif

                if label_path is not None or mode == 'predict':
                    self.images.append(img_path)
                    self.labels.append(label_path)

        print(f"Initialized {mode} dataset with {len(self.images)} images from {len(self.villages)} villages.")

        # Define transforms with additional_targets for NDVI
        if mode == 'train':
            self.transforms = A.Compose([
                A.RandomCrop(width=img_size, height=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], additional_targets={"ndvi": "image"})
            print(f"Data Augmentation enabled for {mode} (RandomCrop, Flips).")
        else:
            self.transforms = A.Compose([
                A.CenterCrop(width=img_size, height=img_size),
            ], additional_targets={"ndvi": "image"})
            print(f"Data Augmentation for {mode} (CenterCrop).")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            # 1. 获取当前样本的文件路径并读取多光谱影像 (以 rasterio 为例)
            img_path = self.images[idx]
            label_path = self.labels[idx]
            
            with rasterio.open(img_path) as src:
                if src.count < 4:
                    raise ValueError(f"Image {img_path} has only {src.count} bands, but 4 are required for NDVI calculation.")
                # 读取原始影像 (B,G,R,NIR)，形状为 (C, H, W)
                image = src.read([1, 2, 3, 4]).astype(np.float32)
            
            if label_path:
                with rasterio.open(label_path) as src:
                    label = src.read(1).astype(np.int64) # (H, W)
            else: # For predict mode
                label = np.zeros((image.shape[1], image.shape[2]), dtype=np.int64)

            # 2. 提取红光 (Red) 和近红外 (NIR) 波段
            # 假设波段顺序为 B, G, R, NIR (1,2,3,4)
            # 读取后 0-indexed: 0:B, 1:G, 2:R, 3:NIR
            red = image[2, :, :]
            nir = image[3, :, :]
            
            # 3. 计算物理先验 NDVI：(NIR - Red) / (NIR + Red)
            # 增强数值稳定性：
            # a. 确保分母不为 0
            denominator = nir + red
            denominator[denominator == 0] = 1e-8 # 避免除以 0
            
            ndvi = (nir - red) / denominator
            
            # b. 处理可能的 NaN/Inf
            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 4. 截断异常值，并严格保持 [-1.0, 1.0] 物理范围
            ndvi = np.clip(ndvi, -1.0, 1.0)
            
            # 5. 调整维度顺序以适配 Albumentations (要求输入形式为 H, W, C)
            # 提取 RGB 波段
            rgb_hwc = image[:3, :, :].transpose(1, 2, 0) 
            ndvi_hwc = np.expand_dims(ndvi, axis=-1) # (H, W) -> (H, W, 1)
            
            # 6. 执行同步数据增强
            augmented = self.transforms(image=rgb_hwc, mask=label, ndvi=ndvi_hwc)
            rgb_aug = augmented['image']
            label_aug = augmented['mask']
            ndvi_aug = augmented['ndvi']
            
            # 7. 归一化和转换为 PyTorch Tensor (H, W, C -> C, H, W)
            # 归一化 RGB 图像到 [-1, 1]
            rgb_normalized = (rgb_aug / 255.0 - 0.5) / 0.5

            rgb_tensor = torch.from_numpy(rgb_normalized.transpose(2, 0, 1).copy()).float()
            # NDVI 已经是 [-1, 1]，只需转换为 Tensor
            ndvi_tensor = torch.from_numpy(ndvi_aug.transpose(2, 0, 1).copy()).float() # 形状为 [1, H, W]
            label_tensor = torch.from_numpy(label_aug.copy()).long()
            
            # Final check for NaN/Inf in tensors
            if torch.isnan(rgb_tensor).any() or torch.isinf(rgb_tensor).any():
                 raise ValueError("NaN/Inf found in RGB tensor")
            if torch.isnan(ndvi_tensor).any() or torch.isinf(ndvi_tensor).any():
                 raise ValueError("NaN/Inf found in NDVI tensor")
            
            # Debug: Check if NDVI is all zeros
            if torch.all(ndvi_tensor == 0):
                # Check if original NIR and Red were all zeros (e.g. black border)
                if np.all(nir == 0) and np.all(red == 0):
                    # This is expected for black borders, but maybe we shouldn't train on it?
                    # But RandomCrop might pick it up.
                    pass
                else:
                    print(f"Warning: NDVI tensor is all zeros for {img_path}")
                    print(f"  Original NIR stats: min={nir.min()}, max={nir.max()}, mean={nir.mean()}")
                    print(f"  Original Red stats: min={red.min()}, max={red.max()}, mean={red.mean()}")
                    print(f"  NDVI before aug stats: min={ndvi.min()}, max={ndvi.max()}, mean={ndvi.mean()}")
                    print(f"  NDVI after aug stats: min={ndvi_aug.min()}, max={ndvi_aug.max()}, mean={ndvi_aug.mean()}")

            # 返回主模态、辅助模态与标签
            return rgb_tensor, ndvi_tensor, label_tensor

        except Exception as e:
            print(f"Error loading sample at index {idx} ({self.images[idx]}): {e}. Skipping.")
            # Return a random different sample to avoid crashing the loader
            new_idx = random.randint(0, len(self.images) - 1)
            return self.__getitem__(new_idx)

import os
import argparse
import sys
import logging
import datetime
import math
import numpy as np
import json
from tqdm import tqdm
import cv2

# --- 1. 路径修复 (确保能导入本地模块) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Local imports
from dataset.dw19c_ndvi import DW19C
from wrappers.dino_mask2former import DinoMask2Former
from utils.metrics import MetricLogger, calculate_semantic_segmentation
import random

def set_global_seed(seed=42):
    """
    固定所有全局随机数种子，确保深度学习模型训练的可复现性。
    """
    # 1. 固定 Python 内置的随机模块
    random.seed(seed)

    # 2. 固定 NumPy 的随机种子
    np.random.seed(seed)

    # 3. 固定 PyTorch 相关的随机种子 (CPU & GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多卡 GPU，需加此行

    # 4. 配置 cuDNN 后端以保证卷积运算的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. 设置系统环境变量，固定 Python Hash 算法的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✅ 成功设置全局随机数种子为: {seed}，消除随机性波动！")


# 立即在脚本最外层调用
set_global_seed(42)


def setup_logger(log_dir):
    """设置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("DINOv3_Train")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 文件日志
    log_file = os.path.join(log_dir, f'train_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def collate_fn_skip_corrupted(batch):
    """
    Custom collate_fn that filters out None values from the batch.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)


def main(args):
    # --- 2. 初始化环境 ---
    torch.backends.cudnn.benchmark = False

    log_dir = os.path.join(args.save_dir, "logs")
    weight_dir = os.path.join(args.save_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info("===============================================")
    logger.info("🚀 DINOv3 + UperHead Training (BF16 Mode)")
    logger.info("===============================================")

    # --- CRITICAL CHECKS ---
    if not os.path.exists(args.dataset_config):
        logger.error(f"❌ Dataset config file not found at: {args.dataset_config}")
        sys.exit(1)

    # Only check for weights if not resuming
    if not args.resume and not os.path.exists(args.weights_path):
        logger.error(f"❌ Backbone weights file not found at: {args.weights_path}")
        sys.exit(1)

    # --- Load Dataset Config ---
    dataset_config = {}
    logger.info(f"📂 Loading dataset config from: {args.dataset_config}")
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = json.load(f)

    if 'num_classes' in dataset_config:
        args.num_classes = dataset_config['num_classes']
        logger.info(f"   -> Overriding num_classes to {args.num_classes}")

    user_class_names = dataset_config.get('class_names', [])
    if len(user_class_names) != args.num_classes:
        logger.warning(
            f"⚠️ Class names count ({len(user_class_names)}) does not match num_classes ({args.num_classes})")
        if len(user_class_names) < args.num_classes:
            user_class_names.extend([f"Class_{i}" for i in range(len(user_class_names), args.num_classes)])

    for k, v in vars(args).items():
        logger.info(f"Config - {k}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3. 数据集加载 ---
    try:
        train_dataset = DW19C(
            data_path=args.data_path,
            mode='train',
            img_size=args.img_size,
            config_path=args.dataset_config
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn_skip_corrupted
        )

        val_dataset = DW19C(
            data_path=args.data_path,
            mode='val',
            img_size=args.img_size,
            config_path=args.dataset_config
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_skip_corrupted
        )

        # DW19C now returns 1 aux channel (the 4th channel)
        aux_chans = 1
        logger.info(f"✅ Data Loaded: Train[{len(train_dataset)}] | Val[{len(val_dataset)}]")
        logger.info(f"ℹ️ Aux Channels: {aux_chans}")

    except Exception as e:
        logger.error(f"❌ Dataset Init Failed: {e}")
        return

    # --- 4. 模型初始化 ---
    logger.info(f"🏗️ Building Model with Backbone: {args.weights_path if not args.resume else 'from checkpoint'}")

    model = DinoMask2Former(
        backbone_weights_path=args.weights_path if not args.resume else "",  # Don't load if resuming
        num_classes=args.num_classes,
        in_chans=3,
        aux_chans=aux_chans
    ).to(device)

    # --- 5. 优化器 ---
    # ================= 1. 定义超参数 =================
    base_lr = args.lr  # 冻结的主干及 Adapter 的基础学习率
    high_lr = base_lr * 10  # 5e-4 -> 2.5e-4，降低倍率以稳定训练
    weight_decay = args.weight_decay
    total_epochs = args.num_epochs  # 您的总训练 Epoch 数
    warmup_epochs = 5  # 预热 Epoch 数

    # ================= 2. 参数分组 =================
    base_params = []
    high_lr_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "csdf_fusion" in name or "gated_lat" in name or "dense_multiscale" in name or "channel_attn" in name:
            high_lr_params.append(param)
        else:
            # 其他所有参与微调的参数（包括预训练的 Norm, 以及 Zero-init 的 HM3E Adapters）
            base_params.append(param)

    optimizer_grouped_parameters = [
        {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": high_lr_params, "lr": high_lr, "weight_decay": weight_decay},
    ]

    # ================= 3. 实例化优化器与调度器 =================
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # 前 5 个 Epoch 线性预热 (1% -> 100%)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # 剩余 Epoch 余弦退火至接近 0
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs),
        eta_min=1e-6
    )

    # 组合调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"🔥 Trainable Params: {trainable_count / 1e6:.2f} M")
    logger.info("Using SequentialLR scheduler with Warmup + Cosine Decay.")

    # --- 7. 损失函数 ---
    # Define class weights
    # Assuming class indices: 14 is Green Space (绿地), 16 is Pylon (电线塔) based on 19c.json order
    # "绿地" is index 14, "电线塔" is index 16
    class_weights = torch.ones(args.num_classes, device=device)
    # class_weights[14] = 10.0  # Increase weight for Green Space
    # class_weights[16] = 10.0  # Increase weight for Pylon

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    logger.info("Using CrossEntropyLoss WITH CLASS WEIGHTS (Green Space & Pylon boosted).")

    # --- 8. 恢复训练 ---
    metric_logger = MetricLogger(num_classes=args.num_classes)
    val_metric_logger = MetricLogger(num_classes=args.num_classes)
    start_epoch = 0
    best_miou = 0.0

    if args.resume and os.path.exists(args.resume):
        logger.info(f"🔄 Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Skip loading optimizer state to reset LR
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # Skip loading scheduler state to reset LR
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)

        logger.info(f"✅ Resume Successful. Starting from Epoch {start_epoch}. Optimizer and Scheduler reset.")
    else:
        logger.info("✨ Starting Training from Scratch.")

    # ====================================================
    # Training Loop
    # ====================================================
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"🚀 Using Mixed Precision: {dtype}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        metric_logger.reset()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.num_epochs} [Train] LR={current_lr:.1e}")

        for i, (optical, aux, masks) in enumerate(pbar):
            if optical is None:
                logger.warning(f"Skipping a batch because all samples in it were corrupted.")
                continue

            optical = optical.to(device)
            aux = aux.to(device)
            masks = masks.to(device)

            if torch.isnan(optical).any() or torch.isinf(optical).any() or torch.isnan(aux).any() or torch.isinf(
                    aux).any():
                logger.warning(f"⚠️ NaN found in input! Skipping batch {i}")
                continue

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=dtype):
                outputs = model(optical, aux)
                loss = criterion(outputs, masks.long())

            if not math.isfinite(loss.item()):
                logger.warning(f"⚠️ Loss is {loss.item()} (NaN or Inf) at batch {i}. Skipping backward pass.")
                # Clear gradients just in case
                optimizer.zero_grad()
                continue

            loss.backward()
            
            # Gradient Clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            # Check for exploding gradients
            if not math.isfinite(grad_norm):
                logger.warning(f"⚠️ Gradient norm is {grad_norm} (NaN or Inf) at batch {i}. Skipping step.")
                optimizer.zero_grad()
                continue
                
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                pred_label = calculate_semantic_segmentation(outputs, target_size=masks.shape[-2:])
                metric_logger.update(pred_label, masks)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Grad": f"{grad_norm:.2f}"})

        scheduler.step()

        train_loss_avg = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_metrics = metric_logger.compute()

        # ====================================================
        # Validation Loop
        # ====================================================
        model.eval()
        val_metric_logger.reset()
        val_total_loss = 0

        logger.info(f"🔍 Validating Epoch {epoch + 1}...")

        with torch.no_grad():
            for optical, aux, masks in tqdm(val_loader, desc="[Val]"):
                if optical is None:
                    continue

                optical = optical.to(device)
                aux = aux.to(device)
                masks = masks.to(device)

                with torch.cuda.amp.autocast(dtype=dtype):
                    outputs = model(optical, aux)
                    loss = criterion(outputs, masks.long())

                val_total_loss += loss.item()
                pred_label = calculate_semantic_segmentation(outputs, target_size=masks.shape[-2:])
                val_metric_logger.update(pred_label, masks)

        val_loss_avg = val_total_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_metrics = val_metric_logger.compute()

        logger.info(
            f"\n📊 Epoch [{epoch + 1}/{args.num_epochs}] Summary:\n"
            f"   Train Loss: {train_loss_avg:.4f} | mIoU: {train_metrics['mIoU']:.4f}\n"
            f"   Val   Loss: {val_loss_avg:.4f}   | mIoU: {val_metrics['mIoU']:.4f} | OA: {val_metrics['OA']:.4f}"
        )

        logger.info("   Val Per-Class IoU:")
        for idx, iou in enumerate(val_metrics['IoU']):
            name = user_class_names[idx] if idx < len(user_class_names) else str(idx)
            mark = "⭐" if iou > 0.5 else "  "
            logger.info(f"   {mark} {name:<10}: {iou:.4f}")

        # --- Checkpoint Saving ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou,
        }

        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint_data, os.path.join(weight_dir, f"epoch_{epoch + 1}.pth"))

        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            checkpoint_data['best_miou'] = best_miou
            save_path = os.path.join(weight_dir, "best_checkpoint_bf32_resume2.pth")
            torch.save(checkpoint_data, save_path)
            logger.info(f"🏆 New Best Record! Saved to {save_path}")

    logger.info("Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv3 + UperHead")

    # Paths
    parser.add_argument("--data_path", type=str,
                        default="/home/ubuntu/workspace/hyb/ht_dataset/v0305_4channel_cunbase_512x512")
    parser.add_argument("--weights_path", type=str, default="dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    parser.add_argument("--save_dir", type=str, default="./output/test-DW19c-3-10-(NDVI+RGB)-removeSGF-test1")
    parser.add_argument("--dataset_config", type=str, default="dataset/19c.json")
    parser.add_argument("--resume", type=str, default=None)

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # Model & Data Config
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    main(args)

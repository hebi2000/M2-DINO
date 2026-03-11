import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from model.vision_transformer import vit_large
from head.uper_head import SimpleUperHead
from model.advanced_fusion import DenseMultiScaleBlock, ChannelAttention, GatedLateralModule


class AuxiliaryExpert(nn.Module):
    """
    Module 1: Auxiliary Expert (Enhanced with Dense Multi-scale & Attention)
    """

    def __init__(self, in_chans=1):
        super().__init__()
        # Stride 4
        self.s1 = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=7, stride=4, padding=3, bias=True),
            nn.ReLU(inplace=True)
        )
        # Stride 8
        self.s2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        # Stride 16
        self.s3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # === 核心升级区：引入密集多尺度与通道注意力 ===
        # 提取大面积地物（如绿地）的连贯性，并提纯近红外特征
        self.dense_multiscale = DenseMultiScaleBlock(in_channels=128, out_channels=128)
        self.channel_attn = ChannelAttention(in_channels=128)
        # ==============================================

        # Dual Output at Stride 16
        self.out_feature = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 1024, 1)
        )
        self.out_gate = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x4 = self.s1(x)
        x8 = self.s2(x4)
        x16 = self.s3(x8)

        # === 唤醒利器：增强空间上下文与过滤噪声，并加入残差旁路 ===
        x16_new = self.dense_multiscale(x16)
        x16_new = self.channel_attn(x16_new)

        # 极其关键：残差连接保护原有特征流，防止初期纯噪声污染！
        x16 = x16 + x16_new
        # ==========================================

        feat = self.out_feature(x16)
        gate = self.out_gate(x16)
        return x4, feat, gate


class DinoMask2Former(nn.Module):  # Class name kept for compatibility, but uses UperHead now
    def __init__(self, backbone_weights_path, num_classes=10, in_chans=3, aux_chans=1):
        super().__init__()

        # Instantiate Backbone
        # Full Adapter Strategy: Inject adapters at shallow and deep layers
        hm3e_indices = [5, 11, 17, 20, 22, 23]
        out_indices = [5, 11, 17, 23]

        self.backbone = vit_large(
            patch_size=16,
            img_size=518,
            layerscale_init=1e-5,
            in_chans=in_chans,
            use_hm3e=True,
            hm3e_indices=hm3e_indices,
            hm3e_rank=32,
            aux_chans=aux_chans,
            out_indices=out_indices
        )

        # Load Backbone Weights
        if os.path.exists(backbone_weights_path):
            print(f"Loading backbone weights from {backbone_weights_path}")
            try:
                state_dict = torch.load(backbone_weights_path, map_location="cpu", weights_only=False)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                if "teacher" in state_dict:
                    state_dict = state_dict["teacher"]
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head")}
                msg = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Backbone weights loaded: {msg}")
            except Exception as e:
                print(f"Error loading backbone weights: {e}")
        else:
            print(f"Warning: Backbone weights not found at {backbone_weights_path}")

        # --- Module 4: Full Adapter Tuning Strategy ---
        # Freeze ALL ViT Blocks
        # Only train Adapters, Norms, and PosEmbed

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False  # Default Freeze

            # Note: We do NOT unfreeze blocks.5 or blocks.23 anymore.
            # The adapters at these layers will handle the adaptation.

            # Unfreeze Norms
            if "norm" in name:  # Catch norm, cls_norm, blocks.x.norm1, etc.
                param.requires_grad = True

            # Unfreeze Positional Embeddings
            if "rope_embed" in name or "pos_embed" in name:
                param.requires_grad = True

            # Unfreeze H-M3E Adapters and Aux Projector
            if "hm3e_adapters" in name or "aux_projector" in name:
                param.requires_grad = True

        # --- Module 1: Auxiliary Expert ---
        self.aux_expert = AuxiliaryExpert(in_chans=aux_chans)

        # --- Module 2: Simplified Fusion (Concat + 1x1 Conv) ---
        embed_dim = 1024
        # Replaced SpatialGatedFusion with a simple 1x1 Conv
        # Input: Concat(RGB_feat, Aux_feat) -> 2 * embed_dim
        # Output: embed_dim
        self.fusion_23 = nn.Conv2d(embed_dim * 2, embed_dim, 1)

        # --- Module 3: Shallow Injection & FPN ---
        out_dim = 256

        # Level 3 (Stride 32) 依然使用简单的 1x1 降维
        self.proj_23 = nn.Conv2d(embed_dim, out_dim, 1)

        # === 核心升级区：唤醒门控侧向连接 ===
        # 替代原本的 self.proj_17 和 self.proj_11
        self.gated_lat_17 = GatedLateralModule(in_channels=embed_dim, out_channels=out_dim)
        self.gated_lat_11 = GatedLateralModule(in_channels=embed_dim, out_channels=out_dim)
        # ==============================================

        # Level 0 (Stride 4) 保留 proj_5，因为此处伴随浅层特征注入
        self.proj_5 = nn.Conv2d(embed_dim, out_dim, 1)
        self.shallow_inject = nn.Conv2d(32, out_dim, 1)
        self.shallow_gate = nn.Sequential(
            nn.Conv2d(out_dim, 1, 1),
            nn.Sigmoid()
        )

        self.embed_dim = out_dim

        # Instantiate Head (Replaced Mask2FormerHead with SimpleUperHead)
        self.head = SimpleUperHead(
            in_channels=self.embed_dim,  # 256
            channels=512,  # Hidden channels for fusion
            num_classes=num_classes
        )

    def forward(self, optical_images, aux_images=None):
        B, C, H, W = optical_images.shape

        # 1. Auxiliary Expert Forward
        # x4: Stride 4, feat: Stride 16, gate: Stride 16
        if aux_images is not None:
            x4, aux_feat, aux_gate = self.aux_expert(aux_images)
        else:
            # Handle case where aux_images might be None
            # Create dummy tensors to avoid crash
            x4 = torch.zeros(B, 32, H // 4, W // 4, device=optical_images.device)
            aux_feat = torch.zeros(B, 1024, H // 16, W // 16, device=optical_images.device)
            aux_gate = torch.zeros(B, 1, H // 16, W // 16, device=optical_images.device)

        # 2. Backbone Forward
        # Pass aux_images to backbone for Adapter interaction
        features_list = self.backbone.forward_features(optical_images, x_aux=aux_images)

        # Reshape tokens to 2D maps
        H_feat, W_feat = H // 16, W // 16
        n_patches = H_feat * W_feat

        feats_2d = []
        for feat in features_list:
            patches = feat[:, -n_patches:, :]
            patches = patches.permute(0, 2, 1).reshape(B, 1024, H_feat, W_feat)
            feats_2d.append(patches)

        f5, f11, f17, f23 = feats_2d

        # 3. Apply Simplified Fusion (Module 2)
        # Only at Layer 23
        # Concatenate RGB features and Aux features
        # f23: [B, 1024, H/16, W/16]
        # aux_feat: [B, 1024, H/16, W/16]
        # aux_gate is ignored

        # Ensure aux_feat matches f23 size (just in case of rounding issues, though strides match)
        if aux_feat.shape[-2:] != f23.shape[-2:]:
            aux_feat = F.interpolate(aux_feat, size=f23.shape[-2:], mode='bilinear', align_corners=False)

        cat_feat = torch.cat([f23, aux_feat], dim=1)  # [B, 2048, H/16, W/16]
        f23_fused = self.fusion_23(cat_feat)  # [B, 1024, H/16, W/16]

        # 4. Build FPN Pyramid
        # Level 3 (Stride 32): From f23_fused
        p23 = self.proj_23(f23_fused)
        p23 = F.avg_pool2d(p23, kernel_size=2, stride=2)

        # Level 2 (Stride 16): Gated Fusion (替代原有的 p17_lat + p23_up)
        # 注意：GatedLateralModule 内部自带 lateral_conv，输入需为 Backbone 原始特征 f17
        p23_up = F.interpolate(p23, size=f17.shape[-2:], mode='bilinear', align_corners=False)
        p17 = self.gated_lat_17(lateral_feat=f17, upsampled_feat=p23_up)

        # Level 1 (Stride 8): Gated Fusion (替代原有的 p11_lat + p17_up)
        p17_up = F.interpolate(p17, size=f11.shape[-2:], mode='bilinear', align_corners=False)
        p11_pre = self.gated_lat_11(lateral_feat=f11, upsampled_feat=p17_up)
        p11 = F.interpolate(p11_pre, scale_factor=2.0, mode='bilinear', align_corners=False)

        # Level 0 (Stride 4): From f5 (Module 3: Shallow Injection 保留原有巧妙设计)
        p5_lat = self.proj_5(f5)
        p5_up = F.interpolate(p5_lat, scale_factor=4.0, mode='bilinear', align_corners=False)

        aux_inject_feat = self.shallow_inject(x4)
        aux_inject_gate = self.shallow_gate(aux_inject_feat)
        aux_inject = aux_inject_feat * aux_inject_gate

        p11_up = F.interpolate(p11, size=p5_up.shape[-2:], mode='bilinear', align_corners=False)

        # Final Fusion for P4
        p5 = p5_up + p11_up + aux_inject

        multi_scale_features = {
            "1": p5,  # Stride 4
            "2": p11,  # Stride 8
            "3": p17,  # Stride 16
            "4": p23,  # Stride 32
        }

        # Head Forward
        outputs = self.head(multi_scale_features)

        # Upsample to original resolution (Stride 4 -> Stride 1)
        outputs = F.interpolate(outputs, size=(H, W), mode='bilinear', align_corners=False)

        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
from .advanced_fusion import CSDFModule

class Aux_Projector(nn.Module):
    """
    通用辅助影像编码器
    """
    def __init__(self, in_chans=1, patch_size=16, embed_dim=1024):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_aux):
        x = self.proj(x_aux)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class ReliabilityRouter(nn.Module):
    def __init__(self, dim, num_experts=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, num_experts)
        )

    def forward(self, x):
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        return weights

class HM3E_Adapter(nn.Module):
    """
    H-M3E 适配器：用于多模态融合和特征适配
    """
    def __init__(self, dim, rank=32):
        super().__init__()
        self.router = ReliabilityRouter(dim, num_experts=3)

        # Expert 1: Optical (LoRA-style)
        self.expert_opt = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, dim, bias=False)
        )

        # Expert 2: Aux (LoRA-style)
        self.expert_aux = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.GELU(),
            nn.Linear(rank, dim, bias=False)
        )

        # Expert 3: Spatial (Conv)
        self.expert_spatial = nn.Sequential(
            nn.Linear(dim, rank, bias=False),
            nn.Conv2d(rank, rank, kernel_size=3, padding=2, dilation=2, groups=rank, bias=False),
            nn.Linear(rank, dim, bias=False)
        )

        self.csdf_fusion = CSDFModule(in_channels=dim)

        # Zero-init last layers for stability
        nn.init.zeros_(self.expert_opt[-1].weight)
        nn.init.zeros_(self.expert_aux[-1].weight)
        nn.init.zeros_(self.expert_spatial[-1].weight)

    def forward(self, x_rgb, H, W, x_aux_token=None):
        # x_rgb: [B, N, C]

        weights = self.router(x_rgb)
        alpha = weights[:, :, 0:1]
        beta = weights[:, :, 1:2]
        gamma = weights[:, :, 2:3]

        # self.last_beta = beta.detach().cpu()          #visual router!!!

        out_opt = self.expert_opt(x_rgb)

        if x_aux_token is not None:
            out_aux = self.expert_aux(x_aux_token)
        else:
            # 如果没有 Aux 输入（例如在浅层只做 RGB 适配），则 Aux 分支为 0
            out_aux = torch.zeros_like(x_rgb)

        # Spatial Expert
        B, N, C = x_rgb.shape
        has_cls = N != H * W

        if has_cls:
            cls_token = x_rgb[:, 0:1, :]
            patch_tokens = x_rgb[:, 1:, :]
        else:
            patch_tokens = x_rgb

        x_2d = self.expert_spatial[0](patch_tokens).transpose(1, 2).reshape(B, -1, H, W)
        x_2d = self.expert_spatial[1](x_2d)
        out_spatial_patches = self.expert_spatial[2](x_2d.flatten(2).transpose(1, 2))

        if has_cls:
            out_spatial_cls = torch.zeros_like(cls_token)
            out_spatial = torch.cat([out_spatial_cls, out_spatial_patches], dim=1)
        else:
            out_spatial = out_spatial_patches

        aggregated_feat = (alpha * out_opt) + (beta * out_aux) + (gamma * out_spatial)

        # CSDF Fusion (Only on patches)
        if has_cls:
            agg_patches = aggregated_feat[:, 1:, :]
            agg_cls = aggregated_feat[:, 0:1, :]
        else:
            agg_patches = aggregated_feat

        agg_patches_2d = agg_patches.transpose(1, 2).reshape(B, C, H, W)
        refined_patches_2d = self.csdf_fusion(agg_patches_2d)
        refined_patches = refined_patches_2d.flatten(2).transpose(1, 2)

        if has_cls:
            out = torch.cat([agg_cls, refined_patches], dim=1)
        else:
            out = refined_patches

        return out + x_rgb

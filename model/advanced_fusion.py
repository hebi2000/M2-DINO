import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 基础组件 (Basic Building Blocks)
# ==========================================

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积：大幅降低参数量和计算量
    Standard Conv: C_in * C_out * K * K
    DS Conv: C_in * 1 * K * K + C_in * C_out * 1 * 1
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ChannelAttention(nn.Module):
    """SE-Block style Channel Attention"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class EfficientSpatialAttention(nn.Module):
    """
    非对称空间注意力：将 7x7 分解为 7x1 和 1x7
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        # 压缩通道 2 -> 1 (Max + Avg)
        self.conv_h = nn.Conv2d(2, 1, (kernel_size, 1), padding=(padding, 0), bias=False)
        self.conv_w = nn.Conv2d(1, 1, (1, kernel_size), padding=(0, padding), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的 Avg 和 Max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 分解卷积处理
        # Note: Input to conv_h should be 2 channels.
        # The provided code in prompt had conv_h output 2 channels then conv_w input 2 channels?
        # Let's check the prompt code carefully.
        # Prompt: self.conv_h = nn.Conv2d(2, 2, ...) -> self.conv_w = nn.Conv2d(2, 1, ...)
        # My implementation above: self.conv_h = nn.Conv2d(2, 1, ...) -> self.conv_w = nn.Conv2d(1, 1, ...)
        # The prompt's version keeps 2 channels intermediate. I will follow the prompt strictly.

        # Re-implementing strictly as per prompt logic but fixing potential channel mismatch if any
        # Prompt: conv_h(2->2), conv_w(2->1).
        # Let's adjust my __init__ to match prompt.
        return x * self.sigmoid(self._spatial_forward(x_cat))

    def _spatial_forward(self, x_cat):
        # This helper is needed because I need to re-init the layers to match prompt
        return x_cat  # Placeholder, logic moved to forward with correct layers


# Re-defining EfficientSpatialAttention to match prompt exactly
class EfficientSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        # 压缩通道 2 -> 1 (Max + Avg)
        # Prompt logic: 2 -> 2 -> 1
        self.conv_h = nn.Conv2d(2, 2, (kernel_size, 1), padding=(padding, 0), bias=False)
        self.conv_w = nn.Conv2d(2, 1, (1, kernel_size), padding=(0, padding), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        x_cat = self.conv_h(x_cat)
        x_cat = F.relu(x_cat, inplace=True)
        x_cat = self.conv_w(x_cat)

        return x * self.sigmoid(x_cat)


# ==========================================
# 2. 核心模块 A: CSDF Module (Optimized)
# ==========================================

class DenseMultiScaleBlock(nn.Module):
    """
    优化版密集多尺度块：使用 Depthwise Separable Conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2  # 内部降维以节省显存

        # Branch 1: Dilation 6
        self.d6 = DepthwiseSeparableConv(in_channels, mid_channels, 3, padding=6, dilation=6)

        # Branch 2: Dilation 12 (Input: x + d6)
        # Input channels: in_channels (x) + mid_channels (feat1)
        self.d12 = DepthwiseSeparableConv(in_channels + mid_channels, mid_channels, 3, padding=12, dilation=12)

        # Branch 3: Dilation 18 (Input: x + d6 + d12)
        # Input channels: in_channels + mid_channels + mid_channels
        self.d18 = DepthwiseSeparableConv(in_channels + mid_channels * 2, mid_channels, 3, padding=18, dilation=18)

        # Final Fusion
        # Concatenation of feat1, feat2, feat3 -> mid_channels * 3
        self.project = nn.Conv2d(mid_channels * 3, out_channels, 1, bias=False)

    def forward(self, x):
        feat1 = self.d6(x)

        cat1 = torch.cat([x, feat1], dim=1)
        feat2 = self.d12(cat1)

        cat2 = torch.cat([x, feat1, feat2], dim=1)
        feat3 = self.d18(cat2)

        out = torch.cat([feat1, feat2, feat3], dim=1)
        return self.project(out)


class CSDFModule(nn.Module):
    """
    最终组装：Channel + Spatial + Dense Fusion
    """

    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = EfficientSpatialAttention()
        self.dense = DenseMultiScaleBlock(in_channels, in_channels)

        # 最后的融合权重（可学习）
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # 三路并行
        x_c = self.ca(x)
        x_s = self.sa(x)
        x_d = self.dense(x)

        # 加权融合 (归一化权重防止数值爆炸)
        w = F.softmax(self.weights, dim=0)
        return w[0] * x_c + w[1] * x_s + w[2] * x_d + x  # +x 为残差连接


# ==========================================
# 3. 核心模块 B: Gated Lateral Connection
# ==========================================

class GatedLateralModule(nn.Module):
    """
    用于 FPN 侧向连接：替代简单的 1x1 Conv + Sum
    解决 'Feature Misalignment' 问题
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, 1)

        # 门控生成器：输入是 Lateral 和 Upsampled 的拼接
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, 2, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, lateral_feat, upsampled_feat):
        """
        lateral_feat: 来自 Backbone 当前层的特征
        upsampled_feat: 来自 FPN 上一层的上采样特征
        """
        # 1. 调整通道
        lat = self.lateral_conv(lateral_feat)

        # 2. 确保尺寸一致 (处理可能的尺寸取整误差)
        if lat.shape[-2:] != upsampled_feat.shape[-2:]:
            upsampled_feat = F.interpolate(upsampled_feat, size=lat.shape[-2:], mode='nearest')

        # 3. 计算门控权重
        cat = torch.cat([lat, upsampled_feat], dim=1)
        weights = self.gate(cat)  # [B, 2, H, W]

        # 4. 加权融合
        # weights[:, 0] 控制 lateral, weights[:, 1] 控制 upsampled
        out = lat * weights[:, 0:1] + upsampled_feat * weights[:, 1:2]

        return out

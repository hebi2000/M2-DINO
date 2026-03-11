import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUperHead(nn.Module):
    """
    A lightweight UperNet-style Head.
    It takes multi-scale features, upsamples them to the highest resolution,
    concatenates them, and predicts the segmentation mask.
    """
    def __init__(self, in_channels, channels, num_classes, dropout_ratio=0.1):
        super().__init__()
        
        # Fusion Convolution
        # Concatenating 4 levels of 'in_channels' -> 'channels'
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        # inputs: dict {"1": p5, "2": p11, "3": p17, "4": p23}
        # p5 is Stride 4.
        
        # Get features
        p5 = inputs["1"]
        p11 = inputs["2"]
        p17 = inputs["3"]
        p23 = inputs["4"]
        
        # Upsample all to p5 size (Stride 4)
        target_size = p5.shape[-2:]
        
        p11_up = F.interpolate(p11, size=target_size, mode='bilinear', align_corners=False)
        p17_up = F.interpolate(p17, size=target_size, mode='bilinear', align_corners=False)
        p23_up = F.interpolate(p23, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate
        x = torch.cat([p5, p11_up, p17_up, p23_up], dim=1)
        
        # Fuse
        x = self.fusion_conv(x)
        
        # Predict
        x = self.dropout(x)
        x = self.conv_seg(x)
        
        return x

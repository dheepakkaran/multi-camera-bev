# ============================================================
# bev_encoder.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# LSS kuduthathu "raw" BEV - just camera features-ai kotti vachathu.
# Antha map-la neighbour cells-ku thodarbu (context) illa.
# Ithu antha map-ai innum smart-a maathum: "intha 4m neelamana blob =
# oru car", "intha vari = road".
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : view_transformer.py output [B,64,200,200]
# Output: center_head.py ku [B,128,200,200]
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# ResNet-style downsample (200->100->50) -> aprom upsample thirumba 200.
# Yaen keezha-mela? Downsample panna oru pixel periya area paakkum
# (receptive field). Athu illaina 12m lorry-va oru pixel-la puriya mudiyathu.
# FPN-style: chinna map-ai periya map-oda koottrom (details + context rendum).
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : [B, 64, 200, 200]
# Output: [B, 128, 200, 200]
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Detection head-ku rich features kudukka.
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.scripts.constants import BACKBONE_OUT_CHANNELS, BEV_OUT_CHANNELS


class BasicBlock(nn.Module):
    """ResNet basic block: conv-bn-relu x2 + skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Skip connection: input-ai output-oda neradiya koottrom.
        # Yaen? Deep network-la gradient vanish aagum. Skip = "highway"
        # gradient-ku, so training stable.
        # Shape match aagalaina 1x1 conv poattu adjust pannurom.
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class BEVEncoder(nn.Module):
    """ResNet-18 style encoder + FPN-style decoder on the BEV map."""

    def __init__(self, in_channels: int = BACKBONE_OUT_CHANNELS,
                 out_channels: int = BEV_OUT_CHANNELS):
        super().__init__()

        # --- Encoder (keezha poguthu) ---
        self.layer1 = BasicBlock(in_channels, 64, stride=1)    # [B,64,200,200]
        self.layer2 = BasicBlock(64, 128, stride=2)            # [B,128,100,100]
        self.layer3 = BasicBlock(128, 256, stride=2)           # [B,256,50,50]

        # --- Decoder (mela varuthu) ---
        # lateral = 1x1 conv, channel count-ai samam pannurathu
        self.lat3 = nn.Conv2d(256, out_channels, 1)
        self.lat2 = nn.Conv2d(128, out_channels, 1)
        self.lat1 = nn.Conv2d(64, out_channels, 1)

        # smooth = upsample panna vandha "blocky" artifacts-ai sari pannum
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 64, 200, 200] raw BEV from LSS

        Returns:
            [B, 128, 200, 200] encoded BEV features
        """
        c1 = self.layer1(x)     # [B, 64, 200, 200]  fine details
        c2 = self.layer2(c1)    # [B,128, 100, 100]  medium context
        c3 = self.layer3(c2)    # [B,256,  50,  50]  large context

        # Top-down: periya context-ai chinna details-oda koottrom
        p3 = self.lat3(c3)                                             # 50x50
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode="bilinear",
                                           align_corners=False)        # 100x100
        p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode="bilinear",
                                           align_corners=False)        # 200x200

        return self.smooth(p1)   # [B, 128, 200, 200]

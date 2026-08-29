# ============================================================
# backbone.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Raw photo pixels model-uku periya sathamana data. Athai "features"-a
# maathanum: "inga oru wheel iruku", "inga road edge" nu.
# EfficientNet-B0 (ImageNet-la already train aanathu) antha velai seiyum.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input: dataset.py kudukkura images
# Output: view_transformer.py (LSS) ku pogum
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# timm features_only EfficientNet-B0 -> stride 16 stage eduthu
# (112 channels) -> 1x1 conv -> 64 channels
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : [N, 3, 224, 400]   (N = B*6, ellaa camera-vum sethu)
# Output: [N, 64, 14, 25]    (224/16=14, 400/16=25)
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# ORE backbone 6 camera-kum (shared weights).
# Yaen? Camera ellathukum "car epdi irukkum" nu same knowledge pothum.
# 6 thani backbone = 6x parameters, waste.
#
# ============================================================

import timm
import torch
import torch.nn as nn

from data.scripts.constants import BACKBONE_OUT_CHANNELS


class CameraBackbone(nn.Module):
    """EfficientNet-B0 based per-camera feature extractor."""

    def __init__(self, out_channels: int = BACKBONE_OUT_CHANNELS, pretrained: bool = True):
        super().__init__()

        # features_only=True -> classifier head remove panni,
        # intermediate feature maps mattum thara.
        # out_indices=(3,) -> stride 16 stage (112 channels)
        # Yaen stride 16? stride 32 romba chinnathu (7x13) - chinna object
        # (pedestrian) miss aagum. stride 8 romba periyathu - slow.
        self.net = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),
        )
        in_ch = self.net.feature_info.channels()[0]   # 112

        # 1x1 conv = channel count-ai mattum maathurathu (spatial size same).
        # 112 -> 64: LSS-la memory kammi aagum (D x C outer product varum).
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, 3, 224, 400] normalized images

        Returns:
            [N, 64, 14, 25] feature maps
        """
        feat = self.net(x)[0]        # [N, 112, 14, 25]
        return self.reduce(feat)     # [N, 64, 14, 25]

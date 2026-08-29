# ============================================================
# center_head.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# BEV features-la irunthu ACTUAL answer-a edukka: "inga oru car iruku,
# ithu size, ithu direction".
# Method: CenterPoint (anchor-free). Anchor boxes ellam venaam -
# "ovvoru cell-la object center irukka?" nu mattum kekurom. Simple.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : bev_encoder.py [B,128,200,200]
# Output: training/losses.py compare pannum (target vs prediction)
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 6 thani chinna conv branch. Ovvondrum oru kelvikku badhil:
#   heatmap -> enna class, enga? (10 channels)
#   offset  -> cell-ku ulla exact edam (2)
#   height  -> z (1)
#   size    -> w, l, h (3)
#   rot     -> sin, cos yaw (2)
#   vel     -> vx, vy (2)
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : [B, 128, 200, 200]
# Output: dict of 6 tensors
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Inference-la: heatmap-la peak (local max) edukurom -> antha cell-oda
# regression values-ai padichi 3D box build pannurom.
#
# ============================================================

import torch
import torch.nn as nn

from data.scripts.constants import BEV_OUT_CHANNELS, N_CLASSES


def _branch(in_ch: int, out_ch: int, mid_ch: int = 64) -> nn.Sequential:
    """
    Oru chinna prediction branch: 3x3 conv -> relu -> 1x1 conv.

    Yaen thani thani branch? Ovvoru prediction-um vera vera vishayam.
    "Class enna" venra kelvikku vendiya feature, "size enna"-ku
    vendiya feature vera. Thani branch kuduthaa each-um specialize aagum.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, 1),
    )


class CenterPointHead(nn.Module):
    """CenterPoint-style multi-task detection head on BEV features."""

    def __init__(self, in_channels: int = BEV_OUT_CHANNELS, n_classes: int = N_CLASSES):
        super().__init__()

        self.heatmap = _branch(in_channels, n_classes)   # 10
        self.offset = _branch(in_channels, 2)
        self.height = _branch(in_channels, 1)
        self.size = _branch(in_channels, 3)
        self.rot = _branch(in_channels, 2)
        self.vel = _branch(in_channels, 2)

        # MUKIYAM trick: heatmap last-layer bias-ai -2.19 nu set pannurom.
        # sigmoid(-2.19) = 0.1. Athaavathu training start-la model
        # "ellaa cell-layum 10% chance object iruku" nu nenaikkum.
        # Yaen? 40000 cell-la ~20 mattum object. Bias illaina model
        # "ellathayum object" nu solli loss vaangi, first few epochs
        # waste aagum.
        self.heatmap[-1].bias.data.fill_(-2.19)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, 128, 200, 200]

        Returns:
            dict of raw predictions (heatmap = LOGITS, sigmoid apply pannala.
            Yaen? Loss function-la sigmoid+loss sethu pannurathu numerically
            stable).
        """
        return {
            "heatmap": self.heatmap(x),   # [B,10,200,200] logits
            "offset": self.offset(x),     # [B, 2,200,200]
            "height": self.height(x),     # [B, 1,200,200]
            "size": self.size(x),         # [B, 3,200,200] log-space
            "rot": self.rot(x),           # [B, 2,200,200] sin, cos
            "vel": self.vel(x),           # [B, 2,200,200]
        }

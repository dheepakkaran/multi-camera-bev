# ============================================================
# simplebev.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# 4 pieces-aiyum (backbone, LSS, encoder, head) ore model-a
# ottu-vathu. Train, export, deploy ellathukum ithu thaan entry point.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# backbone.py + view_transformer.py + bev_encoder.py + center_head.py
# -> training/train.py, export/export_onnx.py ithai use pannum.
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# images [B,6,3,224,400] -> flatten to [B*6,...] -> backbone
# -> unflatten -> LSS (K, E use) -> BEV encoder -> head
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : images, intrinsics, extrinsics
# Output: dict of 6 prediction maps
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# model = SimpleBEV(); preds = model(imgs, K, E)
#
# ============================================================

import torch
import torch.nn as nn

from .backbone import CameraBackbone
from .view_transformer import LiftSplatShoot
from .bev_encoder import BEVEncoder
from .center_head import CenterPointHead


class SimpleBEV(nn.Module):
    """Full multi-camera BEV 3D detection model (~5M params)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = CameraBackbone(pretrained=pretrained)
        self.view_transformer = LiftSplatShoot()
        self.bev_encoder = BEVEncoder()
        self.head = CenterPointHead()

    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor,
                extrinsics: torch.Tensor) -> dict:
        """
        Args:
            images    : [B, 6, 3, 224, 400]
            intrinsics: [B, 6, 3, 3]
            extrinsics: [B, 6, 4, 4]

        Returns:
            dict: heatmap [B,10,200,200], offset/rot/vel [B,2,...],
                  height [B,1,...], size [B,3,...]
        """
        B, N = images.shape[:2]

        # 6 camera-vum ORE backbone-la pogum. Athukku batch dim-la
        # merge pannurom: [B,6,3,H,W] -> [B*6,3,H,W]
        # (GPU-ku ithu ore periya batch madhiri - fast)
        feats = self.backbone(images.flatten(0, 1))       # [B*6, 64, 14, 25]
        feats = feats.view(B, N, *feats.shape[1:])        # [B, 6, 64, 14, 25]

        bev = self.view_transformer(feats, intrinsics, extrinsics)  # [B,64,200,200]
        bev = self.bev_encoder(bev)                                 # [B,128,200,200]

        return self.head(bev)


if __name__ == "__main__":
    # Quick shape test - dataset illama, dummy data vachi
    model = SimpleBEV(pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    imgs = torch.randn(1, 6, 3, 224, 400)
    K = torch.eye(3).repeat(1, 6, 1, 1)
    E = torch.eye(4).repeat(1, 6, 1, 1)

    with torch.no_grad():
        out = model(imgs, K, E)
    for k, v in out.items():
        print(f"{k:>8}: {tuple(v.shape)}")

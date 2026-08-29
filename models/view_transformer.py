# ============================================================
# view_transformer.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Ithu thaan project-oda MOOLAI (brain). 6 camera-oda 2D features-ai
# ORE top-down BEV map-a maathurathu. Algorithm peru: Lift-Splat-Shoot.
#
# Problem: Photo-la depth theriyathu. Oru pixel 2m-layum irukkalaam,
# 50m-layum irukkalaam.
# Solution: "Theriyala, so 64 guess panren, ovvondrukum probability
# kudukiren" - LIFT. Appuram antha points-ai BEV grid-la kottrom - SPLAT.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : backbone.py features + dataset.py K, E
# Output: bev_encoder.py
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 1. Frustum create (14x25 pixel x 64 depth = 22400 3D points/camera)
# 2. K-inverse + E use panni antha points-ai ego car frame-ku maathu
# 3. DepthNet: ovvoru pixel-kum 64 depth probability
# 4. feature x depth_prob = "lifted" 3D features
# 5. Antha features-ai 200x200 BEV cell-la scatter-add (SPLAT)
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : feats [B,6,64,14,25], K [B,6,3,3], E [B,6,4,4]
# Output: bev  [B,64,200,200]
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Itha aprom BEV la ellam 2D detection maadhiri handle pannalaam -
# 3D problem 2D problem-a maarudhu. Athu thaan LSS-oda azhagu.
#
# ============================================================

import torch
import torch.nn as nn

from data.scripts.constants import (
    TARGET_H, TARGET_W, D_MIN, D_MAX, N_DEPTHS,
    BEV_H, BEV_W, BEV_RESOLUTION, X_RANGE, Y_RANGE, Z_RANGE,
    BACKBONE_OUT_CHANNELS,
)


class LiftSplatShoot(nn.Module):
    """2D multi-camera features -> BEV feature map."""

    def __init__(self, in_channels: int = BACKBONE_OUT_CHANNELS,
                 feat_h: int = TARGET_H // 16, feat_w: int = TARGET_W // 16):
        super().__init__()
        self.C = in_channels
        self.D = N_DEPTHS
        self.feat_h, self.feat_w = feat_h, feat_w      # 14, 25

        # DepthNet: ovvoru pixel-kum 64 depth-oda probability
        # 1x1 conv pothum - depth pixel-oda local feature-la irunthu varum
        self.depth_net = nn.Conv2d(in_channels, self.D, kernel_size=1)

        # Frustum = "camera munnadi irukkura 3D cone-la irukkura points"
        # register_buffer: learn aagathu, aana .to(device) la kooda varum
        self.register_buffer("frustum", self._create_frustum(), persistent=False)

    def _create_frustum(self) -> torch.Tensor:
        """
        Feature-map pixel x depth combination ellathaiyum create pannurathu.

        Returns:
            [D, feat_h, feat_w, 3] -> ovvoru entry (u, v, d)
            u,v = ORIGINAL image pixel coords (16x stride-ai thirumba potrom,
                  yaen? K matrix original-ku scale pannirukku, feature map-ku illa)
            d   = metres
        """
        # 2m to 50m varai 64 samam distance-la
        ds = torch.linspace(D_MIN, D_MAX, self.D).view(self.D, 1, 1)
        ds = ds.expand(self.D, self.feat_h, self.feat_w)

        # feature pixel (0..24) -> image pixel (0..399)
        xs = torch.linspace(0, TARGET_W - 1, self.feat_w).view(1, 1, self.feat_w)
        xs = xs.expand(self.D, self.feat_h, self.feat_w)
        ys = torch.linspace(0, TARGET_H - 1, self.feat_h).view(1, self.feat_h, 1)
        ys = ys.expand(self.D, self.feat_h, self.feat_w)

        return torch.stack((xs, ys, ds), dim=-1)      # [D, 14, 25, 3]

    def get_geometry(self, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Frustum points-ai EGO CAR coordinates-ku maathurathu.

        Math (simple-a):
          image pixel (u,v) + depth d
            -> camera 3D: K_inverse @ (u*d, v*d, d)
            -> ego 3D   : R @ point + t

        Example: pixel (200,112) center, d=10m, fx=100, cx=200
            x_cam = (200-200)/100 * 10 = 0m   (center-la irundha 0)
            z_cam = 10m

        Args:
            K: [B, 6, 3, 3]
            E: [B, 6, 4, 4]  camera -> ego

        Returns:
            [B, 6, D, H, W, 3] ego coordinates (metres)
        """
        B, N = K.shape[:2]

        # [B,6,D,H,W,3] la broadcast
        points = self.frustum.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1, 1, 1)

        # (u, v, 1) * d -> homogeneous form
        uvd = torch.cat(
            (points[..., :2] * points[..., 2:3], points[..., 2:3]), dim=-1
        )                                              # [B,6,D,H,W,3]

        # K inverse: image -> camera coords
        K_inv = torch.inverse(K.float())               # [B,6,3,3]
        cam_pts = K_inv.view(B, N, 1, 1, 1, 3, 3) @ uvd.unsqueeze(-1)  # [...,3,1]

        # camera -> ego : R @ p + t
        R = E[..., :3, :3].view(B, N, 1, 1, 1, 3, 3)
        t = E[..., :3, 3].view(B, N, 1, 1, 1, 3)
        ego_pts = (R @ cam_pts).squeeze(-1) + t        # [B,6,D,H,W,3]

        return ego_pts

    def lift(self, feats: torch.Tensor) -> torch.Tensor:
        """
        2D features x depth probability = 3D "lifted" features.

        Args:
            feats: [B*6, C, H, W]

        Returns:
            [B*6, C, D, H, W]
        """
        depth_logits = self.depth_net(feats)                  # [N, D, H, W]
        # softmax over depth: "intha pixel 10m-la 70%, 12m-la 20%..."
        depth_prob = depth_logits.softmax(dim=1)              # [N, D, H, W]

        # outer product: feature vector-ai depth probability-la parichi vidurom
        # feat  [N, C, 1, H, W]  x  depth [N, 1, D, H, W]  =  [N, C, D, H, W]
        return feats.unsqueeze(2) * depth_prob.unsqueeze(1)

    def splat(self, volume: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """
        3D points-ai BEV grid cell-la kotturathu (scatter-add).

        Args:
            volume: [B, 6, C, D, H, W] lifted features
            geom  : [B, 6, D, H, W, 3] ego coordinates

        Returns:
            [B, C, 200, 200]
        """
        B, N, C, D, H, W = volume.shape
        device = volume.device

        # [B, 6, D, H, W, C] -> [B, points, C]
        feats = volume.permute(0, 1, 3, 4, 5, 2).reshape(B, -1, C)
        coords = geom.reshape(B, -1, 3)                        # [B, points, 3]

        # metres -> grid index. floor pannurom (0.4m -> cell 0)
        ix = ((coords[..., 0] - X_RANGE[0]) / BEV_RESOLUTION).long()
        iy = ((coords[..., 1] - Y_RANGE[0]) / BEV_RESOLUTION).long()
        iz = coords[..., 2]

        # Grid-uku veliya / romba mela-keezha irukkura points-ai thookurom
        valid = (
            (ix >= 0) & (ix < BEV_W) &
            (iy >= 0) & (iy < BEV_H) &
            (iz >= Z_RANGE[0]) & (iz <= Z_RANGE[1])
        )                                                       # [B, points]

        # Invalid points-ai thooki-podaama, "kuppai thotti" cell-ku
        # anuppurom (index = H*W, kadaisi la oru extra row).
        # Yaen ippdi? boolean mask use pannina output size ovvoru
        # thadavaiyum maarum (dynamic shape) -> ONNX/TensorRT ku kastam.
        # Ippdi panna shape eppovum fix -> export clean-a varum.
        trash = BEV_H * BEV_W
        flat_idx = torch.where(valid, iy * BEV_W + ix,
                               torch.full_like(ix, trash))    # [B, points]

        # Batch loop illaama ORE scatter-la mudikirom.
        # Batch b-oda cells-ai b*(trash+1) offset-la ninaikirom ->
        # ellaa batch-um ore flat array-la, mothal onnu.
        batch_offset = (torch.arange(B, device=device) * (trash + 1)).view(B, 1)
        flat = (flat_idx + batch_offset).reshape(-1)          # [B*points]

        # ORE cell-la pala point vandha koottum (sum pooling).
        # Yaen sum? Oru car mela pala pixel vizhum - ellathaiyum sethu
        # vachaa signal strong aagum.
        #
        # index_add illaama scatter_add use pannurom. Yaen? ONNX exporter
        # index_add-la duplicate index-ai support pannala - aana namakku
        # duplicate THEVAI (athu thaan pooling). scatter_add ONNX-la
        # ScatterElements(reduction='add') aagum, TensorRT-um support pannum.
        idx = flat.unsqueeze(1).expand(-1, C)                 # [B*points, C]
        bev = torch.zeros(B * (trash + 1), C, device=device, dtype=feats.dtype)
        bev = bev.scatter_add(0, idx, feats.reshape(-1, C))

        bev = bev.view(B, trash + 1, C)[:, :trash]   # kuppai thotti thooku
        # [B, H*W, C] -> [B, C, H, W]
        return bev.view(B, BEV_H, BEV_W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, feats: torch.Tensor, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: [B, 6, C, 14, 25]
            K    : [B, 6, 3, 3]
            E    : [B, 6, 4, 4]

        Returns:
            [B, C, 200, 200] BEV features
        """
        B, N = feats.shape[:2]

        geom = self.get_geometry(K, E)                         # [B,6,D,H,W,3]

        volume = self.lift(feats.flatten(0, 1))                # [B*6,C,D,H,W]
        volume = volume.view(B, N, *volume.shape[1:])          # [B,6,C,D,H,W]

        return self.splat(volume, geom)

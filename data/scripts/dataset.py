# ============================================================
# dataset.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# PyTorch DataLoader-ku puriyara madhiri oru Dataset class venum.
# Rendu velai pannuthu:
#   1. Input taiyaar: 6 camera images + K + E   (sample_loader use)
#   2. TARGET taiyaar: ground-truth boxes -> BEV map format
#      (heatmap, offset, size, rot...) -> loss compare panna
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# sample_loader.load_sample() -> input side
# constants.py -> grid size, classes
# training/train.py ithai DataLoader-la wrap pannum.
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# nuScenes boxes (LIDAR frame) -> ego frame -> BEV grid cell (200x200)
# -> antha cell-la gaussian blob varaiyurom (heatmap)
# -> regression values (offset, z, size, yaw, velocity) store pannurom.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# __getitem__(i) -> dict with images/intrinsics/extrinsics + targets
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Model "intha cell-la car iruku" nu predict pannum.
# Namma target "aama/illa" nu solli, difference = loss.
#
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import (
    BEV_H, BEV_W, BEV_RESOLUTION, X_RANGE, Y_RANGE,
    N_CLASSES, NUSCENES_NAME_MAP, CLASS_TO_IDX,
    VAL_SCENES, DATA_ROOT, VERSION,
)
from .sample_loader import load_sample


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """
    Heatmap-la oru cell suthi "mellisaana veliccham" (gaussian) varaiyurathu.

    Yaen point mattum illa, blob?
    Object center exactly oru cell-la nikkathu - konjam adjacent cell-um
    "kittathatta correct" thaan. So neighbours-ku konjam score kudukirom.
    Ithu illaina training romba kastam (200x200 = 40000 cell-la 1 mattum
    correct-nu solli model-ai therikka mudiyathu).

    Args:
        heatmap: [H, W] array, in-place modify aagum
        cx, cy : center cell (column, row)
        radius : blob size in cells
    """
    diameter = 2 * radius + 1
    sigma = diameter / 6.0                      # standard CenterNet choice

    # Chinna gaussian patch create pannurom
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))   # center=1.0, edge~0
    g[g < np.finfo(g.dtype).eps * g.max()] = 0

    H, W = heatmap.shape
    # Grid edge-la object irundha patch veliya poidum -> clip pannurom
    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)
    if right <= 0 or bottom <= 0 or left < 0 or top < 0:
        return

    masked_hm = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    masked_g = g[radius - top:radius + bottom, radius - left:radius + right]
    # maximum: rendu object overlap aana, periya value-ai vachikirom
    np.maximum(masked_hm, masked_g, out=masked_hm)


class NuScenesBEVDataset(Dataset):
    """
    nuScenes mini -> BEV detection dataset.

    Args:
        data_root: "data/nuscenes-mini"
        split: "train" or "val"
        nusc: already-loaded NuScenes object (optional, reuse panna)
    """

    def __init__(self, data_root: str = DATA_ROOT, split: str = "train", nusc=None):
        from nuscenes.nuscenes import NuScenes

        self.data_root = data_root
        self.split = split
        # NuScenes DB load pannurathu ~10 sec edukkum, so oru thadava mattum
        self.nusc = nusc if nusc is not None else NuScenes(
            version=VERSION, dataroot=data_root, verbose=False
        )

        # Scene level-la split pannurom (sample level illa).
        # Yaen? Ore scene-la adjacent frames kittathatta same photo.
        # Train-la oru frame, val-la adjacent frame irundha = cheating.
        self.sample_tokens = []
        for scene in self.nusc.scene:
            is_val = scene["name"] in VAL_SCENES
            if (split == "val") != is_val:
                continue
            token = scene["first_sample_token"]
            while token:
                self.sample_tokens.append(token)
                token = self.nusc.get("sample", token)["next"]

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def _boxes_in_ego(self, sample_token: str) -> list:
        """
        Antha sample-oda ellaa annotation box-aiyum EGO CAR frame-la thara.

        Yaen ego frame? Namma camera extrinsics-um camera->ego thaan.
        Rendum ore frame-la irundha thaan match aagum.

        Returns:
            list of dict: {cls, x, y, z, w, l, h, yaw, vx, vy}
            x = pinnadi(-)/munnadi(+) metres, y = valathu(-)/idathu(+) metres
        """
        from pyquaternion import Quaternion

        sample = self.nusc.get("sample", sample_token)
        # LIDAR_TOP sample_data -> ego pose reference-ku use pannurom
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", lidar_token)
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])

        ego_t = np.array(ego_pose["translation"])
        ego_R_inv = Quaternion(ego_pose["rotation"]).inverse

        boxes = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            name = NUSCENES_NAME_MAP.get(ann["category_name"])
            if name is None:          # namma 10 class-la illa -> skip
                continue

            # Global (world) coords -> ego coords
            center = np.array(ann["translation"]) - ego_t
            center = ego_R_inv.rotate(center)

            rot = ego_R_inv * Quaternion(ann["rotation"])
            yaw = rot.yaw_pitch_roll[0]        # top-down la yaw mattum thevai

            w, l, h = ann["size"]              # nuScenes order: width,length,height

            vel = self.nusc.box_velocity(ann_token)   # global m/s, nan varalam
            if np.any(np.isnan(vel)):
                vx, vy = 0.0, 0.0
            else:
                v_ego = ego_R_inv.rotate(vel)
                vx, vy = float(v_ego[0]), float(v_ego[1])

            boxes.append({
                "cls": CLASS_TO_IDX[name],
                "x": float(center[0]), "y": float(center[1]), "z": float(center[2]),
                "w": float(w), "l": float(l), "h": float(h),
                "yaw": float(yaw), "vx": vx, "vy": vy,
            })
        return boxes

    def _build_targets(self, boxes: list) -> dict:
        """
        Box list -> BEV target maps (model output-oda same shape).

        Returns dict of tensors:
          heatmap  [10,200,200]  0..1, center-la 1.0
          mask     [1,200,200]   1 = inga object center iruku
          offset   [2,200,200]   cell-uku ulla exact position (0..1)
          height   [1,200,200]   z metres
          size     [3,200,200]   log(w), log(l), log(h)
          rot      [2,200,200]   sin(yaw), cos(yaw)
          vel      [2,200,200]   vx, vy
        """
        heatmap = np.zeros((N_CLASSES, BEV_H, BEV_W), dtype=np.float32)
        mask = np.zeros((1, BEV_H, BEV_W), dtype=np.float32)
        offset = np.zeros((2, BEV_H, BEV_W), dtype=np.float32)
        height = np.zeros((1, BEV_H, BEV_W), dtype=np.float32)
        size = np.zeros((3, BEV_H, BEV_W), dtype=np.float32)
        rot = np.zeros((2, BEV_H, BEV_W), dtype=np.float32)
        vel = np.zeros((2, BEV_H, BEV_W), dtype=np.float32)

        for b in boxes:
            # metres -> grid cell (float)
            # Example: x = 0m   -> (0 - (-50))/0.5 = 100 = grid center
            #          x = 10m (10m munnadi) -> (10+50)/0.5 = 120
            fx = (b["x"] - X_RANGE[0]) / BEV_RESOLUTION
            fy = (b["y"] - Y_RANGE[0]) / BEV_RESOLUTION
            cx, cy = int(fx), int(fy)

            if not (0 <= cx < BEV_W and 0 <= cy < BEV_H):
                continue                       # 100x100m veliya -> skip

            # Object evlo periyathu-nu paathu blob size decide
            radius = max(2, int(min(b["w"], b["l"]) / BEV_RESOLUTION / 2))
            draw_gaussian(heatmap[b["cls"]], cx, cy, radius)

            mask[0, cy, cx] = 1.0
            # int-la potta appuram missing aana decimal part.
            # Ithu illaina 0.5m varai error (cell size).
            offset[0, cy, cx] = fx - cx
            offset[1, cy, cx] = fy - cy
            height[0, cy, cx] = b["z"]
            # log yaen? size 0.5m to 20m varai varum. log potta range
            # chinnathaagum -> network kathukka easy.
            size[0, cy, cx] = np.log(max(b["w"], 0.1))
            size[1, cy, cx] = np.log(max(b["l"], 0.1))
            size[2, cy, cx] = np.log(max(b["h"], 0.1))
            # yaw-ai neradiya predict panna problem: 0 degree = 360 degree
            # aana number-la romba different. sin/cos-la athu solve aagum.
            rot[0, cy, cx] = np.sin(b["yaw"])
            rot[1, cy, cx] = np.cos(b["yaw"])
            vel[0, cy, cx] = b["vx"]
            vel[1, cy, cx] = b["vy"]

        return {
            "heatmap": torch.from_numpy(heatmap),
            "mask": torch.from_numpy(mask),
            "offset": torch.from_numpy(offset),
            "height": torch.from_numpy(height),
            "size": torch.from_numpy(size),
            "rot": torch.from_numpy(rot),
            "vel": torch.from_numpy(vel),
        }

    def __getitem__(self, idx: int) -> dict:
        token = self.sample_tokens[idx]

        sample = load_sample(self.nusc, token, self.data_root)   # inputs
        targets = self._build_targets(self._boxes_in_ego(token))  # labels

        return {
            "images": sample["images"],            # [6,3,224,400]
            "intrinsics": sample["intrinsics"],    # [6,3,3]
            "extrinsics": sample["extrinsics"],    # [6,4,4]
            "targets": targets,
        }

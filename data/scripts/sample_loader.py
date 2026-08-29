# ============================================================
# sample_loader.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Oru "sample" = oru nodi-la (timestamp) 6 camera-vum eduthu photo.
# camera_loader oru camera-va mattum handle pannum.
# Ithu antha 6-ai stack panni ORE tensor-a thara.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# camera_loader.load_camera() -> 6 thadava call
# dataset.py ithai __getitem__-la koopidum.
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# CAMERAS list order-la loop -> 6 dict -> torch.stack -> batch dim add.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : nusc, sample_token
# Output: {"images":[6,3,224,400], "intrinsics":[6,3,3],
#          "extrinsics":[6,4,4], "sample_token": str}
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Model 6 camera-vum ore neram paakkanum (surround view).
# Antha 6-ai ore tensor-a kudukka ithu thevai.
#
# ============================================================

import torch

from .camera_loader import load_camera
from .constants import CAMERAS


def load_sample(nusc, sample_token: str, data_root: str) -> dict:
    """
    Oru sample-oda 6 camera data-vum load panni stack pannurathu.

    Args:
        nusc: NuScenes object
        sample_token: sample-oda unique id
        data_root: dataset folder

    Returns:
        dict of stacked tensors (mela sonna shapes)
    """
    sample = nusc.get("sample", sample_token)

    images, intrinsics, extrinsics = [], [], []

    # CAMERAS order MUKIYAM - ellaa sample-layum ore order irukkanum,
    # illaina model "front camera" nu nenachi back photo paakkum.
    for cam_name in CAMERAS:
        sd_token = sample["data"][cam_name]        # antha camera-oda photo id
        cam = load_camera(nusc, sd_token, data_root)
        images.append(cam["image"])                # [3,224,400]
        intrinsics.append(cam["intrinsic"])        # [3,3]
        extrinsics.append(cam["extrinsic"])        # [4,4]

    # stack = pudhu dimension add pannurathu (list of 6 -> tensor with 6)
    return {
        "images": torch.stack(images, dim=0),          # [6,3,224,400]
        "intrinsics": torch.stack(intrinsics, dim=0),  # [6,3,3]
        "extrinsics": torch.stack(extrinsics, dim=0),  # [6,4,4]
        "sample_token": sample_token,
    }

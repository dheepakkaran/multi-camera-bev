# ============================================================
# camera_loader.py
# ============================================================
#
# ETHUKU CREATE PANNINOM? (Why does this file exist?)
# ━━━━━━━━━━━━━━━━━━━━━━
# ORE oru camera-vukku vendiya 3 vishayathai edukka:
#   1. Image tensor  -> resize + normalize panniyathu
#   2. K (intrinsics) -> camera-oda "lens math" (3x3)
#   3. E (extrinsics) -> camera car-la enga, epadi thirumbi irukku (4x4)
# 6 camera-kum ithe function-a 6 thadava koopiduvom.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# constants.py-la irunthu TARGET_W/H, SCALE, MEAN/STD edukirom.
# sample_loader.py ithai 6 thadava koopidum.
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# nuScenes DB -> photo path + calibration -> cv2 read -> resize ->
# BGR to RGB -> 0..1 -> normalize -> CHW tensor.
# K matrix-ai resize scale-la multiply (MUKIYAM! illaina 3D thappu).
# E = rotation(quaternion->3x3) + translation -> 4x4 matrix.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : nusc object, sample_data_token (str)
# Output: dict {image [3,224,400], intrinsic [3,3], extrinsic [4,4]}
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Model-uku image mattum pothathu. "Intha pixel real world-la enga?"
# nu kandupidikka K, E rendum kandippa venum. LSS athai use pannum.
#
# ============================================================

import cv2
import numpy as np
import torch
from pyquaternion import Quaternion

from .constants import (
    TARGET_W, TARGET_H, SCALE_W, SCALE_H,
    IMAGENET_MEAN, IMAGENET_STD,
)


def load_image(image_path: str) -> torch.Tensor:
    """
    Oru photo-va padichi, resize + normalize panni tensor-a thara.

    Args:
        image_path: full path, e.g. "data/nuscenes-mini/samples/CAM_FRONT/xxx.jpg"

    Returns:
        torch.Tensor shape [3, 224, 400], dtype float32.
        Values roughly -2.5 to +2.5 (normalize pannathaala, 0-1 illa).
    """
    # cv2 BGR order-la padikkum (Blue,Green,Red) - OpenCV oda pazhaya vazhakkam
    img = cv2.imread(image_path)                      # [900, 1600, 3] uint8
    if img is None:
        raise FileNotFoundError(f"Image kedaikala: {image_path}")

    # 1600x900 -> 400x224. INTER_LINEAR = neighbour pixels average
    # (smooth-a suruki, jagged edges varathu)
    img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB. PyTorch/timm ellam RGB expect pannum.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # [224, 400, 3]

    # uint8 (0..255) -> float32 (0..1)
    img = img.astype(np.float32) / 255.0

    # ImageNet normalize: (x - mean) / std
    # Yaen? Pretrained EfficientNet ithe scale-la kathukkichu.
    # Example: pixel 0.485 -> (0.485-0.485)/0.229 = 0.0 (average pixel)
    img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)

    # HWC -> CHW. PyTorch conv layers channel-first expect pannum.
    img = np.transpose(img, (2, 0, 1))                # [3, 224, 400]

    return torch.from_numpy(np.ascontiguousarray(img))


def scale_intrinsic(K: np.ndarray) -> np.ndarray:
    """
    K matrix-ai resize scale-ku match panna adjust pannurathu.

    Yaen ithu MUKIYAM?
    Original photo-la car center pixel (800, 450)-la irunthuchu nu vachiko.
    Photo-va 4x suruki-tom -> car ippo (200, 112)-la.
    Aana K innum "800, 450" nu solli-kittu irundha, LSS car-ai
    thappana edathula BEV-la potrum. So K-yum suruka vendiyathu.

    Args:
        K: [3,3] original intrinsic matrix
           [[fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]]

    Returns:
        [3,3] scaled K. fx,cx -> * 0.25 ; fy,cy -> * 0.2489
    """
    K = K.copy().astype(np.float32)
    K[0, :] *= SCALE_W    # row 0 = x-axis: fx, cx
    K[1, :] *= SCALE_H    # row 1 = y-axis: fy, cy
    return K


def load_camera(nusc, sample_data_token: str, data_root: str) -> dict:
    """
    Oru camera-oda image + K + E moonum load pannurathu.

    Args:
        nusc: NuScenes devkit object (database)
        sample_data_token: intha oru photo-oda unique id
        data_root: dataset folder path

    Returns:
        dict:
          "image"     -> [3, 224, 400] float32
          "intrinsic" -> [3, 3] float32   (scaled K)
          "extrinsic" -> [4, 4] float32   (camera -> ego car transform)
    """
    import os

    sd = nusc.get("sample_data", sample_data_token)

    # --- 1. Image ---
    image = load_image(os.path.join(data_root, sd["filename"]))

    # --- 2 & 3. Calibration (K and E rendum inga irukku) ---
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    K = scale_intrinsic(np.array(calib["camera_intrinsic"]))

    # E = camera coordinate -> ego (car) coordinate
    # rotation quaternion (4 numbers) -> 3x3 rotation matrix
    R = Quaternion(calib["rotation"]).rotation_matrix        # [3,3]
    t = np.array(calib["translation"], dtype=np.float32)     # [3]  metres

    # 4x4 la pack pannurom:
    #   [ R  t ]
    #   [ 0  1 ]
    # Yaen 4x4? Rotation + translation-a ORE matrix multiply-la
    # mudika mudiyum (homogeneous coordinates trick).
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R
    E[:3, 3] = t

    return {
        "image": image,
        "intrinsic": torch.from_numpy(K),
        "extrinsic": torch.from_numpy(E),
    }

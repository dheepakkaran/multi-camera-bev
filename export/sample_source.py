# ============================================================
# sample_source.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# export/ folder-la irukkura scripts-ku (onnx export, int8 calibration,
# benchmark) real camera data thevai. Aana antha data-ku full nuScenes
# dataset (5 GB) + nuscenes-devkit install pannanum.
#
# Kaggle/Colab-la adhu periya thontharavu. So THEVAIYANA data-va mattum
# oru chinna .npz file-la pack pannurom (~80 MB):
#   - 50 sample-oda images (uint8, normalize panna munnadi)
#   - K, E (camera calibration)
#
# Ippo Kaggle-la dataset-e vendaam, devkit-um vendaam.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bundle create: data/scripts/dataset.py use pannum (Mac-la ORE thadava)
# Bundle padikka: export_onnx.py, calibrate_int8.py, benchmark.py
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# uint8 la store pannurom, float32 la illa. Yaen? 4x chinnathu
# (80 MB vs 322 MB). Normalize deterministic operation, so load
# panra podhu thirumba pannalaam - tholaiyura data onnum illa.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# build_bundle()  -> export/calib_bundle.npz ezhuthum
# load_samples()  -> (images [N,6,3,224,400], K [6,3,3], E [6,4,4])
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Mac-la:    python -m export.sample_source        (bundle create)
# Kaggle-la: automatic - scripts bundle-ai thaana kandupidikkum
#
# ============================================================

import os

import numpy as np
import torch

from data.scripts.constants import IMAGENET_MEAN, IMAGENET_STD

BUNDLE_PATH = "export/calib_bundle.npz"


def build_bundle(data_root: str = "data/nuscenes-mini", n_samples: int = 50,
                 out_path: str = BUNDLE_PATH) -> None:
    """
    nuScenes val-la irunthu N sample eduthu chinna .npz-a pack pannurathu.

    Ithu MAC-la ORE thadava odum. Output-ai Kaggle-ku upload pannuvom.

    Args:
        data_root : nuScenes folder
        n_samples : evlo sample (INT8 calibration-ku 50 pothum)
        out_path  : .npz path
    """
    from data.scripts.dataset import NuScenesBEVDataset
    from data.scripts.camera_loader import load_image
    from data.scripts.constants import CAMERAS
    import cv2
    from data.scripts.constants import TARGET_W, TARGET_H

    ds = NuScenesBEVDataset(data_root, split="val")
    n_samples = min(n_samples, len(ds))

    images_u8 = np.zeros((n_samples, 6, TARGET_H, TARGET_W, 3), dtype=np.uint8)

    for i in range(n_samples):
        sample = ds.nusc.get("sample", ds.sample_tokens[i])
        for c, cam_name in enumerate(CAMERAS):
            sd = ds.nusc.get("sample_data", sample["data"][cam_name])
            img = cv2.imread(os.path.join(data_root, sd["filename"]))
            img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            images_u8[i, c] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (i + 1) % 10 == 0:
            print(f"  packed {i+1}/{n_samples}")

    # Ground truth boxes-um serthu vaikirom - visualization-la
    # "nijam idhu" nu dashed-a kaatta thevai. Kaggle-la dataset
    # illaadhathaala ithu illaina GT varaiya mudiyaadhu.
    gt_boxes = np.array([ds._boxes_in_ego(ds.sample_tokens[i])
                         for i in range(n_samples)], dtype=object)

    # Calibration ellaa sample-um ORE car, so K/E onnu pothum
    s0 = ds[0]
    np.savez_compressed(
        out_path,
        images_u8=images_u8,
        intrinsics=s0["intrinsics"].numpy(),
        extrinsics=s0["extrinsics"].numpy(),
        gt_boxes=gt_boxes,
    )
    mb = os.path.getsize(out_path) / 1e6
    print(f"saved {out_path} ({mb:.1f} MB, {n_samples} samples)")


def load_gt_boxes(bundle_path: str = BUNDLE_PATH,
                  data_root: str = "data/nuscenes-mini") -> list:
    """
    Ground truth boxes - bundle-la irundha adhu, illaina dataset-la irundhu.

    Returns:
        list of list of dict (ovvoru sample-kum boxes)
    """
    if os.path.exists(bundle_path):
        d = np.load(bundle_path, allow_pickle=True)
        if "gt_boxes" in d:
            return list(d["gt_boxes"])
    from data.scripts.dataset import NuScenesBEVDataset
    ds = NuScenesBEVDataset(data_root, split="val")
    return [ds._boxes_in_ego(t) for t in ds.sample_tokens[:50]]


def load_samples(bundle_path: str = BUNDLE_PATH, data_root: str = "data/nuscenes-mini"):
    """
    Camera data load pannurathu - bundle irundha adhu, illaina dataset.

    Returns:
        images: [N, 6, 3, 224, 400] float32 normalized torch tensor
        K     : [6, 3, 3]
        E     : [6, 4, 4]
    """
    if os.path.exists(bundle_path):
        d = np.load(bundle_path, allow_pickle=True)
        # uint8 (0..255) -> normalized float32, load_image-la pannurathe
        img = d["images_u8"].astype(np.float32) / 255.0          # [N,6,H,W,3]
        img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / \
              np.array(IMAGENET_STD, dtype=np.float32)
        img = np.transpose(img, (0, 1, 4, 2, 3))                 # -> [N,6,3,H,W]
        print(f"using bundle {bundle_path} ({len(img)} samples, dataset thevai illa)")
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(d["intrinsics"]),
                torch.from_numpy(d["extrinsics"]))

    # Bundle illa -> full dataset use pannu (Mac-la ithu thaan nadakkum)
    from data.scripts.dataset import NuScenesBEVDataset
    ds = NuScenesBEVDataset(data_root, split="val")
    n = min(50, len(ds))
    print(f"using nuScenes dataset ({n} samples)")
    imgs = torch.stack([ds[i]["images"] for i in range(n)])
    s0 = ds[0]
    return imgs, s0["intrinsics"], s0["extrinsics"]


if __name__ == "__main__":
    build_bundle()

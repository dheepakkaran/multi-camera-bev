# ============================================================
# evaluate.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Loss number "5.9" nu vandha, athu nalla-va kettava nu yaarukum
# theriyathu. Recruiter-ku, paper-ku, resume-ku ORE language venum:
# nuScenes-oda official metric = NDS (nuScenes Detection Score).
#
# NDS = mAP + 5 vera error (position, size, angle, velocity, attribute)
# ellathaiyum sethu oru 0-1 score. Ellarum ithe metric use pannuvanga,
# so namma number-ai vera paper-oda compare panna mudiyum.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : train.py save panna checkpoint + dataset.py val split
# Output: NDS/mAP numbers (Phase 5-la INT8 accuracy drop compare panna
#         ithe file-ai thirumba use pannuvom)
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 1. Model predict -> heatmap
# 2. DECODE: heatmap peak -> 3D box (ithu thaan mukkiyamana part)
# 3. ego coords -> global (world) coords  [nuScenes ippadi thaan kekkum]
# 4. JSON submission file ezhuthu
# 5. Official nuscenes-devkit DetectionEval run pannu
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : --ckpt runs/simplebev/best.pth
# Output: NDS, mAP, per-class AP print + results JSON
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
#   python -m training.evaluate --ckpt runs/simplebev/best.pth
#
# ============================================================

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion
from torch.utils.data import DataLoader

from data.scripts.constants import (
    CLASSES, BEV_RESOLUTION, X_RANGE, Y_RANGE, DATA_ROOT, VERSION,
)
from data.scripts.dataset import NuScenesBEVDataset
from models.simplebev import SimpleBEV


def heatmap_nms(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """
    Heatmap-la LOCAL PEAK-ai mattum vachikirathu (NMS-oda cheap version).

    Problem: oru car-ku suthi 8-9 cell-la high score irukkum (gaussian).
    Antha 9-um "9 car" nu report panna koodathu.

    Solution: 3x3 max-pool pannu. Oru cell-oda value athoda 3x3
    neighbourhood-oda max-ku samam-na, athu thaan peak -> vachiko.
    Illaina 0 aakidu.

    Args:
        heat: [B, 10, 200, 200] sigmoid applied (0..1)

    Returns:
        same shape, peak illaatha cells 0
    """
    pooled = F.max_pool2d(heat, kernel, stride=1, padding=kernel // 2)
    keep = (pooled == heat).float()
    return heat * keep


def decode_predictions(preds: dict, max_objects: int = 200,
                       score_threshold: float = 0.05) -> list:
    """
    Model output maps -> 3D box list (ego coordinates).

    Ithu training-ku ethirmaari: training-la box -> map,
    inga map -> box.

    Args:
        preds: center_head output (heatmap = logits)
        max_objects: oru frame-la athigapatcham evlo box
        score_threshold: ithukku keezha score-ai thooki-du

    Returns:
        list (batch size length) of list of dict boxes
    """
    heat = heatmap_nms(torch.sigmoid(preds["heatmap"]))     # [B,10,H,W]
    B, C, H, W = heat.shape

    # Ellaa class + ellaa cell-aiyum ore flat list-a maathi, top-K edukirom
    scores, idx = heat.view(B, -1).topk(min(max_objects, C * H * W), dim=1)

    cls_idx = idx // (H * W)          # entha class
    pix = idx % (H * W)               # entha cell
    ys = pix // W                     # row
    xs = pix % W                      # column

    results = []
    for b in range(B):
        boxes = []
        for k in range(scores.shape[1]):
            score = float(scores[b, k])
            if score < score_threshold:
                break                 # topk sorted, so appuram ellam chinnathu

            cy, cx = int(ys[b, k]), int(xs[b, k])

            # Cell index -> metres. offset = cell-ku ulla decimal part
            # (training-la namma store panna athe thirumba koottrom)
            ox = float(preds["offset"][b, 0, cy, cx])
            oy = float(preds["offset"][b, 1, cy, cx])
            x = (cx + ox) * BEV_RESOLUTION + X_RANGE[0]
            y = (cy + oy) * BEV_RESOLUTION + Y_RANGE[0]
            z = float(preds["height"][b, 0, cy, cx])

            # Training-la log() poattom, so ippo exp() poattu thirumba edukirom
            w = float(np.exp(preds["size"][b, 0, cy, cx].item()))
            l = float(np.exp(preds["size"][b, 1, cy, cx].item()))
            h = float(np.exp(preds["size"][b, 2, cy, cx].item()))

            # sin, cos -> angle. atan2 use pannurom (quadrant correct-a varum)
            sin_y = float(preds["rot"][b, 0, cy, cx])
            cos_y = float(preds["rot"][b, 1, cy, cx])
            yaw = float(np.arctan2(sin_y, cos_y))

            boxes.append({
                "cls": int(cls_idx[b, k]), "score": score,
                "x": x, "y": y, "z": z, "w": w, "l": l, "h": h, "yaw": yaw,
                "vx": float(preds["vel"][b, 0, cy, cx]),
                "vy": float(preds["vel"][b, 1, cy, cx]),
            })
        results.append(boxes)
    return results


def ego_to_global(box: dict, ego_pose: dict, sample_token: str) -> dict:
    """
    Ego (car) coordinates -> global (world) coordinates.

    Yaen thevai? nuScenes official evaluation global coords-la thaan
    ground truth vachirukku. Namma prediction-um anga kondu poganum,
    illaina "car 10m munnadi" nu sonna, evlo thooram-nu compare panna
    mudiyaathu (car nagarndhindae irukku).

    Args:
        box: decode_predictions kuduthathu (ego frame)
        ego_pose: nuScenes ego_pose record (car world-la enga irukku)
        sample_token: entha frame-oda box nu nuScenes eval kekkum

    Returns:
        dict with global translation / rotation / velocity
    """
    ego_R = Quaternion(ego_pose["rotation"])
    ego_t = np.array(ego_pose["translation"])

    center = ego_R.rotate(np.array([box["x"], box["y"], box["z"]])) + ego_t
    rotation = ego_R * Quaternion(axis=[0, 0, 1], angle=box["yaw"])
    vel = ego_R.rotate(np.array([box["vx"], box["vy"], 0.0]))

    return {
        "sample_token": sample_token,
        "translation": center.tolist(),
        "size": [box["w"], box["l"], box["h"]],   # nuScenes order: w, l, h
        "rotation": list(rotation),               # [w, x, y, z]
        "velocity": [float(vel[0]), float(vel[1])],
        "detection_name": CLASSES[box["cls"]],
        "detection_score": box["score"],
        "attribute_name": "",                     # namma predict pannala
    }


@torch.no_grad()
def run_inference(model, dataset, device, batch_size: int = 2,
                  score_threshold: float = 0.05) -> dict:
    """Val split muzhuthum predict panni, nuScenes format dict thara."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}
    sample_i = 0
    for batch in loader:
        preds = model(batch["images"].to(device),
                      batch["intrinsics"].to(device),
                      batch["extrinsics"].to(device))
        preds = {k: v.float().cpu() for k, v in preds.items()}
        decoded = decode_predictions(preds, score_threshold=score_threshold)

        for boxes in decoded:
            token = dataset.sample_tokens[sample_i]
            sample_i += 1

            # Antha sample-oda car world-la enga irundhuthu?
            sd = dataset.nusc.get(
                "sample_data",
                dataset.nusc.get("sample", token)["data"]["LIDAR_TOP"])
            ego_pose = dataset.nusc.get("ego_pose", sd["ego_pose_token"])

            results[token] = [ego_to_global(b, ego_pose, token) for b in boxes]

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/simplebev/best.pth")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-dir", default="runs/eval")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    dataset = NuScenesBEVDataset(args.data_root, split="val")
    print(f"val samples: {len(dataset)}  | device: {device}")

    model = SimpleBEV(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss'):.4f})")

    results = run_inference(model, dataset, device, args.batch_size,
                            args.score_threshold)
    n_boxes = sum(len(v) for v in results.values())
    print(f"predicted {n_boxes} boxes ({n_boxes / max(len(results),1):.1f} per frame)")

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "results_nusc.json")
    with open(result_path, "w") as f:
        # "meta" = enna sensor use panninom nu nuScenes kekkum.
        # Namma camera mattum -> vera ellam False (Tesla style!)
        json.dump({
            "meta": {"use_camera": True, "use_lidar": False, "use_radar": False,
                     "use_map": False, "use_external": False},
            "results": results,
        }, f)
    print(f"wrote {result_path}")

    # --- Official nuScenes evaluation ---
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    evaluator = DetectionEval(
        dataset.nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path=result_path,
        eval_set="mini_val",
        output_dir=args.output_dir,
        verbose=False,
    )
    metrics, _ = evaluator.evaluate()
    summary = metrics.serialize()

    print("\n" + "=" * 46)
    print(f"  NDS  {summary['nd_score']:.4f}")
    print(f"  mAP  {summary['mean_ap']:.4f}")
    print("=" * 46)
    print("  per-class AP:")
    for name, ap in summary["mean_dist_aps"].items():
        print(f"    {name:22s} {ap:.4f}")
    print("  TP errors (kammi-na nalla):")
    for name, val in summary["tp_errors"].items():
        print(f"    {name:22s} {val:.4f}")


if __name__ == "__main__":
    main()

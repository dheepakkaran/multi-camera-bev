# ============================================================
# make_demo.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# README-ku / LinkedIn-ku oru portfolio demo venum. Recruiter 5 second
# thaan paapaanga - antha 5 second-la "6 camera -> BEV detections" +
# "TensorRT 3.9x speedup" rendum theriyanum.
#
# bev_renderer.py oru frame varaiyum. Ithu:
#   - thodarchiyaana frames (scene-la oru vari, so motion theriyum)
#   - real T4 benchmark numbers-ai padathula-ye potrom
#   - GIF + thani PNG rendum
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# visualization/bev_renderer.py -> draw_bev, denormalize reuse
# training/evaluate.py -> decode_predictions
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Output: runs/demo/demo.gif + frame_*.png
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# python -m visualization.make_demo --frames 20
#
# ============================================================

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.scripts.constants import CAMERAS, CLASSES
from visualization.bev_renderer import denormalize, draw_bev, CLASS_COLORS

# Kaggle T4-la nijamaa alandha numbers. Hardcode pannurom yaen?
# Ithu Mac-la odum, aana kaatta vendiyathu T4 result. Mac-la
# TensorRT-e illa, so anga alandhadhai inga kondu varom.
BENCHMARK = [
    ("PyTorch FP32",  36.40, 27.5, "1.00x", "baseline"),
    ("ONNX Runtime",  33.80, 29.6, "1.08x", "100%"),
    ("TensorRT FP32", 28.21, 35.5, "1.29x", "-"),
    ("TensorRT FP16", 63.10, 15.8, "0.58x", "72%"),
    ("TensorRT INT8",  9.35, 107.0, "3.89x", "53%"),
]


def draw_stats_panel(ax) -> None:
    """
    Benchmark table-ai padathula text-a varaiyurathu.

    Yaen padathula? README-la thani table irundhaalum, GIF-ai
    LinkedIn-la neradiya podumbodhu numbers kooda pogum.
    """
    ax.axis("off")
    ax.set_facecolor("#0a0a14")

    y = 0.97
    ax.text(0.5, y, "Tesla T4  |  6 cameras -> BEV", ha="center",
            color="#e0e0e0", fontsize=11, weight="bold",
            transform=ax.transAxes)

    y -= 0.09
    ax.text(0.02, y, f"{'Backend':<15}{'ms':>7}{'FPS':>7}{'speedup':>9}",
            color="#8a8a9a", fontsize=8.5, family="monospace",
            transform=ax.transAxes)
    y -= 0.035
    ax.plot([0.02, 0.98], [y, y], color="#2a2a3a", lw=0.8,
            transform=ax.transAxes, clip_on=False)

    for name, ms, fps, speed, agree in BENCHMARK:
        y -= 0.055
        # INT8 thaan star - highlight pannurom
        best = name == "TensorRT INT8"
        color = "#00d4ff" if best else "#b0b0c0"
        ax.text(0.02, y, f"{name:<15}{ms:>7.1f}{fps:>7.1f}{speed:>9}",
                color=color, fontsize=8.5, family="monospace",
                weight="bold" if best else "normal",
                transform=ax.transAxes)

    y -= 0.10
    for line, col in [
        ("INT8: 3.89x faster, 17x smaller engine", "#00d4ff"),
        ("(49.4 MB -> 2.9 MB)", "#6a6a7a"),
        ("", "#6a6a7a"),
        ("Trade-off: 53% box agreement -", "#ff8c00"),
        ("quantization hurts an undertrained", "#8a8a9a"),
        ("model. FP32 engine ships instead.", "#8a8a9a"),
    ]:
        ax.text(0.02, y, line, color=col, fontsize=8,
                transform=ax.transAxes)
        y -= 0.045

    y -= 0.03
    ax.plot([0.02, 0.98], [y, y], color="#2a2a3a", lw=0.8,
            transform=ax.transAxes, clip_on=False)
    y -= 0.055
    for line in ["SimpleBEV  5.5M params",
                 "EfficientNet-B0 + Lift-Splat-Shoot",
                 "ResNet-18 FPN + CenterPoint head",
                 "nuScenes mini  |  no LiDAR"]:
        ax.text(0.02, y, line, color="#6a6a7a", fontsize=7.5,
                transform=ax.transAxes)
        y -= 0.04


def render(images, gt_boxes, pred_boxes, out_path, frame_i, n_frames,
           score_thresh=0.2):
    """Oru demo frame: cameras | BEV | stats."""
    fig = plt.figure(figsize=(19, 7.5), facecolor="#0a0a14")
    gs = fig.add_gridspec(3, 5, width_ratios=[0.78, 0.78, 0.1, 1.7, 0.95],
                          hspace=0.1, wspace=0.05)

    layout = [("CAM_FRONT_LEFT", 0, 0), ("CAM_FRONT_RIGHT", 0, 1),
              ("CAM_FRONT",      1, 0), ("CAM_BACK",        1, 1),
              ("CAM_BACK_LEFT",  2, 0), ("CAM_BACK_RIGHT",  2, 1)]
    for cam, r, c in layout:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(denormalize(images[CAMERAS.index(cam)]))
        ax.set_title(cam.replace("CAM_", ""), color="#8a8a9a", fontsize=6.5, pad=1)
        ax.axis("off")

    ax_bev = fig.add_subplot(gs[:, 3])
    draw_bev(ax_bev, gt_boxes, pred_boxes, score_thresh)
    n = sum(1 for b in pred_boxes if b["score"] >= score_thresh)
    ax_bev.set_title(f"Bird's Eye View  |  frame {frame_i+1}/{n_frames}  |  "
                     f"{n} detections", color="#e0e0e0", fontsize=10, pad=8)

    present = sorted({b["cls"] for b in pred_boxes if b["score"] >= score_thresh})
    if present:
        handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=6,
                              color=CLASS_COLORS[c], label=CLASSES[c])
                   for c in present]
        ax_bev.legend(handles=handles, loc="upper left", fontsize=7,
                      facecolor="#14141e", edgecolor="#2a2a3a",
                      labelcolor="#c0c0d0")

    draw_stats_panel(fig.add_subplot(gs[:, 4]))

    fig.text(0.42, 0.02, "white dashed = ground truth   |   filled = prediction",
             ha="center", color="#6a6a7a", fontsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/simplebev/best.pth")
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    ap.add_argument("--out-dir", default="runs/demo")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--ms-per-frame", type=int, default=450)
    args = ap.parse_args()

    import torch
    from data.scripts.dataset import NuScenesBEVDataset
    from models.simplebev import SimpleBEV
    from training.evaluate import decode_predictions

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleBEV(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    ds = NuScenesBEVDataset(args.data_root, split="val")
    n = min(args.frames, len(ds) - args.start)
    paths = []

    for k in range(n):
        i = args.start + k
        s = ds[i]
        with torch.no_grad():
            preds = model(s["images"].unsqueeze(0).to(device),
                          s["intrinsics"].unsqueeze(0).to(device),
                          s["extrinsics"].unsqueeze(0).to(device))
        boxes = decode_predictions({k2: v.cpu() for k2, v in preds.items()})[0]
        gt = ds._boxes_in_ego(ds.sample_tokens[i])

        p = os.path.join(args.out_dir, f"frame_{k:03d}.png")
        render(s["images"].numpy(), gt, boxes, p, k, n, args.score_thresh)
        paths.append(p)
        print(f"  frame {k+1}/{n}")

    from PIL import Image
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128)
              for p in paths]
    gif = os.path.join(args.out_dir, "demo.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=args.ms_per_frame, loop=0, optimize=True)
    print(f"\nsaved {gif} ({os.path.getsize(gif)/1e6:.1f} MB, {n} frames)")
    print(f"saved {n} PNGs in {args.out_dir}/")


if __name__ == "__main__":
    main()

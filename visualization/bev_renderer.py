# ============================================================
# bev_renderer.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Numbers (heatmap tensors) yaarukum puriyaadhu. Recruiter oru PADAM
# paakanum: "6 camera photo -> top-down map-la cars marked".
# Ithu antha padathai varaiyum.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : training/evaluate.py-oda decode_predictions output
#         + dataset._boxes_in_ego (ground truth)
# Output: PNG / GIF -> README-la pogum
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# Matplotlib-la 2 panel: (idathu) 6 camera photos grid,
# (valathu) top-down BEV - ego car center-la, boxes suthi.
# Box varaiyurathu: center + w,l + yaw -> 4 corner points -> polygon.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# render_frame(images, gt_boxes, pred_boxes, out_path)
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# python -m visualization.bev_renderer --ckpt runs/simplebev/best.pth
#
# ============================================================

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # display illa (headless) - file-la mattum save
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from data.scripts.constants import (
    CAMERAS, CLASSES, X_RANGE, Y_RANGE, IMAGENET_MEAN, IMAGENET_STD,
)

# Ovvoru class-kum oru color. Recruiter oru nodi-la puriyanum.
CLASS_COLORS = {
    0: "#00d4ff",   # car          - cyan
    1: "#ff8c00",   # truck        - orange
    2: "#ffd700",   # bus          - gold
    3: "#8b4513",   # trailer      - brown
    4: "#9370db",   # construction - purple
    5: "#ff1493",   # pedestrian   - pink
    6: "#00ff7f",   # motorcycle   - spring green
    7: "#7fff00",   # bicycle      - chartreuse
    8: "#ff4500",   # traffic_cone - red-orange
    9: "#a9a9a9",   # barrier      - grey
}


def denormalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize panna image-ai thirumba paakkura maadhiri maathurathu.

    Args:
        img: [3, H, W] normalized (-2.5 to 2.5)

    Returns:
        [H, W, 3] uint8-range float (0..1) - matplotlib ku
    """
    img = img.transpose(1, 2, 0)                       # CHW -> HWC
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def box_corners(x: float, y: float, w: float, l: float, yaw: float) -> np.ndarray:
    """
    Box center + size + angle -> 4 mooli (corner) points.

    Math simple-a: mudhalla car origin-la irukku nu nenachi 4 corner
    kanakku podurom, appuram yaw angle-la suthi, appuram (x,y) ku nagarthurom.

    Args:
        x, y : center metres (x=munnadi, y=idathu)
        w, l : width, length metres
        yaw  : radians

    Returns:
        [4, 2] corner coordinates
    """
    # Car-oda soontha frame-la 4 corners (length = munnadi axis)
    corners = np.array([
        [ l / 2,  w / 2],
        [ l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2,  w / 2],
    ])

    # 2D rotation matrix - yaw angle-la suthurathu
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])

    return corners @ R.T + np.array([x, y])


def draw_bev(ax, gt_boxes: list, pred_boxes: list, score_thresh: float = 0.2) -> None:
    """
    Top-down BEV panel varaiyurathu.

    GT = white dashed outline (nijam)
    Pred = class color filled (namma model sonnathu)
    """
    ax.set_facecolor("#0a0a14")

    # --- Grid lines (10m ku oru vari) - dooram purinjukka ---
    for r in range(10, 51, 10):
        ax.add_patch(plt.Circle((0, 0), r, fill=False,
                                edgecolor="#2a2a3a", linewidth=0.6, zorder=1))
        ax.text(r * 0.7, r * 0.7, f"{r}m", color="#4a4a5a", fontsize=7, zorder=1)

    # --- Ego car (namma vandi) center-la ---
    ego = box_corners(0, 0, 1.8, 4.5, 0)
    ax.add_patch(Polygon(ego, closed=True, facecolor="#ffffff",
                         edgecolor="#ffffff", linewidth=1, zorder=5))
    ax.arrow(0, 2.5, 0, 3, head_width=1.2, head_length=1.2,
             fc="#ffffff", ec="#ffffff", zorder=5)   # munnadi direction

    # --- Ground truth ---
    for b in gt_boxes:
        pts = box_corners(b["x"], b["y"], b["w"], b["l"], b["yaw"])
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor="#ffffff",
                             linewidth=1.0, linestyle="--", alpha=0.55, zorder=3))

    # --- Predictions ---
    n_drawn = 0
    for b in pred_boxes:
        if b["score"] < score_thresh:
            continue
        color = CLASS_COLORS.get(b["cls"], "#ffffff")
        pts = box_corners(b["x"], b["y"], b["w"], b["l"], b["yaw"])
        ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=0.45,
                             edgecolor=color, linewidth=1.2, zorder=4))
        n_drawn += 1

    ax.set_xlim(Y_RANGE[1], Y_RANGE[0])      # y = idathu, plot-la idathu pakkam
    ax.set_ylim(X_RANGE[0], X_RANGE[1])      # x = munnadi, plot-la mela
    ax.set_aspect("equal")
    ax.set_xlabel("left  <-  y (m)  ->  right", color="#8a8a9a", fontsize=8)
    ax.set_ylabel("back  <-  x (m)  ->  front", color="#8a8a9a", fontsize=8)
    ax.set_title(f"BEV  |  white dashed = ground truth  |  filled = predicted "
                 f"({n_drawn} boxes, score>{score_thresh})",
                 color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#4a4a5a", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2a3a")


def render_frame(images: np.ndarray, gt_boxes: list, pred_boxes: list,
                 out_path: str, score_thresh: float = 0.2) -> None:
    """
    Oru frame-oda full visualization save pannurathu.

    Args:
        images: [6, 3, 224, 400] normalized
        gt_boxes / pred_boxes: dict list (x, y, w, l, yaw, cls, [score])
        out_path: PNG path
    """
    fig = plt.figure(figsize=(17, 7), facecolor="#0a0a14")
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.12, 1.9],
                          hspace=0.08, wspace=0.04)

    # --- Idathu pakkam: 6 camera photos ---
    # Car-la epdi irukko appdiye layout: mela munnadi, keezha pinnadi.
    # Ippdi potta padathai paakkurava-ku surround view neradiya purinjidum.
    layout = [
        ("CAM_FRONT_LEFT", 0, 0), ("CAM_FRONT", 0, 1), ("CAM_FRONT_RIGHT", 0, 2),
        ("CAM_BACK_LEFT",  1, 0), ("CAM_BACK",  1, 1), ("CAM_BACK_RIGHT",  1, 2),
    ]
    for cam, r, c in layout:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(denormalize(images[CAMERAS.index(cam)]))
        ax.set_title(cam.replace("CAM_", ""), color="#8a8a9a", fontsize=7, pad=2)
        ax.axis("off")

    # --- Valathu pakkam: BEV ---
    ax_bev = fig.add_subplot(gs[:, 4])
    draw_bev(ax_bev, gt_boxes, pred_boxes, score_thresh)

    # --- Class color legend ---
    present = sorted({b["cls"] for b in pred_boxes if b["score"] >= score_thresh}
                     | {b["cls"] for b in gt_boxes})
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=7,
                          color=CLASS_COLORS[c], label=CLASSES[c]) for c in present]
    if handles:
        ax_bev.legend(handles=handles, loc="upper left", fontsize=7,
                      facecolor="#14141e", edgecolor="#2a2a3a", labelcolor="#c0c0d0")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/simplebev/best.pth")
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    ap.add_argument("--out-dir", default="runs/viz")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--gif", action="store_true", help="frames-ai GIF-a sethu vai")
    args = ap.parse_args()

    import torch
    from data.scripts.dataset import NuScenesBEVDataset
    from models.simplebev import SimpleBEV
    from training.evaluate import decode_predictions

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SimpleBEV(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    ds = NuScenesBEVDataset(args.data_root, split="val")
    paths = []

    for i in range(min(args.frames, len(ds))):
        s = ds[i]
        with torch.no_grad():
            preds = model(s["images"].unsqueeze(0).to(device),
                          s["intrinsics"].unsqueeze(0).to(device),
                          s["extrinsics"].unsqueeze(0).to(device))
        pred_boxes = decode_predictions({k: v.cpu() for k, v in preds.items()})[0]
        gt_boxes = ds._boxes_in_ego(ds.sample_tokens[i])

        p = os.path.join(args.out_dir, f"frame_{i:03d}.png")
        render_frame(s["images"].numpy(), gt_boxes, pred_boxes, p, args.score_thresh)
        paths.append(p)

    if args.gif and paths:
        from PIL import Image
        frames = [Image.open(p) for p in paths]
        gif_path = os.path.join(args.out_dir, "demo.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=700, loop=0)
        print(f"saved {gif_path}")


if __name__ == "__main__":
    main()

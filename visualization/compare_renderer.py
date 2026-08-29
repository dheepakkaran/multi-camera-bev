# ============================================================
# compare_renderer.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# "INT8-ku maathina 4x fast, accuracy kuraiyala" nu sonna, adhukku
# PROOF venum. Numbers table onnu iruku - aana adhu boring, and
# "boxes appadiye irukka?" nu neradiya kaatta maattadhu.
#
# Ithu rendu backend-oda output-ai PAKKAM-PAKKAM varaiyum, latency +
# agreement numbers-ai padathula-ye potrum. Oru nodi-la puriyum.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# export/backends.py    -> rendu backend-aiyum ore madhiri odukka
# export/sample_source.py -> images + ground truth (dataset thevai illa)
# visualization/bev_renderer.py -> draw_bev reuse pannurom
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 1. Rendu backend-um same frame-la odum
# 2. Latency alakkurom
# 3. AGREEMENT: A-oda box-ku B-la 0.5m ulla match irukka?
#    (INT8 sariyaana velai seiyudha nu ithu thaan neradiyaana answer)
# 4. Matplotlib-la 3 panel: cameras | backend A BEV | backend B BEV
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Output: runs/compare/compare_XXX.png + compare.gif
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Mac-la (test):  python -m visualization.compare_renderer -a pytorch -b onnx
# Kaggle-la:      python -m visualization.compare_renderer -a pytorch -b trt_int8
#
# ============================================================

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.scripts.constants import CAMERAS
from visualization.bev_renderer import denormalize, draw_bev, CLASS_COLORS


def box_agreement(boxes_a: list, boxes_b: list, dist_thresh: float = 0.5) -> float:
    """
    Rendu backend-oda boxes evlo match aaguthu nu %-la thara.

    A-la irukkura ovvoru box-kum, B-la 0.5m ulla same-class box
    irukka nu paakurom.

    Yaen ithu mukkiyam? INT8 quantization apram boxes appadiye
    irundha, "accuracy poganum" nu bayam vendaam nu proof.

    Args:
        boxes_a, boxes_b: decode_predictions output
        dist_thresh: metres

    Returns:
        0..100 percentage
    """
    if not boxes_a:
        return 100.0 if not boxes_b else 0.0

    matched = 0
    for ba in boxes_a:
        for bb in boxes_b:
            if ba["cls"] != bb["cls"]:
                continue
            d = np.hypot(ba["x"] - bb["x"], ba["y"] - bb["y"])
            if d <= dist_thresh:
                matched += 1
                break
    return 100.0 * matched / len(boxes_a)


def render_comparison(images: np.ndarray, gt_boxes: list,
                      result_a: dict, result_b: dict,
                      out_path: str, score_thresh: float = 0.2) -> None:
    """
    Oru frame-oda comparison padam.

    Args:
        images  : [6,3,224,400] normalized
        gt_boxes: ground truth list
        result_a/b: {"name":, "boxes":, "latency_ms":, "extra": str}
        out_path: PNG path
    """
    fig = plt.figure(figsize=(19, 7.5), facecolor="#0a0a14")
    # 3 row x 2 col la 6 camera -> BEV panel-oda uyaram-ku match aagum,
    # kaali idam varaadhu
    gs = fig.add_gridspec(3, 5, width_ratios=[0.8, 0.8, 0.12, 1.45, 1.45],
                          hspace=0.1, wspace=0.06)

    # --- Idathu: 6 camera (context-ku) ---
    layout = [("CAM_FRONT_LEFT", 0, 0), ("CAM_FRONT_RIGHT", 0, 1),
              ("CAM_FRONT",      1, 0), ("CAM_BACK",        1, 1),
              ("CAM_BACK_LEFT",  2, 0), ("CAM_BACK_RIGHT",  2, 1)]
    for cam, r, c in layout:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(denormalize(images[CAMERAS.index(cam)]))
        ax.set_title(cam.replace("CAM_", ""), color="#8a8a9a", fontsize=6, pad=1)
        ax.axis("off")

    # --- Rendu BEV panel ---
    for col, res in [(3, result_a), (4, result_b)]:
        ax = fig.add_subplot(gs[:, col])
        draw_bev(ax, gt_boxes, res["boxes"], score_thresh)

        # Class legend - valathu panel-la mattum (rendu-layum vendaam)
        if col == 4:
            from data.scripts.constants import CLASSES
            present = sorted({b["cls"] for b in res["boxes"]
                              if b["score"] >= score_thresh})
            if present:
                handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                                      markersize=6, color=CLASS_COLORS[c],
                                      label=CLASSES[c]) for c in present]
                ax.legend(handles=handles, loc="upper left", fontsize=6.5,
                          facecolor="#14141e", edgecolor="#2a2a3a",
                          labelcolor="#c0c0d0")

        n = sum(1 for b in res["boxes"] if b["score"] >= score_thresh)
        fps = 1000.0 / res["latency_ms"] if res["latency_ms"] else 0
        title = (f"{res['name']}\n"
                 f"{res['latency_ms']:.1f} ms   |   {fps:.0f} FPS   |   "
                 f"{n} boxes")
        if res.get("extra"):
            title += f"\n{res['extra']}"
        ax.set_title(title, color="#e0e0e0", fontsize=10, pad=8)

    fig.text(0.5, 0.015,
             "white dashed = ground truth    |    filled = model prediction",
             ha="center", color="#6a6a7a", fontsize=8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=125, bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--backend-a", default="pytorch")
    ap.add_argument("-b", "--backend-b", default="onnx")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out-dir", default="runs/compare")
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--ckpt", default="runs/simplebev/best.pth")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    import torch
    from export.backends import get_backend
    from export.sample_source import load_samples, load_gt_boxes
    from training.evaluate import decode_predictions

    imgs, K, E = load_samples()
    gts = load_gt_boxes()
    n_frames = min(args.frames, len(imgs))

    print(f"loading backends: {args.backend_a}  vs  {args.backend_b}")
    ba = get_backend(args.backend_a, ckpt=args.ckpt, K=K, E=E)
    bb = get_backend(args.backend_b, ckpt=args.ckpt, K=K, E=E)

    # --- Latency ORE thadava alakirom (frame-kku maaraadhu) ---
    probe = imgs[0].numpy()
    lat_a = ba.measure_latency(probe, warmup=5, runs=20)
    lat_b = bb.measure_latency(probe, warmup=5, runs=20)
    print(f"  {ba.name:30s} {lat_a:7.2f} ms")
    print(f"  {bb.name:30s} {lat_b:7.2f} ms")

    paths, agreements = [], []
    for i in range(n_frames):
        img = imgs[i].numpy()

        pa = decode_predictions({k: torch.from_numpy(v)
                                 for k, v in ba.run(img).items()})[0]
        pb = decode_predictions({k: torch.from_numpy(v)
                                 for k, v in bb.run(img).items()})[0]

        pa_f = [x for x in pa if x["score"] >= args.score_thresh]
        pb_f = [x for x in pb if x["score"] >= args.score_thresh]
        agree = box_agreement(pa_f, pb_f)
        agreements.append(agree)

        p = os.path.join(args.out_dir, f"compare_{i:03d}.png")
        render_comparison(
            img, list(gts[i]),
            {"name": ba.name, "boxes": pa, "latency_ms": lat_a},
            {"name": bb.name, "boxes": pb, "latency_ms": lat_b,
             "extra": f"{agree:.0f}% boxes match baseline   |   "
                      f"{lat_a/lat_b:.2f}x speedup"},
            p, args.score_thresh,
        )
        paths.append(p)
        print(f"  frame {i}: {len(pa_f)} vs {len(pb_f)} boxes, {agree:.0f}% match")

    mean_agree = float(np.mean(agreements))
    print(f"\n{'='*52}")
    print(f"  {ba.name}  ->  {bb.name}")
    print(f"  speedup        {lat_a/lat_b:.2f}x")
    print(f"  box agreement  {mean_agree:.1f}%  ({n_frames} frames)")
    print(f"{'='*52}")

    if args.gif and paths:
        from PIL import Image
        frames = [Image.open(p) for p in paths]
        g = os.path.join(args.out_dir, "compare.gif")
        frames[0].save(g, save_all=True, append_images=frames[1:],
                       duration=900, loop=0)
        print(f"saved {g}")

    # Summary-ai file-la vachikirom - README-la use panna
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(f"| Backend | Latency | FPS | Speedup | Box agreement |\n")
        f.write(f"|---|---|---|---|---|\n")
        f.write(f"| {ba.name} | {lat_a:.1f} ms | {1000/lat_a:.1f} | 1.00x | baseline |\n")
        f.write(f"| {bb.name} | {lat_b:.1f} ms | {1000/lat_b:.1f} | "
                f"{lat_a/lat_b:.2f}x | {mean_agree:.1f}% |\n")
    print(f"saved {args.out_dir}/summary.md")


if __name__ == "__main__":
    main()

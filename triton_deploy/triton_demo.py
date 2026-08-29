# ============================================================
# triton_demo.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Triton server odudhu nu solradhukku PROOF venum - adhuvum
# paakkuravanukku puriyara madhiri.
#
# Ithu:
#   1. Triton server-ai start pannum
#   2. Ovvoru frame-aiyum client vazhiya server-ku anuppum
#   3. Nadakkuradhai padama varaiyum - pipeline-la enna stage,
#      evlo neram aachu, evlo detections
#   4. GIF-a sethu tharum
#
# Technical terms-um irukkum (TensorRT INT8, dynamic batching),
# simple explanation-um irukkum ("server = model-ai vachirukka
# kadai, request anuppina answer tharum").
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# triton_deploy/pytriton_server.py -> server start
# pytriton.client.ModelClient      -> request anuppurathu
# visualization/bev_renderer.py    -> BEV varaiya
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Output: runs/triton_demo/demo.gif + frame_*.png + summary.md
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# python -m triton_deploy.triton_demo --frames 16 --precision int8
#
# ============================================================

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data.scripts.constants import CAMERAS, CLASSES

HEAD_NAMES = ["heatmap", "offset", "height", "size", "rot", "vel"]

# Pipeline-la irukkura stage-gal. (technical peru, simple explanation)
STAGES = [
    ("6 CAMERAS",      "car mela 6 photo"),
    ("TRITON SERVER",  "model-ai vachirukka service"),
    ("TensorRT INT8",  "NVIDIA optimizer, 3.9x fast"),
    ("BEV + BOXES",    "top-down map + detections"),
]


def draw_pipeline(ax, active: int, stats: dict) -> None:
    """
    Pipeline diagram - ippo enna stage nadakkuthu nu light aagum.

    Args:
        active: enna stage highlight pannanum (0..3)
        stats : live numbers dict
    """
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    n = len(STAGES)
    # Box size-ai chinnadha vachikirom - keezha live stats-ku idam venum
    box_h, gap = 1.0, 0.5
    y = 9.7

    for i, (title, simple) in enumerate(STAGES):
        top = y - i * (box_h + gap)
        on = i == active
        ax.add_patch(FancyBboxPatch(
            (0.6, top - box_h), 8.8, box_h,
            boxstyle="round,pad=0.12",
            facecolor="#0f2b3d" if on else "#12121c",
            edgecolor="#00d4ff" if on else "#2a2a3a",
            linewidth=2.0 if on else 1.0))
        ax.text(5.0, top - 0.38, title, ha="center",
                color="#00d4ff" if on else "#7a7a8a",
                fontsize=10, weight="bold")
        ax.text(5.0, top - 0.78, simple, ha="center",
                color="#c0c0d0" if on else "#5a5a6a", fontsize=7.8)

        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (5.0, top - box_h), (5.0, top - box_h - gap + 0.08),
                arrowstyle="-|>", mutation_scale=13,
                color="#00d4ff" if on else "#2a2a3a", linewidth=1.6))

    # --- keezha live numbers ---
    y0 = y - n * (box_h + gap) + gap - 0.45
    ax.plot([0.6, 9.4], [y0, y0], color="#2a2a3a", lw=0.9)
    y0 -= 0.55
    ax.text(0.6, y0, f"request  #{stats['req']} / {stats['total']}",
            color="#e0e0e0", fontsize=9.5, family="monospace")
    y0 -= 0.55
    ax.text(0.6, y0, f"this one  {stats['ms']:6.1f} ms",
            color="#00d4ff", fontsize=9.5, family="monospace", weight="bold")
    y0 -= 0.5
    ax.text(0.6, y0, f"average   {stats['avg']:6.1f} ms   "
                     f"({1000/max(stats['avg'],0.01):.0f} FPS)",
            color="#b0b0c0", fontsize=9.5, family="monospace")
    y0 -= 0.5
    ax.text(0.6, y0, f"detected  {stats['boxes']} objects",
            color="#b0b0c0", fontsize=9.5, family="monospace")


def draw_notes(ax, server_info: dict) -> None:
    """Server settings + beginner-friendly explanation."""
    ax.axis("off")
    y = 0.98
    ax.text(0.0, y, "WHAT IS RUNNING", color="#e0e0e0", fontsize=9.5,
            weight="bold", transform=ax.transAxes)
    y -= 0.075

    rows = [
        ("Triton Inference Server", "NVIDIA-oda production server.", "#00d4ff"),
        ("", "Model ithukkulla irukku; naama", "#8a8a9a"),
        ("", "network vazhiya request anuppurom.", "#8a8a9a"),
        ("", "", "#8a8a9a"),
        (f"instances: {server_info['instances']}",
         "antha model-oda parallel copies -", "#00d4ff"),
        ("", "pala request ore neram odum.", "#8a8a9a"),
        ("", "", "#8a8a9a"),
        (f"dynamic batching: max {server_info['max_batch']}",
         "request vandhavudan odaama konjam", "#00d4ff"),
        ("", "kaathirundhu, pala request-ai sethu", "#8a8a9a"),
        ("", "ORE velai-a GPU-ku kudukkum.", "#8a8a9a"),
        ("", "", "#8a8a9a"),
        (f"engine: TensorRT {server_info['precision'].upper()}",
         "model-ai NVIDIA optimizer vazhiya", "#00d4ff"),
        ("", "potadhu. 32-bit numbers -> 8-bit,", "#8a8a9a"),
        ("", "3.9x fast, 17x chinna file.", "#8a8a9a"),
    ]
    for tech, simple, col in rows:
        if tech:
            ax.text(0.0, y, tech, color=col, fontsize=8.2,
                    family="monospace", transform=ax.transAxes)
            y -= 0.042
        if simple:
            ax.text(0.0, y, simple, color=col if not tech else "#8a8a9a",
                    fontsize=7.8, transform=ax.transAxes)
            y -= 0.042
        if not tech and not simple:
            y -= 0.022


def render_frame(images, gt_boxes, pred_boxes, stats, server_info,
                 out_path, score_thresh=0.2):
    """Oru demo frame varaiyurathu."""
    from visualization.bev_renderer import denormalize, draw_bev, CLASS_COLORS

    fig = plt.figure(figsize=(20, 7.5), facecolor="#0a0a14")
    gs = fig.add_gridspec(3, 6,
                          width_ratios=[0.62, 0.62, 0.95, 0.06, 1.5, 0.85],
                          hspace=0.1, wspace=0.06)

    # cameras
    layout = [("CAM_FRONT_LEFT", 0, 0), ("CAM_FRONT_RIGHT", 0, 1),
              ("CAM_FRONT",      1, 0), ("CAM_BACK",        1, 1),
              ("CAM_BACK_LEFT",  2, 0), ("CAM_BACK_RIGHT",  2, 1)]
    for cam, r, c in layout:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(denormalize(images[CAMERAS.index(cam)]))
        ax.set_title(cam.replace("CAM_", ""), color="#8a8a9a", fontsize=6, pad=1)
        ax.axis("off")

    # pipeline (stage cycle pannurom, so "odudhu" nu theriyum)
    draw_pipeline(fig.add_subplot(gs[:, 2]), stats["req"] % len(STAGES), stats)

    # BEV
    ax_bev = fig.add_subplot(gs[:, 4])
    draw_bev(ax_bev, gt_boxes, pred_boxes, score_thresh)
    ax_bev.set_title("served by Triton  ->  BEV detections",
                     color="#e0e0e0", fontsize=10, pad=8)
    present = sorted({b["cls"] for b in pred_boxes if b["score"] >= score_thresh})
    if present:
        h = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=6,
                        color=CLASS_COLORS[c], label=CLASSES[c]) for c in present]
        ax_bev.legend(handles=h, loc="upper left", fontsize=7,
                      facecolor="#14141e", edgecolor="#2a2a3a",
                      labelcolor="#c0c0d0")

    draw_notes(fig.add_subplot(gs[:, 5]), server_info)

    fig.text(0.5, 0.015,
             "NVIDIA Triton Inference Server  +  TensorRT  |  "
             "6-camera BEV perception, no LiDAR",
             ha="center", color="#6a6a7a", fontsize=8.5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=105, bbox_inches="tight", facecolor="#0a0a14")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--precision", default="int8",
                    choices=["fp32", "fp16", "int8"])
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--out-dir", default="runs/triton_demo")
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    ap.add_argument("--ms-per-frame", type=int, default=500)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    args = ap.parse_args()

    import torch
    from pytriton.client import ModelClient
    from export.sample_source import load_samples, load_gt_boxes
    from training.evaluate import decode_predictions
    from triton_deploy.pytriton_server import (build_infer_functions,
                                               HEAD_SHAPES)
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton

    imgs, K, E = load_samples(data_root=args.data_root)
    gts = load_gt_boxes(data_root=args.data_root)
    n = min(args.frames, len(imgs))

    # ---------- server start ----------
    print(f"loading {args.instances} TensorRT {args.precision.upper()} instance(s)...")
    funcs = build_infer_functions(args.precision, args.instances,
                                  "runs/simplebev/best.pth", "export/onnx",
                                  "export/engines", K, E)
    triton = Triton()
    triton.bind(
        model_name="bev_ensemble",
        infer_func=funcs,
        inputs=[Tensor(name="images", dtype=np.float32, shape=(6, 3, 224, 400))],
        outputs=[Tensor(name=h, dtype=np.float32, shape=HEAD_SHAPES[h])
                 for h in HEAD_NAMES],
        config=ModelConfig(batching=True, max_batch_size=args.max_batch_size,
                           batcher=DynamicBatcher(
                               max_queue_delay_microseconds=2000,
                               preferred_batch_size=[1, 2, args.max_batch_size])),
    )
    triton.run()
    print("TRITON SERVER READY  (HTTP :8000, gRPC :8001, metrics :8002)\n")

    server_info = {"instances": args.instances,
                   "max_batch": args.max_batch_size,
                   "precision": args.precision}

    try:
        with ModelClient("localhost:8000", "bev_ensemble") as client:
            # warmup - mudhal request eppovum slow (memory alloc, kernel load)
            for _ in range(3):
                client.infer_sample(images=imgs[0].numpy())

            times, paths = [], []
            for i in range(n):
                x = imgs[i].numpy()

                t0 = time.perf_counter()
                res = client.infer_sample(images=x)      # <- REAL request
                ms = (time.perf_counter() - t0) * 1000
                times.append(ms)

                boxes = decode_predictions(
                    {h: torch.from_numpy(res[h][None]) for h in HEAD_NAMES})[0]
                n_box = sum(1 for b in boxes if b["score"] >= args.score_thresh)

                stats = {"req": i + 1, "total": n, "ms": ms,
                         "avg": float(np.mean(times)), "boxes": n_box}
                p = os.path.join(args.out_dir, f"frame_{i:03d}.png")
                render_frame(x, list(gts[i]), boxes, stats, server_info, p,
                             args.score_thresh)
                paths.append(p)
                print(f"  request {i+1:2d}/{n}  {ms:6.1f} ms  {n_box} boxes")
    finally:
        triton.stop()
        print("\nserver stopped")

    med = float(np.median(times))
    print("=" * 58)
    print(f"  Triton median latency : {med:.2f} ms  ({1000/med:.1f} FPS)")
    print(f"  (client-side, network round-trip sethu)")
    print("=" * 58)

    from PIL import Image
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=128)
              for p in paths]
    gif = os.path.join(args.out_dir, "demo.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=args.ms_per_frame, loop=0, optimize=True)
    print(f"saved {gif} ({os.path.getsize(gif)/1e6:.1f} MB)")

    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Server | NVIDIA Triton (PyTriton, in-process) |\n")
        f.write(f"| Engine | TensorRT {args.precision.upper()} |\n")
        f.write(f"| Model instances | {args.instances} |\n")
        f.write(f"| Dynamic batching | max {args.max_batch_size}, 2 ms queue delay |\n")
        f.write(f"| Median latency (client-side) | {med:.2f} ms |\n")
        f.write(f"| Throughput | {1000/med:.1f} FPS |\n")
        f.write(f"| Requests | {n} |\n")
    print(f"saved {args.out_dir}/summary.md")


if __name__ == "__main__":
    main()

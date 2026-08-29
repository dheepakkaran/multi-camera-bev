# ============================================================
# app.py  -  Hugging Face Space (ZeroGPU)
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Yaarum - technical theriyaadhavanga kooda - browser-la thirandhu
# "ithu enna panudhu" nu 10 second-la puriya vaikanum.
#
# Rendu mode:
#   Play  -> ellaa moment-um odi, video madhiri GIF. Ulla enna
#            nadakkudhu nu stage-by-stage theriyum.
#   Single-> oru moment mattum, arithmetic paakka
#
# HF ZeroGPU: GPU eppovum ottikittu irukkaadhu, request vandha
# podhu mattum A100 kedaikkum. Adhaan @spaces.GPU decorator.
#
# ============================================================

import os
import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spaces
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data.scripts.constants import IMAGENET_MEAN, IMAGENET_STD, CAMERAS, CLASSES
from models.simplebev import SimpleBEV
from training.evaluate import decode_predictions
from visualization.bev_renderer import draw_bev, CLASS_COLORS

# Ulla nadakkura 4 step. Ovvondrukum oru technical peru + oru
# saadha vaakkiyam - rendum irundha rendu maadhiri aalukkum puriyum.
STAGES = [
    ("The car looks around",   "6 cameras take a photo at the same instant"),
    ("Each photo is read",     "a neural network picks out shapes and edges"),
    ("Depth is guessed",       "one camera can't measure distance, so the model\nguesses 64 possible distances per pixel"),
    ("A top-down map is drawn", "everything is placed on a 100 x 100 m map\nand objects get a box"),
]

# ---------- data ----------
_d = np.load("assets/samples.npz", allow_pickle=True)
IMAGES_U8 = _d["images_u8"]
GT_BOXES = list(_d["gt_boxes"])
K = torch.from_numpy(_d["intrinsics"]).unsqueeze(0)
E = torch.from_numpy(_d["extrinsics"]).unsqueeze(0)
N_MOMENTS = len(IMAGES_U8)

_mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

MODEL = SimpleBEV(pretrained=False).eval()
MODEL.load_state_dict(
    torch.load("assets/best.pth", map_location="cpu", weights_only=False)["model"])
print("model loaded on CPU")


def normalize(i: int) -> np.ndarray:
    return ((IMAGES_U8[i].astype(np.float32) / 255.0) - _mean) / _std


@spaces.GPU(duration=60)
def run_on_gpu(indices: list):
    """
    Kudutha moments-ai GPU-la odurathu.

    @spaces.GPU: intha function odura podhu mattum HF oru A100
    kudukkum. Athanaala GPU velai ellathaiyum ORE call-la mudikirom -
    ovvoru frame-kum thani-a kettaa, ovvoru thadavaiyum GPU attach
    aaga neram pidikkum.

    Returns:
        (list of preds dict, list of latency ms)
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL.to(dev)
    k, e = K.to(dev), E.to(dev)

    # Warmup - mudhal run eppovum slow (CUDA kernels load aagum)
    with torch.no_grad():
        model(torch.from_numpy(normalize(indices[0])).unsqueeze(0).to(dev), k, e)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        all_preds, all_ms = [], []
        for i in indices:
            x = torch.from_numpy(normalize(i)).unsqueeze(0).to(dev)
            t0 = time.perf_counter()
            preds = model(x, k, e)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            all_ms.append((time.perf_counter() - t0) * 1000)
            all_preds.append({k2: v.cpu().numpy() for k2, v in preds.items()})

    return all_preds, all_ms


def draw_stages(ax, active: int, info: dict) -> None:
    """Idathu panel: 4 step, ippo edhu nadakkudho adhu light aagum."""
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    box_h, gap, y = 1.35, 0.42, 9.8
    for i, (title, simple) in enumerate(STAGES):
        top = y - i * (box_h + gap)
        on = i == active
        ax.add_patch(FancyBboxPatch(
            (0.4, top - box_h), 9.2, box_h, boxstyle="round,pad=0.1",
            facecolor="#0f2b3d" if on else "#12121c",
            edgecolor="#00d4ff" if on else "#2a2a3a",
            linewidth=2.0 if on else 1.0))
        ax.text(5.0, top - 0.42, f"{i+1}.  {title}", ha="center",
                color="#00d4ff" if on else "#7a7a8a",
                fontsize=10, weight="bold")
        for j, line in enumerate(simple.split("\n")):
            ax.text(5.0, top - 0.78 - j * 0.3, line, ha="center",
                    color="#c0c0d0" if on else "#5a5a6a", fontsize=7.5)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch(
                (5.0, top - box_h), (5.0, top - box_h - gap + 0.05),
                arrowstyle="-|>", mutation_scale=11,
                color="#00d4ff" if on else "#2a2a3a", linewidth=1.4))

    y0 = y - len(STAGES) * (box_h + gap) + gap - 0.45
    ax.plot([0.4, 9.6], [y0, y0], color="#2a2a3a", lw=0.9)
    for label, value, col in [
        ("moment", f"{info['i']+1} of {N_MOMENTS}", "#e0e0e0"),
        ("time taken", f"{info['ms']:.0f} ms", "#00d4ff"),
        ("objects found", f"{info['n']}", "#b0b0c0"),
    ]:
        y0 -= 0.55
        ax.text(0.4, y0, f"{label:<14}{value}", color=col,
                fontsize=9.5, family="monospace")


def make_figure(i: int, boxes: list, ms: float, stage: int,
                score_thresh: float) -> np.ndarray:
    """Oru padam: stages | cameras | BEV."""
    shown = [b for b in boxes if b["score"] >= score_thresh]

    fig = plt.figure(figsize=(17, 6.8), facecolor="#0a0a14")
    # Naduvula spacer column-gal (0.05, 0.14) - illaina camera padam
    # BEV-oda y-axis label-la mothum
    gs = fig.add_gridspec(2, 6, width_ratios=[1.0, 0.05, 0.8, 0.8, 0.14, 1.5],
                          hspace=0.06, wspace=0.05)

    draw_stages(fig.add_subplot(gs[:, 0]),
                stage, {"i": i, "ms": ms, "n": len(shown)})

    for cam, r, c in [("CAM_FRONT_LEFT", 0, 2), ("CAM_FRONT_RIGHT", 0, 3),
                      ("CAM_BACK_LEFT", 1, 2), ("CAM_BACK", 1, 3)]:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(IMAGES_U8[i][CAMERAS.index(cam)].transpose(1, 2, 0))
        ax.set_title(cam.replace("CAM_", "").replace("_", " ").title(),
                     color="#8a8a9a", fontsize=7, pad=2)
        ax.axis("off")

    ax = fig.add_subplot(gs[:, 5])
    draw_bev(ax, list(GT_BOXES[i]), boxes, score_thresh)
    ax.set_title("Top-down map built from the cameras",
                 color="#e0e0e0", fontsize=10.5, pad=8)
    present = sorted({b["cls"] for b in shown})
    if present:
        h = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=7,
                        color=CLASS_COLORS[c], label=CLASSES[c]) for c in present]
        ax.legend(handles=h, loc="upper left", fontsize=8,
                  facecolor="#14141e", edgecolor="#2a2a3a", labelcolor="#c0c0d0")

    fig.text(0.62, 0.02,
             "dashed white = where objects really are   |   "
             "solid colour = where the model thinks they are",
             ha="center", color="#6a6a7a", fontsize=8)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def _boxes(preds: dict) -> list:
    return decode_predictions(
        {k2: torch.from_numpy(v) for k2, v in preds.items()})[0]


def play(score_thresh: float, progress=gr.Progress()):
    """Ellaa moment-um odi, video madhiri GIF."""
    progress(0.05, desc="Waking up the GPU...")
    all_preds, all_ms = run_on_gpu(list(range(N_MOMENTS)))

    from PIL import Image
    frames = []
    # Ovvoru moment-kum 4 stage -> stage-gal light aagi nagarum
    for i, (preds, ms) in enumerate(zip(all_preds, all_ms)):
        b = _boxes(preds)
        for stage in range(len(STAGES)):
            progress((i * len(STAGES) + stage) / (N_MOMENTS * len(STAGES)),
                     desc=f"Drawing moment {i+1} of {N_MOMENTS}...")
            frames.append(Image.fromarray(
                make_figure(i, b, ms, stage, score_thresh)))

    out = "/tmp/bev_play.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=380, loop=0, optimize=True)

    avg = float(np.mean(all_ms))
    return out, f"""
### What just happened

The model ran on **{N_MOMENTS} moments** from a real drive in Boston.
Each one took about **{avg:.0f} ms** on the GPU — that is roughly
**{1000/avg:.0f} frames per second**, faster than a car's cameras produce them.

For every moment it did all four steps in the diagram: read six photos,
guessed how far away everything was, drew a top-down map, and put boxes
around what it found.

*This is plain PyTorch. Squeezed through TensorRT on a Tesla T4 the same
model runs in **8.5 ms** — the numbers are in the
[GitHub repo](https://github.com/dheepakkaran/multi-camera-bev).*
"""


def single(moment: int, score_thresh: float):
    """Oru moment mattum."""
    i = int(moment) - 1
    preds, ms = run_on_gpu([i])
    b = _boxes(preds[0])
    shown = [x for x in b if x["score"] >= score_thresh]
    img = make_figure(i, b, ms[0], len(STAGES) - 1, score_thresh)
    return img, f"""
### This moment

| | |
|---|---|
| Time taken | **{ms[0]:.0f} ms** on an A100 GPU |
| Objects found | {len(shown)} |
| Cameras used | 6 |
| LiDAR used | none |
"""


DESC = """
# What a self-driving car sees — using only cameras

Six cameras around a car take a photo at the same instant. This model turns
those six flat photos into a **top-down map** of the world around the car,
with a box around every vehicle and person it finds.

**No LiDAR, no laser scanners, no pre-built maps** — just cameras, which is
the approach Tesla uses.

Press **▶ Play** to watch it work.
"""

with gr.Blocks(title="What a self-driving car sees") as demo:
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column(scale=1):
            play_btn = gr.Button("▶  Play", variant="primary", size="lg")
            gr.Markdown("*Runs 8 moments from a real drive. Takes ~20 seconds.*")

            with gr.Accordion("Try one moment at a time", open=False):
                moment = gr.Slider(
                    1, N_MOMENTS, value=3, step=1,
                    label="Which moment?",
                    info="8 snapshots taken a few seconds apart as the car drives")
                thresh = gr.Slider(
                    0.05, 0.5, value=0.2, step=0.05,
                    label="How sure must the model be?",
                    info="Low = show every guess, including wrong ones. "
                         "High = only confident ones.")
                single_btn = gr.Button("Run this moment")

            stats_md = gr.Markdown()
        with gr.Column(scale=3):
            out = gr.Image(label="", type="filepath", height=520)

    play_btn.click(play, inputs=[thresh], outputs=[out, stats_md])
    single_btn.click(single, inputs=[moment, thresh], outputs=[out, stats_md])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

# ============================================================
# app.py  -  Hugging Face Space (ZeroGPU)
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Project-ai yaarum browser-la thirandhu paakkanum. GitHub-la code
# irukku, aana code padikka time edukkum. Ithu oru click-la
# "6 camera photo -> BEV detections" nu kaattum.
#
# HF ZeroGPU-la odudhu: GPU eppovum ottikittu irukkaadhu, request
# vandha podhu mattum oru A100 kedaikkum. Adhaan @spaces.GPU
# decorator - "intha function-ku GPU venum" nu HF-kku solradhu.
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 1. SimpleBEV model CPU-la load (startup-la)
# 2. Request vandha -> @spaces.GPU function -> model GPU-ku pogum
# 3. 6 camera -> backbone -> LSS -> BEV -> head -> boxes
# 4. Matplotlib-la padam varaiyurathu
#
# Triton inga illa - PyTriton-oda python backend Python 3.8 kekkum,
# HF image-la adhu illa. Triton deployment GPU setup-la odudhu,
# code repo-la iruku (triton_deploy/).
#
# ============================================================

import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spaces
import torch

from data.scripts.constants import IMAGENET_MEAN, IMAGENET_STD, CAMERAS, CLASSES
from models.simplebev import SimpleBEV
from training.evaluate import decode_predictions
from visualization.bev_renderer import draw_bev, CLASS_COLORS

# ---------- data ----------
_d = np.load("assets/samples.npz", allow_pickle=True)
IMAGES_U8 = _d["images_u8"]                       # [8, 6, 3, 224, 400] uint8
GT_BOXES = list(_d["gt_boxes"])
K = torch.from_numpy(_d["intrinsics"]).unsqueeze(0)   # [1, 6, 3, 3]
E = torch.from_numpy(_d["extrinsics"]).unsqueeze(0)   # [1, 6, 4, 4]

_mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

# ---------- model ----------
# Startup-la CPU-la load. ZeroGPU-la CUDA @spaces.GPU function ulla
# mattum thaan kedaikkum, so inga .cuda() pannakoodadhu.
MODEL = SimpleBEV(pretrained=False).eval()
MODEL.load_state_dict(
    torch.load("assets/best.pth", map_location="cpu", weights_only=False)["model"])
print("model loaded on CPU")


@spaces.GPU(duration=30)
def run_model(images_np: np.ndarray):
    """
    Oru frame-ai GPU-la odurathu.

    @spaces.GPU: intha function odura podhu mattum HF oru A100 kudukkum.
    Adhukku appuram GPU vera yaarukkaavadhu poidum - adhaan "Zero" GPU.
    Athanaala model-ai ovvoru call-lum GPU-ku nagarthanum.

    Args:
        images_np: [6, 3, 224, 400] normalized float32

    Returns:
        (preds dict of numpy, latency ms)
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL.to(dev)

    x = torch.from_numpy(images_np).unsqueeze(0).to(dev)
    k, e = K.to(dev), E.to(dev)

    # Warmup - mudhal run eppovum slow (CUDA kernels load aagum).
    # Adhai sethu time alandha number thappa irukkum.
    with torch.no_grad():
        model(x, k, e)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        preds = model(x, k, e)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000

    return {k2: v.cpu().numpy() for k2, v in preds.items()}, ms


def render(frame_idx: int, score_thresh: float):
    """Gradio callback: frame odi, padam + stats thara."""
    i = int(frame_idx) - 1
    images = ((IMAGES_U8[i].astype(np.float32) / 255.0) - _mean) / _std

    preds, ms = run_model(images)
    boxes = decode_predictions(
        {k2: torch.from_numpy(v) for k2, v in preds.items()})[0]
    shown = [b for b in boxes if b["score"] >= score_thresh]

    fig = plt.figure(figsize=(15, 6.5), facecolor="#0a0a14")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.08, 1.7],
                          hspace=0.06, wspace=0.04)
    for cam, r, c in [("CAM_FRONT_LEFT", 0, 0), ("CAM_FRONT", 0, 1),
                      ("CAM_BACK_LEFT", 1, 0), ("CAM_BACK", 1, 1)]:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(IMAGES_U8[i][CAMERAS.index(cam)].transpose(1, 2, 0))
        ax.set_title(cam.replace("CAM_", ""), color="#8a8a9a", fontsize=7, pad=2)
        ax.axis("off")

    ax = fig.add_subplot(gs[:, 3])
    draw_bev(ax, list(GT_BOXES[i]), boxes, score_thresh)
    ax.set_title(f"Bird's Eye View  |  {len(shown)} detections  |  {ms:.0f} ms",
                 color="#e0e0e0", fontsize=11, pad=8)
    present = sorted({b["cls"] for b in shown})
    if present:
        h = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=7,
                        color=CLASS_COLORS[c], label=CLASSES[c]) for c in present]
        ax.legend(handles=h, loc="upper left", fontsize=8,
                  facecolor="#14141e", edgecolor="#2a2a3a", labelcolor="#c0c0d0")

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    stats = f"""
### Ippo enna nadandhudhu

| | |
|---|---|
| Model | SimpleBEV, 5.48 M params |
| Hardware | HF ZeroGPU (NVIDIA A100) |
| Inference | **{ms:.0f} ms** |
| Detections | {len(shown)} (score > {score_thresh}) |

6 camera photo -> EfficientNet-B0 -> Lift-Splat-Shoot -> BEV 200x200
-> CenterPoint head -> 3D boxes. **LiDAR illa.**

*Ithu PyTorch FP32. TensorRT INT8 vachi Tesla T4-la **8.5 ms /
117 FPS** alanden - numbers GitHub README-la.*
"""
    return img, stats


DESC = """
# Multi-Camera BEV Perception

6 surround-view camera photo -> top-down Bird's-Eye-View map with 3D boxes.
**LiDAR illa, HD map illa** - camera mattum (Tesla FSD approach).

Oru frame select panni **Run** click pannu.

[GitHub repo](https://github.com/dheepakkaran/multi-camera-bev) &nbsp;·&nbsp;
TensorRT + NVIDIA Triton benchmarks anga iruku.
"""

with gr.Blocks(title="Multi-Camera BEV Perception") as demo:
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column(scale=1):
            frame = gr.Slider(1, len(IMAGES_U8), value=3, step=1,
                              label="Frame (nuScenes mini val)")
            thresh = gr.Slider(0.05, 0.5, value=0.2, step=0.05,
                               label="Score threshold")
            btn = gr.Button("Run", variant="primary")
            stats_md = gr.Markdown()
        with gr.Column(scale=3):
            out = gr.Image(label="Cameras + BEV", type="numpy")

    btn.click(render, inputs=[frame, thresh], outputs=[out, stats_md])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

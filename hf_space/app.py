# ============================================================
# app.py  -  Hugging Face Space
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Project-ai yaarum browser-la thirandhu paakkanum. GitHub-la code
# irukku, aana code padikka time edukkum. Ithu 5 second-la
# "6 camera photo -> BEV detections" nu kaattum.
#
# Ulla NVIDIA Triton server odudhu (PyTriton). Gradio vெறும் UI -
# antha UI-yum Triton-ku HTTP request thaan anuppudhu. So ithu
# nijamaana deployment, demo-vukkaaga ezhudhina fake illa.
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 1. ONNX models load (camera_backbone + bev_decoder)
# 2. PyTriton server start, "bev_ensemble" nu bind
# 3. Gradio UI -> frame select -> Triton-ku request -> BEV padam
#
# HF free tier CPU mattum, so TensorRT illa - ONNX Runtime thaan.
# T4-la alandha TensorRT numbers README-la iruku.
#
# ============================================================

import time

import gradio as gr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.scripts.constants import IMAGENET_MEAN, IMAGENET_STD, CAMERAS, CLASSES
from training.evaluate import decode_predictions
from visualization.bev_renderer import draw_bev, CLASS_COLORS

HEAD_NAMES = ["heatmap", "offset", "height", "size", "rot", "vel"]
HEAD_SHAPES = {"heatmap": (10, 200, 200), "offset": (2, 200, 200),
               "height": (1, 200, 200), "size": (3, 200, 200),
               "rot": (2, 200, 200), "vel": (2, 200, 200)}

# ---------- data ----------
_d = np.load("assets/samples.npz", allow_pickle=True)
IMAGES_U8 = _d["images_u8"]                 # [8, 6, 3, 224, 400] uint8
GT_BOXES = list(_d["gt_boxes"])
GEOM = np.load("assets/geometry.npy")

_mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def normalize(u8: np.ndarray) -> np.ndarray:
    """uint8 [6,3,224,400] -> ImageNet-normalized float32."""
    return ((u8.astype(np.float32) / 255.0) - _mean) / _std


# ---------- ONNX ----------
import onnxruntime as ort

_bb = ort.InferenceSession("assets/camera_backbone.onnx",
                           providers=["CPUExecutionProvider"])
_dec = ort.InferenceSession("assets/bev_decoder.onnx",
                            providers=["CPUExecutionProvider"])


def run_onnx(images: np.ndarray) -> dict:
    """Rendu ONNX graph vazhiya oru frame."""
    feats = _bb.run(None, {"images": images})[0]
    outs = _dec.run(None, {"features": feats, "geometry": GEOM})
    return dict(zip(HEAD_NAMES, outs))


# ---------- Triton ----------
# PyTriton start aagala-na (HF sandbox restriction) ONNX-ai neradiya
# koopiduvom. UI-la edhu use aagudhu nu honest-a kaattuvom.
TRITON_OK = False
_client = None

try:
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton
    from pytriton.client import ModelClient

    @batch
    def _infer(images):
        outs = {n: [] for n in HEAD_NAMES}
        for sample in images:
            preds = run_onnx(sample)
            for n in HEAD_NAMES:
                outs[n].append(preds[n][0])
        return {n: np.stack(v).astype(np.float32) for n, v in outs.items()}

    _triton = Triton()
    _triton.bind(
        model_name="bev_ensemble",
        infer_func=_infer,
        inputs=[Tensor(name="images", dtype=np.float32, shape=(6, 3, 224, 400))],
        outputs=[Tensor(name=n, dtype=np.float32, shape=HEAD_SHAPES[n])
                 for n in HEAD_NAMES],
        config=ModelConfig(batching=True, max_batch_size=4,
                           batcher=DynamicBatcher(
                               max_queue_delay_microseconds=2000)),
    )
    _triton.run()
    _client = ModelClient("localhost:8000", "bev_ensemble")
    _client.infer_sample(images=normalize(IMAGES_U8[0]))   # warmup
    TRITON_OK = True
    print("Triton server ready")
except Exception as e:
    print(f"Triton start aagala ({type(e).__name__}: {e}) - ONNX neradiya use pannuvom")


def infer(images: np.ndarray) -> dict:
    if TRITON_OK:
        res = _client.infer_sample(images=images)
        return {n: res[n][None] for n in HEAD_NAMES}
    return run_onnx(images)


# ---------- rendering ----------
def render(frame_idx: int, score_thresh: float):
    """Gradio callback: oru frame odi, padam + stats thara."""
    i = int(frame_idx) - 1
    images = normalize(IMAGES_U8[i])

    t0 = time.perf_counter()
    preds = infer(images)
    ms = (time.perf_counter() - t0) * 1000

    boxes = decode_predictions(
        {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in preds.items()})[0]
    shown = [b for b in boxes if b["score"] >= score_thresh]

    fig = plt.figure(figsize=(15, 6.5), facecolor="#0a0a14")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.08, 1.7],
                          hspace=0.06, wspace=0.04)
    layout = [("CAM_FRONT_LEFT", 0, 0), ("CAM_FRONT", 0, 1),
              ("CAM_BACK_LEFT", 1, 0), ("CAM_BACK", 1, 1)]
    for cam, r, c in layout:
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(IMAGES_U8[i][CAMERAS.index(cam)].transpose(1, 2, 0))
        ax.set_title(cam.replace("CAM_", ""), color="#8a8a9a", fontsize=7, pad=2)
        ax.axis("off")

    ax = fig.add_subplot(gs[:, 3])
    draw_bev(ax, list(GT_BOXES[i]), boxes, score_thresh)
    ax.set_title(f"BEV  |  {len(shown)} detections  |  {ms:.0f} ms",
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

    served_by = ("**NVIDIA Triton** (PyTriton, in-process)" if TRITON_OK
                 else "ONNX Runtime (Triton start aagala)")
    stats = f"""
### Ippo enna nadandhudhu

| | |
|---|---|
| Served by | {served_by} |
| Backend | ONNX Runtime, CPU (HF free tier) |
| Latency | **{ms:.0f} ms** |
| Detections | {len(shown)} (score > {score_thresh}) |

6 camera photo -> EfficientNet-B0 -> Lift-Splat-Shoot -> BEV 200x200 ->
CenterPoint head -> 3D boxes.

*Intha Space CPU-la odudhu. Tesla T4-la TensorRT INT8 vachi
**8.5 ms / 117 FPS** alanden - numbers GitHub README-la.*
"""
    return img, stats


# ---------- UI ----------
DESC = """
# Multi-Camera BEV Perception

6 surround-view camera photo -> top-down Bird's-Eye-View map with 3D boxes.
**LiDAR illa, HD map illa** - camera mattum (Tesla FSD approach).

Oru frame select panni **Run** click pannu.

[GitHub repo](https://github.com/dheepakkaran/multi-camera-bev)
"""

with gr.Blocks(title="Multi-Camera BEV Perception",
               theme=gr.themes.Base()) as demo:
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
    demo.load(render, inputs=[frame, thresh], outputs=[out, stats_md])

if __name__ == "__main__":
    import gradio as gr_  # noqa
    demo.launch(server_name="0.0.0.0", server_port=7860)

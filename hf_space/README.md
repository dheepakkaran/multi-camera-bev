---
title: Multi-Camera BEV Perception
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Camera-only 3D BEV object detection from 6 surround cameras
---

# Multi-Camera BEV Perception

Six surround-view cameras go in; a top-down Bird's-Eye-View map with 3D
boxes comes out — **no LiDAR, no HD maps**, the same sensor philosophy as
Tesla FSD.

Pick a frame and hit **Run**.

## What you are looking at

- **Left**: four of the six camera feeds
- **Right**: the BEV map. White dashed = ground truth, filled = model
  prediction. Rings are 10 m apart; the ego vehicle sits at the centre.

## Honest notes

**This Space runs ONNX Runtime on a free CPU tier** — roughly one to two
seconds per frame.

The app tries to serve the model through **NVIDIA Triton** (PyTriton, run
in-process so no Docker is needed) and falls back to calling ONNX Runtime
directly when it cannot start. On this Space it always falls back:
PyTriton's Python backend needs a Python 3.8 interpreter that the Space
image does not provide. The UI tells you which path served your request.

Triton *does* run in the GPU setup — two model instances over TensorRT
INT8 engines, with dynamic batching — and the code for it is in the repo.
There, client-side latency was 48.6 ms against 8.5 ms of actual compute:
the remaining 40 ms is HTTP serialisation of ~10 MB of tensors per
request, which is the thing gRPC and CUDA shared memory exist to fix.

The TensorRT numbers below were measured on a Tesla T4:

| Backend | Latency | FPS | Speedup |
|---|---:|---:|---:|
| PyTorch FP32 | 32.66 ms | 30.6 | 1.00x |
| TensorRT FP32 | 24.34 ms | 41.1 | 1.34x |
| TensorRT INT8 | 8.52 ms | 117.3 | 3.83x |

The model is small (5.5 M parameters) and trained on nuScenes **mini** —
323 training samples against the 28,130 the published numbers use. It
scores 0.043 NDS where the same pipeline fed ground truth scores 0.596,
so what you see are real but weak detections. That gap is training data,
not a bug, and the full reasoning is in the repo.

**Code, benchmarks and engineering notes:**
[github.com/dheepakkaran/multi-camera-bev](https://github.com/dheepakkaran/multi-camera-bev)

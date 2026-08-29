---
title: Multi-Camera BEV Perception
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Camera-only 3D BEV detection served through NVIDIA Triton
---

# Multi-Camera BEV Perception

Six surround-view cameras go in; a top-down Bird's-Eye-View map with 3D
boxes comes out — **no LiDAR, no HD maps**, the same sensor philosophy as
Tesla FSD.

Pick a frame and hit **Run**. The request goes to an **NVIDIA Triton**
server running inside this Space (via PyTriton, so no Docker is needed),
which runs the ONNX pipeline and returns the detection heads.

## What you are looking at

- **Left**: four of the six camera feeds
- **Right**: the BEV map. White dashed = ground truth, filled = model
  prediction. Rings are 10 m apart; the ego vehicle sits at the centre.

## Honest notes

This Space runs on a **free CPU tier**, so inference is ONNX Runtime at
roughly one to two seconds per frame. The TensorRT numbers below were
measured on a Tesla T4:

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

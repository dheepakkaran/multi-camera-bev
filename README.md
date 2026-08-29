# Multi-Camera BEV Perception

Camera-only 3D object detection for autonomous driving. Six surround-view
cameras go in; a top-down Bird's-Eye-View map with 3D boxes comes out —
**no LiDAR, no HD maps**. Optimised with TensorRT and served through
NVIDIA Triton.

![demo](docs/assets/bev_demo.gif)

*Left: the six raw camera feeds. Centre: BEV map — white dashed = ground
truth, filled = model prediction. Right: measured latency on a Tesla T4.*

---

## What this is

A small, complete perception pipeline built end to end: dataloader →
model → training → evaluation → ONNX → TensorRT → Triton → demo. The
model is deliberately tiny (5.5 M parameters) and trained on nuScenes
**mini**, so the interesting part is not the accuracy — it is everything
that surrounds it.

```
6 cameras (900x1600)  --resize + normalise-->  [6, 3, 224, 400]
                                                     |
                       EfficientNet-B0 (shared weights across cameras)
                                                     |
                                            [6, 64, 14, 25]
                                                     |
                     Lift-Splat-Shoot  <-- intrinsics K + extrinsics E
                     (64 depth bins, 2-50 m)
                                                     |
                                        [1, 64, 200, 200]   BEV, 0.5 m/cell
                                                     |
                               ResNet-18 + FPN BEV encoder
                                                     |
                                       [1, 128, 200, 200]
                                                     |
                               CenterPoint head (anchor-free)
                                                     |
     heatmap [10] | offset [2] | height [1] | size [3] | rot [2] | vel [2]
```

| Component | Parameters | Share |
|---|---:|---:|
| EfficientNet-B0 backbone | 3,602,684 | 66% |
| Lift-Splat-Shoot | 4,160 | 0% |
| ResNet-18 + FPN BEV encoder | 1,428,608 | 26% |
| CenterPoint head | 444,436 | 8% |
| **Total** | **5,479,888** | |

### Why Lift-Splat-Shoot?
A camera cannot measure depth. LSS handles that honestly: for every pixel
it predicts a *probability distribution* over 64 candidate depths (2 m to
50 m), "lifts" the pixel's features into all 64 possible 3D positions
weighted by those probabilities, then "splats" (sum-pools) them into a
shared BEV grid. Six cameras splat into the same grid, so multi-camera
fusion falls out for free — and the whole 3D problem becomes a 2D one.

---

## Results

### Latency (Tesla T4, batch 1, 6 cameras)

| Backend | Latency | FPS | Speedup | Box agreement vs FP32 |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 32.66 ms | 30.6 | 1.00x | baseline |
| ONNX Runtime (CUDA) | 30.20 ms | 33.1 | 1.08x | 100% |
| **TensorRT FP32** | **24.34 ms** | **41.1** | **1.34x** | — |
| TensorRT FP16 | 61.56 ms | 16.2 | 0.53x | 66.4% |
| TensorRT INT8 | 8.52 ms | 117.3 | 3.83x | 34.3% |

*Box agreement = fraction of FP32 detections that survive quantisation
with the same class within 0.5 m.*

Engine sizes: camera backbone **49.4 MB → 3.1 MB** (16x smaller) at INT8;
BEV decoder 23.4 MB → 13.7 MB.

**INT8 is the fastest and by far the smallest — and it is not the one I
would ship.** At 34% box agreement it is not producing the same
detections; it emits three to five times as many boxes, most of them
noise. TensorRT FP32 gives a real 1.34x for free with no accuracy
question, and that is the honest recommendation for this model.

![precision comparison](docs/assets/fp16_vs_fp32.gif)

### Detection accuracy (nuScenes **mini**, official `mini_val`)

| | NDS | mAP | car AP | pedestrian AP |
|---|---:|---:|---:|---:|
| Trained model (best epoch 5) | 0.043 | 0.011 | 0.094 | 0.016 |
| **Pipeline ceiling** (oracle) | **0.596** | **0.631** | 0.823 | 0.937 |

**Read this honestly.** nuScenes *mini* has **323 training samples**.
Published SimpleBEV numbers (NDS ≈ 0.30) come from the full nuScenes
trainval set — **28,130 samples, 87x more data**. A 5.5 M-parameter
detector cannot learn 3D geometry from 323 frames, and it does not.
Training loss falls to 0.65 while validation loss climbs from 4.45 to
9.36: textbook overfitting on a tiny set.

The **oracle row** is the number that matters. It pushes *ground-truth*
boxes through the exact same encode → decode → nuScenes-eval path the
model uses. It scores 0.596 rather than 1.0 because:

- three classes (trailer, construction_vehicle, barrier) have **zero
  instances** in `mini_val`, and nuScenes scores an absent class as AP 0 —
  over the seven classes that *are* present, oracle mAP is **0.90**;
- the 0.5 m BEV grid quantises centres, and two objects landing in one
  cell collide.

So the geometry, target encoding, decoding and evaluation are verified
correct. The gap is training data, not code.

---

## Triton deployment

![triton demo](docs/assets/triton_demo.gif)

The model is served through **NVIDIA Triton** (via PyTriton, which runs
the real Triton server in-process — no Docker needed, which matters on
free GPU notebooks).

| | |
|---|---|
| Server | NVIDIA Triton (PyTriton, in-process) |
| Engine | TensorRT INT8 |
| Model instances | 2 (each with its own TensorRT execution context) |
| Dynamic batching | max batch 4, 2 ms queue delay |
| Median latency, client-side | 48.56 ms (20.6 FPS) |

**That 48.56 ms is 40 ms slower than the same engine called directly
(8.52 ms), and the gap is not compute.** Each request ships a
`[6, 3, 224, 400]` float32 input (6.4 MB) and six output maps (3.2 MB)
over HTTP, with NumPy serialisation on both ends — roughly 10 MB per
inference. The model finishes in 8.5 ms and then waits on the wire.

The production fix is gRPC plus CUDA shared memory, so client and server
address the same GPU buffers and the tensors never leave the device. That
is not implemented here; the number above is what an unoptimised HTTP
client actually costs, which felt more useful to measure than to hide.

A conventional Docker-based `model_repository` (ensemble, `instance_group`,
`config.pbtxt`) is also included under `triton_deploy/` for a real GPU host.

---

## Repository layout

```
data/scripts/     constants, camera loader, 6-camera sample loader, Dataset
models/           backbone, view_transformer (LSS), bev_encoder, center_head
training/         losses (focal + L1), train loop, nuScenes NDS evaluation
export/           ONNX export, FP16/INT8 engine build, unified backends, benchmark
triton_deploy/    PyTriton server + Docker model repository + client
visualization/    BEV renderer, precision comparison, demo generator
docker/           Triton docker-compose
```

---

## Running it

```bash
python3 -m venv bev_env && source bev_env/bin/activate
pip install -r requirements.txt
```

Download nuScenes **mini** (v1.0-mini, ~4 GB) from
[nuscenes.org/download](https://www.nuscenes.org/download) into
`data/nuscenes-mini/`.

```bash
python -m training.train --epochs 10           # ~70 s/epoch on Apple MPS
python -m training.evaluate --ckpt runs/simplebev/best.pth
python -m visualization.make_demo --frames 20  # demo GIF
python -m export.export_onnx                   # ONNX + numerical verification
```

On an NVIDIA GPU (T4 / Colab / Kaggle):

```bash
python -m export.calibrate_int8 --precision fp32
python -m export.calibrate_int8 --precision int8
python -m export.benchmark
python -m triton_deploy.triton_demo --frames 16 --precision int8 --instances 2
```

---

## Engineering notes

The parts that cost real debugging time. Most of them are TensorRT 11
behaviour that is not in older tutorials.

**Intrinsics must be rescaled with the image.** Resizing 1600x900 to
400x224 without scaling `K` by the same factors puts every projected point
in the wrong BEV cell. `fx, cx` scale by 0.25; `fy, cy` by 0.2489.

**ONNX cannot export `index_add_` with duplicate indices** — but duplicates
are exactly what BEV sum-pooling needs, since many pixels vote into one
cell. `scatter_add` maps to `ScatterElements(reduction='add')` and exports
cleanly.

**TensorRT needs its plugin registry initialised** before parsing that
graph. Without `trt.init_libnvinfer_plugins(logger, "")` the parser fails
with *"ScatterReduction plugin was not found"* and the BEV decoder cannot
be built at all.

**TensorRT 11 removed the precision builder flags.** There is no
`BuilderFlag.FP16` or `BuilderFlag.INT8` any more — the network is
strongly typed, so precision has to be expressed in the ONNX itself. FP16
became an `onnxconverter_common` conversion; INT8 became explicit QDQ
nodes.

**TensorRT only accepts symmetric quantisation.** ONNX Runtime's default
static quantisation emits non-zero zero-points and int32 biases, and
TensorRT rejects both (*"Non-zero zero point is not supported"*,
*"input has type Int32"*). `ActivationSymmetric`, `WeightSymmetric` and
`QuantizeBias=False` fix it.

**FP16 was 2.3x slower than FP32, and the obvious cause was wrong.**
`onnxconverter_common` keeps `Resize` in fp32 by default, which strands
the FPN upsamples in fp32 islands surrounded by casts — 17 Cast nodes over
9.7 M elements. Clearing `op_block_list` cut that to 11 casts over 1.7 M
elements and changed the latency by 0.4 ms. So the casts were not the
problem; the remaining suspect is the scatter plugin having no fp16
kernel, forcing conversions around the 8.6 M-element lifted volume.
Measured, not resolved.

**Geometry is computed once, on the host.** Camera calibration is fixed on
a given vehicle, so the frustum-to-ego point cloud never changes between
frames. Hoisting it out of the graph removes `torch.inverse` (poorly
supported in ONNX/TensorRT) and avoids recomputing 134,400 points every
frame. It becomes a cached input tensor.

**Never name a directory `triton/` in a project that imports PyTorch.** It
shadows the real `triton` package and breaks `torch._dynamo` with
`module 'triton' has no attribute 'language'`.

**Verify the harness before blaming the model.** Feeding ground truth
through the decode path — the oracle run above — is what proved 0.043 NDS
was a data-quantity problem and not a geometry bug, and it also surfaced
the three classes that are absent from `mini_val` entirely.

---

## Stack

PyTorch · timm (EfficientNet-B0) · nuScenes devkit · ONNX / ONNX Runtime ·
TensorRT · NVIDIA Triton (PyTriton) · Docker · OpenCV · Matplotlib

Benchmarks measured on a Tesla T4 (Kaggle). Training and visualisation run
on Apple Silicon (MPS).

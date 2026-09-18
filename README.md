# Multi-Camera BEV Perception

**Six cameras on a car → a top-down map of everything around it.
No lasers, no radar, no pre-built map of the street.**

[**▶ Live demo**](https://huggingface.co/spaces/dheepakkaran/multi-camera-bev)
 · PyTorch → ONNX → TensorRT → NVIDIA Triton

![demo](docs/assets/bev_demo.gif)

---

## What this does

A self-driving car needs to know what is around it and *where*. The usual
answer is a spinning laser scanner on the roof — accurate, and $10,000 to
$75,000 per car.

This does it with cameras alone. Six of them fire at the same instant, and
the model turns those six flat photos into one map of the world seen from
above, with a box around every car, pedestrian and cyclist it finds. That
is the same bet Tesla made with FSD v12.

## Why that is hard

**A photo has no depth.** Look at a picture of a street: you know the car
is further away than the lamp post, but nothing in the pixels says
*twelve metres*. A small car nearby and a large car far away can occupy
exactly the same pixels.

The trick this model uses is to stop pretending it knows. For every point
in every image it predicts **64 different distances at once** — "maybe
5 m, probably 12 m, unlikely 40 m" — and carries all 64 forward, weighted
by confidence. Those weighted guesses get scattered into a shared 3D map,
and because all six cameras scatter onto the *same* map, they fuse into
one picture with no separate fusion step.

Once everything is on a flat top-down map, finding objects is an ordinary
2D problem. That is
[Lift-Splat-Shoot](https://arxiv.org/abs/2008.05711), and it is the heart
of this project.

---

## How it works

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

| Component | What it does | Parameters |
|---|---|---:|
| EfficientNet-B0 backbone | turns pixels into shapes and textures | 3,602,684 |
| Lift-Splat-Shoot | guesses depth, builds the top-down map | 4,160 |
| ResNet-18 + FPN encoder | adds context across the map | 1,428,608 |
| CenterPoint head | finds object centres, sizes, angles | 444,436 |
| **Total** | | **5,479,888** |

The map covers 100 m × 100 m around the car at 0.5 m per cell.

📘 **[docs/LEARN.md](docs/LEARN.md)** explains every piece of this —
what it does, why it is there, and the trade-offs.

---

## Results

### Speed — the same model, five ways (Tesla T4)

| Backend | Latency | FPS | Speedup | Detections kept |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 32.66 ms | 30.6 | 1.00x | baseline |
| ONNX Runtime (CUDA) | 30.20 ms | 33.1 | 1.08x | 100% |
| **TensorRT FP32** ← shipped | **24.34 ms** | **41.1** | **1.34x** | — |
| TensorRT FP16 | 61.56 ms | 16.2 | 0.53x | 66% |
| TensorRT INT8 | 8.52 ms | 117.3 | 3.83x | 34% |

*"Detections kept" = how many of the FP32 detections survive quantisation
with the same class within half a metre.*

**The fastest option is the one I did not ship.** INT8 runs in 8.5 ms and
shrinks the engine from 49.4 MB to 3.1 MB, but at 34% agreement it is not
the same detector any more — it emits three to five times as many boxes,
mostly noise. TensorRT FP32 gives 1.34x with nothing given up, so that is
what goes to production.

I think this model is unusually fragile under quantisation because it is
under-trained: its confidence scores sit close to the decision threshold,
so small numerical changes flip detections on and off. A well-trained
model would likely survive INT8 far better.

![precision comparison](docs/assets/fp16_vs_fp32.gif)

### Accuracy — and how I checked whose fault it was

| | NDS | mAP | car AP | pedestrian AP |
|---|---:|---:|---:|---:|
| Trained model (best epoch 5) | 0.043 | 0.011 | 0.094 | 0.016 |
| **Pipeline ceiling** (oracle) | **0.596** | **0.631** | 0.823 | 0.937 |

Accuracy is poor. nuScenes **mini** has **323 training samples**; published
numbers for this architecture (≈ 0.30 NDS) come from the full set —
28,130 samples, 87× more. Training loss fell to 0.65 while validation loss
climbed from 4.45 to 9.36. It memorised the training set.

The question worth answering is whether that is the data or my code. So I
fed **ground-truth boxes** through my own target encoder, my own decoder
and the official nuScenes evaluation. A broken pipeline would score near
zero. It scored **0.596** — and the gap from 1.0 is fully explained: three
classes have zero instances in the validation split and nuScenes scores an
absent class as AP 0. Over the seven classes actually present, oracle mAP
is **0.90**.

So the geometry, encoding, decoding and evaluation are correct. What is
missing is data.

---

## Serving it with NVIDIA Triton

![triton demo](docs/assets/triton_demo.gif)

Training a model is one thing; putting it behind an endpoint other
software can call is another. This runs on **NVIDIA Triton** through
PyTriton, which hosts the real Triton server inside a Python process — no
Docker, which matters on free GPU notebooks.

| | |
|---|---|
| Engine | TensorRT INT8 |
| Model instances | 2, each with its own TensorRT execution context |
| Dynamic batching | max batch 4, 2 ms queue delay |
| Median latency, client-side | 48.56 ms |

**The server is six times slower than the engine, and none of it is
compute.** The model still finishes in 8.5 ms. Every request ships a
`[6, 3, 224, 400]` float32 input (6.4 MB) and six output maps (3.2 MB)
over HTTP with NumPy serialisation at both ends — about 10 MB per
inference. The rest of the 48 ms is the model waiting on the wire.

The fix is gRPC with CUDA shared memory, so client and server point at the
same GPU buffers and the tensors never get copied. I did not implement it.
The number above is what an unoptimised HTTP client actually costs, which
seemed more useful to measure than to leave out.

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
python -m visualization.make_demo --frames 20  # the demo GIF above
python -m export.export_onnx                   # ONNX + numerical check
```

On an NVIDIA GPU (T4 / Colab / Kaggle):

```bash
python -m export.calibrate_int8 --precision fp32
python -m export.calibrate_int8 --precision int8
python -m export.benchmark
python -m triton_deploy.triton_demo --frames 16 --precision int8 --instances 2
```

```
data/scripts/     constants, camera loader, 6-camera sample loader, Dataset
models/           backbone, view_transformer (LSS), bev_encoder, center_head
training/         losses (focal + L1), train loop, nuScenes NDS evaluation
export/           ONNX export, TensorRT engines, unified backends, benchmark
triton_deploy/    PyTriton server + live demo client
visualization/    BEV renderer, precision comparison, demo generators
hf_space/         the Gradio demo (deploy.sh pushes it)
docs/LEARN.md     every concept in this project, explained
```

---

## What broke along the way

Five problems that cost real time. Worth writing down because none of them
are in the tutorials.

**The camera intrinsics did not survive the resize.** I shrank the images
from 1600×900 to 400×224 and forgot that the K matrix is measured in
pixels. Every projected point landed in the wrong cell. `fx, cx` scale by
0.25, `fy, cy` by 0.2489.

**ONNX would not export the BEV pooling.** The pooling sums many pixels
into the same output cell, which means duplicate indices, and
`index_add_` does not support those during export. `scatter_add` does the
same thing and exports cleanly.

**Then TensorRT could not parse it** — *"ScatterReduction plugin was not
found"*. TensorRT ships the plugin but does not load it unless you ask:
one call to `init_libnvinfer_plugins` before parsing.

**TensorRT 11 has no precision flags.** Every tutorial says
`config.set_flag(BuilderFlag.FP16)`. In TensorRT 11 that attribute simply
does not exist — precision now has to come from the ONNX itself. FP16
became an `onnxconverter_common` conversion, INT8 became quantisation
nodes written into the graph.

**TensorRT only accepts symmetric quantisation.** ONNX Runtime's defaults
produce non-zero zero-points and int32 biases and TensorRT rejects both.
`ActivationSymmetric`, `WeightSymmetric` and `QuantizeBias=False` fixed it.

One thing I could not resolve: **FP16 came out 2.3× slower than FP32.** My
first theory was cast overhead — the conversion leaves the upsampling
layers in FP32 and wraps them in conversions, 17 casts over 9.7 M
elements. I cut that to 1.7 M elements and the latency moved by 0.4 ms, so
that was not it. My remaining suspect is the scatter having no FP16
kernel, forcing conversions around a large tensor. Measured, not resolved.

---

## Stack

PyTorch · timm (EfficientNet-B0) · nuScenes devkit · ONNX / ONNX Runtime ·
TensorRT · NVIDIA Triton (PyTriton) · Gradio · OpenCV · Matplotlib

Benchmarks on a Tesla T4 (Kaggle). Training and visualisation on Apple
Silicon (MPS). Live demo on Hugging Face ZeroGPU.

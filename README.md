# Multi-Camera BEV Perception

**Six cameras on a car. One map of everything around it. No lasers.**

[**▶ Live demo**](https://huggingface.co/spaces/dheepakkaran/multi-camera-bev)
 · PyTorch → ONNX → TensorRT → NVIDIA Triton
 · 📘 [Full explanation of every concept](docs/LEARN.md)

![demo](docs/assets/bev_demo.gif)

---

## 1. The problem

A self-driving car needs to know what is around it, and how far away it
is.

Most cars answer this with **LiDAR** — a spinning laser on the roof. It
measures distance directly and it is very accurate. It also costs
$10,000 to $75,000 per car. You cannot put that on a car you want to
sell.

Cameras cost a few hundred dollars. But a camera has one big weakness:

> **A photo does not contain distance.**

Hold your thumb up at arm's length. It can cover a car parked 20 metres
away. On the camera sensor, your thumb and that car make the same size
patch. Nothing in the pixels says which one is near.

So the question this project answers is:

> Can you build a system that takes six ordinary camera photos and works
> out where every car and person is, in 3D, without a laser?

---

## 2. What I chose to build (and what I left out)

This is a learning project, not a product. So I picked a scope I could
actually finish and understand.

### In scope

| Choice | Why |
|---|---|
| Cameras only | This is the hard and interesting version of the problem |
| nuScenes **mini** dataset | 4 GB, fits on a laptop. The full set is 350 GB |
| A small model (5.5 M parameters) | Trains in an hour on a laptop GPU |
| Full path: data → model → training → export → server | I wanted the whole pipeline, not just a training notebook |
| Measure everything | Numbers you did not measure are not results |

### Out of scope

| Left out | Why |
|---|---|
| Beating published accuracy | Impossible with 323 training samples. See §6 |
| LiDAR or radar fusion | Would hide the interesting problem |
| Tracking objects over time | Single frame is enough to learn from |
| Inventing a new architecture | See §3 — I am implementing known ideas, not inventing |
| Multi-GPU training | One GPU, one laptop |

Being clear about this matters. A project that quietly pretends to be
state of the art is worse than one that says what it is.

---

## 3. Where the ideas come from

**I did not invent any of the architecture.** Every part of this model is
a published idea that I read about and implemented. That is on purpose —
the goal was to learn by building something real, not to do research.

| Part of this project | Comes from | Year |
|---|---|---|
| Turning camera images into a top-down map | [Lift, Splat, Shoot](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf) — Philion & Fidler, NVIDIA Toronto AI Lab | ECCV 2020 |
| Finding objects as peaks on a heatmap | [Objects as Points](https://arxiv.org/abs/1904.07850) — Zhou et al. | 2019 |
| Doing that for 3D driving scenes | [CenterPoint](https://arxiv.org/abs/2006.11275) — Yin, Zhou & Krähenbühl | 2020 |
| Keeping the whole thing simple | [Simple-BEV](https://simple-bev.github.io/simple_bev_sep30.pdf) — Harley et al. | 2022 |
| The image backbone | [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le | 2019 |
| Handling rare positive examples | [Focal Loss](https://arxiv.org/abs/1708.02002) — Lin et al. | 2017 |
| Combining coarse and fine features | [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) — Lin et al. | 2017 |
| The dataset and scoring | [nuScenes](https://www.nuscenes.org/) — Caesar et al. | 2020 |

**What is mine:** the implementation, the training, every measurement in
§6 and §7, the oracle test in §6, the decision not to ship INT8, and all
the debugging in §8.

---

## 4. The solution

The pipeline has four stages. Each one solves one problem.

```
6 cameras (900×1600)  →  resize + normalise  →  [6, 3, 224, 400]
                                                      │
        ┌─────────────────────────────────────────────┘
        ▼
   ① Backbone        EfficientNet-B0, shared across all six cameras
        │            [6, 3, 224, 400] → [6, 64, 14, 25]
        ▼
   ② View transform  Lift-Splat-Shoot ← camera calibration (K, E)
        │            64 depth guesses per pixel → one top-down map
        │            → [1, 64, 200, 200]   (200×200 cells, 0.5 m each)
        ▼
   ③ BEV encoder     ResNet-18 + FPN
        │            → [1, 128, 200, 200]
        ▼
   ④ Detection head  CenterPoint, anchor-free
                     heatmap [10] · offset [2] · height [1]
                     size [3] · rotation [2] · velocity [2]
```

### ① Backbone — reading the photos

**Problem.** Raw pixels are not useful. The model needs to know "there is
a wheel here", not "this pixel is grey".

**Solution.** A convolutional network that has already learned what edges
and shapes look like, from 1.2 million ImageNet photos. We only have 323
driving photos — far too few to learn vision from scratch.

**What I did.** EfficientNet-B0, pretrained, with one shared copy for all
six cameras. A car looks like a car whichever camera sees it, so six
separate copies would waste parameters.

**Result.** `[6, 3, 224, 400] → [6, 64, 14, 25]`. 3.6 M parameters.

### ② View transform — the hard part

**Problem.** Six flat photos. No distance information. Need one 3D map.

**Solution — Lift-Splat-Shoot.** Stop pretending you know the distance.
For every point in every photo, guess **64 distances at once** — "maybe
5 m, probably 12 m, unlikely 40 m" — and keep all 64, each weighted by
how likely it is.

Then use the camera calibration to work out where each of those guesses
would be in the real world, and drop them all onto one flat map. All six
cameras drop onto the **same** map. Where two cameras agree, the guesses
add up and get strong. Where they disagree, the wrong guesses stay faint.

**Why this is clever:** the cameras get combined for free. There is no
separate "fusion" step. And after this, a 3D problem has become an
ordinary 2D one.

**What I did.** Implemented the frustum geometry, the depth prediction,
and the scatter into the grid. The trickiest part was making the camera
calibration correct — see §8.

**Result.** `[6, 64, 14, 25] + calibration → [1, 64, 200, 200]`.
134,400 3D points per frame, landing in 40,000 map cells. Only
**4,160 parameters** — this stage is mostly geometry, not learning.

### ③ BEV encoder — adding context

**Problem.** Straight after the splat, each map cell only knows what
landed in it. Imagine looking at a map through a straw that shows one
0.5 m square. You see something metallic. Car? Truck? Road sign? You
cannot tell without seeing the surrounding area.

**Solution.** Shrink the map down so each cell has "seen" more area, then
grow it back to full size. Add skip connections so the fine detail
returns too.

**What I did.** A small ResNet-18 with an FPN-style decoder.

**Result.** `[1, 64, 200, 200] → [1, 128, 200, 200]`. 1.4 M parameters.

### ④ Detection head — finding the objects

**Problem.** Given the map, where are the objects and how big are they?

**Solution — CenterPoint.** Predict a heatmap where each object's
**centre** is a bright spot. Then read the box properties at that spot.

This works especially well here because in a top-down map, two objects
**cannot overlap** — two things cannot be in the same place. So one peak
means exactly one object, with no ambiguity.

**What I did.** Six output maps per cell: class heatmap, sub-cell offset,
height, size, rotation, velocity. Size is predicted in log space because
objects range from a 0.5 m cone to a 20 m truck. Rotation is predicted as
sin and cos because 0° and 360° are the same heading but very different
numbers.

**Result.** 444,436 parameters. Total model: **5,479,888**.

---

## 5. How I built it

The order mattered. Each step had to be verified before the next one
could be trusted.

| Step | What I did | How I knew it worked |
|---|---|---|
| 1 | Load nuScenes, six cameras per frame, with calibration | Checked that the front camera's position came out as 1.7 m forward, 1.5 m up — which is where a windscreen camera actually is |
| 2 | Build the model | Checked every tensor shape against the design |
| 3 | Check the geometry | Projected each camera's frustum and confirmed the front camera's points land in front of the car, the rear camera's behind |
| 4 | Turn boxes into training targets | See step 6 |
| 5 | Train | Watched train loss and validation loss separately |
| 6 | **Oracle test** | Pushed ground truth through my own encoder → my own decoder → official scoring. This checks the pipeline with the model removed |
| 7 | Export to ONNX | Compared PyTorch and ONNX outputs number by number (max difference 1e-5) |
| 8 | Build TensorRT engines | FP32, FP16, INT8 |
| 9 | Benchmark all five backends | Median of 50 runs after 10 warm-up runs |
| 10 | Check the detections still match | Not just speed — compared every backend's boxes against FP32 |
| 11 | Serve through Triton | Measured client-side latency, found it was 6× the engine time, worked out why |

Step 6 and step 10 are the two that most projects skip. They are also the
two that produced the most useful findings.

---

## 6. Result: accuracy

| | NDS | mAP | car AP | pedestrian AP |
|---|---:|---:|---:|---:|
| Trained model (best epoch 5) | 0.043 | 0.011 | 0.094 | 0.016 |
| **Pipeline ceiling** (oracle test) | **0.596** | **0.631** | 0.823 | 0.937 |

**The accuracy is poor.** Being straight about why:

nuScenes mini has **323 training samples**. The published number for this
kind of architecture — about 0.30 NDS — comes from the full nuScenes set,
which has **28,130 samples. 87 times more.**

The model memorised the small set:

```
epoch  5:   train loss 4.29    validation loss 4.45     ← best
epoch 50:   train loss 0.65    validation loss 9.36     ← memorised
```

### But whose fault is it — the data, or my code?

Guessing is useless, so I designed a test.

I took the **ground-truth boxes** and pushed them through my own target
encoder, then my own decoder, then the official nuScenes scoring. **The
model is not involved at all.** Only my pipeline.

If any of my code were wrong, this would score near zero.

**It scored 0.596.** And I could explain the gap from 1.0:

- Three classes (trailer, construction_vehicle, barrier) have **zero
  instances** in the validation split. nuScenes scores an absent class as
  AP 0 and averages over all ten. Over the seven classes that are
  actually present, the oracle scores **0.90 mAP**.
- The 0.5 m grid rounds object centres, and two objects landing in the
  same cell collide.

**Conclusion: the geometry, the encoding, the decoder and the evaluation
are all correct. What is missing is data.**

---

## 7. Result: speed

The same model, run five different ways, on a Tesla T4.

| Backend | Latency | FPS | Speedup | Detections kept |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 32.66 ms | 30.6 | 1.00× | baseline |
| ONNX Runtime (CUDA) | 30.20 ms | 33.1 | 1.08× | 100% |
| **TensorRT FP32** ← shipped | **24.34 ms** | **41.1** | **1.34×** | — |
| TensorRT FP16 | 61.56 ms | 16.2 | 0.53× | 66% |
| TensorRT INT8 | 8.52 ms | 117.3 | 3.83× | 34% |

*"Detections kept" means: of the boxes FP32 found, how many does this
backend also find — same class, centre within half a metre?*

### The fastest one is not the one I shipped

INT8 runs in 8.5 ms and shrinks the engine from 49.4 MB to 3.1 MB. But
only **34%** of the original detections survive. It also produces three
to five times as many boxes, most of them noise.

**A detector that is fast and wrong is worse than one that is slower and
right.** TensorRT FP32 gives 1.34× with nothing given up. That is what
ships.

Why is this model so fragile under INT8? I think because it is
under-trained. Its confidence scores sit close to the decision threshold,
so small numerical changes flip detections on and off. A well-trained
model would likely survive INT8 far better.

![precision comparison](docs/assets/fp16_vs_fp32.gif)

---

## 8. Result: serving

![triton demo](docs/assets/triton_demo.gif)

**Problem.** A script that loads a model is not a deployment. In real use
the model sits behind an address, several programs call it, and you
update it without changing their code.

**Solution.** NVIDIA Triton, through PyTriton — which runs the real
Triton server inside a Python process, so no Docker is needed.

| | |
|---|---|
| Engine | TensorRT INT8 |
| Model copies running in parallel | 2 |
| Dynamic batching | max batch 4, 2 ms wait |
| Median latency seen by the client | 48.56 ms |

### The server is 6× slower than the engine, and none of it is compute

The model still finishes in 8.5 ms. So where do the other 40 ms go?

```
input     6 × 3 × 224 × 400 × 4 bytes  =  6.45 MB
outputs   20 × 200 × 200 × 4 bytes     =  3.20 MB
                                  total ≈ 9.65 MB   per request
```

Nearly 10 MB, converted to and from NumPy, sent over HTTP, **every single
inference**. The model spends most of the request waiting for data to
arrive.

The standard fix is gRPC with CUDA shared memory: the client and server
point at the same GPU memory, so the data never gets copied. I did not
implement it. But measuring this, and knowing what the fix is, was more
useful than leaving the number out.

---

## 9. What broke

Five problems that cost real time. None of them are in the tutorials.

**1. The camera calibration did not survive resizing the images.** The K
matrix is measured in pixels. I shrank the images from 1600×900 to
400×224 and forgot to shrink K with them. Result: a car directly in front
of the vehicle was placed 9.7 metres to the left. Every frame, silently.
The fix is one line; finding it took much longer.

**2. ONNX would not export the map pooling.** The pooling adds many
points into the same cell, which means repeated indices. PyTorch's
`index_add_` does that, but the ONNX exporter refuses it. `scatter_add`
does the same thing and exports fine.

**3. Then TensorRT could not read that operation.** *"ScatterReduction
plugin was not found in the plugin registry."* TensorRT ships the plugin
but does not load it unless you ask. One line before parsing fixed it.
"Not found" turned out to mean "not loaded".

**4. TensorRT 11 removed the precision settings.** Every tutorial says
`config.set_flag(BuilderFlag.FP16)`. In TensorRT 11 that does not exist —
I printed the available options to confirm. NVIDIA moved to *strongly
typed* networks, where precision comes from the ONNX file itself. So FP16
became a conversion of the ONNX, and INT8 became quantisation nodes
written into the graph.

**5. TensorRT only accepts symmetric quantisation.** ONNX Runtime's
defaults produce a form TensorRT rejects, in two different ways. Three
settings fixed it.

**One I could not solve: FP16 came out 2.3× slower than FP32.** My first
theory was conversion overhead — the FP16 conversion left the upsampling
layers in FP32 and wrapped them in conversions, 17 of them over 9.7
million values. I cut that to 1.7 million and the latency moved by
0.4 ms. So that was not the cause. My remaining suspect is the scatter
operation having no FP16 version, forcing conversions around a large
tensor. Measured, not resolved.

---

## 10. What I would do next

1. **Train on the full nuScenes set.** Everything else checks out. Data
   is the only thing holding accuracy back.
2. **Retest INT8 after that.** I expect a well-trained model to survive
   quantisation far better, which would make the 3.83× usable.
3. **gRPC with CUDA shared memory** for serving, since I measured that
   the data transfer dominates.
4. **Finish the FP16 investigation** — build an engine without the
   scatter and see whether the slowdown disappears.

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
docs/LEARN.md     every concept above, explained from scratch
```

---

## Stack

PyTorch · timm (EfficientNet-B0) · nuScenes devkit · ONNX / ONNX Runtime ·
TensorRT · NVIDIA Triton (PyTriton) · Gradio · OpenCV · Matplotlib

Benchmarks on a Tesla T4 (Kaggle). Training and visualisation on Apple
Silicon (MPS). Live demo on Hugging Face ZeroGPU.

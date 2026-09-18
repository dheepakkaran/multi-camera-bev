# Understanding this project

Everything in this repo, explained at the level you need to talk about it.
Each section has the idea in plain words, the number from this project, and
the question an interviewer is likely to ask with an answer you can give.

You do not need to go deeper than what is written here.

---

## The 60-second version

Say this when someone asks "tell me about your project":

> I built a 3D object detection system for self-driving cars that uses only
> cameras — no LiDAR. Six cameras around the car feed into a model that
> produces a top-down map of everything nearby, with a box around each car
> and pedestrian.
>
> The interesting part for me was the deployment side. I exported it to
> ONNX, built TensorRT engines at three precisions, and served it through
> NVIDIA Triton. TensorRT FP32 gave a clean 1.34x speedup. INT8 was 3.8x
> faster but only reproduced a third of the original detections, so I
> didn't ship it — that trade-off was the most useful thing I learned.
>
> Accuracy is low because I trained on nuScenes mini, which is 323 samples.
> I verified that was a data problem and not a code problem by running
> ground truth through my own decoder and evaluation.

That covers: what, why, what you measured, and what you know is weak.

---

## Part 1 — The problem

### Bird's Eye View (BEV)

**Idea.** Instead of detecting objects separately in each camera image, put
everything onto one flat map seen from above, with the car at the centre.
Then detection becomes a 2D problem on that map.

**In this project.** The map is 200 × 200 cells, each cell 0.5 m, so it
covers 100 m × 100 m around the car.

> **Q: Why BEV instead of detecting in each camera image?**
>
> Two reasons. A car can appear in two cameras at once, and merging those
> two detections afterwards is messy — in BEV they land on the same cell
> automatically. And planning happens in BEV anyway: the module that
> decides where to steer wants a map, not image coordinates.

### Camera-only

**Idea.** Most self-driving stacks use LiDAR — a spinning laser that
measures distance directly. It costs $10,000–$75,000 per car. Cameras cost
a few hundred dollars.

> **Q: Why camera-only? What do you lose?**
>
> Cost and scalability — you can put cameras on every car you sell. What
> you lose is direct depth measurement, so the model has to infer distance,
> and it will be less accurate than LiDAR at range. Tesla made this bet
> with FSD v12; most other companies still use LiDAR.

### Why depth is the hard part

**Idea.** A photo is a projection. A small car nearby and a large car far
away can occupy exactly the same pixels. Nothing in a single image tells
you which it is.

> **Q: How can you get 3D from a 2D image?**
>
> You can't, from one image, with certainty. What the model does instead is
> predict a distribution: for every point it says "5 m is unlikely, 12 m is
> probable, 40 m is unlikely" across 64 possible distances. It carries all
> 64 forward weighted by confidence rather than committing to one. Where
> several cameras overlap, the wrong guesses disagree and cancel out.

---

## Part 2 — The model

### EfficientNet-B0 backbone

**Idea.** A pretrained convolutional network that turns pixels into
features — edges, textures, shapes. Pretrained on ImageNet, so it already
knows generic visual patterns and we don't have to learn them from 323
samples.

**In this project.** 3.6 M of the model's 5.5 M parameters. Input
[6, 3, 224, 400] → output [6, 64, 14, 25]. Stride 16, so the feature map
is 16× smaller than the image.

> **Q: Why share one backbone across all six cameras?**
>
> A car looks like a car regardless of which camera sees it, so the same
> weights work for all six. Six separate backbones would be six times the
> parameters for no benefit. It also means the batch dimension is just the
> six images, which is convenient.

> **Q: Why stride 16 and not 32?**
>
> Stride 32 gives a 7 × 13 feature map, which is too coarse — a pedestrian
> at 30 m would be under one cell. Stride 8 would be more detailed but four
> times the compute in the transformer that follows.

### Lift-Splat-Shoot (LSS)

This is the core of the project. Three steps:

**Lift.** For each pixel in the feature map, predict a probability over 64
depth bins from 2 m to 50 m. Multiply the pixel's feature vector by each
depth probability. One pixel becomes 64 weighted points in 3D space.

**Splat.** Use the camera calibration to work out where each of those
points actually is in the world, then drop it into the BEV grid cell it
lands in. Multiple points landing in one cell are summed.

**Shoot.** (Not used here — it's about trajectory planning in the original
paper.)

**In this project.** 6 cameras × 64 depths × 14 × 25 = 134,400 points per
frame, scattered into 40,000 BEV cells. Only 4,160 parameters — almost all
of the work is geometry, not learning.

> **Q: Walk me through Lift-Splat-Shoot.**
>
> Each pixel gets a distribution over 64 candidate depths. That turns one
> 2D pixel into 64 3D points, each weighted by how likely that depth is.
> Camera calibration tells you where those points are in the car's
> coordinate frame, so you drop them onto a flat grid and sum whatever
> lands in each cell. All six cameras drop onto the same grid, which is
> what fuses them — there's no separate fusion step.

> **Q: What are the intrinsics and extrinsics?**
>
> Intrinsics, the K matrix, describe the lens: focal length and where the
> optical centre sits in the image. Extrinsics, the E matrix, describe
> where the camera is bolted on the car and which way it points. You need
> both to go from a pixel plus a depth to a position in the world.

> **Q: You resize images from 1600×900 to 400×224. Does that affect K?**
>
> Yes, and this was a bug I had to fix. K is in pixels, so if the image
> shrinks, K has to shrink with it. fx and cx scale by 0.25, fy and cy by
> 0.2489. Without it every point lands in the wrong cell and the whole map
> is wrong.

### BEV encoder (ResNet-18 + FPN)

**Idea.** The splatted map is just camera features dropped in place — no
cell knows about its neighbours. A small convolutional network adds that
context: downsample to see large objects, upsample back for detail.

**In this project.** [1, 64, 200, 200] → [1, 128, 200, 200]. 1.4 M
parameters.

> **Q: Why do you need this at all?**
>
> A truck is 12 m long — 24 cells. One cell alone can't tell you it's a
> truck. Downsampling widens what each output cell has seen, and the FPN
> skip connections bring the fine detail back.

### CenterPoint head

**Idea.** Anchor-free detection. Instead of pre-defined box shapes, predict
a heatmap where each object's centre is a peak, then regress the box
properties at that peak.

**In this project.** Six outputs per cell: class heatmap (10 classes),
sub-cell offset, height, size, rotation as sin/cos, velocity.

> **Q: Why anchor-free?**
>
> Anchors mean choosing box sizes and aspect ratios up front and tuning
> them. In BEV objects don't overlap — two cars can't be in the same place
> — so a centre point is unambiguous and there's nothing to tune.

> **Q: Why predict rotation as sin and cos instead of the angle?**
>
> Because 0° and 360° are the same heading but very different numbers. The
> loss would punish the model for being right. sin and cos are continuous
> across the wrap-around.

> **Q: Why log of the size?**
>
> Sizes range from a 0.5 m traffic cone to a 20 m truck. In log space that
> range is much narrower, which is easier for the network to regress.

---

## Part 3 — Training and evaluation

### The losses

**Focal loss** on the heatmap. Of 40,000 cells, maybe 50 contain an object.
Plain cross-entropy would let the model predict "empty" everywhere and
score well. Focal loss down-weights the easy empty cells so the rare
positives dominate.

**L1 loss** on the regression outputs, applied only at cells that actually
contain an object centre.

> **Q: Why focal loss?**
>
> Class imbalance. 50 positives against 40,000 negatives. Focal loss scales
> down the contribution of examples the model already gets right, so the
> gradient comes from the hard cases.

### Gaussian targets

The heatmap target isn't a single 1 at the centre — it's a small Gaussian
blob. Getting a cell one off shouldn't be punished as hard as being
completely wrong, and a single 1 in 40,000 cells is too sparse to train on.

### NDS and mAP

**mAP** here uses centre distance, not box overlap — a detection counts if
its centre is within 0.5, 1, 2 or 4 m of a real object.

**NDS** is nuScenes' combined score: mAP plus penalties for errors in
position, size, orientation, velocity and attributes.

> **Q: Your NDS is 0.043. What happened?**
>
> nuScenes mini has 323 training samples. The published number for this
> architecture, around 0.30, comes from the full set — 28,130 samples, 87
> times more. Training loss went to 0.65 while validation loss climbed from
> 4.45 to 9.36, so it memorised the training set.
>
> What I did about it was check whether the problem was my code or the
> data. I fed ground-truth boxes through my own encoder, decoder and the
> official evaluation. If the pipeline were broken that would score near
> zero too — it scored 0.596, and the gap from 1.0 is explained: three
> classes have no instances in the validation split and nuScenes scores an
> absent class as zero. Over the seven classes actually present it was
> 0.90. So the pipeline is correct and the model is under-trained.

This is the strongest answer in the whole project. It shows you isolated a
variable instead of guessing.

---

## Part 4 — Making it fast

### ONNX

**Idea.** A model saved as a `.pth` file only means something to PyTorch.
ONNX is a common format — a description of the computation graph that other
tools can read. TensorRT reads ONNX.

**In this project.** Two graphs: `camera_backbone` and `bev_decoder`. After
exporting, the same input is run through both PyTorch and ONNX and the
outputs compared — they agree to 1e-5.

> **Q: Why check the numbers after exporting?**
>
> Export can silently change behaviour — an operator gets mapped
> differently, a shape gets fixed when it shouldn't be. If you don't
> compare against the original you find out later, after you've already
> built engines and measured latency on a model that isn't your model.

### TensorRT

**Idea.** NVIDIA's inference optimiser. It takes your graph and rebuilds it
for one specific GPU: fusing layers so intermediate results stay in fast
memory, picking the fastest kernel for each operation, and optionally using
lower precision.

> **Q: What does TensorRT actually do to make it faster?**
>
> Three things mainly. It fuses operations — a convolution, batch norm and
> ReLU become one kernel instead of three round trips to memory. It
> benchmarks several implementations of each layer on your actual GPU and
> keeps the fastest. And it can run in lower precision.

### FP32, FP16, INT8

Numbers are stored with fewer bits. Less memory to move, and on modern
GPUs, dedicated hardware for the smaller types.

| | Bits | Effect |
|---|---|---|
| FP32 | 32 | baseline |
| FP16 | 16 | half the memory traffic |
| INT8 | 8 | quarter the memory, needs calibration |

> **Q: Why is INT8 faster?**
>
> Four times less data moving between memory and the compute units, and the
> GPU has instructions that do INT8 arithmetic several values at a time.
> Most of inference is memory-bound, so shrinking the numbers helps more
> than you'd expect.

> **Q: How does the model know how to convert float to int8?**
>
> Calibration. You run real images through it and record the range of
> values each layer produces. Then you map that range onto -128 to 127. I
> used 16 validation samples and percentile calibration, which ignores the
> extreme outliers so the common values get more of the range.

### What actually happened

| Backend | Latency | Speedup | Detections kept |
|---|---:|---:|---:|
| PyTorch FP32 | 32.66 ms | 1.00x | baseline |
| ONNX Runtime | 30.20 ms | 1.08x | 100% |
| TensorRT FP32 | 24.34 ms | 1.34x | — |
| TensorRT FP16 | 61.56 ms | 0.53x | 66% |
| TensorRT INT8 | 8.52 ms | 3.83x | 34% |

> **Q: INT8 is 3.8x faster. Why didn't you use it?**
>
> Because it isn't the same model any more. I measured how many of the FP32
> detections survived quantisation with the same class within half a metre
> — only 34%. It also emitted three to five times as many boxes, mostly
> noise. A detector that's fast and wrong is worse than one that's slower
> and right, so I shipped FP32, which gave 1.34x with nothing given up.
>
> I think this model is unusually fragile under quantisation because it's
> under-trained — its confidence scores sit close to the threshold, so
> small numerical changes flip detections on and off. A well-trained model
> would likely hold up much better.

> **Q: Your FP16 is slower than FP32. Why?**
>
> I don't fully know, and I'd rather say that than guess. What I ruled out:
> my first theory was cast overhead, because the FP16 conversion left the
> upsampling layers in FP32 and wrapped them in conversions — 17 casts over
> 9.7 M elements. I removed most of those, down to 1.7 M elements, and the
> latency moved by 0.4 ms. So that wasn't it. My remaining theory is the
> scatter operation in the BEV pooling having no FP16 kernel, forcing
> conversions around a large tensor, but I haven't confirmed it.

That answer is fine. "I measured, I ruled out the obvious cause, here's my
remaining hypothesis" is a better answer than a confident wrong one.

---

## Part 5 — Serving it

### Why a server

A script that loads a model and runs it is not deployable. In production
the model sits behind an endpoint: several clients call it, it batches
their requests, it reports metrics, it can be updated without redeploying
the caller.

### Triton

NVIDIA's inference server. Handles batching, multiple model copies,
versioning and metrics. Normally runs in Docker.

**PyTriton** is the same server run inside a Python process — pip
installable, no Docker. That is what this project uses, because free GPU
notebooks don't give you Docker.

> **Q: What is dynamic batching?**
>
> Instead of running each request the moment it arrives, the server waits a
> couple of milliseconds to see if more arrive, then runs them together.
> GPUs are much more efficient on one batch of four than on four separate
> calls. You trade a little latency for a lot of throughput.

> **Q: Why two model instances?**
>
> Each instance has its own TensorRT execution context. Those aren't thread
> safe, so with one instance concurrent requests queue up behind each
> other. Two instances can genuinely run at the same time.

### The result, and what it taught me

| | |
|---|---|
| Median latency through Triton | 48.56 ms |
| Same engine called directly | 8.52 ms |

> **Q: Why is it six times slower through the server?**
>
> It isn't compute — the model still finishes in 8.5 ms. Each request ships
> 6.4 MB of input and 3.2 MB of output over HTTP with NumPy serialisation
> at both ends. About 10 MB per inference, and that's what the extra 40 ms
> is. The model spends most of the request waiting on the wire.
>
> The standard fix is gRPC with CUDA shared memory, so client and server
> point at the same GPU buffers and the tensors never get copied. I didn't
> implement it, but that's where I'd go next.

This is a genuinely good thing to have measured. Most people benchmark the
model and stop.

---

## Part 6 — The bugs

Five things that cost real time. These are yours — you hit them and worked
through them, so talk about them as a story, not as expertise.

**1. Rescaling the camera intrinsics.** Resized the images, forgot K is in
pixels. Everything landed in the wrong cell.

**2. ONNX wouldn't export the BEV pooling.** The pooling sums many pixels
into one cell, which means duplicate indices, which `index_add_` doesn't
support in ONNX export. Switched to `scatter_add`, which does.

**3. TensorRT couldn't parse that operation** — "ScatterReduction plugin
was not found". TensorRT ships plugins but doesn't load them unless you
ask: one call to `init_libnvinfer_plugins` fixed it.

**4. TensorRT 11 removed the precision flags.** The tutorials all say
`config.set_flag(BuilderFlag.FP16)`. In TensorRT 11 that attribute doesn't
exist — precision now comes from the ONNX itself. Converted the ONNX to
FP16 with `onnxconverter_common`; for INT8, wrote quantisation nodes into
the graph.

**5. TensorRT only accepts symmetric quantisation.** ONNX Runtime's
defaults produce non-zero zero-points and int32 biases; TensorRT rejects
both. Three settings fixed it.

> **Q: What was the hardest part?**
>
> Getting TensorRT to accept the graph. The BEV pooling scatters many
> values into the same output cell, and that pattern kept hitting
> limitations — first ONNX export wouldn't do it, then TensorRT couldn't
> parse it, then quantisation produced a form it rejected. Each one was a
> different fix. Reading the actual error, finding what changed between
> TensorRT versions, and testing one thing at a time is what got through
> it.

---

## Part 7 — Questions you should expect

**"What would you do differently?"**
> Train on the full nuScenes set. Everything else in the pipeline checks
> out; accuracy is the one thing held back purely by data. Second would be
> gRPC with shared memory for the serving path, since I measured that the
> HTTP transfer dominates.

**"What's the weakest part?"**
> Accuracy, and I know exactly why. Also that I never resolved the FP16
> slowdown — I narrowed it down but ran out of GPU time.

**"How long did this take?"**
> Answer honestly with your real timeline.

**"Did you use AI tools?"**
> Yes, and I'd say so. Everyone does now. What matters is that you can
> explain every part — which is what this document is for.

**"Explain the maths of Lift-Splat-Shoot."**
> Give the plain version from Part 2 first. If they push for the matrix
> algebra: a pixel (u, v) at depth d becomes K⁻¹·(u·d, v·d, d) in camera
> coordinates, then R·p + t puts it in the car's frame. That's as deep as
> you need to go.

**If you don't know something:**
> "I didn't go that deep on that part — here's what I do know, and here's
> how I'd find out." That is a completely acceptable answer for an intern.
> Pretending is not.

---

## What to put on a resume

Defensible:

- Built a camera-only 3D object detection pipeline (BEV) on nuScenes:
  six-camera input, Lift-Splat-Shoot view transform, CenterPoint head
- Exported to ONNX with numerical verification; built TensorRT engines at
  FP32/FP16/INT8 and benchmarked all five backends on a Tesla T4
- Achieved 1.34x inference speedup with TensorRT FP32 at no accuracy cost;
  measured INT8 at 3.83x but only 34% detection agreement, and documented
  why it wasn't shipped
- Deployed through NVIDIA Triton with dynamic batching and two model
  instances; identified that HTTP serialisation, not compute, dominated
  end-to-end latency

Do not write:

- "3.9x speedup" without the accuracy caveat
- any NDS number — bring it up only if asked

# Understanding this project

Everything in this repo, explained from the beginning. No prior knowledge
assumed beyond basic Python and having heard of neural networks.

Each topic has four parts:

- **The idea** — in plain words, with a scenario
- **The maths** — worked with real numbers from this project
- **In this project** — where it actually appears in the code
- **Learn more** — one thing to read, one thing to watch

At the end of each part there are interview questions with answers you can
actually say out loud.

---

## Contents

**Part 0** — [The 60-second version](#part-0--the-60-second-version)

**Part 1 — Cameras and geometry**
[1.1 What a camera actually does](#11-what-a-camera-actually-does) ·
[1.2 Intrinsics: the K matrix](#12-intrinsics-the-k-matrix) ·
[1.3 Extrinsics: the E matrix](#13-extrinsics-the-e-matrix) ·
[1.4 Pixel to 3D point](#14-pixel-to-3d-point) ·
[1.5 The BEV grid](#15-the-bev-grid)

**Part 2 — The model**
[2.1 The backbone](#21-the-backbone) ·
[2.2 Lift-Splat-Shoot](#22-lift-splat-shoot) ·
[2.3 The BEV encoder](#23-the-bev-encoder) ·
[2.4 The detection head](#24-the-detection-head)

**Part 3 — Training**
[3.1 Focal loss](#31-focal-loss) ·
[3.2 Gaussian targets](#32-gaussian-targets) ·
[3.3 How accuracy is measured](#33-how-accuracy-is-measured) ·
[3.4 The oracle test](#34-the-oracle-test)

**Part 4 — Making it fast**
[4.1 ONNX](#41-onnx) ·
[4.2 TensorRT](#42-tensorrt) ·
[4.3 FP32, FP16, INT8](#43-fp32-fp16-int8) ·
[4.4 Calibration](#44-calibration) ·
[4.5 What actually happened](#45-what-actually-happened)

**Part 5 — Serving**
[5.1 Why a server](#51-why-a-server) ·
[5.2 Triton](#52-triton)

**Part 6** — [The bugs](#part-6--the-bugs)
**Part 7** — [Interview questions](#part-7--interview-questions)
**Part 8** — [A study plan](#part-8--a-study-plan)

---

## Part 0 — The 60-second version

Say this when someone asks "tell me about your project":

> I built a 3D object detection system for self-driving cars that uses
> only cameras — no LiDAR. Six cameras around the car feed into a model
> that produces a top-down map of everything nearby, with a box around
> each car and pedestrian.
>
> The part I spent most time on was deployment. I exported it to ONNX,
> built TensorRT engines at three precisions, and served it through NVIDIA
> Triton. TensorRT FP32 gave a clean 1.34x speedup. INT8 was 3.8x faster
> but only reproduced a third of the original detections, so I didn't ship
> it — working out that trade-off was the most useful thing I learned.
>
> Accuracy is low because I trained on nuScenes mini, which is 323
> samples. I confirmed that was a data problem and not a code problem by
> running ground truth through my own decoder and evaluation.

That covers what, why, what you measured, and what you know is weak. Four
things, sixty seconds.

---

# Part 1 — Cameras and geometry

This is the part people skip and then cannot answer questions about. It is
also the least difficult — it is secondary-school geometry with matrices
wrapped around it.

## 1.1 What a camera actually does

### The idea

A camera flattens the world. Light from a 3D scene passes through a small
hole (the lens) and lands on a flat sensor. Everything on a straight line
from the hole lands on the *same* pixel.

**The scenario.** Hold your thumb at arm's length. Now look at a car
parked 20 metres away. Your thumb can completely cover the car. On your
retina — or a camera sensor — the thumb and the car produce exactly the
same size patch. Nothing in that patch says which one is near.

That is the entire problem this project solves. A photo tells you
*direction* perfectly and *distance* not at all.

### The maths

The pinhole model. A 3D point at distance `z` with horizontal offset `x`
lands at pixel offset `u` from the image centre:

```
u = f · x / z
```

where `f` is the focal length in pixels. That is one equation, and it is
similar triangles from geometry class.

**Worked example.** A camera with focal length 1266 pixels. A car 1.8 m
wide sits 20 m away:

```
pixel width = 1266 × 1.8 / 20 = 114 pixels
```

The same car at 40 m:

```
pixel width = 1266 × 1.8 / 40 = 57 pixels
```

Twice as far, half the size. Now go backwards: you see something 57 pixels
wide. Is it a 1.8 m car at 40 m, or a 0.9 m motorbike at 20 m? **The pixels
cannot tell you.** You need either a second viewpoint, a prior about
object sizes, or a model that learns to guess.

### Learn more

- 📖 [OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
  — the reference, skim the first section
- 🎥 [Computer Vision: The Camera Matrix](https://www.youtube.com/watch?v=Hz8kz5aeQ44)
  — derives the pinhole model and both matrices from scratch. **Watch this one.**

---

## 1.2 Intrinsics: the K matrix

### The idea

K describes the camera itself — properties that do not change when you
move it. Focal length, and where the optical centre lands on the sensor.

**The scenario.** Two cameras bolted side by side on the same car, one
wide-angle and one telephoto. Same position, same direction, completely
different images. The difference is entirely in K.

### The maths

```
      ⎡ fx   0   cx ⎤
K  =  ⎢  0  fy   cy ⎥
      ⎣  0   0    1 ⎦
```

- `fx, fy` — focal length in pixels (two values because pixels are not
  always perfectly square)
- `cx, cy` — the optical centre, usually near the middle of the image

**Real values from this project** (nuScenes CAM_FRONT, original resolution
1600×900):

```
fx = 1266.4    cx = 816.3
fy = 1266.4    cy = 491.5
```

Note `cx = 816` when the image is 1600 wide — close to the middle (800)
but not exactly. Lenses are not mounted perfectly.

### The rescaling trap — and a real bug I hit

K is measured **in pixels**. Resize the image and K is wrong.

This project resizes 1600×900 → 400×224:

```
scale_x = 400 / 1600 = 0.25
scale_y = 224 / 900  = 0.2489
```

So:

```
fx: 1266.4 × 0.25   = 316.6
cx:  816.3 × 0.25   = 204.1
fy: 1266.4 × 0.2489 = 315.2
cy:  491.5 × 0.2489 = 122.3
```

**What happens if you forget.** A car sits dead centre, 20 m ahead. In the
resized image it is at pixel u = 204. Project it back with the *unscaled*
K:

```
x = (u − cx) · z / fx = (204 − 816.3) × 20 / 1266.4 = −9.7 m
```

The car is directly in front of you and the model places it **9.7 metres
to the left**. Every object in every frame is wrong, and the loss does not
go down, and nothing in the error message tells you why.

This is in [`data/scripts/camera_loader.py`](../data/scripts/camera_loader.py),
function `scale_intrinsic`.

> **Q: You resize the images. Does that affect the intrinsics?**
>
> Yes, and I got this wrong the first time. K is in pixel units, so
> shrinking the image by four times means fx and cx shrink by four times
> too. If you don't do it, every projected point lands in the wrong place.
> My scale factors are 0.25 horizontally and 0.2489 vertically, because
> 1600→400 and 900→224 are not the same ratio.

---

## 1.3 Extrinsics: the E matrix

### The idea

E describes *where the camera is* and *which way it points*, relative to
the car. It does not describe the camera at all — swap in a different lens
and E is unchanged.

**The scenario.** The front camera and the rear camera both see "a car
15 m away, straight ahead". One means 15 m in front of you, the other
means 15 m behind. E is what turns those two identical statements into
different positions on the map.

### The maths

Rotation and translation packed into one 4×4:

```
      ⎡         | tx ⎤
E  =  ⎢    R    | ty ⎥        R is 3×3 rotation
      ⎢         | tz ⎥        t is where the camera sits
      ⎣ 0  0  0 |  1 ⎦
```

To move a point from camera coordinates to car coordinates:

```
p_car = R · p_camera + t
```

**Real values from this project** (CAM_FRONT):

```
t = [1.70, 0.02, 1.51]
```

The front camera is 1.70 m ahead of the car's origin, 0.02 m off centre,
1.51 m off the ground. That is a camera mounted near the top of the
windscreen, which is exactly right.

**Why 4×4 and not just R and t separately?** Because with the extra row
you can write rotation *and* translation as one matrix multiply, and chain
several transforms by multiplying matrices. This is called *homogeneous
coordinates* and it is the reason almost all 3D graphics uses 4×4
matrices.

**Worked example.** A point 10 m in front of the front camera, dead
centre, at camera height:

```
p_camera = [0, 0, 10]          (camera frame: z is forward)
```

CAM_FRONT's rotation is roughly "camera z → car x", so:

```
p_car ≈ [10, 0, 0] + [1.70, 0.02, 1.51]
      = [11.70, 0.02, 1.51]
```

11.7 m ahead of the car's centre, not 10 — because the camera itself is
1.7 m forward. That offset matters at close range.

> **Q: What is the difference between intrinsics and extrinsics?**
>
> Intrinsics are about the camera: focal length and optical centre. They
> change if you swap the lens. Extrinsics are about where the camera is
> bolted on the car and which way it faces. They change if you move the
> camera. You need both — intrinsics get you from a pixel to a direction,
> extrinsics get you from that direction to a position in the world.

---

## 1.4 Pixel to 3D point

### Putting it together

Given a pixel `(u, v)` and a guessed depth `d`, where is it in the world?

```
step 1   camera coords:  p_cam = K⁻¹ · (u·d, v·d, d)
step 2   car coords:     p_car = R · p_cam + t
```

Two matrix multiplies. That is the whole of the geometry in this project.

### Worked example with real numbers

Take CAM_FRONT after rescaling: `fx = 316.6, cx = 204.1, fy = 315.2,
cy = 122.3`. Pick the pixel at `(250, 130)` and guess a depth of `15 m`.

**Step 1.** Applying K⁻¹ works out to:

```
x_cam = (u − cx) · d / fx = (250 − 204.1) × 15 / 316.6 = +2.17 m
y_cam = (v − cy) · d / fy = (130 − 122.3) × 15 / 315.2 = +0.37 m
z_cam = d                                              = 15.00 m
```

So in the camera's own frame: 2.17 m right, 0.37 m down, 15 m forward.
Sensible — the pixel was right of centre and slightly below it.

**Step 2.** Rotate into the car's frame and add the camera's position. For
CAM_FRONT that comes to roughly:

```
p_car ≈ [16.7, −2.2, 1.1]
```

16.7 m ahead, 2.2 m to the right, 1.1 m off the ground.

**Now the important part.** Change the depth guess to 30 m and redo it:

```
x_cam = (250 − 204.1) × 30 / 316.6 = +4.35 m
p_car ≈ [31.7, −4.4, 0.7]
```

The *same pixel* now sits at a completely different place. That is what
"the model has to guess depth" means in concrete terms, and it is exactly
why Lift-Splat-Shoot carries 64 of these at once instead of picking one.

This is in [`models/view_transformer.py`](../models/view_transformer.py),
function `get_geometry`.

---

## 1.5 The BEV grid

### The idea

A top-down map stored as a 2D array, like a chessboard drawn on the road
with the car at the centre square.

### The maths

```
grid       200 × 200 cells
cell size  0.5 m
coverage   100 m × 100 m
car        at the centre, cell (100, 100)
```

Converting a position in metres to a cell index:

```
cell = (metres − range_min) / cell_size
```

**Worked examples:**

| Position | Calculation | Cell |
|---|---|---|
| 0 m (the car itself) | (0 + 50) / 0.5 | 100 |
| 10 m ahead | (10 + 50) / 0.5 | 120 |
| 25 m ahead | (25 + 50) / 0.5 | 150 |
| 7.3 m ahead | (7.3 + 50) / 0.5 | 114.6 → cell 114, remainder 0.6 |

That last row matters. Cell indices are integers, so 7.3 m and 7.4 m land
in the same cell. Rounding to the cell loses up to 0.5 m of accuracy.

**The fix: predict the remainder.** The model outputs an extra "offset"
value per object — the 0.6 in the example above. Add it back at decode
time and you recover the exact position:

```
recovered = (114 + 0.6) × 0.5 − 50 = 7.3 m       exact
```

This is the `offset` head, and it is why the detection head has six
outputs instead of five.

> **Q: Why 0.5 m cells? Why not finer?**
>
> It is a memory and compute trade-off. 200×200 at 128 channels is already
> 5 million numbers per frame. Halving the cell size quadruples that. At
> 0.5 m the quantisation error is at most 0.25 m, and I recover most of
> that with the offset head — which is cheaper than a finer grid.

---

# Part 2 — The model

## 2.1 The backbone

### The idea

A convolutional network that turns raw pixels into *features* — numbers
that describe what is in a region rather than its colour. Early layers
find edges, later layers find wheels and windows.

**Why pretrained?** This project has 323 training samples. Learning what an
edge looks like from 323 photos is hopeless. EfficientNet-B0 has already
learned that from 1.2 million ImageNet photos, so we start from there and
only learn the driving-specific part.

### In this project

```
input     [6, 3, 224, 400]        6 cameras, RGB, 224×400
output    [6, 64, 14, 25]         6 feature maps
```

The spatial size shrinks by 16× (224/16 = 14, 400/16 = 25). This is called
the **stride**. Each of the 14×25 output cells summarises a 16×16 patch of
the original image.

**Parameters:** 3.6 M of the model's 5.5 M total. The backbone is most of
the model.

> **Q: Why one backbone shared across all six cameras?**
>
> A car looks like a car whichever camera sees it, so the same weights
> work for all six. Six separate backbones would be six times the
> parameters learning the same thing. Practically it also means I can
> stack the six images into one batch and do a single forward pass.

> **Q: Why stride 16?**
>
> Stride 32 gives a 7×13 map — a pedestrian at 30 m would be smaller than
> one cell, so it would be gone. Stride 8 gives four times the cells,
> which makes the view transformer four times more expensive, and that
> stage is already the bottleneck. 16 was the middle.

### Learn more

- 📖 [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
  — the clearest written explanation of convolutions there is
- 🎥 [Search: "convolutional neural network explained visually"](https://www.youtube.com/results?search_query=convolutional+neural+network+explained+visually)
  — 3Blue1Brown and StatQuest both have good ones

---

## 2.2 Lift-Splat-Shoot

This is the heart of the project. If you learn one thing properly, learn
this.

### The idea

**The scenario.** You are handed six photographs taken at the same moment
from six points on a car, and asked to draw a map of where everything is.
You cannot, because no photo tells you distance.

So you cheat, honestly. For every point in every photo you write down *all*
the places it might be — "if it's 5 m away it's here, if it's 12 m away
it's there" — with a confidence on each. Then you drop all those
possibilities onto one map. Where several cameras agree, the confidences
stack up. Where they disagree, the wrong guesses stay faint.

### The three steps

**Lift.** For each cell of the feature map, predict a probability
distribution over 64 candidate depths from 2 m to 50 m. Multiply the
cell's 64-dimensional feature vector by each depth probability.

```
feature    [64]        what is here
depth      [64]        how likely each distance is
outer product → [64 features × 64 depths]
```

One 2D cell becomes 64 points in 3D, each carrying a scaled copy of the
feature.

**Splat.** Use the geometry from Part 1.4 to work out where each of those
points is in the car's frame, find which BEV cell it lands in, and **sum**
everything that lands in the same cell.

**Shoot.** Not used here — in the original paper it refers to planning
trajectories on the resulting map.

### The numbers

```
6 cameras × 64 depths × 14 × 25 cells = 134,400 points per frame
scattered into                           40,000 BEV cells
```

**Parameters: 4,160.** Almost nothing. The depth prediction is a single
1×1 convolution; everything else is geometry, which has no parameters to
learn. This is the most important stage of the model and the smallest.

### Worked example

Take one cell of the CAM_FRONT feature map. The depth network outputs 64
numbers; after softmax, suppose the top few are:

```
depth  8 m  → 0.05
depth 10 m  → 0.22
depth 12 m  → 0.41      ← most likely
depth 14 m  → 0.19
depth 16 m  → 0.08
everything else         ≈ 0.05 combined
```

The cell's feature vector has 64 numbers. The point placed at 12 m gets
`0.41 × feature`, the point at 10 m gets `0.22 × feature`, and so on. All
64 get placed. The model never picks one — it keeps all of them, weighted.

If the object really is at 12 m, the CAM_FRONT_LEFT camera also drops a
strong contribution at that same world position, and the two sum. The
guesses at 8 m and 16 m land where the other camera put nothing, so they
stay weak.

### Why sum and not average or max?

Sum-pooling means more evidence gives a stronger signal, which is what you
want — a cell seen confidently by two cameras should look different from
one seen faintly by one. Averaging would throw that away.

### In this project

[`models/view_transformer.py`](../models/view_transformer.py):
`_create_frustum`, `get_geometry`, `lift`, `splat`.

> **Q: Walk me through Lift-Splat-Shoot.**
>
> Each pixel of the feature map gets a probability distribution over 64
> candidate depths. That turns one 2D cell into 64 points in 3D, each
> carrying the feature scaled by how likely that depth is. Camera
> calibration tells me where those points are in the car's frame, so I
> drop them onto a flat 200×200 grid and sum whatever lands in each cell.
> All six cameras drop onto the same grid, and that is what fuses them —
> there is no separate fusion step. After that it is a 2D detection
> problem.

> **Q: What happens to the wrong depth guesses?**
>
> They get placed too, just weakly, and they land in cells where no other
> camera put anything. The BEV encoder that follows learns to ignore
> diffuse low-magnitude regions and respond to the concentrated ones. It
> is not that wrong guesses are removed — they are outvoted.

### Learn more

- 📖 [Lift, Splat, Shoot — the original paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf)
  — read the intro and Section 3 only
- 📖 [Lift Splat Shoot Explained (blog, with code)](https://akashprakas.github.io/akashBlog/posts/2025-11-15-LiftSplatShoot.html)
  — walks the actual implementation, closest to this repo
- 🎥 [Lift, Splat, Shoot](https://www.youtube.com/watch?v=fqIwhE2dmWk)
- 💻 [Original code (NVIDIA Toronto AI Lab)](https://github.com/nv-tlabs/lift-splat-shoot)

---

## 2.3 The BEV encoder

### The idea

Straight out of the splat, each BEV cell only knows what was dropped into
it. No cell knows about its neighbours.

**The scenario.** You are looking at the map through a straw that shows
one 0.5 m square. You see "something metallic". Is it a car, a truck, or a
road sign? You cannot tell — you need to see the surrounding 4 m.

The encoder gives cells that wider view.

### How

Downsample, then upsample back:

```
200×200  →  100×100  →  50×50   (each cell now "sees" more area)
                  ↓
200×200  ←  100×100  ←  50×50   (back to full resolution)
```

Every downsample doubles how much of the original map each cell has been
influenced by. This is called the **receptive field**. Going down to 50×50
means one cell there has seen roughly 16 m of road.

The upward path adds **skip connections** — the fine 200×200 detail is
added back so you get context *and* precision. That is the FPN, Feature
Pyramid Network.

### The numbers

```
input   [1,  64, 200, 200]
output  [1, 128, 200, 200]
params  1.4 M
```

> **Q: Why do you need this at all?**
>
> A truck is 12 m long, which is 24 cells. A single cell cannot contain
> enough information to say "truck". Downsampling widens what each cell
> has seen so large objects become recognisable, and the skip connections
> bring the fine detail back so I do not lose small objects like
> pedestrians.

### Learn more

- 📖 [Feature Pyramid Networks, explained](https://paperswithcode.com/method/fpn)
- 🎥 [Search: "ResNet skip connections explained"](https://www.youtube.com/results?search_query=resnet+skip+connections+explained)

---

## 2.4 The detection head

### The idea

Anchor-free detection. Older detectors place thousands of candidate boxes
of pre-chosen sizes and classify each one. CenterPoint does something
simpler: predict a heatmap where each object's **centre** is a bright
spot, then read off the box properties at that spot.

**Why this works well in BEV.** In a camera image two objects can overlap
— a pedestrian in front of a car occupies the same pixels. In a top-down
map they cannot: two physical objects cannot be in the same place. So one
peak = one object, with no ambiguity.

### The six outputs

Every one of the 200×200 cells produces:

| Output | Size | What it means |
|---|---|---|
| `heatmap` | 10 | is there an object centre here, per class |
| `offset` | 2 | the sub-cell remainder from §1.5 |
| `height` | 1 | z position, metres |
| `size` | 3 | log width, log length, log height |
| `rot` | 2 | sin and cos of the heading |
| `vel` | 2 | velocity in x and y |

### Why log of the size

Object sizes in nuScenes span a 0.5 m traffic cone to a 20 m truck. Asking
a network to output numbers across that range is awkward — the loss is
dominated by the big ones.

```
traffic cone:  0.5 m  →  log(0.5)  = −0.69
car:           1.8 m  →  log(1.8)  = +0.59
truck:        12.0 m  →  log(12.0) = +2.48
```

A 40× range becomes a range of about 3. Much easier to regress. Decoding
is just `exp()`.

### Why sin and cos instead of the angle

A car pointing north can be described as 0° or as 360°. Same heading,
wildly different numbers — so if the truth is 0° and the model says 359°,
a plain loss punishes it as if it were completely wrong.

```
truth   0°      →  sin = 0.000,  cos = 1.000
model 359°      →  sin = −0.017, cos = 1.000
                   difference ≈ 0.017          tiny, correct
```

```
as raw angles:     |0 − 359| = 359             enormous, wrong
```

sin and cos are continuous across the wrap-around. Decode with
`atan2(sin, cos)`.

> **Q: Why anchor-free?**
>
> Anchors mean choosing box sizes and aspect ratios up front and tuning
> them per dataset. In BEV objects cannot overlap, so a centre point is
> unambiguous and there is nothing to tune. It is also fewer outputs —
> one prediction per cell instead of one per anchor per cell.

> **Q: Why predict rotation as two numbers?**
>
> Because angles wrap. 0° and 360° are the same heading but the loss would
> see a huge error. sin and cos are continuous, so being 1° off always
> costs about the same regardless of where you are on the circle.

### Learn more

- 📖 [CenterNet Explained: Anchor-Free Object Detection (LearnOpenCV)](https://learnopencv.com/centernet-anchor-free-object-detection-explained/)
  — the clearest write-up; this project's head is the same idea
- 📖 [CenterPoint paper](https://arxiv.org/abs/2006.11275) — skim Section 3
- 🎥 [Search: "CenterNet objects as points explained"](https://www.youtube.com/results?search_query=centernet+objects+as+points+explained)

---

# Part 3 — Training

## 3.1 Focal loss

### The problem it solves

The heatmap has 40,000 cells. A typical frame has about 50 objects. So
**0.125%** of cells are positive.

**The scenario.** A student is asked 40,000 yes/no questions where the
answer is "no" 39,950 times. Answering "no" to everything scores 99.875%.
The student learns nothing and the score looks excellent.

That is what plain cross-entropy does here.

### How focal loss fixes it

It scales down the loss from examples the model already gets right, so the
gradient comes from the hard ones.

```
focal loss  =  (1 − p)^γ  ×  −log(p)          with γ = 2
              └─────────┘    └────────┘
              the new bit    normal loss
```

### Worked example

An easy negative — an empty cell the model correctly calls empty with
p = 0.99 of being correct:

```
plain:  −log(0.99)               = 0.010
focal:  (1 − 0.99)² × 0.010      = 0.0001 × 0.010  = 0.0000010
```

A hard positive — a real object the model is unsure about, p = 0.3:

```
plain:  −log(0.3)                = 1.204
focal:  (1 − 0.3)² × 1.204       = 0.49 × 1.204    = 0.590
```

Compare the ratios:

```
plain:  1.204 / 0.010     =    120× more weight on the hard case
focal:  0.590 / 0.0000010 = 590,000× more weight on the hard case
```

With 39,950 easy negatives, the plain version lets them collectively
drown out the 50 positives. Focal loss makes each easy negative
essentially free.

### In this project

[`training/losses.py`](../training/losses.py). Focal on the heatmap, plain
L1 on the six regression outputs — and the L1 is applied **only at cells
that contain a real object centre**, because "the size of the object in
this empty cell" is meaningless.

> **Q: Why focal loss and not cross-entropy?**
>
> Class imbalance. About 50 positive cells against 40,000 negatives. With
> cross-entropy the model can predict "empty" everywhere and the loss
> looks fine. Focal loss multiplies each example's loss by (1−p)², so
> confident correct predictions contribute almost nothing and the
> gradient comes from the cases the model is actually getting wrong.

### Learn more

- 📖 [Focal Loss paper (RetinaNet)](https://arxiv.org/abs/1708.02002) — Section 3 is short and readable
- 🎥 [Search: "focal loss explained class imbalance"](https://www.youtube.com/results?search_query=focal+loss+explained+class+imbalance)

---

## 3.2 Gaussian targets

### The idea

The training target for the heatmap is not a single 1 at the object
centre. It is a small blurred blob.

**Two reasons:**

1. **A single 1 in 40,000 zeros is too sparse to train on.** Almost every
   gradient step sees only background.
2. **Being one cell off is nearly right.** A hard target punishes
   "adjacent cell" exactly as hard as "other side of the map", which is
   not what you want to teach.

### What it looks like

For an object at cell (100, 120) with radius 3:

```
cell     (100,120)  (101,120)  (102,120)  (103,120)
target      1.00       0.80       0.41       0.14
```

The peak is still unambiguous, but neighbours get partial credit.

The radius scales with object size — a truck gets a wider blob than a
traffic cone, because "one cell off" is proportionally less wrong for a
big object.

### In this project

[`data/scripts/dataset.py`](../data/scripts/dataset.py), function
`draw_gaussian`. When two objects overlap, the maximum is kept rather than
the sum, so a peak never exceeds 1.

---

## 3.3 How accuracy is measured

### mAP — but by distance, not overlap

Ordinary object detection asks "do the predicted and true boxes overlap by
more than 50%?" nuScenes instead asks **"is the predicted centre within X
metres of the true centre?"** and averages over X = 0.5, 1, 2 and 4 m.

**Why.** At 40 m a box is only a few cells wide, so overlap becomes an
unstable measure. Distance is what actually matters for driving: knowing a
car is within half a metre of where you thought is what keeps you from
hitting it.

### NDS — the combined score

```
NDS = ½ × mAP  +  ½ × (average of five "how wrong were you" scores)
```

The five are errors in translation, scale, orientation, velocity and
attribute. Each is turned into a 0–1 score where 1 is perfect. So a model
that finds every object but gets all the sizes wrong is penalised, not
just one that misses objects.

### What this project scored

| | NDS | mAP |
|---|---:|---:|
| Trained model | 0.043 | 0.011 |
| Oracle (ground truth in) | 0.596 | 0.631 |

### Why it is low, honestly

nuScenes **mini** has **323 training samples**. The published figure for
this architecture, about 0.30 NDS, comes from the full set — **28,130
samples, 87× more data**.

The evidence it is overfitting, not broken:

```
epoch  5:  train 4.29   val 4.45      ← best
epoch 50:  train 0.65   val 9.36      ← memorised
```

Training loss keeps falling. Validation loss more than doubles. That is
the textbook signature.

### Learn more

- 📖 [nuScenes detection task and metrics](https://www.nuscenes.org/object-detection)
  — the official definition of NDS
- 🎥 [Search: "mean average precision mAP explained object detection"](https://www.youtube.com/results?search_query=mean+average+precision+mAP+explained+object+detection)

---

## 3.4 The oracle test

**This is the strongest thing in the whole project.** Learn to tell it
well.

### The situation

NDS came out at 0.043. Two possible explanations:

1. The model is under-trained (a data problem)
2. Something in my code is broken — the geometry, the target encoding, the
   decoder, or the evaluation (a bug)

Guessing between them is useless. So I designed a test that separates
them.

### The test

Take the **ground-truth boxes**. Push them through my own target encoder —
the same code that turns boxes into heatmaps for training. Then push the
result back through my own decoder — the same code that turns model
outputs into boxes. Then score it with the official nuScenes evaluation.

**The model is not involved at all.** Only my pipeline.

```
ground truth boxes
    ↓  my encoder
heatmap / offset / size / rot
    ↓  my decoder
predicted boxes
    ↓  official nuScenes eval
score
```

If any of my code were wrong, this scores near zero.

### The result

```
Oracle NDS  0.596      mAP  0.631
```

Not 1.0 — so I had to explain the gap, which turned out to be two things:

**1. Three classes have zero instances in the validation split.** trailer,
construction_vehicle and barrier never appear. nuScenes scores an absent
class as AP 0, and averages over all ten. Over the seven classes actually
present:

```
car 0.823 · truck 1.000 · bus 1.000 · pedestrian 0.937
motorcycle 0.981 · bicycle 0.663 · traffic_cone 0.902

mean = 0.901
```

**2. The 0.5 m grid.** Two objects landing in the same cell collide and
one overwrites the other.

### The conclusion

The geometry, target encoding, decoder and evaluation are all correct. The
pipeline's ceiling on this data is about 0.90 mAP. The model is at 0.011.
**The gap is data, not code.**

> **Q: Your NDS is 0.043. What happened?**
>
> nuScenes mini has 323 training samples — the published number for this
> architecture comes from the full set, which is 87 times larger. Training
> loss went to 0.65 while validation loss climbed from 4.45 to 9.36, so it
> memorised the training set.
>
> But I didn't want to just assume that. I fed ground-truth boxes through
> my own encoder, my own decoder and the official evaluation, with the
> model taken out of the loop. If my pipeline were broken that would score
> near zero. It scored 0.596, and the gap from 1.0 is fully explained —
> three classes have no instances in the validation split and nuScenes
> scores those as zero, so over the seven classes that are actually there
> it was 0.90. So the code is correct and the model is under-trained.

Practise saying that until it is natural. It shows you isolated a variable
instead of guessing, which is the single most valuable engineering habit
an interviewer can see.

---

# Part 4 — Making it fast

## 4.1 ONNX

### The idea

A `.pth` file only means something to PyTorch. ONNX is a **shared
description of the computation** — a list of operations and how they
connect — that other tools can read.

**The scenario.** You write a recipe in Tamil. Only Tamil speakers can
cook it. Translate it into a notation everyone understands — ingredient
list, numbered steps — and any kitchen can. ONNX is that notation for
neural networks.

### Why bother

TensorRT cannot read PyTorch. Neither can ONNX Runtime, or most mobile and
embedded runtimes. ONNX is the common entry point to all of them.

### In this project

Two graphs, not one:

```
camera_backbone.onnx    images [6,3,224,400]  →  features [6,64,14,25]
bev_decoder.onnx        features + geometry   →  6 detection heads
```

Splitting them means the backbone can be scaled independently when served
— it is the expensive part and the one you would replicate first.

### Verifying the export

After exporting, the same input goes through both PyTorch and ONNX and the
outputs are compared:

```
backbone   max diff  1.07e-05   OK
heatmap    max diff  4.77e-06   OK
offset     max diff  5.96e-06   OK
...
```

Differences around 1e-5 are just floating-point ordering. Anything larger
would mean the export changed the model.

> **Q: Why check the numbers after exporting?**
>
> Because export can silently change behaviour — an operator gets mapped
> to a slightly different implementation, or a shape gets baked in when it
> should stay dynamic. If you do not compare against the original you find
> out much later, after you have already built engines and measured
> latency on a model that is not your model.

### Learn more

- 📖 [ONNX — official intro](https://onnx.ai/onnx/intro/)
- 📖 [PyTorch: Exporting to ONNX](https://pytorch.org/docs/stable/onnx.html)
- 🎥 [Search: "ONNX explained model deployment"](https://www.youtube.com/results?search_query=onnx+explained+model+deployment)
- 🔧 [Netron](https://netron.app) — drag an `.onnx` file in and see the
  graph. **Do this with your own file, it makes everything concrete.**

---

## 4.2 TensorRT

### The idea

NVIDIA's inference optimiser. It takes your graph and **rebuilds it for
one specific GPU**.

**The scenario.** A recipe says: boil water, add rice, drain, heat pan,
add rice. A good cook notices you can skip the draining and cook it in one
pan. Same dish, fewer steps, less washing up. TensorRT does that to a
neural network.

### The three things it does

**1. Layer fusion.** A convolution followed by batch-norm followed by
ReLU is three separate GPU kernels, each reading from and writing to slow
memory. TensorRT fuses them into one kernel where the intermediate values
never leave fast on-chip memory.

```
before:  conv → [write to memory] → BN → [write] → ReLU → [write]
after:   conv+BN+ReLU as one kernel → [write once]
```

**2. Kernel auto-tuning.** For each layer there are several possible GPU
implementations. TensorRT actually **runs** them on your GPU during the
build and keeps the fastest. This is why building an engine takes minutes,
and why an engine built for a T4 will not run on a different GPU.

**3. Precision.** Optionally run in FP16 or INT8. That is §4.3.

### Why the engine is GPU-specific

The auto-tuning is measured on the actual hardware. A kernel that is
fastest on a T4 may be slower on an A100 — different core counts,
different memory bandwidth. So `.plan` files are not portable, which is a
real operational constraint.

> **Q: What does TensorRT actually do to make it faster?**
>
> Three things. It fuses operations, so a conv, batch-norm and ReLU become
> one kernel instead of three round trips to memory. It benchmarks several
> implementations of each layer on the actual GPU during the build and
> keeps the fastest. And optionally it runs in lower precision. In my case
> FP32 alone — just fusion and tuning, no precision change — gave 1.34x.

### Learn more

- 📖 [How to Speed Up Deep Learning Inference Using TensorRT (NVIDIA blog)](https://developer.nvidia.com/blog/speed-up-inference-tensorrt/)
  — start here
- 📖 [TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- 🎥 [Search: "TensorRT tutorial ONNX inference"](https://www.youtube.com/results?search_query=tensorrt+tutorial+onnx+inference)

---

## 4.3 FP32, FP16, INT8

### The idea

How many bits you use to store each number.

| | Bits | Range | Typical use |
|---|---|---|---|
| FP32 | 32 | ±3.4×10³⁸ | training, default |
| FP16 | 16 | ±65,504 | inference |
| INT8 | 8 | −128 to +127 | inference, needs calibration |

### Why fewer bits is faster

Most of inference is **memory-bound**, not compute-bound. The GPU spends
more time waiting for numbers to arrive than multiplying them. Halve the
size of each number and you halve the waiting.

**The scenario.** You need to move 1,000 books across town. Your van
carries 100. That is 10 trips. Now the books are half the size — 5 trips.
You did not drive faster; there was just less to carry.

On top of that, modern GPUs have dedicated hardware that does several
INT8 operations in the time of one FP32 operation.

### The maths of INT8 quantisation

You are mapping a range of real numbers onto the 256 integers from −128
to +127.

```
scale = max_absolute_value / 127
quantised = round(value / scale)
recovered = quantised × scale
```

**Worked example.** Suppose a layer's activations range over ±4.2:

```
scale = 4.2 / 127 = 0.0331
```

| Real value | ÷ scale | Rounded | × scale | Error |
|---:|---:|---:|---:|---:|
| 1.000 | 30.2 | 30 | 0.993 | 0.007 |
| 2.500 | 75.5 | 76 | 2.516 | 0.016 |
| 0.010 | 0.30 | 0 | 0.000 | **0.010** |
| 4.200 | 126.9 | 127 | 4.204 | 0.004 |

Notice the third row. Small values round to **zero** and vanish
completely. That is the real danger of INT8 — not that large values become
slightly wrong, but that small ones disappear.

### Why that mattered here

This model's detection confidences sit close to the decision threshold.
Small numerical changes flip detections on and off. Measured result:

```
INT8:  3.83x faster, but only 34% of the FP32 detections survive
```

A better-trained model with confident, well-separated scores would survive
this much better. Ours is under-trained, so its outputs are fragile.

### Learn more

- 📖 [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
  — genuinely excellent, lots of diagrams
- 🎥 [From FP32 to INT8: Post-Training Quantization Explained in PyTorch](https://www.youtube.com/watch?v=7a8b6hgOjgc)
- 🎥 [Quantization: A Beginner's Guide to Model Optimization](https://www.youtube.com/watch?v=qN5TwGwXpdo)

---

## 4.4 Calibration

### The problem

To compute `scale = max / 127` you need to know the max. But activations
depend on the input, and you do not know them until you run the model.

### The answer

Run real data through it and watch.

```
1. take ~16 real validation images
2. run them through the FP32 model
3. record the range of values each layer produces
4. choose a scale per layer from those ranges
```

### Why not just use the maximum

One freak outlier wrecks it. Suppose a layer normally produces values in
±3, but one pixel in one image produces 47:

```
using max:          scale = 47 / 127 = 0.370
                    a typical value of 1.0 → round(2.7) = 3 → 1.11
                    11% error on every normal value, to accommodate one outlier
```

**Percentile calibration** throws away the extreme 0.001% first:

```
using 99.999th pct: scale = 3.1 / 127 = 0.0244
                    a typical value of 1.0 → round(41) = 41 → 1.0004
                    0.04% error; the outlier gets clipped, and it was noise anyway
```

This project uses percentile calibration for exactly this reason. I tried
entropy and min-max too — both were worse on this model.

> **Q: How does the model know how to convert float to int8?**
>
> Calibration. You run real images through the FP32 model and record the
> range of values each layer produces, then map that range onto −128 to
> 127. I used 16 validation samples with percentile calibration, which
> discards the extreme outliers before choosing the range — otherwise one
> freak value forces a coarse scale on everything else.

---

## 4.5 What actually happened

| Backend | Latency | FPS | Speedup | Detections kept |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 32.66 ms | 30.6 | 1.00x | baseline |
| ONNX Runtime (CUDA) | 30.20 ms | 33.1 | 1.08x | 100% |
| **TensorRT FP32** ← shipped | **24.34 ms** | **41.1** | **1.34x** | — |
| TensorRT FP16 | 61.56 ms | 16.2 | 0.53x | 66% |
| TensorRT INT8 | 8.52 ms | 117.3 | 3.83x | 34% |

Engine size: **49.4 MB → 3.1 MB** at INT8.

### "Detections kept" — why I measured this

Speed numbers on their own are meaningless if the model stopped working.
So for every backend I compared its detections against the FP32 ones: same
class, centre within 0.5 m. That fraction is the number in the last
column.

> **Q: INT8 is 3.8x faster. Why didn't you ship it?**
>
> Because it is not the same model any more. Only 34% of the FP32
> detections survived, and it emitted three to five times as many boxes,
> mostly noise. A detector that is fast and wrong is worse than one that
> is slower and right. TensorRT FP32 gave 1.34x with nothing given up, so
> that is what ships.
>
> I think this model is unusually fragile under quantisation because it is
> under-trained — its confidence scores sit close to the threshold, so
> small numerical changes flip detections. A well-trained model would
> likely hold up much better.

> **Q: Your FP16 is slower than FP32. Why?**
>
> I do not fully know, and I would rather say that than guess. What I
> ruled out: my first theory was cast overhead, because the FP16
> conversion left the upsampling layers in FP32 and wrapped them in
> conversions — 17 cast operations over 9.7 million elements. I cut that
> down to 1.7 million and the latency moved by 0.4 ms. So that was not it.
> My remaining theory is the scatter operation in the BEV pooling having
> no FP16 kernel, which would force conversions around a large tensor, but
> I have not confirmed it.

That is a good answer. "I measured, I ruled out the obvious cause, here is
my remaining hypothesis" beats a confident wrong answer every time.

---

# Part 5 — Serving

## 5.1 Why a server

A script that loads a model and runs it is not a deployment.

**The scenario.** Your model works on your laptop. Now the perception
team, the logging team and the simulation team all want to call it. Do
they each copy your script and your weights? What happens when you improve
the model — do you email everyone?

A server solves this: the model lives in one place behind an address.
Callers send data and get results. You update the model without anyone
changing their code.

A server also does things a script cannot:

- **batching** across callers who do not know about each other
- **concurrency** — several copies running at once
- **metrics** — latency and throughput, exposed for monitoring
- **versioning** — roll forward and back

---

## 5.2 Triton

### What it is

NVIDIA's inference server. Handles all of the above, supports TensorRT,
ONNX, PyTorch and TensorFlow models side by side.

Normally it runs in a Docker container. **PyTriton** is the same server
run inside a Python process — pip-installable, no Docker. This project
uses PyTriton because free GPU notebooks do not give you Docker.

### Dynamic batching

**The idea.** Instead of running each request the moment it arrives, wait
a couple of milliseconds to see whether more arrive, then run them
together.

**The scenario.** A lift on the ground floor. Someone presses the button.
Do you go up immediately, or hold the doors two seconds in case someone
else is walking over? Holding briefly means fewer trips overall.

**Why it works on a GPU.** A GPU running one image uses a fraction of its
cores. Running four costs barely more time than one:

```
4 requests, one at a time:   4 × 8.5 ms  = 34 ms
4 requests, batched:              ~11 ms
```

You pay a little latency on the first request to gain a lot of throughput.

**In this project:** max batch 4, 2 ms queue delay.

### Model instances

Each instance is a separate copy of the model with its own TensorRT
execution context. Those contexts are **not thread-safe**, so with one
instance concurrent requests queue behind each other.

**In this project:** 2 instances.

### The result — and the interesting part

| | |
|---|---:|
| Median latency through Triton | 48.56 ms |
| Same engine called directly | 8.52 ms |

**Six times slower, and none of it is compute.**

Work out the data volume per request:

```
input    6 × 3 × 224 × 400 × 4 bytes  =  6.45 MB
outputs  20 × 200 × 200 × 4 bytes     =  3.20 MB
                                  total ≈ 9.65 MB
```

Nearly 10 MB over HTTP, serialised to and from NumPy at both ends, for
every single inference. The model finishes in 8.5 ms and then waits on the
wire.

**The fix** is gRPC with CUDA shared memory: client and server point at
the same GPU buffers, so the tensors never get copied at all. Not
implemented here — but knowing *why* it is slow, and what the fix is, is
the point.

> **Q: What is dynamic batching?**
>
> Instead of running each request as it arrives, the server waits a couple
> of milliseconds to see if more come in, then runs them as one batch.
> GPUs are far more efficient on one batch of four than on four separate
> calls, so you trade a small amount of latency for a large gain in
> throughput.

> **Q: Why two model instances?**
>
> Each instance has its own TensorRT execution context, and those are not
> thread-safe. With a single instance, two concurrent requests serialise
> behind each other. Two instances can genuinely run at the same time.

> **Q: Why is your server six times slower than the engine?**
>
> It is not compute — the model still finishes in 8.5 ms. Each request
> ships about 10 MB: 6.4 MB of input images and 3.2 MB of output maps,
> serialised through NumPy over HTTP at both ends. That transfer is the
> other 40 ms. The standard fix is gRPC with CUDA shared memory so the
> tensors stay on the GPU, which is where I would go next.

### Learn more

- 📖 [Triton: Dynamic Batching & Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)
  — official, with diagrams
- 📖 [Triton conceptual guide (GitHub)](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide)
- 🎥 [Search: "Triton Inference Server tutorial"](https://www.youtube.com/results?search_query=nvidia+triton+inference+server+tutorial)

---

# Part 6 — The bugs

Five problems that cost real time. **These are yours.** You hit them, you
worked through them. Talk about them as a story, not as expertise —
nobody expects an intern to know TensorRT internals, but everybody
respects someone who reads an error message carefully.

### 1. The intrinsics did not survive the resize

Resized the images from 1600×900 to 400×224 and forgot that K is measured
in pixels. Every projected point landed in the wrong cell. Covered in
§1.2 with the numbers.

**What it taught me:** when a geometric pipeline produces plausible-looking
but wrong output, check the units first.

### 2. ONNX would not export the BEV pooling

The splat sums many points into the same output cell. In PyTorch that is
`index_add_`. On export:

```
ONNX export does not support exporting 'index_add_()'
with duplicated values in 'index'
```

Duplicates are the *whole point* — that is what pooling means. Switched to
`scatter_add`, which does the same thing and maps to a supported ONNX
operator.

### 3. Then TensorRT could not parse it

```
Assertion failed: plugin != nullptr:
ScatterReduction plugin was not found in the plugin registry!
```

TensorRT ships the plugin but does not load it unless you ask. One line
before parsing:

```python
trt.init_libnvinfer_plugins(logger, "")
```

**What it taught me:** "not found" sometimes means "not loaded", not
"does not exist".

### 4. TensorRT 11 has no precision flags

Every tutorial says:

```python
config.set_flag(trt.BuilderFlag.FP16)
```

In TensorRT 11 that attribute does not exist. I printed what *did* exist:

```python
>>> [f for f in dir(trt.BuilderFlag) if not f.startswith("_")]
['DEBUG', 'DIRECT_IO', ..., 'TF32', ...]        # no FP16, no INT8
```

NVIDIA moved to *strongly typed* networks — precision now comes from the
ONNX file itself, not a builder flag. So FP16 became an
`onnxconverter_common` conversion of the ONNX, and INT8 became
quantisation nodes written into the graph.

**What it taught me:** when the API does not match the tutorial, print
`dir()` on the object and look at what is actually there.

### 5. TensorRT only accepts symmetric quantisation

Two errors in sequence:

```
input has type Int32 but must have type FP8, FP4, Int4, or Int8
Non-zero zero point is not supported
```

ONNX Runtime's defaults quantise biases to int32 and use non-zero
zero-points. TensorRT rejects both. Three settings:

```python
extra_options={
    "QuantizeBias": False,
    "ActivationSymmetric": True,
    "WeightSymmetric": True,
}
```

*Symmetric* means the range is centred on zero — `[−4.2, +4.2]` maps to
`[−127, +127]` with zero mapping to zero. *Asymmetric* allows an offset,
which is more accurate for one-sided distributions but needs an extra
addition per operation, which TensorRT's kernels do not implement.

> **Q: What was the hardest part of the project?**
>
> Getting TensorRT to accept the graph. The BEV pooling scatters many
> values into the same output cell, and that one pattern kept hitting
> limits — first ONNX export would not do it, then TensorRT could not
> parse it, then quantisation produced a form it rejected. Three
> different fixes for what is conceptually one operation. Reading the
> actual error instead of searching for the symptom, and changing one
> thing at a time, is what got through it.

---

# Part 7 — Interview questions

### Opening

**"Tell me about this project."** → Part 0, the 60-second version.

**"Why did you build it?"**
> I wanted a project that went all the way from data to a served
> endpoint, not just a training notebook. Autonomous driving perception
> was interesting because the camera-only version has a genuinely hard
> sub-problem — depth — and because the deployment side is where I wanted
> to learn.

### Technical

**"How do you get 3D from 2D images?"** → §2.2

**"What are intrinsics and extrinsics?"** → §1.3

**"Why BEV instead of per-camera detection?"**
> A car can appear in two cameras at once, and merging those detections
> afterwards is fiddly — in BEV they land on the same cell automatically.
> And planning wants a map anyway, so producing one directly skips a
> conversion.

**"Why anchor-free detection?"** → §2.4

**"Why focal loss?"** → §3.1

**"What does TensorRT do?"** → §4.2

**"Why is INT8 faster?"** → §4.3

**"What is dynamic batching?"** → §5.2

### The uncomfortable ones

**"Your accuracy is terrible."**
> It is, and I know exactly why — 323 training samples. What I would point
> to instead is that I proved it was the data and not my code, by running
> ground truth through my own pipeline and scoring 0.90 mAP on the classes
> present. [continue with §3.4]

**"You got 3.8x with INT8 but didn't use it. Isn't that a waste?"**
> The measurement is the deliverable. I now know that this model loses
> two-thirds of its detections under INT8 and why, which is more useful
> than shipping something fast and broken. If I retrained on the full
> dataset I would test INT8 again, because I expect a well-trained model
> to hold up.

**"Why couldn't you solve the FP16 problem?"** → §4.5

**"How much of this did you write yourself?"**
> I used AI assistance, like most people do now. The architecture is from
> a published paper, which I reimplemented. What I can do is explain every
> piece of it, which is what matters.

Say this plainly. Nobody is impressed by a denial and everybody is
suspicious of one.

**"What would you do differently?"**
> Train on the full nuScenes set — everything else in the pipeline checks
> out, so data is the one thing holding accuracy back. Then gRPC with
> shared memory for serving, since I measured that the transfer dominates.

**"What is the weakest part?"**
> Accuracy, and the unresolved FP16 slowdown. I narrowed that one down but
> ran out of free GPU time before I could confirm the cause.

### When you do not know

> "I did not go that deep on that part. Here is what I do know, and here
> is how I would find out."

That is a completely acceptable answer for an intern. Guessing is not.

---

# Part 8 — A study plan

Three weeks, an hour a day. Do it in this order — each part depends on the
one before.

### Week 1 — Geometry and the model

| Day | Read | Watch | Then do |
|---|---|---|---|
| 1 | §1.1, §1.2 | [Camera Matrix](https://www.youtube.com/watch?v=Hz8kz5aeQ44) | Work the K rescaling by hand |
| 2 | §1.3, §1.4 | (same video, second half) | Work the pixel→3D example with different numbers |
| 3 | §1.5, §2.1 | CNN video | Open `camera_loader.py`, match it to §1.2 |
| 4 | §2.2 | [Lift, Splat, Shoot](https://www.youtube.com/watch?v=fqIwhE2dmWk) | Read `view_transformer.py` line by line |
| 5 | §2.2 again | [LSS blog](https://akashprakas.github.io/akashBlog/posts/2025-11-15-LiftSplatShoot.html) | Explain LSS out loud, no notes |
| 6 | §2.3, §2.4 | [CenterNet](https://learnopencv.com/centernet-anchor-free-object-detection-explained/) | Read `center_head.py` |
| 7 | — | — | Say the whole of Part 0 out loud |

### Week 2 — Training and optimisation

| Day | Read | Watch | Then do |
|---|---|---|---|
| 8 | §3.1, §3.2 | Focal loss video | Recompute the focal loss example |
| 9 | §3.3, §3.4 | — | Practise the oracle answer until fluent |
| 10 | §4.1 | ONNX video | Open your `.onnx` in [Netron](https://netron.app) |
| 11 | §4.2 | [NVIDIA TensorRT blog](https://developer.nvidia.com/blog/speed-up-inference-tensorrt/) | — |
| 12 | §4.3 | [FP32→INT8](https://www.youtube.com/watch?v=7a8b6hgOjgc) | Redo the quantisation table with scale 2.0 |
| 13 | §4.4, §4.5 | [Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) | — |
| 14 | — | — | Explain the whole benchmark table out loud |

### Week 3 — Serving, bugs, practice

| Day | Read | Then do |
|---|---|---|
| 15 | §5.1, §5.2 | Read `pytriton_server.py` |
| 16 | [Triton dynamic batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html) | Work out the 10 MB per request yourself |
| 17 | Part 6 | Explain each bug in two sentences |
| 18 | Part 7 | Answer every question out loud, timed |
| 19 | — | Re-read the whole README as if you had never seen it |
| 20 | — | Mock interview — have someone ask from Part 7 |
| 21 | — | Fix whatever you stumbled on |

### The honest test

You are ready when you can:

1. Give the 60-second version without notes
2. Explain LSS to someone who has never heard of it
3. Work the K-rescaling arithmetic on a whiteboard
4. Tell the oracle-test story and say why it mattered
5. Explain why you did not ship the fastest option
6. Say "I don't know" about the FP16 slowdown, and say what you ruled out

If you can do those six, you can hold a thirty-minute conversation about
this project with anyone.

---

# Appendix — All resources in one place

### Autonomous driving perception
- 📖 [nuScenes dataset](https://www.nuscenes.org/nuscenes) · [detection metrics](https://www.nuscenes.org/object-detection)
- 📖 [Lift, Splat, Shoot — paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf) · [code](https://github.com/nv-tlabs/lift-splat-shoot) · [blog walkthrough](https://akashprakas.github.io/akashBlog/posts/2025-11-15-LiftSplatShoot.html)
- 📖 [Simple-BEV paper](https://simple-bev.github.io/simple_bev_sep30.pdf) — what this project's architecture is based on
- 🎥 [Lift, Splat, Shoot](https://www.youtube.com/watch?v=fqIwhE2dmWk)

### Camera geometry
- 📖 [OpenCV camera calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- 🎥 [Computer Vision: The Camera Matrix](https://www.youtube.com/watch?v=Hz8kz5aeQ44)

### Detection
- 📖 [CenterNet explained (LearnOpenCV)](https://learnopencv.com/centernet-anchor-free-object-detection-explained/)
- 📖 [CenterPoint paper](https://arxiv.org/abs/2006.11275)
- 📖 [Focal Loss paper](https://arxiv.org/abs/1708.02002)
- 📖 [CS231n convolutional networks](https://cs231n.github.io/convolutional-networks/)

### Deployment
- 📖 [ONNX intro](https://onnx.ai/onnx/intro/) · [PyTorch ONNX export](https://pytorch.org/docs/stable/onnx.html)
- 🔧 [Netron — visualise any ONNX file](https://netron.app)
- 📖 [Speed up inference with TensorRT (NVIDIA)](https://developer.nvidia.com/blog/speed-up-inference-tensorrt/)
- 📖 [TensorRT quick start](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- 📖 [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- 🎥 [FP32 to INT8 post-training quantization in PyTorch](https://www.youtube.com/watch?v=7a8b6hgOjgc)
- 🎥 [Quantization: a beginner's guide](https://www.youtube.com/watch?v=qN5TwGwXpdo)

### Serving
- 📖 [Triton dynamic batching & concurrency](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)
- 📖 [Triton conceptual guide](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide)
- 📖 [PyTriton](https://github.com/triton-inference-server/pytriton)

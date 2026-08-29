# ============================================================
# benchmark.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# "3x fast panniten" nu resume-la sonna, PROOF venum. Ithu antha proof
# table-ai generate pannuthu: ovvoru backend-um evlo milliseconds
# edukkuthu, FPS evlo.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : runs/simplebev/best.pth, export/onnx/*.onnx, export/engines/*.plan
# Output: runs/benchmark.md (README-la neradiya paste pannalaam)
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# Ovvoru backend-kum: 10 warmup run + 50 timed run -> median edukirom.
#
# Yaen WARMUP? First run eppovum slow (GPU kernel load, memory alloc,
# cache cold). Athai sethu average pannina number thappu.
# Yaen MEDIAN, average illa? Oru run-la OS vera velai pannina spike
# varum. Median antha outlier-ai thaakkaadhu.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Output table: backend | latency (ms) | FPS | speedup
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Mac-la: PyTorch + ONNX Runtime numbers varum (TensorRT illa).
# Kaggle T4-la: ellaam varum (FP32/FP16/INT8/Triton).
#
# ============================================================

import argparse
import os
import time

import numpy as np


def timeit(fn, warmup: int = 10, runs: int = 50) -> float:
    """
    Oru function evlo neram edukkuthu nu median-a alakkurathu.

    Returns:
        median latency in milliseconds
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)   # ms

    return float(np.median(times))


def bench_pytorch(device_name: str, onnx_dir: str, ckpt: str, data_root: str) -> float:
    """PyTorch FP32 baseline - ithu thaan 'before' number."""
    import torch
    from models.simplebev import SimpleBEV
    from export.sample_source import load_samples

    device = torch.device(device_name)
    model = SimpleBEV(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])

    samples, K6, E6 = load_samples(data_root=data_root)
    imgs = samples[0].unsqueeze(0).to(device)
    K = K6.unsqueeze(0).to(device)
    E = E6.unsqueeze(0).to(device)

    def run():
        with torch.no_grad():
            model(imgs, K, E)
        # GPU async-a velai seiyum - sync pannaama time alandha thappu
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    return timeit(run)


def bench_onnxruntime(onnx_dir: str, data_root: str) -> tuple:
    """ONNX Runtime - PyTorch overhead illaama, graph optimized."""
    import onnxruntime as ort
    from export.sample_source import load_samples

    # CUDA kekirom, aana nijamaa edhu use aagudhu nu apram paapom -
    # ORT CUDA provider load fail aana silent-a CPU-ku vizhundhudum
    want = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    samples, _, _ = load_samples(data_root=data_root)
    imgs = samples[0].numpy()
    geom = np.load(os.path.join(onnx_dir, "geometry.npy"))

    bb = ort.InferenceSession(f"{onnx_dir}/camera_backbone.onnx", providers=want)
    dec = ort.InferenceSession(f"{onnx_dir}/bev_decoder.onnx", providers=want)
    ep = bb.get_providers()[0]              # NADANTHATHU, keattathu illa

    def run():
        feats = bb.run(None, {"images": imgs})[0]           # [6,64,14,25]
        dec.run(None, {"features": feats, "geometry": geom})

    return timeit(run, warmup=5, runs=20), ep.replace("ExecutionProvider", "")


def bench_tensorrt(engine_dir: str, precision: str, onnx_dir: str,
                   data_root: str, ckpt: str) -> tuple:
    """
    TensorRT engine - NVIDIA GPU la mattum.

    export/backends.py-la irukkura TensorRTBackend-ai use pannurom
    (inga thirumba ezhudhina rendu edathula bug fix panna vendiyirukkum).

    Returns:
        (median latency ms, mode string)
    """
    from export.backends import get_backend
    from export.sample_source import load_samples

    imgs, K, E = load_samples(data_root=data_root)
    b = get_backend(f"trt_{precision}", ckpt=ckpt, onnx_dir=onnx_dir,
                    engine_dir=engine_dir, K=K, E=E)
    return b.measure_latency(imgs[0].numpy()), b.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/simplebev/best.pth")
    ap.add_argument("--onnx-dir", default="export/onnx")
    ap.add_argument("--engine-dir", default="export/engines")
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    ap.add_argument("--out", default="runs/benchmark.md")
    args = ap.parse_args()

    import torch
    if torch.cuda.is_available():
        dev, dev_name = "cuda", torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        dev, dev_name = "mps", "Apple Silicon GPU (MPS)"
    else:
        dev, dev_name = "cpu", "CPU"
    print(f"device: {dev_name}\n")

    results = []

    print("PyTorch FP32...")
    baseline = bench_pytorch(dev, args.onnx_dir, args.ckpt, args.data_root)
    results.append(("PyTorch FP32", baseline, dev.upper()))
    print(f"  {baseline:.2f} ms")

    if os.path.exists(f"{args.onnx_dir}/camera_backbone.onnx"):
        print("ONNX Runtime...")
        try:
            ms, ort_dev = bench_onnxruntime(args.onnx_dir, args.data_root)
            results.append((f"ONNX Runtime", ms, ort_dev))
            print(f"  {ms:.2f} ms")
        except Exception as e:
            print(f"  skip: {e}")

    for precision in ["fp32", "fp16", "int8"]:
        if not os.path.exists(f"{args.engine_dir}/camera_backbone_{precision}.plan"):
            continue
        print(f"TensorRT {precision.upper()}...")
        try:
            ms, label = bench_tensorrt(args.engine_dir, precision,
                                       args.onnx_dir, args.data_root, args.ckpt)
            results.append((label, ms, "CUDA"))
            print(f"  {ms:.2f} ms")
        except Exception as e:
            print(f"  skip: {type(e).__name__}: {str(e)[:90]}")

    # --- Markdown table ---
    lines = [
        f"# Inference Benchmark",
        "",
        f"Hardware: **{dev_name}**  |  Input: 6 cameras x 3x224x400  |  batch 1",
        "",
        "| Backend | Runs on | Latency (ms) | FPS | Speedup |",
        "|---|---|---|---|---|",
    ]
    for name, ms, run_dev in results:
        lines.append(f"| {name} | {run_dev} | {ms:.2f} | {1000/ms:.1f} | {baseline/ms:.2f}x |")
    if len({d for _, _, d in results}) > 1:
        lines += ["", "> **Kavanam:** rows different device-la odudhu - neradiya compare",
                  "> panna koodadhu. Sariyaana comparison-ku ellaam ORE GPU-la odanum",
                  "> (Kaggle T4). Mac-la ONNX Runtime CPU-la mattum thaan odum."]
    if not any("TensorRT" in n for n, _, _ in results):
        lines += ["", "> TensorRT rows illa - NVIDIA GPU + built engines thevai.",
                  "> `python -m export.calibrate_int8 --precision fp16` odichi, "
                  "appuram ithai thirumba odu."]
    if any("split" in n for n, _, _ in results):
        lines += ["", "> **split** = TensorRT-ku `scatter_add` (BEV pooling) support",
                  "> illa, so antha oru step mattum PyTorch-la odudhu. Meedhi",
                  "> ellaam (backbone + BEV encoder + head) TensorRT-la."]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()

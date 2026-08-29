# ============================================================
# infer_client.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Triton server-ku request anuppi, BEV detections vaangura client.
# Ithu thaan "production-la epdi use pannuvom" nu kaattura file.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Server: docker/docker-compose-triton.yml
# Models: triton/model_repository/ensemble_bev
# Decode: training/evaluate.py-oda decode_predictions reuse pannurom
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 1. Server health check
# 2. images + geometry -> ensemble_bev ku gRPC request
# 3. 6 head outputs vaangi -> decode -> 3D boxes
# 4. Latency measure (client side, network sethu)
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : nuScenes val sample
# Output: detected boxes list + latency print
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# NVIDIA GPU machine-la Triton start panniyathukku apram:
#   python triton_deploy/client/infer_client.py --url localhost:8001
#
# Folder peru "triton" illa "triton_deploy" yaen? Project root-la
# "triton/" nu folder irundha, PyTorch import panra REAL triton
# package-ai athu shadow pannidum -> "module triton has no attribute
# language" nu error varum. Ithu naan neradiya sandhichathu.
#
# ============================================================

import argparse
import os
import sys
import time

import numpy as np

# Script-a odumbodhu project root sys.path-la illa - athai serkurom
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="localhost:8001", help="Triton gRPC address")
    ap.add_argument("--model", default="ensemble_bev")
    ap.add_argument("--onnx-dir", default="export/onnx")
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        print("tritonclient illa. Install pannu:")
        print("  pip install tritonclient[grpc]")
        return

    client = grpcclient.InferenceServerClient(url=args.url)

    if not client.is_server_ready():
        print(f"Triton server ready illa @ {args.url}")
        print("Mudhalla server start pannu:")
        print("  docker compose -f docker/docker-compose-triton.yml up")
        return
    print(f"server ready @ {args.url}")
    print(f"model '{args.model}' ready:", client.is_model_ready(args.model))

    # --- Input thayaar ---
    from data.scripts.dataset import NuScenesBEVDataset
    ds = NuScenesBEVDataset(args.data_root, split="val")
    images = np.ascontiguousarray(ds[0]["images"].numpy())          # [6,3,224,400]
    # geometry ORE thadava load - camera calibration maaraadhu
    geometry = np.ascontiguousarray(np.load(f"{args.onnx_dir}/geometry.npy"))

    inputs = [
        grpcclient.InferInput("images", images.shape, "FP32"),
        grpcclient.InferInput("geometry", geometry.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(images)
    inputs[1].set_data_from_numpy(geometry)

    head_names = ["heatmap", "offset", "height", "size", "rot", "vel"]
    outputs = [grpcclient.InferRequestedOutput(n) for n in head_names]

    # --- Oru request: correct-a varuthaa nu check ---
    res = client.infer(args.model, inputs, outputs=outputs)
    preds = {n: res.as_numpy(n) for n in head_names}
    for n, v in preds.items():
        print(f"  {n:8s} {v.shape}")

    # --- Decode: raw heatmap -> 3D boxes ---
    import torch
    from training.evaluate import decode_predictions
    boxes = decode_predictions({k: torch.from_numpy(v) for k, v in preds.items()})[0]
    print(f"\ndetected {len(boxes)} boxes (score > threshold)")
    for b in boxes[:5]:
        print(f"  cls {b['cls']:2d} | score {b['score']:.3f} | "
              f"pos ({b['x']:+6.1f}, {b['y']:+6.1f}, {b['z']:+5.1f}) m")

    # --- Latency: warmup + median ---
    for _ in range(10):
        client.infer(args.model, inputs, outputs=outputs)

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        client.infer(args.model, inputs, outputs=outputs)
        times.append((time.perf_counter() - t0) * 1000)

    ms = float(np.median(times))
    print(f"\nTriton ensemble: {ms:.2f} ms  |  {1000/ms:.1f} FPS  "
          f"(client-side, network sethu)")


if __name__ == "__main__":
    main()

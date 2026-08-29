# Inference Benchmark

Hardware: **Tesla T4**  |  Input: 6 cameras x 3x224x400  |  batch 1

| Backend | Runs on | Latency (ms) | FPS | Speedup |
|---|---|---|---|---|
| PyTorch FP32 | CUDA | 32.66 | 30.6 | 1.00x |
| ONNX Runtime | CUDA | 30.20 | 33.1 | 1.08x |
| TensorRT FP32 | CUDA | 24.34 | 41.1 | 1.34x |
| TensorRT FP16 | CUDA | 61.56 | 16.2 | 0.53x |
| TensorRT INT8 | CUDA | 8.52 | 117.3 | 3.83x |

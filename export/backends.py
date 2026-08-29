# ============================================================
# backends.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Ore model-ai 3 vazhiyila odalaam: PyTorch, ONNX Runtime, TensorRT.
# Ovvondrukum API vera vera (torch tensor vs numpy vs GPU pointer).
#
# Comparison panra podhu antha vithiyasam thontharavu. So ellathaiyum
# ORE interface-la mudurom:
#     backend.run(images) -> {"heatmap": ..., "offset": ...}
#     backend.latency_ms
#
# Ippo compare_renderer.py "edhu backend nu theriyaama" velai seiyum.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : export/onnx/*.onnx, export/engines/*.plan, runs/simplebev/best.pth
# Output: visualization/compare_renderer.py, export/benchmark.py use pannum
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# Ovvoru class-um load pannurathu + run pannurathu + time alakkurathu.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# run(images [6,3,224,400] numpy) -> dict of 6 numpy arrays
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# b = get_backend("trt_int8")
# preds = b.run(images)
# print(b.measure_latency(images))
#
# ============================================================

import os
import time

import numpy as np

HEAD_NAMES = ["heatmap", "offset", "height", "size", "rot", "vel"]


class Backend:
    """Ellaa backend-kum common base."""

    name = "base"

    def run(self, images: np.ndarray) -> dict:
        """images [6,3,224,400] -> {head_name: numpy array}"""
        raise NotImplementedError

    def measure_latency(self, images: np.ndarray, warmup: int = 10,
                        runs: int = 50) -> float:
        """
        Median latency in ms.

        Warmup yaen? First few run eppovum slow (memory alloc, kernel
        load). Adhai sethu average pannina number poi solli-idum.
        """
        for _ in range(warmup):
            self.run(images)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.run(images)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))


class PyTorchBackend(Backend):
    """Baseline - namma original model."""

    name = "PyTorch FP32"

    def __init__(self, ckpt: str, K, E, device: str = None):
        import torch
        from models.simplebev import SimpleBEV

        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.torch = torch
        self.device = torch.device(device)
        self.name = f"PyTorch FP32 ({device})"

        self.model = SimpleBEV(pretrained=False).to(self.device).eval()
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device)["model"])

        self.K = K.unsqueeze(0).to(self.device)
        self.E = E.unsqueeze(0).to(self.device)

    def run(self, images: np.ndarray) -> dict:
        t = self.torch
        x = t.from_numpy(np.ascontiguousarray(images)).unsqueeze(0).to(self.device)
        with t.no_grad():
            preds = self.model(x, self.K, self.E)
        # GPU async - sync pannaama time alandha thappu
        if self.device.type == "cuda":
            t.cuda.synchronize()
        elif self.device.type == "mps":
            t.mps.synchronize()
        return {k: v.cpu().numpy() for k, v in preds.items()}


class ONNXBackend(Backend):
    """ONNX Runtime - rendu graph (backbone + decoder)."""

    name = "ONNX Runtime"

    def __init__(self, onnx_dir: str):
        import onnxruntime as ort

        # CUDA-vai KEKIROM, aana kedaikkum-nu உறுதி illa.
        # ORT CUDA provider load fail aana (driver/CUDA version mismatch)
        # silent-a CPU-ku vizhundhudum. So KEATTATHAI illa, NADANTHATHAI
        # report pannanum - illaina benchmark table poi solli-idum.
        # (Ithu naan Kaggle-la sandhicha bug: table "CUDA" nu kaattudhu,
        #  nijamaa CPU-la 736 ms odindhurundhudhu.)
        want = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.bb = ort.InferenceSession(f"{onnx_dir}/camera_backbone.onnx",
                                       providers=want)
        self.dec = ort.InferenceSession(f"{onnx_dir}/bev_decoder.onnx",
                                        providers=want)
        actual = self.bb.get_providers()[0]      # nijamaa edhu use aagudhu
        self.name = f"ONNX Runtime ({actual.replace('ExecutionProvider','')})"
        self.geom = np.load(f"{onnx_dir}/geometry.npy")

    def run(self, images: np.ndarray) -> dict:
        feats = self.bb.run(None, {"images": np.ascontiguousarray(images)})[0]
        outs = self.dec.run(None, {"features": feats, "geometry": self.geom})
        return dict(zip(HEAD_NAMES, outs))


class TensorRTBackend(Backend):
    """
    TensorRT engine backend - NVIDIA GPU la mattum.

    RENDU mode:
      "fused" : camera_backbone -> bev_decoder   (LSS-um TRT-la)
      "split" : camera_backbone -> [PyTorch LSS] -> bev_head

    Yaen split? TensorRT-ku `scatter_add` (ONNX ScatterElements with
    reduction) support illa. Adhu thaan BEV pooling. So scatter-ai
    mattum PyTorch-la vachikitu, meedhi ellathaiyum TRT-la potrom.
    Heavy compute (backbone + 200x200 encoder/head) TRT-la-ye irukku.

    Buffers-ku pycuda illaama torch CUDA tensors use pannurom - split
    mode-la PyTorch-um sethu velai seiya vendiyirukku, so ore memory
    world-la irundha copy thevai illa.
    """

    def __init__(self, engine_dir: str, precision: str, onnx_dir: str,
                 ckpt: str = "runs/simplebev/best.pth", K=None, E=None):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.trt = trt
        self.device = torch.device("cuda")
        self.name = f"TensorRT {precision.upper()}"

        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")       # built-in plugins
        self.runtime = trt.Runtime(logger)

        self.bb_eng, self.bb_ctx, self.bb_buf = self._load(
            f"{engine_dir}/camera_backbone_{precision}.plan")

        fused = f"{engine_dir}/bev_decoder_{precision}.plan"
        split = f"{engine_dir}/bev_head_{precision}.plan"

        if os.path.exists(fused):
            self.mode = "fused"
            self.dec_eng, self.dec_ctx, self.dec_buf = self._load(fused)
            geom = np.load(f"{onnx_dir}/geometry.npy")
            self.dec_buf["geometry"].copy_(
                torch.from_numpy(geom).to(self.device))
        elif os.path.exists(split):
            self.mode = "split"
            self.name += " (split: LSS in PyTorch)"
            self.dec_eng, self.dec_ctx, self.dec_buf = self._load(split)

            # LSS-ku PyTorch model thevai (scatter anga nadakkum)
            from models.simplebev import SimpleBEV
            m = SimpleBEV(pretrained=False).eval()
            m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
            self.lss = m.view_transformer.to(self.device).eval()
            self.K = K.unsqueeze(0).to(self.device)
            self.E = E.unsqueeze(0).to(self.device)
            # geometry ORE thadava - camera calibration maaraadhu
            with torch.no_grad():
                self.geom = self.lss.get_geometry(self.K, self.E)
        else:
            raise FileNotFoundError(
                f"bev_decoder-um bev_head-um illa ({precision}) - "
                f"calibrate_int8.py odichiyaa?")

        print(f"  TRT backend mode: {self.mode}")

    def _load(self, path: str):
        """Engine load panni, ovvoru I/O tensor-kum torch buffer alloc."""
        torch = self.torch
        with open(path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        ctx = engine.create_execution_context()

        bufs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            bufs[name] = torch.empty(shape, dtype=torch.float32,
                                     device=self.device)
        return engine, ctx, bufs

    def _run_engine(self, ctx, bufs):
        """Buffer address kudthu engine-ai odurathu."""
        for name, t in bufs.items():
            ctx.set_tensor_address(name, t.data_ptr())
        ctx.execute_async_v3(self.torch.cuda.current_stream().cuda_stream)
        self.torch.cuda.synchronize()

    def run(self, images: np.ndarray) -> dict:
        torch = self.torch

        # --- Stage 1: camera backbone (TRT) ---
        self.bb_buf["images"].copy_(
            torch.from_numpy(np.ascontiguousarray(images)).to(self.device))
        self._run_engine(self.bb_ctx, self.bb_buf)
        feats = self.bb_buf["features"]              # [6,64,14,25]

        # --- Stage 2 ---
        if self.mode == "fused":
            self.dec_buf["features"].copy_(feats)
        else:
            # LSS PyTorch-la (scatter TRT-la aagaadhu)
            with torch.no_grad():
                vol = self.lss.lift(feats)                   # [6,C,D,H,W]
                bev = self.lss.splat(vol.unsqueeze(0), self.geom)
            self.dec_buf["bev"].copy_(bev)

        # --- Stage 3: decoder / head (TRT) ---
        self._run_engine(self.dec_ctx, self.dec_buf)
        return {n: self.dec_buf[n].cpu().numpy() for n in HEAD_NAMES}


def get_backend(kind: str, ckpt: str = "runs/simplebev/best.pth",
                onnx_dir: str = "export/onnx",
                engine_dir: str = "export/engines", K=None, E=None) -> Backend:
    """
    Peru kudutha backend-ai create panni thara.

    Args:
        kind: "pytorch" | "onnx" | "trt_fp32" | "trt_fp16" | "trt_int8"
    """
    if kind == "pytorch":
        return PyTorchBackend(ckpt, K, E)
    if kind == "onnx":
        return ONNXBackend(onnx_dir)
    if kind.startswith("trt_"):
        return TensorRTBackend(engine_dir, kind.split("_")[1], onnx_dir,
                               ckpt=ckpt, K=K, E=E)
    raise ValueError(f"theriyaadha backend: {kind}")


def available_backends(onnx_dir: str = "export/onnx",
                       engine_dir: str = "export/engines") -> list:
    """Intha machine-la edhu edhu odum nu paathu list thara."""
    out = ["pytorch"]
    if os.path.exists(f"{onnx_dir}/camera_backbone.onnx"):
        try:
            import onnxruntime  # noqa: F401
            out.append("onnx")
        except ImportError:
            pass
    for p in ["fp32", "fp16", "int8"]:
        has_bb = os.path.exists(f"{engine_dir}/camera_backbone_{p}.plan")
        has_dec = (os.path.exists(f"{engine_dir}/bev_decoder_{p}.plan") or
                   os.path.exists(f"{engine_dir}/bev_head_{p}.plan"))
        if has_bb and has_dec:      # rendum irundha thaan odum
            out.append(f"trt_{p}")
    return out

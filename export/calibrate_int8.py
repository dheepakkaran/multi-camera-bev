# ============================================================
# calibrate_int8.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Model-oda ellaa number-um FP32 (32 bit). INT8 (8 bit) la maathina
# 4x memory kammi, 3-4x fast. Aana problem: 32 bit-la irundha
# -3.4e38 to +3.4e38 range, INT8-la -128 to +127 mattum thaan.
#
# "Calibration" = konjam real photos-ai model-la anuppi, ovvoru layer-la
# actual numbers enna range-la varuthu nu paakurathu. Appuram antha
# range-ai -128..127 ku fit pannurathu (scale factor kandupidikkurathu).
#
# Analogy: veyil-la photo edukka exposure set pannurathu maadhiri.
# Munnadi konjam photo eduthu paathu "intha veliccham range-la iruku"
# nu therinjaa, correct-a set pannalaam.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : export/onnx/*.onnx (export_onnx.py kudukkurathu)
# Output: export/engines/*.plan -> triton/model_repository/ ku pogum
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# TensorRT builder + EntropyCalibrator2 + 50 nuScenes val samples.
# EntropyCalibrator2 yaen? CNN-ku ithu thaan best (KL-divergence use
# panni "evlo information loss aaguthu" nu minimize pannum).
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : onnx path, precision (fp16/int8), calibration samples
# Output: .plan engine file
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# Ithu MAC-LA ODAATHU (NVIDIA GPU thevai).
# Kaggle/Colab T4-la odanum:
#   !python -m export.calibrate_int8 --precision int8
#
# ============================================================

import argparse
import os

import numpy as np

# TensorRT Mac-la illa. Import fail aana clean-a solli veliya varanum,
# illaina "ModuleNotFoundError" nu confusing-a irukkum.
try:
    import tensorrt as trt
    import pycuda.autoinit          # noqa: F401  (CUDA context create pannum)
    import pycuda.driver as cuda
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# TensorRT 10-la pazhaya calibrator API (implicit quantization) REMOVE
# pannitaanga. Ippo "explicit quantization" thaan - ONNX graph-la-ye
# QuantizeLinear/DequantizeLinear (QDQ) nodes potrom, TensorRT athai
# padichi INT8 engine build pannum.
#
# Analogy: munnadi "TensorRT-e ne range kandupudi" nu sonnom.
# Ippo "naame range kandupidichi graph-la ezhuthi kudukirom" - control
# namma kitta, result-um predictable.
TRT_HAS_CALIBRATOR = TRT_AVAILABLE and hasattr(trt, "IInt8EntropyCalibrator2")


class EntropyCalibrator(trt.IInt8EntropyCalibrator2 if TRT_HAS_CALIBRATOR else object):
    """
    TensorRT-ku calibration data feed pannura class.

    TensorRT engine build panra podhu "adutha batch kudu" nu ketkum ->
    get_batch() call aagum -> namma GPU memory pointer kudukirom.
    """

    def __init__(self, data: dict, cache_file: str):
        """
        Args:
            data: {input_name: numpy array [N, ...]}  N = calibration samples
            cache_file: calibration result-ai save panna (2nd time fast)
        """
        super().__init__()
        self.data = data
        self.names = list(data.keys())
        self.n = len(next(iter(data.values())))
        self.idx = 0
        self.cache_file = cache_file

        # GPU-la ovvoru input-kum idam eduthu vachikirom (oru sample size)
        self.device_mem = {
            name: cuda.mem_alloc(arr[0].nbytes) for name, arr in data.items()
        }

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names, p_str=None):
        """Adutha calibration sample-ai GPU-ku anuppurathu."""
        if self.idx >= self.n:
            return None                      # mudinjuthu nu solrathu
        for name in self.names:
            sample = np.ascontiguousarray(self.data[name][self.idx])
            cuda.memcpy_htod(self.device_mem[name], sample)
        self.idx += 1
        if self.idx % 10 == 0:
            print(f"  calibration {self.idx}/{self.n}")
        return [int(self.device_mem[n]) for n in names]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class _ORTCalibrationReader:
    """
    onnxruntime.quantization-ku calibration data feed pannurathu.

    quantize_static() "adutha batch kudu" nu ketkum -> get_next() call
    aagum -> namma oru sample kudukirom -> None vandha mudinjadhu.
    """

    def __init__(self, data: dict):
        self.data = data
        self.n = len(next(iter(data.values())))
        self.i = 0

    def get_next(self):
        if self.i >= self.n:
            return None
        batch = {k: np.ascontiguousarray(v[self.i]) for k, v in self.data.items()}
        self.i += 1
        if self.i % 10 == 0:
            print(f"  calibration {self.i}/{self.n}")
        return batch

    def rewind(self):
        self.i = 0


def convert_onnx_fp16(onnx_path: str, out_path: str) -> str:
    """
    ONNX model-oda weights + activations-ai FP16-a maathurathu.

    Yaen ithu thevai? TensorRT 8-la `config.set_flag(BuilderFlag.FP16)`
    nu sonna pothum - TRT-e ellathaiyum fp16-la odum.
    TensorRT 11-la antha flag-e ILLA. Ippo "strongly typed" vazhi:
    ONNX-la enna type irukko, TRT adhe type-la odum.

    So precision-ai namma ONNX-la-ye set pannanum.

    keep_io_types=True yaen? Input/output-ai FP32-a-ve vaikirom.
    Ulla mattum fp16. Ippdi panna namma buffers/client code
    maatha vendaam - ulla mattum fast aagum.

    op_block_list=[] yaen MUKKIYAM?
    Converter-oda default block list-la `Resize` iruku - "ithu fp32-la
    thaan iruken" nu vidum. Namma FPN upsample-ku Resize use pannurom.
    Appo graph-la fp32 "theevu" (island) uruvaagum, athu suthi Cast
    node varum. Naan alandhu paathen:
        default        : 17 casts, 9,740,808 elements
        op_block_list=[]: 11 casts, 1,740,800 elements   (5.6x kammi)
    Antha extra cast-aala FP16 engine FP32 vida SLOW aachu (63 vs 28 ms).
    Ellathaiyum fp16-la convert panna antha overhead poidum.

    Args:
        onnx_path: FP32 onnx
        out_path : FP16 onnx

    Returns:
        out_path
    """
    import onnx
    from onnxconverter_common import float16

    print(f"  converting {os.path.basename(onnx_path)} -> FP16 ...")
    model = onnx.load(onnx_path)
    model16 = float16.convert_float_to_float16(
        model, keep_io_types=True, op_block_list=[])
    onnx.save(model16, out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"  saved {out_path} ({mb:.1f} MB)")
    return out_path


def quantize_onnx_qdq(onnx_path: str, out_path: str, data: dict) -> str:
    """
    ONNX model-la QDQ nodes potu INT8 version create pannurathu.

    Enna nadakkuthu? 50 real sample-ai model-la anuppi, ovvoru layer-la
    numbers enna range-la varuthu nu paakuthu. Appuram antha range-ai
    graph-la-ye "intha layer -128..127 ku ippdi scale pannu" nu
    ezhuthi vaikkuthu (QuantizeLinear / DequantizeLinear nodes).

    TensorRT antha nodes-ai paathu "aha, INT8 nu solriye" nu engine
    build pannum.

    Args:
        onnx_path: FP32 onnx
        out_path : INT8 (QDQ) onnx
        data     : {input_name: [N, ...] numpy}

    Returns:
        out_path
    """
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Pre-process: shape inference + optimization. Ithu illaina
    # quantization konjam layer-ai miss pannum.
    pre_path = onnx_path.replace(".onnx", "_pre.onnx")
    quant_pre_process(onnx_path, pre_path, skip_symbolic_shape=True)

    print(f"  quantizing {os.path.basename(onnx_path)} ...")
    quantize_static(
        pre_path, out_path,
        calibration_data_reader=_ORTCalibrationReader(data),
        quant_format=QuantFormat.QDQ,     # TensorRT QDQ format kekkum
        activation_type=QuantType.QInt8,  # signed int8 - TRT-ku ithu thaan
        weight_type=QuantType.QInt8,
        per_channel=True,                 # ovvoru conv filter-kum thani scale
                                          # -> accuracy nalla irukkum
        # Entropy vs MinMax vs Percentile moonaiyum Mac-la test panniten.
        # Entropy/MinMax rendum 76% feature error kuduthuchu, Percentile
        # 55% - so Percentile thaan intha model-ku best.
        calibrate_method=__import__(
            "onnxruntime.quantization", fromlist=["CalibrationMethod"]
        ).CalibrationMethod.Percentile,
        # QuantizeBias=False MUKKIYAM: ORT default-a conv bias-ai int32-la
        # quantize pannum. TensorRT 11 int32 DequantizeLinear-ai accept
        # panna maattaadhu ("input has type Int32 but must have type
        # FP8, FP4, Int4, or Int8"). Bias-ai FP32-la vittaa TRT santhosham.
        extra_options={
            # ORT default-a conv bias-ai int32-la quantize pannum.
            # TRT 11 int32 DequantizeLinear-ai accept panna maattaadhu.
            "QuantizeBias": False,
            # TensorRT SYMMETRIC quantization mattum thaan support pannum
            # (zero_point = 0). ORT default-la activation-ku non-zero
            # zero_point varum -> "Non-zero zero point is not supported"
            # nu parse fail aagum. Ithu naan Kaggle-la sandhichadhu.
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    os.remove(pre_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"  saved {out_path} ({mb:.1f} MB)")
    return out_path


def build_engine(onnx_path: str, engine_path: str, precision: str,
                 calibrator=None, workspace_gb: int = 4) -> None:
    """
    ONNX -> TensorRT engine (.plan) build pannurathu.

    Args:
        onnx_path   : input .onnx
        engine_path : output .plan
        precision   : "fp32" | "fp16" | "int8"
        calibrator  : int8-ku mattum thevai
    """
    logger = trt.Logger(trt.Logger.WARNING)

    # TensorRT-oda built-in plugin library-ai load pannurom.
    # Ithu illaina sila ONNX op (ScatterElements madhiri) "plugin not
    # found in registry" nu fail aagum.
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)

    # TensorRT 8-la EXPLICIT_BATCH flag kandippa kudukkanum.
    # TRT 10+ la explicit batch thaan default, antha flag deprecated
    # (TRT 11-la remove aagirukalaam). Rendaiyum handle pannurom.
    try:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    except AttributeError:
        network = builder.create_network()

    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise RuntimeError(f"parse fail: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    # BuilderFlag-la enna iruku nu version-ku version maarum.
    # TRT 11-la FP16 flag remove pannitaanga (strongly-typed networks
    # thaan pudhu vazhi). So irukkuradhai paathu use pannurom.
    def try_flag(name: str) -> bool:
        flag = getattr(trt.BuilderFlag, name, None)
        if flag is None:
            print(f"  BuilderFlag.{name} illa (TRT {trt.__version__}) - skip")
            return False
        config.set_flag(flag)
        return True

    if precision == "fp16":
        if not (try_flag("FP16") or try_flag("HALF")):
            print("  FP16 flag illa - ONNX-la-ye precision irukkanum")
    elif precision == "int8":
        try_flag("INT8")
        # FP16-um on pannurom: INT8-la convert panna mudiyaadha layer
        # irundha FP16-ku fallback aagum (FP32 vida fast)
        try_flag("FP16")
        if calibrator is not None:
            config.int8_calibrator = calibrator
        # calibrator None-na, ONNX-la-ye QDQ nodes irukkum (explicit
        # quantization) - TensorRT athai thaana padichikkum

    print(f"building {precision.upper()} engine... (5-10 min edukkum)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build fail")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"  saved {engine_path} ({size_mb:.1f} MB)")


def collect_calibration_data(data_root: str, n_samples: int, onnx_dir: str) -> tuple:
    """
    nuScenes val-la irunthu N sample eduthu calibration data-va thayaar pannurathu.

    Yaen val samples? Real-world distribution venum. Random noise use
    pannina, calibration thappu range kudukkum -> accuracy drop aagum.

    Returns:
        (backbone_data, decoder_data) - rendum dict of numpy arrays
    """
    import torch
    from models.simplebev import SimpleBEV
    from export.sample_source import load_samples

    imgs, _, _ = load_samples(data_root=data_root)          # [N,6,3,224,400]
    n_samples = min(n_samples, len(imgs))
    print(f"calibration data: {n_samples} samples")

    model = SimpleBEV(pretrained=False).eval()
    ckpt = torch.load("runs/simplebev/best.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    geom = np.load(os.path.join(onnx_dir, "geometry.npy"))   # [6,D,14,25,3]

    images, feats = [], []
    with torch.no_grad():
        for i in range(n_samples):
            img = imgs[i]                               # [6,3,224,400]
            images.append(img.numpy())
            feats.append(model.backbone(img).numpy())   # [6,64,14,25]

    # Decoder-ku kammi sample. Yaen? ORT calibration ovvoru tensor-oda
    # data-vaiyum memory-la vachikkum. Decoder-la [1,64,200,200] madhiri
    # periya tensors iruku - 50 sample potta RAM theendhu process
    # kill aagum (Mac-la naan ithai sandhichen).
    # Backbone-ku 16, decoder-ku 8. Yaen ippdi kammi?
    # ORT calibration ovvoru tensor-oda data-vaiyum memory-la vachikkum.
    # Kaggle-la 50 sample potta process KILL aachu (OOM).
    n_bb = min(16, n_samples)
    n_dec = min(8, n_samples)
    return (
        {"images": np.stack(images[:n_bb])},
        {"features": np.stack(feats[:n_dec]),
         "geometry": np.repeat(geom[None], n_dec, 0)},
    )


def collect_bev_calibration(data_root: str, n_samples: int, onnx_dir: str) -> dict:
    """bev_head-ku calibration data (BEV feature maps)."""
    import torch
    from export.sample_source import load_samples
    from models.simplebev import SimpleBEV

    imgs, K, E = load_samples(data_root=data_root)
    n_samples = min(n_samples, len(imgs))
    model = SimpleBEV(pretrained=False).eval()
    model.load_state_dict(
        torch.load("runs/simplebev/best.pth", map_location="cpu")["model"])

    bevs = []
    with torch.no_grad():
        for i in range(n_samples):
            f = model.backbone(imgs[i]).unsqueeze(0)
            bevs.append(model.view_transformer(f, K.unsqueeze(0),
                                               E.unsqueeze(0)).numpy()[0])
    return {"bev": np.stack(bevs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="export/onnx")
    ap.add_argument("--engine-dir", default="export/engines")
    ap.add_argument("--precision", default="int8", choices=["fp32", "fp16", "int8"])
    ap.add_argument("--calib-samples", type=int, default=50)
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    args = ap.parse_args()

    if not TRT_AVAILABLE:
        print("TensorRT illa - ithu NVIDIA GPU-la mattum odum.")
        print("Kaggle/Colab T4-la ippdi odu:")
        print(f"  !python -m export.calibrate_int8 --precision {args.precision}")
        return

    os.makedirs(args.engine_dir, exist_ok=True)

    print(f"TensorRT {trt.__version__} | "
          f"calibrator API: {'irukku' if TRT_HAS_CALIBRATOR else 'illa (TRT 10+)'}")

    sources = {"camera_backbone": f"{args.onnx_dir}/camera_backbone.onnx",
               "bev_decoder": f"{args.onnx_dir}/bev_decoder.onnx"}
    calibs = {"camera_backbone": None, "bev_decoder": None}

    if args.precision == "fp16":
        # TRT 11-la FP16 builder flag illa - ONNX-la-ye fp16 aakkanum
        for name in sources:
            sources[name] = convert_onnx_fp16(
                sources[name], f"{args.onnx_dir}/{name}_fp16.onnx")

    if args.precision == "int8":
        bb_data, dec_data = collect_calibration_data(
            args.data_root, args.calib_samples, args.onnx_dir
        )
        if TRT_HAS_CALIBRATOR:
            # TensorRT 8 vazhi - TRT-e calibration pannum
            calibs["camera_backbone"] = EntropyCalibrator(
                bb_data, f"{args.engine_dir}/backbone_calib.cache")
            calibs["bev_decoder"] = EntropyCalibrator(
                dec_data, f"{args.engine_dir}/decoder_calib.cache")
        else:
            # TensorRT 10 vazhi - namma ONNX-la QDQ nodes potrom
            for name, data in [("camera_backbone", bb_data),
                               ("bev_decoder", dec_data)]:
                sources[name] = quantize_onnx_qdq(
                    sources[name],
                    f"{args.onnx_dir}/{name}_int8.onnx",
                    data,
                )

    for name in ["camera_backbone", "bev_decoder"]:
        try:
            build_engine(
                sources[name],
                f"{args.engine_dir}/{name}_{args.precision}.plan",
                args.precision, calibs[name],
            )
        except RuntimeError as e:
            if name != "bev_decoder":
                raise
            # bev_decoder-la scatter_add iruku (BEV pooling). TensorRT
            # antha op-ai support pannala. So SPLIT vazhi:
            # scatter-ai PyTorch-la vachikitu, meedhi (encoder + head)
            # mattum TRT-la potrom. Heavy compute anga thaan iruku.
            print(f"\n  bev_decoder TRT-la aagala: {str(e)[:60]}")
            print("  -> bev_head (encoder+head mattum) try pannurom")
            head_onnx = f"{args.onnx_dir}/bev_head.onnx"
            if not os.path.exists(head_onnx):
                print(f"  {head_onnx} illa - export_onnx.py thirumba odu")
                continue
            if args.precision == "int8":
                bev_data = collect_bev_calibration(
                    args.data_root, 8, args.onnx_dir)
                head_onnx = quantize_onnx_qdq(
                    head_onnx, f"{args.onnx_dir}/bev_head_int8.onnx", bev_data)
            build_engine(head_onnx,
                         f"{args.engine_dir}/bev_head_{args.precision}.plan",
                         args.precision, None)
    print("ENGINES READY ->", args.engine_dir)


if __name__ == "__main__":
    main()

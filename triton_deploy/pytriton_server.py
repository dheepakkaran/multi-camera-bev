# ============================================================
# pytriton_server.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Model-ai oru SERVICE-a maathurathu. Ippo varaikum model namma
# script ulla thaan odunuchu. Production-la appdi illa - model oru
# server-la irukkum, vera applications HTTP/gRPC-la request anuppum.
#
# NVIDIA Triton (ippo "Dynamo-Triton") thaan antha server.
#
# Yaen PyTriton, saadha Triton illa?
# Saadha Triton Docker container-la varum (~15 GB image). Kaggle/Colab-la
# Docker illa. PyTriton = ATHE Triton server, aana Python process
# ulla-ye odum. pip install pothum. Server nijam, endpoint nijam,
# batching nijam - Docker mattum thaan illa.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# export/backends.py -> TensorRT engines-ai load pannurathu
# triton_deploy/client/pytriton_client.py -> ithukku request anuppum
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# 1. TensorRT backend load (oru instance-ku onnu)
# 2. Antha backend-ai oru python function-a wrap pannurathu
# 3. Triton-la "bev_ensemble" nu peru vachi bind pannurathu
# 4. Server start -> HTTP :8000, gRPC :8001, metrics :8002
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Endpoint input : images [6, 3, 224, 400]
# Endpoint output: heatmap, offset, height, size, rot, vel
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# python -m triton_deploy.pytriton_server --precision int8 --instances 2
#
# ============================================================

import argparse
import time

import numpy as np

HEAD_NAMES = ["heatmap", "offset", "height", "size", "rot", "vel"]

# Ovvoru head-um enna shape (batch dim illaama - Triton adhai thaana serkkum)
HEAD_SHAPES = {
    "heatmap": (10, 200, 200),
    "offset":  (2, 200, 200),
    "height":  (1, 200, 200),
    "size":    (3, 200, 200),
    "rot":     (2, 200, 200),
    "vel":     (2, 200, 200),
}


def build_infer_functions(precision: str, instances: int, ckpt: str,
                          onnx_dir: str, engine_dir: str, K, E) -> list:
    """
    Triton-ku kudukka vendiya inference function-gal create pannurathu.

    Ovvoru instance-kum THANI TensorRT backend. Yaen thani?
    TensorRT execution context thread-safe illa - rendu request ore
    context-ai ore neram use panna crash aagum. Thani context-na
    Triton antha requests-ai NIJAMAAVE parallel-a odalaam.

    Ithu thaan real Triton-oda `instance_group { count: N }` settings-ku
    samam. NVIDIA interview-la ketkura vishayam.

    Args:
        instances: evlo parallel copy venum
        K, E: camera calibration (split mode-ku thevai)

    Returns:
        list of callables - Triton ithai list-a-ve accept pannum
    """
    from pytriton.decorators import batch
    from export.backends import get_backend

    funcs = []
    for i in range(instances):
        backend = get_backend(f"trt_{precision}", ckpt=ckpt, onnx_dir=onnx_dir,
                              engine_dir=engine_dir, K=K, E=E)
        print(f"  instance {i+1}/{instances} ready: {backend.name}")

        # @batch decorator: Triton pala request-ai sethu oru batch-a
        # kudukkum. Namma engine batch=1 fix, so loop podurom.
        # (Engine-ai batch>1-ku build pannalaam, aana appo memory
        #  athigam - portfolio project-ku ithu pothum.)
        def make_fn(be):
            @batch
            def infer(images):
                outs = {n: [] for n in HEAD_NAMES}
                for sample in images:                  # [B,6,3,224,400]
                    preds = be.run(sample)
                    for n in HEAD_NAMES:
                        outs[n].append(preds[n][0])    # batch dim thooku
                return {n: np.stack(v).astype(np.float32)
                        for n, v in outs.items()}
            return infer

        funcs.append(make_fn(backend))

    return funcs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="int8",
                    choices=["fp32", "fp16", "int8"])
    ap.add_argument("--instances", type=int, default=2,
                    help="parallel model copies (real Triton instance_group)")
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--queue-delay-us", type=int, default=2000,
                    help="dynamic batching-ku evlo neram kaathirukkanum")
    ap.add_argument("--ckpt", default="runs/simplebev/best.pth")
    ap.add_argument("--onnx-dir", default="export/onnx")
    ap.add_argument("--engine-dir", default="export/engines")
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    ap.add_argument("--serve", action="store_true",
                    help="server-ai nirutthaama odavidu (Ctrl+C varaikum)")
    args = ap.parse_args()

    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton
    from export.sample_source import load_samples

    _, K, E = load_samples(data_root=args.data_root)

    print(f"loading {args.instances} TensorRT {args.precision.upper()} instance(s)...")
    infer_funcs = build_infer_functions(
        args.precision, args.instances, args.ckpt,
        args.onnx_dir, args.engine_dir, K, E)

    triton = Triton()
    triton.bind(
        model_name="bev_ensemble",
        # LIST kuduthaa Triton pala instance create pannum.
        # Real Triton-la ithu config.pbtxt-la instance_group { count: N }
        infer_func=infer_funcs,
        inputs=[Tensor(name="images", dtype=np.float32,
                       shape=(6, 3, 224, 400))],
        outputs=[Tensor(name=n, dtype=np.float32, shape=HEAD_SHAPES[n])
                 for n in HEAD_NAMES],
        config=ModelConfig(
            batching=True,
            max_batch_size=args.max_batch_size,
            # Dynamic batching: request vandhavudan odaama, konjam neram
            # (2 ms) kaathirundhu vera request vandha sethu odum.
            # GPU-ku oru periya velai kudukkurathu, chinna velai
            # nirayya kudukkuradha vida efficient.
            batcher=DynamicBatcher(
                max_queue_delay_microseconds=args.queue_delay_us,
                preferred_batch_size=[1, 2, args.max_batch_size],
            ),
        ),
    )

    triton.run()          # non-blocking - background-la server start aagum
    print("\n" + "=" * 58)
    print("  TRITON SERVER READY")
    print(f"  model      : bev_ensemble")
    print(f"  precision  : TensorRT {args.precision.upper()}")
    print(f"  instances  : {args.instances}  (parallel model copies)")
    print(f"  batching   : dynamic, max {args.max_batch_size}, "
          f"{args.queue_delay_us} us queue delay")
    print(f"  HTTP :8000  |  gRPC :8001  |  metrics :8002")
    print("=" * 58 + "\n")

    if args.serve:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("stopping server...")
    return triton


if __name__ == "__main__":
    main()

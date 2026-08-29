# ============================================================
# export_onnx.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# PyTorch model .pth file python-ku mattum thaan puriyum. Production-la
# TensorRT / Triton use pannanum-na ONNX nu oru "common language"-ku
# maathanum. ONNX = model-oda graph-ai standard format-la ezhuthurathu.
#
# Model-ai RENDU thani graph-a export pannurom:
#   1. camera_backbone -> 6 camera-kum PARALLEL-a odalaam (Triton la
#      instance_group count=6). Ithu thaan NVIDIA interview talking point.
#   2. bev_decoder     -> LSS + encoder + head. Oru thadava mattum.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : runs/simplebev/best.pth (training/train.py kudukkurathu)
# Output: export/onnx/*.onnx -> calibrate_int8.py, triton/ ku pogum
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# Geometry (frustum -> ego points) model-uku VELIYA compute pannurom.
# Yaen? 2 karanam:
#   (a) torch.inverse ONNX/TensorRT la support kammi
#   (b) Camera calibration car-la FIX. So geometry ORE thadava compute
#       panni cache pannalaam - inference-la thirumba thirumba vendaam.
#       Ithu oru real optimization.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# camera_backbone.onnx : images [6,3,224,400] -> feats [6,64,14,25]
# bev_decoder.onnx     : feats [6,64,14,25] + geom [6,64,14,25,3]
#                        -> heatmap/offset/height/size/rot/vel
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# ONNX -> TensorRT engine (.plan) -> Triton serve.
#
# ============================================================

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from data.scripts.constants import (
    N_CAMERAS, TARGET_H, TARGET_W, N_DEPTHS,
    BACKBONE_OUT_CHANNELS, BEV_OUT_CHANNELS,
)
from models.simplebev import SimpleBEV

HEAD_NAMES = ["heatmap", "offset", "height", "size", "rot", "vel"]


class BackboneONNX(nn.Module):
    """Graph 1: 6 camera images -> feature maps. Sutha CNN, export easy."""

    def __init__(self, model: SimpleBEV):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [6, 3, 224, 400] -> [6, 64, 14, 25]
        return self.backbone(images)


class DecoderONNX(nn.Module):
    """
    Graph 2: features + precomputed geometry -> detection heads.

    Geometry-ai input-a vaangurom (model ulla compute pannala) - mela
    sonna maadhiri torch.inverse-ai graph-la irunthu thavirkka.
    """

    def __init__(self, model: SimpleBEV):
        super().__init__()
        self.lss = model.view_transformer
        self.bev_encoder = model.bev_encoder
        self.head = model.head

    def forward(self, feats: torch.Tensor, geom: torch.Tensor) -> tuple:
        """
        Args:
            feats: [6, 64, 14, 25]      <- camera_backbone output NERADIYA
            geom : [6, D, 14, 25, 3]    ego-frame points (metres)

        Returns:
            tuple of 6 tensors, HEAD_NAMES order-la

        Yaen batch dim illa? Triton ensemble-la backbone output-ai
        neradiya inga connect pannanum. Naduvula reshape panna Triton-la
        vazhi illa, so shape exactly match aaganum.
        """
        # batch dim-ai ULLA add pannikirom (B=1, oru car oru neram)
        volume = self.lss.lift(feats)                     # [6, C, D, H, W]
        volume = volume.unsqueeze(0)                      # [1, 6, C, D, H, W]
        geom = geom.unsqueeze(0)                          # [1, 6, D, H, W, 3]
        bev = self.lss.splat(volume, geom)                # [1, 64, 200, 200]
        bev = self.bev_encoder(bev)                       # [1, 128, 200, 200]
        preds = self.head(bev)
        return tuple(preds[k] for k in HEAD_NAMES)


class BevHeadONNX(nn.Module):
    """
    Graph 3 (fallback): BEV map -> detection heads. LSS ILLAAMA.

    Yaen ithu thevai? TensorRT-ku `scatter_add` (ONNX ScatterElements
    with reduction) support illa - "ScatterReduction plugin not found"
    nu parse fail aagum. Aana scatter thaan BEV pooling.

    Solution: pipeline-ai pirikirom -
        camera_backbone  -> TensorRT
        lift + splat     -> PyTorch (scatter inga, GPU-la fast)
        bev_head         -> TensorRT   <- ithu
    Heavy compute (encoder + head, 200x200 la) TRT-la-ye irukku,
    so speedup meedhi kedaikkum.
    """

    def __init__(self, model: SimpleBEV):
        super().__init__()
        self.bev_encoder = model.bev_encoder
        self.head = model.head

    def forward(self, bev: torch.Tensor) -> tuple:
        """
        Args:
            bev: [1, 64, 200, 200]  LSS output

        Returns:
            tuple of 6 tensors (HEAD_NAMES order)
        """
        preds = self.head(self.bev_encoder(bev))
        return tuple(preds[k] for k in HEAD_NAMES)


def compute_geometry(model: SimpleBEV, K: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """
    Frustum points-ai ego coordinates-a maathi thara (host-la, ORE thadava).

    Args:
        K: [1, 6, 3, 3]
        E: [1, 6, 4, 4]

    Returns:
        [1, 6, D, 14, 25, 3] float32
    """
    with torch.no_grad():
        return model.view_transformer.get_geometry(K, E)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/simplebev/best.pth")
    ap.add_argument("--out-dir", default="export/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--data-root", default="data/nuscenes-mini")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")     # export eppovum CPU-la - deterministic

    # --- Model load ---
    model = SimpleBEV(pretrained=False).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch','?')})")

    # --- Real sample edukirom (dummy illa) ---
    # Yaen real? Geometry real calibration-la irunthu varanum, illaina
    # export aana model thappana points-ai use pannum.
    from export.sample_source import load_samples
    imgs, K6, E6 = load_samples(data_root=args.data_root)
    images = imgs[0].to(device)                             # [6,3,224,400]
    K = K6.unsqueeze(0).to(device)                          # [1,6,3,3]
    E = E6.unsqueeze(0).to(device)                          # [1,6,4,4]

    geom = compute_geometry(model, K, E)[0]                 # [6,D,14,25,3]
    np.save(os.path.join(args.out_dir, "geometry.npy"), geom.numpy())
    print(f"geometry cached {tuple(geom.shape)} -> {args.out_dir}/geometry.npy")

    # ================= Graph 1: camera backbone =================
    bb = BackboneONNX(model).eval()
    bb_path = os.path.join(args.out_dir, "camera_backbone.onnx")
    torch.onnx.export(
        bb, (images,), bb_path,
        input_names=["images"], output_names=["features"],
        opset_version=args.opset, do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {bb_path}")

    # ================= Graph 2: bev decoder =================
    with torch.no_grad():
        feats = model.backbone(images)                     # [6,64,14,25]

    dec = DecoderONNX(model).eval()
    dec_path = os.path.join(args.out_dir, "bev_decoder.onnx")
    torch.onnx.export(
        dec, (feats, geom), dec_path,
        input_names=["features", "geometry"], output_names=HEAD_NAMES,
        opset_version=args.opset, do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {dec_path}")

    # ============ Graph 3: bev_head (TRT fallback) ============
    # bev_decoder-la scatter iruku, TensorRT adhai support pannala.
    # So scatter illaadha version-um export pannurom.
    with torch.no_grad():
        bev = model.view_transformer(feats.unsqueeze(0), K, E)  # [1,64,200,200]

    head = BevHeadONNX(model).eval()
    head_path = os.path.join(args.out_dir, "bev_head.onnx")
    torch.onnx.export(
        head, (bev,), head_path,
        input_names=["bev"], output_names=HEAD_NAMES,
        opset_version=args.opset, do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {head_path}")

    # ================= Verify: PyTorch vs ONNX =================
    # Export aanathu SARIYA velai seiyudha nu check pannurom.
    # Numbers konjam maaralaam (float rounding) aana 1e-3 ku ulla irukkanum.
    #
    # onnxruntime illaina verification skip pannuvom - export aana
    # .onnx file-gal sari-a thaan iruku, adhai TensorRT use pannalaam.
    # (Kaggle-la ORT install fail aana podhu ithu illaama crash aachu.)
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime illa - verification skip "
              "(.onnx files export aagiduchu, TensorRT use pannalaam)")
        return

    with torch.no_grad():
        torch_feats = model.backbone(images)
        torch_out = dec(feats, geom)

    sess_bb = ort.InferenceSession(bb_path, providers=["CPUExecutionProvider"])
    onnx_feats = sess_bb.run(None, {"images": images.numpy()})[0]
    d = np.abs(onnx_feats - torch_feats.numpy()).max()
    print(f"  backbone  max diff {d:.2e}  {'OK' if d < 1e-3 else 'FAIL'}")

    sess_dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    onnx_out = sess_dec.run(None, {"features": feats.numpy(), "geometry": geom.numpy()})
    ok = True
    for name, o, t in zip(HEAD_NAMES, onnx_out, torch_out):
        d = np.abs(o - t.numpy()).max()
        ok &= d < 1e-3
        print(f"  {name:8s}  max diff {d:.2e}  {'OK' if d < 1e-3 else 'FAIL'}")
    print("ONNX EXPORT VERIFIED" if ok else "ONNX MISMATCH - paaru!")


if __name__ == "__main__":
    main()

# ============================================================
# train.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Model-ai kathukka vaikkurathu. Loop simple:
#   data edu -> model predict -> loss kanakku -> gradient -> weights update
# Itha 50 epoch x 71 batch thadava seiyum.
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# dataset.py (data) + simplebev.py (model) + losses.py (loss)
# Output: runs/simplebev/best.pth checkpoint
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# AdamW optimizer + cosine LR schedule + AMP (GPU-la mattum)
# + val loss paathu best checkpoint save.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : configs/simplebev.yaml
# Output: checkpoints + console log (+ MLflow install pannirundha)
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
#   python -m training.train --config configs/simplebev.yaml
#   python -m training.train --smoke        # dataset illama 2 step test
#
# ============================================================

import argparse
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from models.simplebev import SimpleBEV
from training.losses import CenterPointLoss


def get_device() -> torch.device:
    """CUDA > MPS (Mac GPU) > CPU nu order-la pick pannum."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to(obj, device):
    """dict-ku ulla irukkura ellaa tensor-aiyum device-ku anuppurathu."""
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    return obj


def run_epoch(model, loader, criterion, optimizer, device, scaler,
              cfg: dict, train: bool = True) -> dict:
    """
    ORE epoch (dataset muzhuthum oru thadava) run pannurathu.

    Args:
        train: True -> weights update aagum. False -> validation mattum.

    Returns:
        dict of average losses
    """
    model.train() if train else model.eval()

    totals, n_batches = {}, 0
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"

    for step, batch in enumerate(loader):
        images = batch["images"].to(device)
        K = batch["intrinsics"].to(device)
        E = batch["extrinsics"].to(device)
        targets = move_to(batch["targets"], device)

        # torch.set_grad_enabled: val time-la gradient store pannaama
        # irundha memory kammi + fast
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(images, K, E)
                loss, parts = criterion(preds, targets)

        if train:
            optimizer.zero_grad(set_to_none=True)     # pazhaya gradient clear
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)            # clip pannurathukku munnadi
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient romba periyathaa poga koodathu (NaN varum)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step()

        parts["total"] = float(loss.detach())
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        if train and step % cfg["train"]["log_every"] == 0:
            print(f"    step {step:4d} | loss {parts['total']:.4f} "
                  f"| hm {parts['heatmap']:.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def smoke_test(device: torch.device) -> None:
    """
    Dataset illama, dummy random data vachi 2 step train panni paakkurathu.

    Yaen ithu venum? nuScenes download panna munnadiye "shapes correct-a?
    loss NaN varutha? backward velai seiyutha?" nu therinjikkalaam.
    """
    from data.scripts.constants import N_CLASSES, BEV_H, BEV_W

    print(f"SMOKE TEST on {device} (dummy data, no dataset needed)")
    model = SimpleBEV(pretrained=False).to(device)
    criterion = CenterPointLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 1
    images = torch.randn(B, 6, 3, 224, 400, device=device)
    K = torch.eye(3, device=device).repeat(B, 6, 1, 1)
    K[:, :, 0, 0] = 100.0; K[:, :, 1, 1] = 100.0    # fx, fy
    K[:, :, 0, 2] = 200.0; K[:, :, 1, 2] = 112.0    # cx, cy (image center)
    E = torch.eye(4, device=device).repeat(B, 6, 1, 1)

    targets = {
        "heatmap": torch.zeros(B, N_CLASSES, BEV_H, BEV_W, device=device),
        "mask": torch.zeros(B, 1, BEV_H, BEV_W, device=device),
        "offset": torch.zeros(B, 2, BEV_H, BEV_W, device=device),
        "height": torch.zeros(B, 1, BEV_H, BEV_W, device=device),
        "size": torch.zeros(B, 3, BEV_H, BEV_W, device=device),
        "rot": torch.zeros(B, 2, BEV_H, BEV_W, device=device),
        "vel": torch.zeros(B, 2, BEV_H, BEV_W, device=device),
    }
    # Oru fake car: grid center (100,100), class 0
    targets["heatmap"][0, 0, 100, 100] = 1.0
    targets["mask"][0, 0, 100, 100] = 1.0
    targets["size"][0, :, 100, 100] = torch.tensor([0.5, 1.5, 0.5], device=device)

    for step in range(2):
        t0 = time.time()
        preds = model(images, K, E)
        loss, parts = criterion(preds, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"  step {step}: loss={float(loss.detach()):.4f}  "
              f"hm={parts['heatmap']:.4f} size={parts['size']:.4f}  "
              f"({time.time() - t0:.1f}s)")
        assert torch.isfinite(loss), "Loss NaN/Inf aayiduchu!"

    print("SMOKE TEST PASSED - forward + loss + backward ellam OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simplebev.yaml")
    parser.add_argument("--smoke", action="store_true",
                        help="dataset illama dummy data vachi 2 step test")
    parser.add_argument("--epochs", type=int, default=None, help="config-ai override")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    device = get_device()

    if args.smoke:
        smoke_test(device)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    # --- Data ---
    from data.scripts.dataset import NuScenesBEVDataset

    train_ds = NuScenesBEVDataset(cfg["data_root"], split="train")
    # nusc object-ai reuse pannurom - rendaavathu thadava load panna 10s waste
    val_ds = NuScenesBEVDataset(cfg["data_root"], split="val", nusc=train_ds.nusc)
    print(f"train {len(train_ds)} samples | val {len(val_ds)} samples")

    common = dict(batch_size=cfg["train"]["batch_size"],
                  num_workers=cfg["train"]["num_workers"],
                  pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    # --- Model / loss / optimizer ---
    model = SimpleBEV(pretrained=True).to(device)
    criterion = CenterPointLoss(**{f"w_{k}": v for k, v in cfg["loss_weights"].items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    # Cosine: LR mella mella 0 varai kurayum. Mudivula chinna steps ->
    # model settle aagum (romba periya step-la minimum-ai thaandi poidum).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.amp.GradScaler(enabled=(cfg["train"]["amp"] and device.type == "cuda"))

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # MLflow irundha use pannuvom, illaina normal-a odum (optional dependency)
    try:
        import mlflow
        mlflow.set_experiment("simplebev")
        mlflow.start_run()
        mlflow.log_params({**cfg["train"], **cfg["loss_weights"]})
    except Exception:
        mlflow = None
        print("mlflow illa - console log mattum")

    best_val = float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['train']['epochs']}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        tr = run_epoch(model, train_loader, criterion, optimizer, device,
                       scaler, cfg, train=True)
        va = run_epoch(model, val_loader, criterion, optimizer, device,
                       scaler, cfg, train=False)
        scheduler.step()

        print(f"  train {tr['total']:.4f} | val {va['total']:.4f}")

        if mlflow:
            mlflow.log_metrics({f"train_{k}": v for k, v in tr.items()}, step=epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in va.items()}, step=epoch)

        # Best model mattum save - last epoch always best-a irukkathu
        if va["total"] < best_val:
            best_val = va["total"]
            path = os.path.join(cfg["output_dir"], "best.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": best_val}, path)
            print(f"  saved {path} (best val {best_val:.4f})")

    if mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()

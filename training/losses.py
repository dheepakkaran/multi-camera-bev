# ============================================================
# losses.py
# ============================================================
#
# ETHUKU CREATE PANNINOM?
# ━━━━━━━━━━━━━━━━━━━━━━
# Model-oda prediction-um, namma target-um evlo different nu oru
# NUMBER-a solla. Antha number thaan loss. Loss kammi = model nalla.
#
# Rendu vidhamana loss thevai:
#   1. heatmap ("inga car iruka?") -> classification -> FOCAL loss
#   2. size/offset/rot ("evlo periyathu?") -> regression -> L1 loss
#
# MUNADI FILE ODA CONNECTION:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# Input : center_head.py predictions + dataset.py targets
# Output: train.py backward() pannum
#
# INNER OPERATIONS:
# ━━━━━━━━━━━━━━━━
# CenterNet-style gaussian focal loss + masked L1.
#
# INPUT / OUTPUT:
# ━━━━━━━━━━━━━━
# Input : preds dict, targets dict
# Output: (total_loss tensor, dict of individual losses for logging)
#
# EPADI USE AAGUM:
# ━━━━━━━━━━━━━━━
# loss, parts = criterion(preds, targets); loss.backward()
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """
    CenterNet gaussian focal loss.

    PROBLEM ithu solve pannuthu:
    200x200x10 = 400,000 cell. Athula ~20 cell-la mattum object.
    Normal cross-entropy use panna, model "ellame background" nu
    sonnaale 99.99% correct! So kathukka maatan.

    SOLUTION - focal loss:
      - Easy example (already correct) -> loss-ai romba kammi pannu
      - Hard example (thappu) -> loss-ai perusa vachiru
      - Center pass-a irukkura cell (gaussian 0.9) -> konjam mattum thandi

    Example numbers:
      target=1 (car center), pred=0.9 (nalla) -> (1-0.9)^2 = 0.01 -> small
      target=1,               pred=0.1 (mosam) -> (1-0.1)^2 = 0.81 -> BIG
      target=0.9 (center pakkathula), pred=0.8 -> (1-0.9)^4 = 0.0001 weight
        -> "ithu kittathatta correct, vidu"

    Args:
        pred_logits: [B, 10, 200, 200] RAW logits (sigmoid apply pannala)
        target     : [B, 10, 200, 200] 0..1 gaussian heatmap

    Returns:
        scalar loss
    """
    # clamp: log(0) = -infinity varathu-nu thadukka
    pred = torch.sigmoid(pred_logits).clamp(min=1e-4, max=1 - 1e-4)

    pos_mask = target.eq(1.0).float()     # exact center cells mattum
    neg_mask = 1.0 - pos_mask

    # Positive: model 1.0 sollanum. Sollala-na loss perusu.
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask

    # Negative: model 0 sollanum. Aana center pakkathula (target=0.9)
    # irundha (1-target)^beta chinnathaagi, thandanai kammi aagum.
    neg_loss = (
        -torch.log(1 - pred)
        * torch.pow(pred, alpha)
        * torch.pow(1 - target, beta)
        * neg_mask
    )

    n_pos = pos_mask.sum()
    if n_pos == 0:
        # Intha frame-la object-e illa -> negative loss mattum
        return neg_loss.sum()
    # Object count-la divide: 3 car frame-um 30 car frame-um samam weight
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Object center cell-la MATTUM L1 loss.

    Yaen mask? Background cell-la "car size enna" nu kekkurathu arthamillai.
    Anga target 0, aana antha 0-ai model kathukka thevai illa.

    Args:
        pred  : [B, C, 200, 200]
        target: [B, C, 200, 200]
        mask  : [B, 1, 200, 200]  1 = object center

    Returns:
        scalar loss
    """
    # L1 = |pred - target|. Yaen L2 illa? Outlier (romba periya lorry)
    # irundha L2 loss vedithu poidum. L1 stable.
    loss = torch.abs(pred - target) * mask     # mask broadcast aagum
    return loss.sum() / (mask.sum() * pred.shape[1] + 1e-4)


class CenterPointLoss(nn.Module):
    """
    Ellaa 6 head-oda loss-aiyum weighted-a koottrathu.

    Weights yaen? Heatmap thaan mukkiyam (object iruka illaya).
    Regression correct-a irunthum heatmap thappu-na box-e varathu.
    """

    def __init__(self, w_heatmap: float = 1.0, w_offset: float = 1.0,
                 w_height: float = 1.0, w_size: float = 1.0,
                 w_rot: float = 1.0, w_vel: float = 0.5):
        super().__init__()
        self.weights = {
            "heatmap": w_heatmap, "offset": w_offset, "height": w_height,
            "size": w_size, "rot": w_rot, "vel": w_vel,
        }

    def forward(self, preds: dict, targets: dict) -> tuple:
        """
        Args:
            preds  : center_head output dict
            targets: dataset targets dict (mask-um ullathu)

        Returns:
            (total_loss, {name: float}) - rendaavathu logging-ku
        """
        mask = targets["mask"]

        parts = {"heatmap": focal_loss(preds["heatmap"], targets["heatmap"])}
        for key in ("offset", "height", "size", "rot", "vel"):
            parts[key] = masked_l1(preds[key], targets[key], mask)

        total = sum(self.weights[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}

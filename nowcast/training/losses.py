"""Segmentation-style losses for multi-hazard, multi-lead occurrence maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nowcast import config


def _stack(value: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Stack a hazard dict into ``(B, N_HAZARDS, ...)``; pass tensors through."""
    if isinstance(value, dict):
        return torch.stack([value[hazard] for hazard in config.HAZARDS], dim=1)
    return value


def masked_bce_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Binary cross-entropy on logits, optionally averaged over a boolean mask."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Soft Dice loss (``1 - Dice`` coefficient) over all elements."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss (Lin et al., 2017) with soft-target support."""
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


class MultiTaskLoss(nn.Module):
    """Weighted BCE + Dice, summed over hazards then averaged.

    Accepts either a head-output dict (hazard -> ``(B, L, H, W)``) or a stacked
    ``(B, N_HAZARDS, N_LEADS, H, W)`` tensor for both predictions and targets.
    """

    def __init__(
        self,
        hazard_weights: dict[str, float] | None = None,
        bce: float = 1.0,
        dice: float = 0.5,
    ) -> None:
        super().__init__()
        self.bce = bce
        self.dice = dice
        self.hazard_weights = hazard_weights or {hazard: 1.0 for hazard in config.HAZARDS}

    def forward(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        targets: dict[str, torch.Tensor] | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the scalar multi-task loss."""
        pred = _stack(outputs)
        target = _stack(targets)
        parts: list[torch.Tensor] = []
        for i, hazard in enumerate(config.HAZARDS):
            weight = float(self.hazard_weights.get(hazard, 1.0))
            logit_i = pred[:, i]
            target_i = target[:, i]
            mask_i = mask[:, i] if mask is not None else None
            term = self.bce * masked_bce_with_logits(logit_i, target_i, mask_i)
            term = term + self.dice * dice_loss(logit_i, target_i)
            parts.append(weight * term)
        return torch.stack(parts).sum() / len(parts)

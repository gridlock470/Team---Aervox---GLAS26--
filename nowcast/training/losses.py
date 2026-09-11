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
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy on logits, optionally averaged over a boolean mask.

    ``pos_weight`` multiplies the positive term of the BCE (see
    :func:`torch.nn.functional.binary_cross_entropy_with_logits`); values above
    1.0 make rare positive cells cost more than the abundant negatives. It
    composes with ``mask``; ``pos_weight=None`` is the unweighted BCE.
    """
    weight: torch.Tensor | None = None
    if pos_weight is not None:
        weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=weight
    )
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


def hazard_weights_from_base_rates(
    targets: dict[str, torch.Tensor] | torch.Tensor,
    threshold: float = config.LABEL_OCCURRENCE_THRESHOLD,
) -> dict[str, float]:
    """Inverse-frequency hazard weights from the positive base rate of ``targets``.

    ``targets`` is a hazard dict or a ``(B, N_HAZARDS, ...)`` tensor of soft
    labels; a cell counts as positive at ``>= threshold``. The returned weights
    are normalised to mean 1.0, so swapping them into :class:`MultiTaskLoss`
    re-balances the hazards without changing the overall loss scale. A hazard
    with no positives is floored at one positive cell, so its weight is large
    but finite rather than ``inf``.
    """
    target = _stack(targets)
    n_cells = float(target[:, 0].numel())
    floor = 1.0 / max(n_cells, 1.0)
    rates: list[float] = []
    for i in range(len(config.HAZARDS)):
        positives = float((target[:, i] >= threshold).sum().item())
        rates.append(max(positives / max(n_cells, 1.0), floor))
    raw = [1.0 / rate for rate in rates]
    mean = sum(raw) / len(raw)
    return {
        hazard: raw[i] / mean for i, hazard in enumerate(config.HAZARDS)
    }


class MultiTaskLoss(nn.Module):
    """Weighted BCE + Dice + focal, summed over hazards then averaged.

    BCE and Dice train against the *soft* probabilistic target (per the label
    contract). Accepts either a head-output dict (hazard -> ``(B, L, H, W)``) or
    a stacked ``(B, N_HAZARDS, N_LEADS, H, W)`` tensor for both predictions and
    targets.

    The focal term is ON by default at ``config.FOCAL_WEIGHT`` (1.0). Positive
    cells are ~1e-4 to 1e-5 of all cell-hours, so BCE + Dice alone are nearly
    minimised by an all-negative prediction; ``focal_gamma`` (2.0) scales each
    cell's loss by ``(1 - p_t) ** gamma``, which suppresses the flood of easy,
    confidently-correct negatives by ~100x and leaves the gradient dominated by
    the hard cells near the storm. ``focal_alpha`` (0.25) is the static share of
    the loss given to positive cells; it stays small because gamma already
    supplies the rare-class emphasis. Weight 1.0 is chosen because the
    alpha/gamma factors already put the focal term about an order of magnitude
    below plain BCE, so it contributes without dominating. Pass ``focal=0.0``
    for the previous BCE + Dice behaviour. ``bce`` and ``dice`` are unchanged.

    ``pos_weight`` (default ``None``) is forwarded to
    :func:`masked_bce_with_logits`; see ``config.BCE_POS_WEIGHT`` for the
    recommended non-default value.
    """

    def __init__(
        self,
        hazard_weights: dict[str, float] | None = None,
        bce: float = 1.0,
        dice: float = 0.5,
        focal: float = config.FOCAL_WEIGHT,
        focal_alpha: float = config.FOCAL_ALPHA,
        focal_gamma: float = config.FOCAL_GAMMA,
        pos_weight: float | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce = bce
        self.dice = dice
        self.focal = focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
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
            term = self.bce * masked_bce_with_logits(
                logit_i, target_i, mask_i, self.pos_weight
            )
            term = term + self.dice * dice_loss(logit_i, target_i)
            if self.focal > 0.0:
                term = term + self.focal * focal_loss(
                    logit_i, target_i, self.focal_alpha, self.focal_gamma
                )
            parts.append(weight * term)
        return torch.stack(parts).sum() / len(parts)

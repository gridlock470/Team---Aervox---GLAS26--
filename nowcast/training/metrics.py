"""Verification metrics: CSI/POD/FAR plus PR-AUC and calibration error."""

from __future__ import annotations

import math

import torch
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification import binary_average_precision

from nowcast import config


def _tensor(value: object) -> torch.Tensor:
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def _finite(value: object, default: float = 0.0) -> float:
    number = float(value)
    return number if math.isfinite(number) else default


def _contingency(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_pos = preds >= threshold
    true_pos = targets >= threshold
    hits = (pred_pos & true_pos).sum().float()
    false_alarms = (pred_pos & ~true_pos).sum().float()
    misses = (~pred_pos & true_pos).sum().float()
    return hits, false_alarms, misses


def csi(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Critical success index (threat score) in ``[0, 1]``."""
    hits, false_alarms, misses = _contingency(_tensor(preds), _tensor(targets), threshold)
    denom = hits + false_alarms + misses
    return _finite(hits / denom) if denom > 0 else 0.0


def pod(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Probability of detection (hit rate) in ``[0, 1]``."""
    hits, _, misses = _contingency(_tensor(preds), _tensor(targets), threshold)
    denom = hits + misses
    return _finite(hits / denom) if denom > 0 else 0.0


def far(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """False-alarm ratio in ``[0, 1]``."""
    hits, false_alarms, _ = _contingency(_tensor(preds), _tensor(targets), threshold)
    denom = hits + false_alarms
    return _finite(false_alarms / denom) if denom > 0 else 0.0


def pr_auc(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Average precision (area under the precision-recall curve)."""
    target_bin = (_tensor(targets) >= threshold).long().flatten()
    if target_bin.unique().numel() < 2:
        return 0.0
    prob = _tensor(preds).flatten().clamp(0.0, 1.0)
    return _finite(binary_average_precision(prob, target_bin))


def calibration_error(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, n_bins: int = 10
) -> float:
    """Expected calibration error via :class:`BinaryCalibrationError`."""
    target_bin = (_tensor(targets) >= threshold).long().flatten()
    prob = _tensor(preds).flatten().clamp(0.0, 1.0)
    metric = BinaryCalibrationError(n_bins=n_bins)
    return _finite(metric(prob, target_bin))


class NowcastMetrics:
    """Bundle computing overall and per-hazard / per-lead means on stacked tensors.

    Inputs are ``(B, N_HAZARDS, N_LEADS, H, W)`` probability and target tensors.
    Every value returned by :meth:`compute` is finite.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def compute(self, probs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Return a flat dict of finite metric values."""
        probs = _tensor(probs)
        targets = _tensor(targets)
        result: dict[str, float] = {
            "csi": csi(probs, targets, self.threshold),
            "pod": pod(probs, targets, self.threshold),
            "far": far(probs, targets, self.threshold),
            "pr_auc": pr_auc(probs, targets, self.threshold),
            "ece": calibration_error(probs, targets, threshold=self.threshold),
        }
        for i, hazard in enumerate(config.HAZARDS):
            result[f"csi/{hazard}"] = csi(probs[:, i], targets[:, i], self.threshold)
        for j, lead in enumerate(config.LEAD_TIMES_H):
            result[f"csi/lead_{lead}h"] = csi(
                probs[:, :, j], targets[:, :, j], self.threshold
            )
        return result

    __call__ = compute

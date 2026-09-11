"""Verification metrics: CSI/POD/FAR plus PR-AUC and calibration error.

Targets are probabilistic; every consumer here binarises the target at
:data:`nowcast.config.LABEL_OCCURRENCE_THRESHOLD` (overridable per call). CSI /
POD / FAR additionally threshold the *predictions* at ``threshold``. Degenerate
cases (no observed positives, no predicted positives, single-class target)
return ``nan`` -- never ``0.0`` -- so a silent no-op is distinguishable from a
genuinely bad score; :class:`NowcastMetrics` aggregates nan-aware.
"""

from __future__ import annotations

import math

import torch
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification import binary_average_precision

from nowcast import config


def _tensor(value: object) -> torch.Tensor:
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def _target_threshold(value: float | None) -> float:
    return config.LABEL_OCCURRENCE_THRESHOLD if value is None else value


def _nanmean(values: list[float]) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    finite = tensor[~torch.isnan(tensor)]
    return float(finite.mean()) if finite.numel() else math.nan


def _contingency(
    preds: torch.Tensor,
    targets: torch.Tensor,
    pred_threshold: float,
    target_threshold: float,
) -> tuple[float, float, float]:
    pred_pos = preds >= pred_threshold
    true_pos = targets >= target_threshold
    hits = float((pred_pos & true_pos).sum())
    false_alarms = float((pred_pos & ~true_pos).sum())
    misses = float((~pred_pos & true_pos).sum())
    return hits, false_alarms, misses


def csi(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    *,
    target_threshold: float | None = None,
) -> float:
    """Critical success index (threat score); ``nan`` when no cell is hit/missed/alarmed."""
    hits, false_alarms, misses = _contingency(
        _tensor(preds), _tensor(targets), threshold, _target_threshold(target_threshold)
    )
    denom = hits + false_alarms + misses
    return hits / denom if denom > 0 else math.nan


def pod(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    *,
    target_threshold: float | None = None,
) -> float:
    """Probability of detection (hit rate); ``nan`` when there are no observed positives."""
    hits, _, misses = _contingency(
        _tensor(preds), _tensor(targets), threshold, _target_threshold(target_threshold)
    )
    denom = hits + misses
    return hits / denom if denom > 0 else math.nan


def far(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    *,
    target_threshold: float | None = None,
) -> float:
    """False-alarm ratio; ``nan`` when nothing is predicted positive."""
    hits, false_alarms, _ = _contingency(
        _tensor(preds), _tensor(targets), threshold, _target_threshold(target_threshold)
    )
    denom = hits + false_alarms
    return false_alarms / denom if denom > 0 else math.nan


def pr_auc(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_threshold: float | None = None,
) -> float:
    """Average precision; ``nan`` when the binarised target is single-class."""
    target_bin = (_tensor(targets) >= _target_threshold(target_threshold)).long().flatten()
    if target_bin.unique().numel() < 2:
        return math.nan
    prob = _tensor(preds).flatten().clamp(0.0, 1.0)
    return float(binary_average_precision(prob, target_bin))


def calibration_error(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_threshold: float | None = None,
    n_bins: int = 10,
) -> float:
    """Expected calibration error against the hard-binarised target."""
    target_bin = (_tensor(targets) >= _target_threshold(target_threshold)).long().flatten()
    if target_bin.numel() == 0:
        return math.nan
    prob = _tensor(preds).flatten().clamp(0.0, 1.0)
    return float(BinaryCalibrationError(n_bins=n_bins)(prob, target_bin))


class NowcastMetrics:
    """Compute per-(hazard, lead) scores and their nan-aware means.

    Inputs are ``(B, N_HAZARDS, N_LEADS, H, W)`` probability and target tensors.
    :meth:`compute` returns headline means (``csi``/``pod``/``far``/``pr_auc``,
    all ``nanmean`` over hazard*lead), per-hazard ``csi/<hazard>``, overall
    ``ece``, and ``n_pos/<hazard>/lead_<k>h`` positive counts so a degenerate
    split is visible rather than silently zero.
    """

    def __init__(
        self, pred_threshold: float = 0.5, target_threshold: float | None = None
    ) -> None:
        self.pred_threshold = pred_threshold
        self.target_threshold = _target_threshold(target_threshold)

    def compute(self, probs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Return a flat dict of metric values (headline values are finite or ``nan``)."""
        probs = _tensor(probs)
        targets = _tensor(targets)
        all_csi: list[float] = []
        all_pod: list[float] = []
        all_far: list[float] = []
        all_ap: list[float] = []
        result: dict[str, float] = {}
        for i, hazard in enumerate(config.HAZARDS):
            hazard_csi: list[float] = []
            for j, lead in enumerate(config.LEAD_TIMES_H):
                pred = probs[:, i, j]
                target = targets[:, i, j]
                value_csi = csi(
                    pred, target, self.pred_threshold,
                    target_threshold=self.target_threshold,
                )
                value_pod = pod(
                    pred, target, self.pred_threshold,
                    target_threshold=self.target_threshold,
                )
                value_far = far(
                    pred, target, self.pred_threshold,
                    target_threshold=self.target_threshold,
                )
                value_ap = pr_auc(
                    pred, target, target_threshold=self.target_threshold
                )
                hazard_csi.append(value_csi)
                all_csi.append(value_csi)
                all_pod.append(value_pod)
                all_far.append(value_far)
                all_ap.append(value_ap)
                n_pos = float((target >= self.target_threshold).sum())
                result[f"n_pos/{hazard}/lead_{lead}h"] = n_pos
            result[f"csi/{hazard}"] = _nanmean(hazard_csi)
        result["csi"] = _nanmean(all_csi)
        result["pod"] = _nanmean(all_pod)
        result["far"] = _nanmean(all_far)
        result["pr_auc"] = _nanmean(all_ap)
        result["ece"] = calibration_error(
            probs, targets, target_threshold=self.target_threshold
        )
        return result

    __call__ = compute

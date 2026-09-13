"""Verification metrics: CSI/POD/FAR/bias plus PR-AUC, Brier and calibration.

Targets are probabilistic; every consumer here binarises the target at
:data:`nowcast.config.LABEL_OCCURRENCE_THRESHOLD` (overridable per call). CSI /
POD / FAR / frequency bias additionally threshold the *predictions* at
``threshold``. Degenerate cases (no observed positives, no predicted positives,
single-class target) return ``nan`` -- never ``0.0`` -- so a silent no-op is
distinguishable from a genuinely bad score; :class:`NowcastMetrics` aggregates
nan-aware.

At convective base rates (order 1e-4 positives) a CSI quoted at a fixed 0.5
threshold badly understates skill, so :func:`best_csi` sweeps a grid and
:class:`NowcastMetrics` reports the winning value *and* the threshold that
achieved it. Accuracy and ROC-AUC are deliberately absent: both look excellent
on rare events and mean nothing here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.functional.classification import binary_average_precision

from nowcast import config

#: Default sweep grid -- coarse 0.05 steps plus finer low values, because the
#: CSI-maximising threshold for a 1-in-10,000 event sits well below 0.05.
DEFAULT_CSI_THRESHOLDS: tuple[float, ...] = (0.01, 0.02, 0.03) + tuple(
    round(0.05 * step, 2) for step in range(1, 20)
)


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


def frequency_bias(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    *,
    target_threshold: float | None = None,
) -> float:
    """Frequency bias ``(hits + false alarms) / (hits + misses)``.

    The fourth canonical nowcasting score: ``> 1`` means the model raises more
    events than were observed (over-forecasting), ``< 1`` fewer. CSI hides this
    because it mixes placement and frequency error. ``nan`` when there are no
    observed positives, matching :func:`pod`.
    """
    hits, false_alarms, misses = _contingency(
        _tensor(preds), _tensor(targets), threshold, _target_threshold(target_threshold)
    )
    denom = hits + misses
    return (hits + false_alarms) / denom if denom > 0 else math.nan


def _csi_curve(
    preds: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Sequence[float],
    target_threshold: float,
) -> list[float]:
    """CSI at each threshold, computed from one pass over the score vectors."""
    true_pos = targets >= target_threshold
    pos_scores = preds[true_pos]
    neg_scores = preds[~true_pos]
    n_pos = float(pos_scores.numel())
    values: list[float] = []
    for threshold in thresholds:
        hits = float((pos_scores >= threshold).sum())
        false_alarms = float((neg_scores >= threshold).sum())
        denom = hits + false_alarms + (n_pos - hits)
        values.append(hits / denom if denom > 0 else math.nan)
    return values


def _best_from_curves(
    curves: list[list[float]], thresholds: Sequence[float]
) -> tuple[float, float]:
    """Pick the threshold maximising the mean CSI over a FIXED set of curves.

    Curves containing any nan are dropped up front rather than per threshold.
    A nan-aware mean taken independently at each threshold averages a different
    population at each point, so a threshold at which most (hazard, lead)
    slices are degenerate can win on the few that survive: a case with 3
    positives in 38,400 cells scored ``csi_best = 1.0`` while ``csi`` and
    ``pod`` were both 0.0. Thresholds are only comparable over the same slices.
    """
    usable = [c for c in curves if not any(math.isnan(v) for v in c)]
    if not usable:
        return math.nan, math.nan

    best_value, best_threshold = math.nan, math.nan
    for index, threshold in enumerate(thresholds):
        value = _nanmean([curve[index] for curve in usable])
        if math.isnan(value):
            continue
        if math.isnan(best_value) or value > best_value:
            best_value, best_threshold = value, threshold
    return best_value, best_threshold


def best_csi(
    preds: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Sequence[float] | None = None,
    *,
    target_threshold: float | None = None,
) -> tuple[float, float]:
    """Return ``(best_csi, threshold)`` over a probability grid.

    Defaults to :data:`DEFAULT_CSI_THRESHOLDS`. A CSI is meaningless without the
    threshold it was achieved at, so both are returned. ``(nan, nan)`` when
    every threshold is degenerate.
    """
    grid = tuple(DEFAULT_CSI_THRESHOLDS if thresholds is None else thresholds)
    curve = _csi_curve(
        _tensor(preds), _tensor(targets), grid, _target_threshold(target_threshold)
    )
    return _best_from_curves([curve], grid)


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


def brier_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_threshold: float | None = None,
) -> float:
    """Mean squared error of the probability against the hard-binarised target.

    ``0.0`` for perfect confident forecasts, ``1.0`` for confidently wrong ones;
    ``nan`` on empty input.
    """
    target_bin = (_tensor(targets) >= _target_threshold(target_threshold)).float().flatten()
    if target_bin.numel() == 0:
        return math.nan
    prob = _tensor(probs).flatten().clamp(0.0, 1.0)
    return float(((prob - target_bin) ** 2).mean())


def reliability(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
    *,
    target_threshold: float | None = None,
) -> float:
    """``1 - ECE`` clamped to ``[0, 1]`` -- a "how trustworthy is this probability" readout.

    ``1.0`` means the forecast probabilities match observed frequencies in every
    bin; ``nan`` on empty input.
    """
    ece = calibration_error(
        probs, targets, target_threshold=target_threshold, n_bins=n_bins
    )
    if math.isnan(ece):
        return math.nan
    return min(1.0, max(0.0, 1.0 - ece))


def event_mask_from_targets(
    targets: torch.Tensor, *, target_threshold: float | None = None
) -> torch.Tensor:
    """Boolean ``(B,)`` mask of samples containing at least one observed positive.

    Convenience for :meth:`NowcastMetrics.compute`'s ``event_mask``.
    """
    targets = _tensor(targets)
    positive = targets >= _target_threshold(target_threshold)
    return positive.reshape(targets.shape[0], -1).any(dim=1)


class NowcastMetrics:
    """Compute per-(hazard, lead) scores and their nan-aware means.

    Inputs are ``(B, N_HAZARDS, N_LEADS, H, W)`` probability and target tensors.
    :meth:`compute` returns headline means (``csi``/``pod``/``far``/
    ``frequency_bias``/``pr_auc``, all ``nanmean`` over hazard*lead), per-hazard
    ``csi/<hazard>`` and ``frequency_bias/<hazard>``, the skill-optimal
    ``csi_best`` with its ``csi_best_threshold`` (and per-hazard
    ``csi_best/<hazard>``/``csi_best_threshold/<hazard>``), overall
    ``ece``/``brier``/``reliability`` with per-lead ``brier/lead_<k>h`` and
    ``reliability/lead_<k>h``, and ``n_pos/<hazard>/lead_<k>h`` positive counts
    so a degenerate split is visible rather than silently zero.

    Passing ``event_mask`` additionally emits a ``storm/`` block scored only on
    the selected samples -- aggregate CSI over all windows is flattered by easy
    quiet windows, so storm-window skill is the number that matters.
    """

    def __init__(
        self,
        pred_threshold: float = 0.5,
        target_threshold: float | None = None,
        *,
        sweep_thresholds: Sequence[float] | None = None,
    ) -> None:
        self.pred_threshold = pred_threshold
        self.target_threshold = _target_threshold(target_threshold)
        self.sweep_thresholds = tuple(
            DEFAULT_CSI_THRESHOLDS if sweep_thresholds is None else sweep_thresholds
        )

    def _core(
        self, probs: torch.Tensor, targets: torch.Tensor, *, detailed: bool
    ) -> dict[str, float]:
        grid = self.sweep_thresholds
        all_csi: list[float] = []
        all_pod: list[float] = []
        all_far: list[float] = []
        all_bias: list[float] = []
        all_ap: list[float] = []
        all_curves: list[list[float]] = []
        result: dict[str, float] = {}
        for i, hazard in enumerate(config.HAZARDS):
            hazard_csi: list[float] = []
            hazard_bias: list[float] = []
            hazard_curves: list[list[float]] = []
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
                value_bias = frequency_bias(
                    pred, target, self.pred_threshold,
                    target_threshold=self.target_threshold,
                )
                value_ap = pr_auc(
                    pred, target, target_threshold=self.target_threshold
                )
                curve = _csi_curve(pred, target, grid, self.target_threshold)
                hazard_csi.append(value_csi)
                hazard_bias.append(value_bias)
                hazard_curves.append(curve)
                all_csi.append(value_csi)
                all_pod.append(value_pod)
                all_far.append(value_far)
                all_bias.append(value_bias)
                all_ap.append(value_ap)
                all_curves.append(curve)
                if detailed:
                    n_pos = float((target >= self.target_threshold).sum())
                    result[f"n_pos/{hazard}/lead_{lead}h"] = n_pos
            if detailed:
                result[f"csi/{hazard}"] = _nanmean(hazard_csi)
                result[f"frequency_bias/{hazard}"] = _nanmean(hazard_bias)
                value, threshold = _best_from_curves(hazard_curves, grid)
                result[f"csi_best/{hazard}"] = value
                result[f"csi_best_threshold/{hazard}"] = threshold
        result["csi"] = _nanmean(all_csi)
        result["pod"] = _nanmean(all_pod)
        result["far"] = _nanmean(all_far)
        result["frequency_bias"] = _nanmean(all_bias)
        result["pr_auc"] = _nanmean(all_ap)
        value, threshold = _best_from_curves(all_curves, grid)
        result["csi_best"] = value
        result["csi_best_threshold"] = threshold
        result["ece"] = calibration_error(
            probs, targets, target_threshold=self.target_threshold
        )
        result["brier"] = brier_score(
            probs, targets, target_threshold=self.target_threshold
        )
        result["reliability"] = reliability(
            probs, targets, target_threshold=self.target_threshold
        )
        if detailed:
            for j, lead in enumerate(config.LEAD_TIMES_H):
                lead_probs = probs[:, :, j]
                lead_targets = targets[:, :, j]
                result[f"brier/lead_{lead}h"] = brier_score(
                    lead_probs, lead_targets, target_threshold=self.target_threshold
                )
                result[f"reliability/lead_{lead}h"] = reliability(
                    lead_probs, lead_targets, target_threshold=self.target_threshold
                )
        return result

    def compute(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        *,
        event_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Return a flat dict of metric values (headline values are finite or ``nan``).

        ``event_mask`` is a boolean ``(B,)`` selector -- see
        :func:`event_mask_from_targets` -- restricting the extra ``storm/``
        scores to windows that actually contain an event. Omitting it leaves the
        output identical to the all-window scores.
        """
        probs = _tensor(probs)
        targets = _tensor(targets)
        result = self._core(probs, targets, detailed=True)
        if event_mask is not None:
            mask = _tensor(event_mask).bool().reshape(-1)
            result["storm/n_samples"] = float(mask.sum())
            storm = self._core(probs[mask], targets[mask], detailed=False)
            for key, value in storm.items():
                result[f"storm/{key}"] = value
        return result

    __call__ = compute

"""Confidence calibration for predicted hazard probabilities."""

from __future__ import annotations

import math

import torch
from torch import nn

from nowcast import config


def _flatten(value: object) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.detach().reshape(-1).float()


def _hard_targets(value: object, target_threshold: float | None) -> torch.Tensor:
    """Binarise probabilistic targets -- temperature scaling and reliability
    diagrams calibrate against hard labels (Guo et al., 2017)."""
    threshold = (
        config.LABEL_OCCURRENCE_THRESHOLD
        if target_threshold is None
        else target_threshold
    )
    return (_flatten(value) >= threshold).float()


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling (Guo et al., 2017).

    Learns one positive scalar ``T`` (via its log) that divides the logits;
    :meth:`fit` optimises ``T`` on held-out logits/targets with LBFGS.
    """

    def __init__(self, init_temp: float = 1.5) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temp))))

    @property
    def temperature(self) -> torch.Tensor:
        """The current positive temperature ``T``."""
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits ``logits / T``."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.1,
        max_iter: int = 100,
        target_threshold: float | None = None,
    ) -> TemperatureScaler:
        """Optimise ``T`` to minimise BCE against the hard-binarised target."""
        flat_logits = _flatten(logits)
        flat_targets = _hard_targets(targets, target_threshold)
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)
        loss_fn = nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn(flat_logits / self.temperature, flat_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self


def apply_temperature(
    logits: torch.Tensor, temperature: float | torch.Tensor
) -> torch.Tensor:
    """Divide ``logits`` by a scalar ``temperature``."""
    temp = (
        temperature
        if torch.is_tensor(temperature)
        else torch.tensor(float(temperature))
    )
    return logits / temp


def _bin_edges(n_bins: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n_bins + 1)


def _bin_mask(prob: torch.Tensor, low: torch.Tensor, high: torch.Tensor, last: bool):
    return (prob >= low) & (prob <= high if last else prob < high)


def reliability_curve(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
    *,
    target_threshold: float | None = None,
) -> dict[str, list[float]]:
    """Binned confidence vs. empirical frequency for a reliability diagram.

    The target is hard-binarised so ``accuracy`` is a genuine event frequency.
    """
    prob = _flatten(probs)
    target = _hard_targets(targets, target_threshold)
    edges = _bin_edges(n_bins)
    centers, accuracy, confidence, counts = [], [], [], []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        in_bin = _bin_mask(prob, low, high, last=i == n_bins - 1)
        centers.append(float((low + high) / 2))
        if bool(in_bin.any()):
            accuracy.append(float(target[in_bin].mean()))
            confidence.append(float(prob[in_bin].mean()))
            counts.append(float(in_bin.sum()))
        else:
            accuracy.append(float("nan"))
            confidence.append(float("nan"))
            counts.append(0.0)
    return {
        "bin_center": centers,
        "accuracy": accuracy,
        "confidence": confidence,
        "count": counts,
    }


def expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
    *,
    target_threshold: float | None = None,
) -> float:
    """Weighted mean gap between confidence and accuracy across bins (hard target).

    ``nan`` on the empty-tensor edge case -- there is no calibration to report,
    which is distinct from perfect (``0.0``) calibration.
    """
    prob = _flatten(probs)
    target = _hard_targets(targets, target_threshold)
    edges = _bin_edges(n_bins)
    total = prob.numel()
    if total == 0:
        return math.nan
    ece = torch.tensor(0.0)
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        in_bin = _bin_mask(prob, low, high, last=i == n_bins - 1)
        if bool(in_bin.any()):
            gap = (target[in_bin].mean() - prob[in_bin].mean()).abs()
            ece = ece + (in_bin.sum() / total) * gap
    return float(ece)

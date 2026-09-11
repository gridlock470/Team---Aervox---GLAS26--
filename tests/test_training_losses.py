"""Tests for the rare-event loss weighting: pos_weight, focal and base rates."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nowcast import config, schema
from nowcast.training.losses import (
    MultiTaskLoss,
    focal_loss,
    hazard_weights_from_base_rates,
    masked_bce_with_logits,
)

_SHAPE = (2, schema.N_HAZARDS, schema.N_LEADS, 8, 9)


def _logits_targets(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(_SHAPE, generator=generator),
        torch.rand(_SHAPE, generator=generator),
    )


def _rare_batch(
    n_positive: int = 6, shape: tuple[int, ...] = (4, 1, 2, 8, 9)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mostly easy negatives (confident, correct) plus a few hard positives."""
    logits = torch.full(shape, -6.0)
    targets = torch.zeros(shape)
    flat_logits = logits.view(-1)
    flat_targets = targets.view(-1)
    step = flat_targets.numel() // n_positive
    for k in range(n_positive):
        idx = k * step
        flat_targets[idx] = 1.0
        flat_logits[idx] = -1.0  # positive the model currently gets wrong
    return logits.requires_grad_(True), targets


def _positive_gradient_share(loss: torch.Tensor, logits: torch.Tensor,
                             targets: torch.Tensor) -> float:
    """Fraction of the total |dloss/dlogit| mass that sits on positive cells."""
    (grad,) = torch.autograd.grad(loss, logits)
    magnitude = grad.abs()
    positive = magnitude[targets > 0.0].sum()
    return float(positive / magnitude.sum().clamp_min(1e-12))


def test_pos_weight_none_reproduces_plain_bce():
    logits, targets = _logits_targets()
    expected = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.equal(masked_bce_with_logits(logits, targets), expected)
    assert torch.equal(
        masked_bce_with_logits(logits, targets, None, None), expected
    )
    mask = torch.zeros_like(targets)
    mask[..., :3, :] = 1.0
    assert torch.equal(
        masked_bce_with_logits(logits, targets, mask, None),
        masked_bce_with_logits(logits, targets, mask),
    )


def test_pos_weight_raises_positive_cost_only():
    logits, targets = _rare_batch()
    logits = logits.detach()
    positives = (targets > 0.0).float()
    negatives = 1.0 - positives

    on_pos_plain = masked_bce_with_logits(logits, targets, positives)
    on_pos_weighted = masked_bce_with_logits(logits, targets, positives, 10.0)
    assert on_pos_weighted > on_pos_plain

    on_neg_plain = masked_bce_with_logits(logits, targets, negatives)
    on_neg_weighted = masked_bce_with_logits(logits, targets, negatives, 10.0)
    assert torch.isclose(on_neg_weighted, on_neg_plain)


def test_pos_weight_shifts_gradient_towards_positives():
    logits, targets = _rare_batch()
    plain = _positive_gradient_share(
        masked_bce_with_logits(logits, targets), logits, targets
    )
    weighted = _positive_gradient_share(
        masked_bce_with_logits(logits, targets, None, config.BCE_POS_WEIGHT),
        logits,
        targets,
    )
    assert weighted > plain


def test_focal_downweights_easy_negatives_relative_to_bce():
    logits, targets = _rare_batch()
    bce_share = _positive_gradient_share(
        masked_bce_with_logits(logits, targets), logits, targets
    )
    focal_share = _positive_gradient_share(
        focal_loss(logits, targets, config.FOCAL_ALPHA, config.FOCAL_GAMMA),
        logits,
        targets,
    )
    assert focal_share > bce_share


def test_focal_is_enabled_by_default():
    logits, targets = _logits_targets(2)
    assert MultiTaskLoss().focal == config.FOCAL_WEIGHT
    assert config.FOCAL_WEIGHT > 0.0
    assert not torch.isclose(
        MultiTaskLoss()(logits, targets), MultiTaskLoss(focal=0.0)(logits, targets)
    )


def test_hazard_weights_favour_the_rarest_hazard():
    targets = torch.zeros(_SHAPE)
    counts = {0: 40, 1: 4, 2: 20}
    for index, count in counts.items():
        targets[0, index].reshape(-1)[:count] = 1.0
    weights = hazard_weights_from_base_rates(targets)

    assert set(weights) == set(config.HAZARDS)
    assert all(torch.isfinite(torch.tensor(value)) for value in weights.values())
    rarest = config.HAZARDS[1]
    assert weights[rarest] == max(weights.values())
    assert weights[rarest] > weights[config.HAZARDS[0]]
    assert abs(sum(weights.values()) / len(weights) - 1.0) < 1e-6


def test_hazard_weights_finite_with_a_zero_positive_hazard():
    targets = torch.zeros(_SHAPE)
    targets[0, 0].reshape(-1)[:50] = 1.0  # only hazard 0 has any positives
    weights = hazard_weights_from_base_rates(targets)

    values = torch.tensor(list(weights.values()))
    assert torch.isfinite(values).all()
    assert (values > 0.0).all()
    assert abs(float(values.mean()) - 1.0) < 1e-6
    assert weights[config.HAZARDS[0]] < weights[config.HAZARDS[1]]


def test_hazard_weights_respect_the_occurrence_threshold():
    targets = torch.full(_SHAPE, config.LABEL_OCCURRENCE_THRESHOLD - 0.01)
    weights = hazard_weights_from_base_rates(targets)
    # No cell clears the threshold, so every hazard is floored equally.
    assert all(abs(value - 1.0) < 1e-6 for value in weights.values())


def test_multitask_loss_finite_on_degenerate_batches():
    logits = torch.randn(_SHAPE, generator=torch.Generator().manual_seed(3))
    loss_fn = MultiTaskLoss()
    for targets in (torch.zeros(_SHAPE), torch.ones(_SHAPE)):
        value = loss_fn(logits, targets)
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_multitask_loss_is_differentiable():
    logits, targets = _logits_targets(4)
    logits = logits.clone().requires_grad_(True)
    weights = hazard_weights_from_base_rates(targets)
    loss = MultiTaskLoss(
        hazard_weights=weights, pos_weight=config.BCE_POS_WEIGHT
    )(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0

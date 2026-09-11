"""Tests for the rare-event verification metrics.

Every tensor here is synthetic and hand-built so the expected contingency table
is exact -- these assertions are about arithmetic, not about model quality.
"""

from __future__ import annotations

import math

import pytest
import torch

from nowcast import schema
from nowcast.training.metrics import (
    NowcastMetrics,
    best_csi,
    brier_score,
    csi,
    event_mask_from_targets,
    far,
    frequency_bias,
    pod,
    reliability,
)

_SHAPE = (4, schema.N_HAZARDS, schema.N_LEADS, 8, 9)


def _binary_targets(seed: int = 0, rate: float = 0.5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(_SHAPE, generator=generator) < rate).float()


def test_perfect_prediction_scores_are_extremal():
    targets = _binary_targets(0)
    assert csi(targets, targets) == 1.0
    assert pod(targets, targets) == 1.0
    assert far(targets, targets) == 0.0
    assert frequency_bias(targets, targets) == 1.0


def test_all_zero_prediction_scores_zero_not_nan():
    # A model that predicts nothing must score a visible zero, not vanish.
    targets = _binary_targets(1)
    preds = torch.zeros(_SHAPE)
    assert csi(preds, targets) == 0.0
    assert pod(preds, targets) == 0.0
    assert frequency_bias(preds, targets) == 0.0
    assert math.isnan(far(preds, targets))  # nothing predicted positive


def test_frequency_bias_detects_over_and_under_forecasting():
    targets = torch.zeros(_SHAPE)
    targets[:, :, :, 0, :4] = 1.0  # 4 positives per (sample, hazard, lead)
    over = targets.clone()
    over[:, :, :, 1, :4] = 1.0  # 8 predicted -> bias 2.0
    under = torch.zeros(_SHAPE)
    under[:, :, :, 0, :2] = 1.0  # 2 predicted -> bias 0.5
    assert frequency_bias(over, targets) == 2.0
    assert frequency_bias(under, targets) == 0.5


def test_threshold_sweep_beats_fixed_half_at_low_base_rate():
    # 1-in-1000 base rate; the model ranks correctly but is under-confident, so
    # CSI at 0.5 is zero while the optimal threshold sits near 0.1.
    generator = torch.Generator().manual_seed(7)
    targets = torch.zeros(_SHAPE)
    targets[0, 0, 0, 0, :3] = 1.0
    preds = torch.rand(_SHAPE, generator=generator) * 0.02
    preds[targets > 0] = 0.2
    fixed = csi(preds, targets, 0.5)
    best_value, best_threshold = best_csi(preds, targets)
    assert fixed == 0.0
    assert best_value > fixed
    assert best_threshold < 0.5


def test_brier_score_bounds():
    targets = _binary_targets(2)
    assert brier_score(targets, targets) == 0.0
    assert brier_score(1.0 - targets, targets) == 1.0


def test_reliability_is_a_unit_interval_readout():
    targets = _binary_targets(3)
    value = reliability(targets, targets)
    assert 0.0 <= value <= 1.0
    assert value == 1.0  # perfectly calibrated hard forecasts


def test_storm_subset_csi_differs_from_all_window_csi():
    targets = torch.zeros(_SHAPE)
    probs = torch.zeros(_SHAPE)
    # Samples 2 and 3 are storm windows: observed events, imperfectly forecast.
    targets[2:, :, :, 0, :4] = 1.0
    probs[2:, :, :, 0, :3] = 0.9
    # Samples 0 and 1 are quiet but the model still raises false alarms there.
    probs[:2, :, :, 5, :2] = 0.9
    mask = event_mask_from_targets(targets)
    assert mask.tolist() == [False, False, True, True]

    scores = NowcastMetrics().compute(probs, targets, event_mask=mask)
    assert scores["storm/n_samples"] == 2.0
    assert scores["storm/csi"] != scores["csi"]
    assert scores["storm/csi"] > scores["csi"]


def test_compute_preserves_existing_keys_and_adds_new_ones():
    generator = torch.Generator().manual_seed(4)
    targets = (torch.rand(_SHAPE, generator=generator) < 0.05).float()
    probs = torch.rand(_SHAPE, generator=generator)
    scores = NowcastMetrics().compute(probs, targets)
    for key in ("csi", "pod", "far", "pr_auc", "ece"):
        assert math.isfinite(scores[key])
    assert any(key.startswith("csi/") for key in scores)
    assert any(key.startswith("n_pos/") for key in scores)
    for key in ("frequency_bias", "brier", "reliability", "csi_best"):
        assert math.isfinite(scores[key])
    assert scores["csi_best"] >= scores["csi"]
    assert 0.0 < scores["csi_best_threshold"] <= 1.0
    assert math.isfinite(scores["brier/lead_1h"])
    assert math.isfinite(scores["reliability/lead_1h"])
    assert "storm/csi" not in scores  # no mask -> no storm block


def test_degenerate_input_returns_nan_and_never_raises():
    probs = torch.full(_SHAPE, 0.1)
    targets = torch.zeros(_SHAPE)  # no positives anywhere
    assert math.isnan(frequency_bias(probs, targets))
    # A (hazard, lead) slice with no observed positives can never produce a
    # meaningful CSI, so it is excluded from the threshold comparison. With no
    # positives anywhere every slice is excluded, there is nothing to maximise,
    # and best_csi is nan -- consistent with csi() being nan on the same input.
    # Comparing thresholds over a slice population that changes with the
    # threshold is what previously let csi_best reach 1.0 while pod was 0.0.
    best_value, best_threshold = best_csi(probs, targets)
    assert math.isnan(best_value)
    assert math.isnan(best_threshold)
    assert math.isnan(csi(probs, targets, 0.5))
    assert brier_score(probs, targets) == pytest.approx(0.1**2)

    scores = NowcastMetrics().compute(
        probs, targets, event_mask=event_mask_from_targets(targets)
    )
    assert math.isnan(scores["csi"])
    assert math.isnan(scores["csi_best"])  # nothing observed -> nothing to maximise
    assert scores["storm/n_samples"] == 0.0
    assert math.isnan(scores["storm/csi"])

"""Tests for temperature scaling, reliability curves and MC-dropout."""

from __future__ import annotations

import torch

from nowcast import config, schema
from nowcast.models.multitask import MultiTaskNowcastNet
from nowcast.testing import synthetic
from nowcast.uncertainty.calibration import (
    TemperatureScaler,
    apply_temperature,
    expected_calibration_error,
    reliability_curve,
)
from nowcast.uncertainty.mc_dropout import enable_mc_dropout, mc_dropout_predict


def _overconfident_case(n: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    latent = torch.randn(n, generator=generator)
    targets = torch.bernoulli(torch.sigmoid(latent), generator=generator)
    logits = latent * 4.0  # inflated -> over-confident
    return logits, targets


def test_temperature_scaling_lowers_ece():
    logits, targets = _overconfident_case()
    before = expected_calibration_error(torch.sigmoid(logits), targets)
    scaler = TemperatureScaler()
    scaler.fit(logits, targets)
    after = expected_calibration_error(
        torch.sigmoid(scaler(logits).detach()), targets
    )
    assert scaler.temperature.item() > 1.0
    assert after < before


def test_apply_temperature_matches_division():
    logits, _ = _overconfident_case(128)
    assert torch.allclose(apply_temperature(logits, 2.0), logits / 2.0)


def test_reliability_curve_shape_and_ece_bounds():
    logits, targets = _overconfident_case(512)
    curve = reliability_curve(torch.sigmoid(logits), targets, n_bins=8)
    assert len(curve["bin_center"]) == 8
    assert len(curve["accuracy"]) == 8
    ece = expected_calibration_error(torch.sigmoid(logits), targets)
    assert 0.0 <= ece <= 1.0


def test_mc_dropout_predict_returns_mean_and_std():
    net = MultiTaskNowcastNet(backbone_kwargs={"hidden": 8, "depth": 1})
    assert enable_mc_dropout(net) > 0
    x, _ = synthetic.make_batch(2, seed=0)
    mean, std = mc_dropout_predict(net, torch.from_numpy(x), n=5)
    expected = (2, schema.N_HAZARDS, schema.N_LEADS, *config.GRID_SHAPE)
    assert mean.shape == expected
    assert std.shape == expected
    assert float(std.sum()) > 0.0
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()

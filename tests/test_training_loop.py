"""Tests for losses, verification metrics and the LightningModule wiring."""

from __future__ import annotations

import math

import torch

from nowcast import config, schema
from nowcast.training.lit_module import LitNowcast
from nowcast.training.losses import (
    MultiTaskLoss,
    dice_loss,
    focal_loss,
    masked_bce_with_logits,
)
from nowcast.training.metrics import (
    NowcastMetrics,
    calibration_error,
    csi,
    far,
    pod,
    pr_auc,
)
from nowcast.training.train import build, load_config

_SHAPE = (2, schema.N_HAZARDS, schema.N_LEADS, 8, 9)
_CONFIG = (
    config.PROJECT_ROOT
    / "nowcast"
    / "training"
    / "configs"
    / "convlstm_small.yaml"
)


def _logits_targets(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(_SHAPE, generator=generator),
        torch.rand(_SHAPE, generator=generator),
    )


def test_elementwise_losses_are_finite_scalars():
    logits, targets = _logits_targets()
    for value in (
        masked_bce_with_logits(logits, targets),
        dice_loss(logits, targets),
        focal_loss(logits, targets),
    ):
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_masked_bce_respects_mask():
    logits, targets = _logits_targets()
    mask = torch.zeros_like(targets)
    mask[..., :3, :] = 1.0
    assert not torch.isclose(
        masked_bce_with_logits(logits, targets),
        masked_bce_with_logits(logits, targets, mask),
    )


def test_multitask_loss_accepts_dict_and_tensor():
    logits, targets = _logits_targets()
    as_dict = {hazard: logits[:, i] for i, hazard in enumerate(config.HAZARDS)}
    loss_fn = MultiTaskLoss()
    assert torch.isclose(loss_fn(as_dict, targets), loss_fn(logits, targets))
    assert loss_fn(logits, targets).ndim == 0


def test_csi_pod_far_in_unit_interval():
    probs, targets = _logits_targets(1)
    probs = probs.sigmoid()
    for metric in (csi, pod, far):
        value = metric(probs, targets, 0.5)
        assert 0.0 <= value <= 1.0


def test_perfect_prediction_metric_extremes():
    targets = (torch.rand(_SHAPE) > 0.5).float()
    assert csi(targets, targets, 0.5) == 1.0
    assert pod(targets, targets, 0.5) == 1.0
    assert far(targets, targets, 0.5) == 0.0


def test_nowcast_metrics_bundle_all_finite():
    probs, targets = _logits_targets(2)
    scores = NowcastMetrics().compute(probs.sigmoid(), targets)
    assert {"csi", "pod", "far", "pr_auc", "ece"} <= set(scores)
    assert all(math.isfinite(value) for value in scores.values())
    assert any(key.startswith("csi/") for key in scores)
    assert math.isfinite(pr_auc(probs.sigmoid(), (targets > 0.5).float()))
    assert math.isfinite(calibration_error(probs.sigmoid(), targets))


def test_configure_optimizers_is_adamw_and_cosine():
    lit = LitNowcast(backbone_kwargs={"hidden": 8, "depth": 1}, max_epochs=5)
    opt_cfg = lit.configure_optimizers()
    assert isinstance(opt_cfg["optimizer"], torch.optim.AdamW)
    assert isinstance(
        opt_cfg["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert lit.hparams.lr == 3e-4
    assert "backbone_kwargs" in lit.hparams


def test_train_config_parses_and_builds():
    cfg = load_config(_CONFIG)
    assert cfg["model"]["backbone"]["hidden"] == 16
    cfg["data"] = {"source": "synthetic", "n_hours": 56, "batch_size": 2}
    datamodule, module = build(cfg)
    assert isinstance(module, LitNowcast)
    datamodule.setup("fit")
    assert len(datamodule.train_dataloader()) >= 1

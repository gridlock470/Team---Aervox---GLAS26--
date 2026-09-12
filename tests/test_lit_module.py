"""Regression for the val-epoch NaN-poisoning bug in ``LitNowcast``.

``NowcastMetrics`` deliberately returns ``nan`` for any batch with zero true
positives (see metrics.py docstrings) -- that is correct, per-batch behavior.
The bug was one level up: Lightning's default epoch reduction for
``self.log(..., on_epoch=True)`` is a plain ``mean()`` over the per-step
values, which is NOT nan-aware. With ``batch_size=4`` in the real config and
cloudburst/flash-flood this rare, at least one validation batch per epoch has
zero positives across every hazard x lead, so its per-batch CSI is nan and
that single nan poisons the whole epoch's logged ``val_csi`` -- matching the
observed 25-epoch run where val_csi/val_csi_best/val_storm_csi were nan
throughout, even though the val split (and most individual batches) had real
positives.

This test reproduces the minimal case: one batch with a positive-everywhere
target (finite per-batch metrics) and one batch with an all-zero target (nan
per-batch metrics), run through a real Lightning epoch. It fails before the
fix (poisoned to nan) and passes after (nanmean skips the degenerate batch).
"""

from __future__ import annotations

import math

import lightning as L
import torch
from torch.utils.data import DataLoader

from nowcast import config, schema
from nowcast.data.transforms import TERRAIN_PLANES
from nowcast.training.lit_module import LitNowcast

_H, _W = config.GRID_SHAPE


def _base(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(schema.N_CHANNELS, config.INPUT_SEQ_LEN, _H, _W, generator=g)
    terrain = torch.randn(TERRAIN_PLANES, _H, _W, generator=g)
    return x, terrain


def _mixed_sample(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch with BOTH classes present, for every hazard/lead.

    Every per-batch score in NowcastMetrics needs a true positive to be
    non-nan; pr_auc additionally needs a true *negative* in the same batch
    (it is undefined for a single-class target, by design -- see
    metrics.py's ``pr_auc`` docstring). A checkerboard is the simplest
    pattern that guarantees both classes in every (hazard, lead) slice
    regardless of grid size.
    """
    x, terrain = _base(seed)
    row = torch.arange(_H).unsqueeze(1)
    col = torch.arange(_W).unsqueeze(0)
    checkerboard = ((row + col) % 2 == 0).float()
    y = checkerboard.expand(schema.N_HAZARDS, schema.N_LEADS, _H, _W).clone()
    return x, y, terrain


def _all_negative_sample(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch with zero true positives anywhere -- the degenerate case that
    NowcastMetrics scores as nan (see metrics.py), and that used to poison
    the whole epoch's logged value once one of these landed in a real
    validation split."""
    x, terrain = _base(seed)
    y = torch.zeros(schema.N_HAZARDS, schema.N_LEADS, _H, _W)
    return x, y, terrain


def _dataloader() -> DataLoader:
    # One batch that must score finite (both classes present) and one that
    # must score nan (no cell is a positive) -- batch_size=1 so each sample is
    # its own Lightning step/batch, matching how the real rare-event splits
    # produce all-negative batches at batch_size=4.
    samples = [_mixed_sample(seed=0), _all_negative_sample(seed=1)]
    return DataLoader(samples, batch_size=1)


def test_val_epoch_metrics_survive_one_all_negative_batch():
    lit = LitNowcast(backbone_kwargs={"hidden": 4, "depth": 1})
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(lit, train_dataloaders=_dataloader(), val_dataloaders=_dataloader())

    metrics = trainer.callback_metrics
    for key in ("val_csi", "val_csi_best", "val_pr_auc", "val_frequency_bias"):
        value = float(metrics[key])
        assert math.isfinite(value), (
            f"{key}={value} -- one all-negative validation batch poisoned the "
            f"whole epoch's logged value; epoch reduction must be nan-aware"
        )
    if "val_storm_csi" in metrics:
        value = float(metrics["val_storm_csi"])
        assert math.isfinite(value), f"val_storm_csi={value}"

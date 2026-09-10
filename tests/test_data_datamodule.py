"""Tests for feature normalisation, NowcastDataset windowing and the datamodule."""

from __future__ import annotations

import tempfile
from pathlib import Path

import lightning as L
import torch

from nowcast import config, schema
from nowcast.data.datamodule import NowcastDataModule, NowcastDataset
from nowcast.data.transforms import (
    Normalizer,
    compute_norm_stats,
    load_norm_stats,
    save_norm_stats,
)
from nowcast.testing import synthetic
from nowcast.training.lit_module import LitNowcast


def test_compute_norm_stats_covers_feature_channels():
    stats = compute_norm_stats(synthetic.make_feature_datacube(n_hours=10, seed=0))
    assert set(stats) == set(schema.FEATURE_CHANNELS)
    assert all(set(entry) == {"mean", "std"} for entry in stats.values())


def test_normalizer_round_trips():
    stats = {
        channel: {"mean": float(i), "std": float(i + 1)}
        for i, channel in enumerate(schema.FEATURE_CHANNELS)
    }
    norm = Normalizer(stats)
    x = torch.randn(schema.N_CHANNELS, 4, 8, 9)
    assert torch.allclose(norm.inverse(norm(x)), x, atol=1e-5)
    x3 = torch.randn(schema.N_CHANNELS, 8, 9)
    assert norm(x3).shape == x3.shape


def test_save_and_load_norm_stats():
    stats = compute_norm_stats(synthetic.make_feature_datacube(n_hours=8, seed=0))
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = save_norm_stats(stats, Path(tmp_dir) / "norm.json")
        loaded = load_norm_stats(path)
    assert loaded.keys() == stats.keys()
    first = schema.FEATURE_CHANNELS[0]
    assert loaded[first]["mean"] == stats[first]["mean"]


def test_dataset_item_shapes():
    features = synthetic.make_feature_datacube(n_hours=20, seed=0)
    labels = synthetic.make_targets(n_hours=20, seed=0)
    dataset = NowcastDataset(features, labels)
    assert len(dataset) == 20 - config.INPUT_SEQ_LEN + 1
    x, y, terrain = dataset[0]
    assert tuple(x.shape) == schema.INPUT_SHAPE
    assert tuple(y.shape) == schema.TARGET_SHAPE
    assert tuple(terrain.shape) == (
        len(schema.FLASH_FLOOD_EXTRA_CHANNELS),
        *config.GRID_SHAPE,
    )
    assert torch.isfinite(x).all()


def test_dataset_applies_normalizer():
    features = synthetic.make_feature_datacube(n_hours=16, seed=0)
    labels = synthetic.make_targets(n_hours=16, seed=0)
    stats = compute_norm_stats(features)
    raw = NowcastDataset(features, labels)[0][0]
    normed = NowcastDataset(features, labels, norm=Normalizer(stats))[0][0]
    assert not torch.allclose(raw, normed)


def test_datamodule_fast_dev_run_with_lit_module():
    datamodule = NowcastDataModule.from_synthetic(n_hours=56, batch_size=2)
    datamodule.setup("fit")
    assert len(datamodule.train_dataloader()) >= 1
    assert len(datamodule.val_dataloader()) >= 1

    lit = LitNowcast(backbone_kwargs={"hidden": 8, "depth": 1})
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(lit, datamodule=datamodule)

    batch = next(iter(datamodule.train_dataloader()))
    with torch.no_grad():
        probs = lit.model.predict_proba(batch[0], batch[2])
    assert probs.shape[1:] == (
        schema.N_HAZARDS,
        schema.N_LEADS,
        *config.GRID_SHAPE,
    )

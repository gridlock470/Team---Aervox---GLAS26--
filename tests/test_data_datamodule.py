"""Tests for feature normalisation, NowcastDataset windowing and the datamodule."""

from __future__ import annotations

import tempfile
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch

from nowcast import config, schema
from nowcast.data.datamodule import NowcastDataModule, NowcastDataset
from nowcast.data.transforms import (
    TERRAIN_PLANES,
    Normalizer,
    compute_norm_stats,
    load_norm_stats,
    resolve_norm_stats,
    save_norm_stats,
    transform_terrain,
)
from nowcast.features.assemble import assemble_features
from nowcast.features.labels import build_labels
from nowcast.testing import synthetic
from nowcast.training.lit_module import LitNowcast


def test_compute_norm_stats_covers_feature_channels():
    stats = compute_norm_stats(synthetic.make_feature_datacube(n_hours=10, seed=0))
    assert set(stats) == set(schema.FEATURE_CHANNELS)
    assert all(set(entry) == {"mean", "std"} for entry in stats.values())


def test_compute_norm_stats_respects_date_range():
    features = synthetic.make_feature_datacube(n_hours=48, seed=0)
    times = pd.date_range("2018-01-01", periods=48, freq="h")
    features = features.assign_coords(time=times)
    full = compute_norm_stats(features)
    windowed = compute_norm_stats(features, date_range=("2018-01-01", "2018-01-01"))
    assert full["t2m"]["mean"] != windowed["t2m"]["mean"]


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


def test_resolve_norm_stats_computes_then_loads():
    features = synthetic.make_feature_datacube(n_hours=10, seed=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "norm_stats.json"
        first = resolve_norm_stats(features, path=path)
        assert path.exists()
        second = resolve_norm_stats("unused-once-cached", path=path)  # loads, no compute
    assert first == second


def test_transform_terrain_is_standardised():
    datacube = synthetic.make_datacube(n_hours=6, seed=0)
    terrain = transform_terrain(
        datacube["flow_accumulation"].values, datacube["hand"].values
    )
    assert terrain.shape == (TERRAIN_PLANES, *config.GRID_SHAPE)
    for plane in terrain:
        assert abs(float(plane.mean())) < 0.5
        assert abs(float(plane.std()) - 1.0) < 0.2


def test_dataset_item_shapes_and_normalisation():
    features = synthetic.make_feature_datacube(n_hours=24, seed=0)
    labels = synthetic.make_targets(n_hours=24, seed=0)
    stats = compute_norm_stats(features)
    dataset = NowcastDataset(features, labels, norm=Normalizer(stats))
    assert len(dataset) == 24 - config.INPUT_SEQ_LEN + 1

    x, y, terrain = dataset[0]
    assert tuple(x.shape) == schema.INPUT_SHAPE
    assert tuple(y.shape) == schema.TARGET_SHAPE
    assert tuple(terrain.shape) == (TERRAIN_PLANES, *config.GRID_SHAPE)
    assert torch.isfinite(x).all()
    assert terrain.mean(dim=(1, 2)).abs().max() < 0.5

    # Stats came from the *entire* 24h cube, so normalising that same cube
    # (not overlapping sliding windows, which would double-count the middle
    # hours and bias the sample mean) should land each channel at mean~0/std~1.
    whole_cube = torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(dataset._series(c, 0, 24)))
            for c in schema.FEATURE_CHANNELS
        ]
    )
    normalised = Normalizer(stats)(whole_cube)
    per_channel = normalised.reshape(schema.N_CHANNELS, -1)
    assert per_channel.mean(dim=1).abs().max() < 0.5
    assert (per_channel.std(dim=1) - 1.0).abs().max() < 0.5


def test_dataset_uses_explicit_terrain_source_from_assembled_features():
    datacube = synthetic.make_datacube(n_hours=20, seed=0)
    features = assemble_features(datacube)  # 20 FEATURE_CHANNELS only, no flow_direction
    labels = build_labels(datacube)
    dataset = NowcastDataset(features, labels, terrain=datacube)
    x, y, terrain = dataset[0]
    assert tuple(x.shape) == schema.INPUT_SHAPE
    assert tuple(y.shape) == schema.TARGET_SHAPE
    assert tuple(terrain.shape) == (TERRAIN_PLANES, *config.GRID_SHAPE)


def _multi_range_features(per_range: int = 15):
    n = per_range * 3
    features = synthetic.make_feature_datacube(n_hours=n, seed=0)
    labels = synthetic.make_targets(n_hours=n, seed=0)
    times = pd.to_datetime(
        list(pd.date_range("2018-03-01", periods=per_range, freq="h"))
        + list(pd.date_range("2020-02-01", periods=per_range, freq="h"))
        + list(pd.date_range("2020-08-01", periods=per_range, freq="h"))
    )
    return (
        features.assign_coords(time=times),
        labels.assign_coords(time=times),
    )


def test_datamodule_date_range_split_is_disjoint():
    features, labels = _multi_range_features(per_range=15)
    datacube = synthetic.make_datacube(n_hours=6, seed=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        dm = NowcastDataModule(
            features=features,
            labels=labels,
            terrain=datacube,
            batch_size=2,
            norm_stats_path=Path(tmp_dir) / "norm_stats.json",
        )
        dm.setup("fit")
        assert (Path(tmp_dir) / "norm_stats.json").exists()
    train_t = set(dm._train.time_values().tolist())
    val_t = set(dm._val.time_values().tolist())
    test_t = set(dm._test.time_values().tolist())
    assert val_t & test_t == set()
    assert train_t & val_t == set()
    assert len(dm.val_dataloader()) >= 1


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
    assert np.isfinite(batch[2].numpy()).all()

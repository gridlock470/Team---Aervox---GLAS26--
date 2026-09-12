"""Tests for the pixel-dataset builder (``nowcast.baseline.dataset``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from nowcast import config, schema
from nowcast.baseline.dataset import (
    PixelDataset,
    make_pixel_dataset,
    split_by_date_range,
)
from nowcast.features.labels import build_labels
from nowcast.testing import synthetic

_HAS_METPY = importlib.util.find_spec("metpy") is not None
pytestmark = pytest.mark.skipif(not _HAS_METPY, reason="metpy is required")

if _HAS_METPY:
    from nowcast.features.assemble import assemble_features

_MAX_LEAD = max(config.LEAD_TIMES_H)


def _cube(n_hours: int = 14, seed: int = 4):
    return synthetic.make_datacube(n_hours=n_hours, seed=seed).isel(
        lat=slice(0, 8), lon=slice(0, 8)
    )


@pytest.fixture(scope="module")
def feat_and_labels():
    cube = _cube()
    return assemble_features(cube), build_labels(cube), cube.sizes["time"]


def test_shapes_are_consistent(feat_and_labels):
    features, labels, n_hours = feat_and_labels
    data = make_pixel_dataset(features, labels, n_per_time=40, seed=0)
    n_valid = n_hours - _MAX_LEAD
    assert data.X.shape[0] == data.y.shape[0] == data.times.shape[0]
    assert data.X.shape[1] == len(schema.FEATURE_CHANNELS)
    assert data.y.shape[1] == schema.N_HAZARDS * schema.N_LEADS
    assert 0 < data.X.shape[0] <= n_valid * 40
    assert data.feature_names == list(schema.FEATURE_CHANNELS)
    assert len(data.target_names) == schema.N_HAZARDS * schema.N_LEADS


def test_no_non_finite_rows(feat_and_labels):
    features, labels, _ = feat_and_labels
    data = make_pixel_dataset(features, labels, n_per_time=None)
    assert np.isfinite(data.X).all()
    assert np.isfinite(data.y).all()


def test_label_horizon_drop_excludes_tail(feat_and_labels):
    features, labels, n_hours = feat_and_labels
    kept = make_pixel_dataset(features, labels, n_per_time=None)
    full = make_pixel_dataset(
        features, labels, n_per_time=None, drop_label_horizon=False
    )
    assert {pd.Timestamp(t) for t in kept.times} <= {
        pd.Timestamp(t) for t in full.times
    }
    latest_kept = max(pd.Timestamp(t) for t in kept.times)
    latest_all = max(pd.Timestamp(t) for t in full.times)
    assert latest_all - latest_kept == pd.Timedelta(hours=_MAX_LEAD)


def test_static_channel_values_come_from_the_2d_field(feat_and_labels):
    features, labels, _ = feat_and_labels
    data = make_pixel_dataset(features, labels, n_per_time=20, seed=1)
    col = data.feature_names.index("elevation")
    field_values = set(np.unique(features["elevation"].values.astype("float32")))
    assert set(np.unique(data.X[:, col])).issubset(field_values)


def test_split_by_date_range_partitions_rows_disjointly():
    cube = synthetic.make_datacube(n_hours=48, seed=5).isel(
        lat=slice(0, 6), lon=slice(0, 6)
    )
    # Straddle the configured train/val boundary rather than a hard-coded date,
    # so the fixture follows the contract when the split ranges move.
    boundary = pd.Timestamp(config.VAL_DATE_RANGE[0]) - pd.Timedelta(days=1)
    stamps = pd.date_range(boundary, periods=48, freq="1h")
    cube = cube.assign_coords(time=stamps)
    features = assemble_features(cube)
    labels = build_labels(cube)

    train, val = split_by_date_range(features, labels, n_per_time=10)
    assert isinstance(train, PixelDataset)
    assert isinstance(val, PixelDataset)
    assert train.X.shape[0] > 0
    assert val.X.shape[0] > 0

    train_days = {pd.Timestamp(t).date().isoformat() for t in train.times}
    val_days = {pd.Timestamp(t).date().isoformat() for t in val.times}
    # The claim is that rows partition across the configured boundary -- the
    # day before it lands in train, the boundary day in val -- not that the
    # boundary sits on any particular calendar date.
    assert train_days == {boundary.date().isoformat()}
    assert val_days == {config.VAL_DATE_RANGE[0]}
    assert train_days.isdisjoint(val_days)


def test_split_by_date_range_empty_val_returns_empty_dataset():
    cube = _cube(n_hours=14, seed=6)  # synthetic default dates: 2018 only
    features = assemble_features(cube)
    labels = build_labels(cube)
    train, val = split_by_date_range(features, labels, n_per_time=10)
    assert train.X.shape[0] > 0
    assert val.X.shape[0] == 0
    assert val.X.shape == (0, len(schema.FEATURE_CHANNELS))

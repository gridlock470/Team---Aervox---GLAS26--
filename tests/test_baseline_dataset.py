"""Tests for the pixel-dataset builder (``nowcast.baseline.dataset``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from nowcast import schema
from nowcast.baseline.dataset import PixelDataset, make_pixel_dataset, split_by_year
from nowcast.features.labels import build_labels
from nowcast.testing import synthetic

_HAS_METPY = importlib.util.find_spec("metpy") is not None
pytestmark = pytest.mark.skipif(not _HAS_METPY, reason="metpy is required")

if _HAS_METPY:
    from nowcast.features.assemble import assemble_features


def _cube(n_hours: int = 3, seed: int = 4):
    return synthetic.make_datacube(n_hours=n_hours, seed=seed).isel(
        lat=slice(0, 8), lon=slice(0, 8)
    )


@pytest.fixture(scope="module")
def feat_and_labels():
    cube = _cube()
    return assemble_features(cube), build_labels(cube)


def test_shapes_are_consistent(feat_and_labels):
    features, labels = feat_and_labels
    data = make_pixel_dataset(features, labels, n_per_time=40, seed=0)
    assert data.X.shape[0] == data.y.shape[0] == data.times.shape[0]
    assert data.X.shape[1] == len(schema.FEATURE_CHANNELS)
    assert data.y.shape[1] == schema.N_HAZARDS * schema.N_LEADS
    assert data.X.shape[0] <= 3 * 40
    assert data.feature_names == list(schema.FEATURE_CHANNELS)
    assert len(data.target_names) == schema.N_HAZARDS * schema.N_LEADS


def test_no_non_finite_rows(feat_and_labels):
    features, labels = feat_and_labels
    data = make_pixel_dataset(features, labels, n_per_time=None)
    assert np.isfinite(data.X).all()
    assert np.isfinite(data.y).all()


def test_static_channel_values_come_from_the_2d_field(feat_and_labels):
    features, labels = feat_and_labels
    data = make_pixel_dataset(features, labels, n_per_time=20, seed=1)
    col = data.feature_names.index("elevation")
    field_values = set(np.unique(features["elevation"].values.astype("float32")))
    assert set(np.unique(data.X[:, col])).issubset(field_values)


def test_split_by_year_partitions_rows():
    cube = synthetic.make_datacube(n_hours=6, seed=5).isel(
        lat=slice(0, 6), lon=slice(0, 6)
    )
    stamps = pd.to_datetime(
        [
            "2018-06-01",
            "2018-06-02",
            "2018-06-03",
            "2020-06-01",
            "2020-06-02",
            "2020-06-03",
        ]
    )
    cube = cube.assign_coords(time=stamps)
    features = assemble_features(cube)
    labels = build_labels(cube)

    train, val = split_by_year(features, labels, n_per_time=10)
    assert isinstance(train, PixelDataset)
    assert isinstance(val, PixelDataset)
    assert train.X.shape[0] > 0
    assert val.X.shape[0] > 0
    assert train.X.shape[0] <= 3 * 10
    assert val.X.shape[0] <= 3 * 10
    assert {pd.Timestamp(t).year for t in train.times} == {2018}
    assert {pd.Timestamp(t).year for t in val.times} == {2020}

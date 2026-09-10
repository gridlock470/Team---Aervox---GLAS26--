"""Tests for the assembled feature cube (``nowcast.features.assemble``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from nowcast import schema
from nowcast.testing import synthetic

_HAS_METPY = importlib.util.find_spec("metpy") is not None
pytestmark = pytest.mark.skipif(not _HAS_METPY, reason="metpy is required")

if _HAS_METPY:
    from nowcast.features.assemble import assemble_features, open_features, write_features


@pytest.fixture(scope="module")
def features():
    cube = synthetic.make_datacube(n_hours=3, seed=1).isel(
        lat=slice(0, 8), lon=slice(0, 8)
    )
    return assemble_features(cube)


def test_exact_channel_set_and_order(features):
    assert set(features.data_vars) == set(schema.FEATURE_CHANNELS)
    assert list(features.data_vars) == list(schema.FEATURE_CHANNELS)


def test_all_channels_float32(features):
    for channel in schema.FEATURE_CHANNELS:
        assert features[channel].dtype == np.float32, channel


def test_static_and_dynamic_dims(features):
    for channel in schema.FEATURE_CHANNELS:
        if channel in schema.STATIC_CHANNELS:
            assert tuple(features[channel].dims) == ("lat", "lon"), channel
        else:
            assert tuple(features[channel].dims) == ("time", "lat", "lon"), channel


def test_all_values_finite(features):
    for channel in schema.FEATURE_CHANNELS:
        assert np.isfinite(features[channel].values).all(), channel


def test_zarr_roundtrip(tmp_path, features):
    path = write_features(features, tmp_path / "features.zarr")
    reopened = open_features(path)
    assert set(reopened.data_vars) == set(schema.FEATURE_CHANNELS)
    np.testing.assert_allclose(
        reopened["t2m"].values, features["t2m"].values, rtol=1e-6
    )
    np.testing.assert_allclose(
        reopened["elevation"].values, features["elevation"].values, rtol=1e-6
    )


def test_fills_injected_nans(features):
    cube = synthetic.make_datacube(n_hours=3, seed=9).isel(
        lat=slice(0, 6), lon=slice(0, 6)
    )
    t2m = cube["t2m"].values.copy()
    t2m[0, 0, 0] = np.nan
    cube["t2m"] = (("time", "lat", "lon"), t2m)
    out = assemble_features(cube)
    assert np.isfinite(out["t2m"].values).all()

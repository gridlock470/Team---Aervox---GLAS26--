"""Tests for :mod:`nowcast.common.grid`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from nowcast import config
from nowcast.common import grid


def _raw_field(res: float = 0.25) -> xr.DataArray:
    """A coarse field spanning wider than the pilot bbox, descending latitude."""
    lat = np.arange(config.BBOX_NORTH + 1.0, config.BBOX_SOUTH - 1.0, -res)
    lon = np.arange(config.BBOX_WEST - 1.0, config.BBOX_EAST + 1.0, res)
    time = np.arange(3)
    data = (
        lat[None, :, None] * 0.1
        + lon[None, None, :] * 0.01
        + time[:, None, None]
    )
    return xr.DataArray(
        data.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="field",
    )


def test_target_grid_matches_config():
    tg = grid.target_grid()
    np.testing.assert_allclose(tg["lat"].values, config.GRID_LAT)
    np.testing.assert_allclose(tg["lon"].values, config.GRID_LON)
    assert tg["lat"].size, tg["lon"].size == config.GRID_SHAPE


def test_regrid_to_target_shape_and_coords():
    out = grid.regrid_to_target(_raw_field())
    assert out.shape == (3, *config.GRID_SHAPE)
    np.testing.assert_allclose(out["lat"].values, config.GRID_LAT, atol=1e-6)
    np.testing.assert_allclose(out["lon"].values, config.GRID_LON, atol=1e-6)
    assert np.isfinite(out.values).all()


def test_regrid_to_target_handles_descending_lat():
    desc = _raw_field()
    asc = desc.sortby("lat")
    out_desc = grid.regrid_to_target(desc)
    out_asc = grid.regrid_to_target(asc)
    xr.testing.assert_allclose(out_desc, out_asc)


def test_regrid_nearest_method_runs():
    out = grid.regrid_to_target(_raw_field(), method="nearest")
    assert out.shape == (3, *config.GRID_SHAPE)


def test_crop_bbox_reduces_extent_but_covers_grid():
    cropped = grid.crop_bbox(_raw_field(), pad=0.5)
    assert cropped["lat"].min() <= config.GRID_LAT[0]
    assert cropped["lat"].max() >= config.GRID_LAT[-1]
    assert cropped["lat"].size < _raw_field()["lat"].size


def test_crop_bbox_requires_coords():
    bad = xr.DataArray(np.zeros((2, 2)), dims=("a", "b"))
    with pytest.raises(KeyError):
        grid.crop_bbox(bad)


def test_regrid_dataset_input():
    ds = _raw_field().to_dataset(name="field")
    out = grid.regrid_to_target(ds)
    assert isinstance(out, xr.Dataset)
    assert out["field"].shape == (3, *config.GRID_SHAPE)

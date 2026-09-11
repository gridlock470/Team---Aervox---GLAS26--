"""Tests for :mod:`nowcast.common.io`."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nowcast.common import io


@pytest.fixture
def tmp_path():
    """Isolated temp dir (the shared pytest tmp root is permission-locked here)."""
    d = Path(tempfile.mkdtemp(prefix="sih-io-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _scrambled_ds() -> xr.Dataset:
    """A dataset with raw coord names, descending lat and 0-360 lon."""
    lat = np.array([30.0, 29.0, 28.0, 27.0])  # descending
    lon = np.array([340.0, 350.0, 355.0])  # 0-360 convention
    time = np.arange(2)
    data = np.random.default_rng(0).random((2, 4, 3)).astype("float32")
    return xr.Dataset(
        {"TMP_2m": (("valid_time", "latitude", "longitude"), data)},
        coords={"valid_time": time, "latitude": lat, "longitude": lon},
    )


def test_standardize_coords_renames_and_orders():
    out = io.standardize_coords(_scrambled_ds())
    assert set(out.coords) >= {"lat", "lon", "time"}
    assert "latitude" not in out.coords and "valid_time" not in out.coords
    assert io._is_ascending(out["lat"].values)
    assert io._is_ascending(out["lon"].values)
    assert (out["lon"].values <= 180).all() and (out["lon"].values >= -180).all()
    assert out["TMP_2m"].dims == ("time", "lat", "lon")


def test_standardize_coords_renames_vertical_coord():
    ds = xr.Dataset(
        {"t": (("time", "plev", "lat", "lon"), np.zeros((1, 3, 2, 2)))},
        coords={"time": [0], "plev": [1000.0, 850.0, 500.0], "lat": [1, 2], "lon": [3, 4]},
    )
    out = io.standardize_coords(ds)
    assert "level" in out.coords
    assert out["level"].dtype == np.float64


def test_open_netcdf_roundtrip(tmp_path):
    ds = _scrambled_ds()
    path = tmp_path / "raw.nc"
    ds.to_netcdf(path)
    loaded = io.open_netcdf(path)
    assert loaded["TMP_2m"].dims == ("time", "lat", "lon")
    assert io._is_ascending(loaded["lat"].values)


def test_open_netcdf_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.open_netcdf(tmp_path / "nope.nc")


def test_write_and_open_zarr(tmp_path):
    ds = io.standardize_coords(_scrambled_ds())
    store = tmp_path / "cube.zarr"
    returned = io.write_zarr(ds, store)
    assert returned == store and store.exists()
    reopened = io.open_zarr(store)
    xr.testing.assert_allclose(ds["TMP_2m"], reopened["TMP_2m"])


def test_open_zarr_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.open_zarr(tmp_path / "absent.zarr")


def test_standardize_coords_keeps_ecmwf_temperature_variable():
    """``t`` is ECMWF's short name for temperature, not a time coordinate.

    Regression: the coordinate alias ``t -> time`` used to fire on data
    variables too, so an ERA5/IMDAA pressure-level file carrying both ``t``
    and ``valid_time`` raised "the new name 'time' conflicts".
    """
    ds = xr.Dataset(
        {
            "t": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros((1, 2, 2, 2))),
            "r": (("valid_time", "pressure_level", "latitude", "longitude"), np.zeros((1, 2, 2, 2))),
        },
        coords={
            "valid_time": [0],
            "pressure_level": [1000.0, 850.0],
            "latitude": [2.0, 1.0],
            "longitude": [3.0, 4.0],
        },
    )
    out = io.standardize_coords(ds)
    assert "t" in out.data_vars, "temperature must survive standardization"
    assert "time" in out.coords and "valid_time" not in out.coords
    assert "level" in out.coords and "pressure_level" not in out.coords
    assert set(out.data_vars) == {"t", "r"}

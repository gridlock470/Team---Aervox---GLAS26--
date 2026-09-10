"""Tests for :mod:`nowcast.ingest.merge_precip`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from nowcast import config
from nowcast.ingest.merge_precip import merge_precip


def _precip(value: float, *, nan_mask: np.ndarray | None = None) -> xr.DataArray:
    h, w = config.GRID_SHAPE
    arr = np.full((3, h, w), value, dtype="float32")
    if nan_mask is not None:
        arr[:, nan_mask] = np.nan
    return xr.DataArray(
        arr,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(3), "lat": config.GRID_LAT, "lon": config.GRID_LON},
        name="precip",
    )


def test_mera_preferred_where_finite():
    h, w = config.GRID_SHAPE
    hole = np.zeros((h, w), dtype=bool)
    hole[:5, :5] = True
    mera = _precip(4.0, nan_mask=hole)
    imerg = _precip(9.0)
    out = merge_precip(mera=mera, imerg=imerg)
    assert out.name == "precip"
    # MERA value kept where finite
    assert float(out.isel(time=0).values[10, 10]) == pytest.approx(4.0)
    # IMERG fills the hole
    assert float(out.isel(time=0).values[0, 0]) == pytest.approx(9.0)


def test_all_nan_falls_through_to_zero():
    h, w = config.GRID_SHAPE
    allnan = np.ones((h, w), dtype=bool)
    out = merge_precip(mera=_precip(1.0, nan_mask=allnan), imerg=_precip(2.0, nan_mask=allnan))
    assert float(out.max()) == 0.0
    assert np.isfinite(out.values).all()


def test_accepts_dataset_and_dataarray():
    ds = _precip(3.0).to_dataset(name="precip")
    out = merge_precip(mera=ds)
    assert float(out.mean()) == pytest.approx(3.0)


def test_priority_order_mera_over_insat():
    out = merge_precip(mera=_precip(1.0), insat_qpe=_precip(7.0))
    assert float(out.mean()) == pytest.approx(1.0)


def test_requires_at_least_one_source():
    with pytest.raises(ValueError):
        merge_precip()


def test_dataset_without_precip_raises():
    bad = xr.Dataset({"t2m": _precip(1.0).rename("t2m")})
    with pytest.raises(KeyError):
        merge_precip(mera=bad)

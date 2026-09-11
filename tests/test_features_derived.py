"""Unit tests for the individual derived-feature functions.

All tests run on small spatial subsets of the synthetic datacube so the
column-wise MetPy loops stay fast.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from nowcast.features import kinematics, moisture, satellite, thermo
from nowcast.testing import synthetic

_HAS_METPY = importlib.util.find_spec("metpy") is not None
metpy_only = pytest.mark.skipif(not _HAS_METPY, reason="metpy is required")


def _cube(n_hours: int = 6, seed: int = 3):
    return synthetic.make_datacube(n_hours=n_hours, seed=seed).isel(
        lat=slice(0, 8), lon=slice(0, 8)
    )


def test_iwv_prefers_tcwv_when_present():
    cube = _cube()
    iwv = moisture.integrated_water_vapour(cube)
    assert set(iwv.dims) == {"time", "lat", "lon"}
    assert iwv.dtype == np.float32
    assert np.isfinite(iwv.values).all()
    assert float(iwv.min()) >= 0.0
    np.testing.assert_allclose(iwv.values, cube["tcwv"].values, rtol=1e-5)


@metpy_only
def test_iwv_pressure_level_integral_fallback():
    cube = _cube().drop_vars("tcwv")
    iwv = moisture.integrated_water_vapour(cube)
    assert set(iwv.dims) == {"time", "lat", "lon"}
    assert iwv.dtype == np.float32
    assert np.isfinite(iwv.values).all()
    assert float(iwv.min()) >= 0.0


def test_iwv_raises_without_tcwv_or_levels():
    cube = _cube().drop_vars("tcwv").drop_dims("level")
    with pytest.raises(KeyError):
        moisture.integrated_water_vapour(cube)


def test_iwv_tendency_first_step_backfilled():
    cube = _cube()
    iwv = cube["tcwv"].rename("iwv")
    tend = moisture.iwv_tendency(iwv, hours=1)
    assert tend.sizes["time"] == iwv.sizes["time"]
    np.testing.assert_allclose(
        tend.isel(time=0).values, tend.isel(time=1).values
    )


def test_ctt_drop_rate_positive_when_cooling():
    cube = _cube()
    ctt = cube["ctt"].values.copy()
    ctt[1:] = ctt[0] - 5.0  # step then hold: cooling between t0 and t1
    ctt[2:] = ctt[1] - 5.0
    cube["ctt"] = (("time", "lat", "lon"), ctt)
    rate = satellite.ctt_drop_rate(cube, hours=1)
    assert set(rate.dims) == {"time", "lat", "lon"}
    assert float(rate.isel(time=2).mean()) > 0.0


@metpy_only
def test_lifted_index_shape_and_mostly_finite():
    li = thermo.lifted_index(_cube())
    assert set(li.dims) == {"time", "lat", "lon"}
    assert li.dtype == np.float32
    assert np.isfinite(li.values).mean() > 0.5


@metpy_only
def test_ensure_cape_cin_passthrough():
    cube = _cube()
    cape, cin = thermo.ensure_cape_cin(cube)
    np.testing.assert_allclose(cape.values, cube["cape"].values)
    np.testing.assert_allclose(cin.values, cube["cin"].values)


@metpy_only
def test_ensure_cape_cin_computes_when_absent():
    cube = _cube(n_hours=3).drop_vars(["cape", "cin"]).isel(
        lat=slice(0, 3), lon=slice(0, 3)
    )
    cape, cin = thermo.ensure_cape_cin(cube)
    assert set(cape.dims) == {"time", "lat", "lon"}
    finite = np.isfinite(cape.values)
    assert finite.mean() > 0.5
    assert float(np.nanmin(cape.values)) >= 0.0
    assert float(np.nanmax(cin.values)) <= 1e-6  # CIN is non-positive


def test_convergence_850_positive_for_convergent_flow():
    cube = _cube()
    lon = cube["lon"].values
    lat = cube["lat"].values
    u = np.zeros_like(cube["u"].values)
    v = np.zeros_like(cube["v"].values)
    u[:] = -(lon[None, None, None, :] - lon.mean())
    v[:] = -(lat[None, None, :, None] - lat.mean())
    cube["u"] = (("time", "level", "lat", "lon"), u)
    cube["v"] = (("time", "level", "lat", "lon"), v)
    conv = kinematics.convergence_850(cube)
    assert set(conv.dims) == {"time", "lat", "lon"}
    assert float(conv.mean()) > 0.0


def test_bulk_shear_non_negative_and_named():
    cube = _cube()
    s01 = kinematics.bulk_shear(cube, 0.0, 1000.0)
    s06 = kinematics.bulk_shear(cube, 0.0, 6000.0)
    assert s01.name == "shear_0_1km"
    assert s06.name == "shear_0_6km"
    assert set(s01.dims) == {"time", "lat", "lon"}
    assert float(s01.min()) >= 0.0
    assert float(s06.min()) >= 0.0
    assert np.isfinite(s06.values).all()

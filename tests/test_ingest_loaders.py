"""Loader tests for every ingest source, exercised on tiny synthetic files."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nowcast import config
from nowcast.ingest import dem as dem_ingest
from nowcast.ingest import imdaa, imerg, insat, mera


@pytest.fixture
def tmp_path():
    """Isolated temp dir (the shared pytest tmp root is permission-locked here)."""
    d = Path(tempfile.mkdtemp(prefix="sih-ing-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


_RES = 0.25
_LAT = np.arange(32.0, 26.99, -_RES)  # descending, wider than bbox
_LON = np.arange(76.0, 82.01, _RES)
_TIME = np.arange(4)


def _field(*shape: int, base: float = 0.0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(sum(shape))
    return (base + scale * rng.random(shape)).astype("float32")


def _write(ds: xr.Dataset, path, engine=None):
    ds.to_netcdf(path, engine=engine)
    return path


# ---------------------------------------------------------------------------
# IMDAA
# ---------------------------------------------------------------------------
def _imdaa_single() -> xr.Dataset:
    dims = ("time", "latitude", "longitude")
    shp = (len(_TIME), len(_LAT), len(_LON))
    return xr.Dataset(
        {
            "TMP_2m": (dims, _field(*shp, base=290.0, scale=10.0)),
            "UGRD_10m": (dims, _field(*shp, base=-3.0, scale=6.0)),
            "VGRD_10m": (dims, _field(*shp, base=-3.0, scale=6.0)),
            "PRMSL_msl": (dims, _field(*shp, base=100_500.0, scale=200.0)),
            "PWAT_eatm": (dims, _field(*shp, base=30.0, scale=20.0)),
            "APCP_sfc": (dims, np.full(shp, 2.0, dtype="float32")),
            "CAPE_sfc": (dims, _field(*shp, base=200.0, scale=1500.0)),
            "CIN_sfc": (dims, _field(*shp, base=-80.0, scale=60.0)),
        },
        coords={"time": _TIME, "latitude": _LAT, "longitude": _LON},
    )


def _imdaa_levels() -> xr.Dataset:
    levs = np.array([1000, 925, 850, 700, 500, 300, 250, 200], dtype="float64")
    dims = ("time", "level", "latitude", "longitude")
    shp = (len(_TIME), len(levs), len(_LAT), len(_LON))
    return xr.Dataset(
        {
            "TMP_prl": (dims, _field(*shp, base=250.0, scale=40.0)),
            "RH_prl": (dims, _field(*shp, base=10.0, scale=80.0)),
            "HGT_prl": (dims, _field(*shp, base=5000.0, scale=100.0)),
            "UGRD_prl": (dims, _field(*shp, base=-10.0, scale=30.0)),
            "VGRD_prl": (dims, _field(*shp, base=-10.0, scale=30.0)),
        },
        coords={"time": _TIME, "level": levs, "latitude": _LAT, "longitude": _LON},
    )


def test_load_imdaa_single_level(tmp_path):
    path = _write(_imdaa_single(), tmp_path / "imdaa_sfc.nc")
    ds = imdaa.load_imdaa_single_level(path)
    for name in ("t2m", "u10", "v10", "prmsl", "tcwv", "precip", "cape", "cin"):
        assert name in ds, name
        assert ds[name].dims == ("time", "lat", "lon")
    np.testing.assert_allclose(ds["lat"].values, config.GRID_LAT, atol=1e-6)
    np.testing.assert_allclose(ds["lon"].values, config.GRID_LON, atol=1e-6)
    # APCP 2 mm over a 1 h window -> 2 mm/h
    np.testing.assert_allclose(ds["precip"].values, 2.0, atol=1e-3)
    assert float(ds["t2m"].mean()) > 250.0


def test_load_imdaa_pressure_level(tmp_path):
    path = _write(_imdaa_levels(), tmp_path / "imdaa_prl.nc")
    ds = imdaa.load_imdaa_pressure_level(path)
    for name in ("t", "rh", "z", "u", "v"):
        assert ds[name].dims == ("time", "level", "lat", "lon")
    np.testing.assert_array_equal(
        ds["level"].values, np.asarray(config.PRESSURE_LEVELS_HPA, dtype="float64")
    )
    # geopotential height (~5000 m) -> geopotential (m2 s-2), ~5000 * 9.81
    assert float(ds["z"].mean()) > 40_000.0


# ---------------------------------------------------------------------------
# MERA
# ---------------------------------------------------------------------------
def test_load_mera_cumulative_precip(tmp_path):
    dims = ("time", "latitude", "longitude")
    shp = (len(_TIME), len(_LAT), len(_LON))
    # running total in metres: 0, 1mm, 2mm, 3mm  -> 1 mm/h rate after step 0
    tp = np.stack(
        [np.full(shp[1:], k * 1.0e-3, dtype="float32") for k in range(len(_TIME))]
    )
    ds = xr.Dataset({"tp": (dims, tp)}, coords={"time": _TIME, "latitude": _LAT, "longitude": _LON})
    path = _write(ds, tmp_path / "mera.nc")
    out = mera.load_mera(path)
    assert out["precip"].dims == ("time", "lat", "lon")
    assert out["precip"].attrs["units"] == "mm h-1"
    np.testing.assert_allclose(out["precip"].isel(time=slice(1, None)).values, 1.0, atol=1e-2)
    assert float(out["precip"].min()) >= 0.0


# ---------------------------------------------------------------------------
# IMERG
# ---------------------------------------------------------------------------
def test_load_imerg(tmp_path):
    dims = ("time", "lat", "lon")
    shp = (len(_TIME), len(_LAT), len(_LON))
    ds = xr.Dataset(
        {"precipitation": (dims, _field(*shp, base=0.0, scale=8.0))},
        coords={"time": _TIME, "lat": np.sort(_LAT), "lon": _LON},
    )
    path = _write(ds, tmp_path / "imerg.nc")
    out = imerg.load_imerg(path, group=None)
    assert out["precip"].dims == ("time", "lat", "lon")
    np.testing.assert_allclose(out["lat"].values, config.GRID_LAT, atol=1e-6)
    assert float(out["precip"].min()) >= 0.0


# ---------------------------------------------------------------------------
# INSAT L1C + QPE  (HDF5 / .h5)
# ---------------------------------------------------------------------------
def _insat_l1c() -> xr.Dataset:
    dims = ("time", "lat", "lon")
    shp = (2, len(_LAT), len(_LON))
    return xr.Dataset(
        {
            "IMG_TIR1_TEMP": (dims, _field(*shp, base=200.0, scale=100.0)),
            "IMG_WV_TEMP": (dims, _field(*shp, base=220.0, scale=30.0)),
        },
        coords={"time": np.arange(2), "lat": np.sort(_LAT), "lon": _LON},
    )


def test_load_insat_l1c_h5(tmp_path):
    path = tmp_path / "3DIMG_L1C.h5"
    _write(_insat_l1c(), path, engine="h5netcdf")
    ds = insat.load_insat_l1c(path)
    assert ds["ctt"].dims == ("time", "lat", "lon")
    assert ds["wv_bt"].dims == ("time", "lat", "lon")
    np.testing.assert_allclose(ds["lon"].values, config.GRID_LON, atol=1e-6)
    assert 150.0 < float(ds["ctt"].mean()) < 330.0


def test_load_insat_qpe_h5(tmp_path):
    dims = ("time", "lat", "lon")
    shp = (2, len(_LAT), len(_LON))
    ds = xr.Dataset(
        {"HEM_GPI": (dims, _field(*shp, base=0.0, scale=12.0))},
        coords={"time": np.arange(2), "lat": np.sort(_LAT), "lon": _LON},
    )
    path = tmp_path / "3DIMG_QPE.h5"
    _write(ds, path, engine="h5netcdf")
    out = insat.load_insat_qpe(path)
    assert out["precip"].dims == ("time", "lat", "lon")
    assert float(out["precip"].min()) >= 0.0


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------
def test_load_dem_netcdf(tmp_path):
    lat = np.arange(32.0, 26.99, -0.1)
    lon = np.arange(76.0, 82.01, 0.1)
    elev = (
        1000.0
        + 200.0 * np.sin(lat[:, None] / 2.0)
        + 100.0 * np.cos(lon[None, :] / 2.0)
    ).astype("float32")
    ds = xr.Dataset({"elevation": (("lat", "lon"), elev)}, coords={"lat": lat, "lon": lon})
    path = _write(ds, tmp_path / "dem.nc")
    da = dem_ingest.load_dem(path)
    assert da.name == "elevation"
    assert da.dims == ("lat", "lon")
    assert da.shape == config.GRID_SHAPE
    np.testing.assert_allclose(da["lat"].values, config.GRID_LAT, atol=1e-6)
    assert np.isfinite(da.values).all()


def test_fetch_glo30_offline_raises(tmp_path, monkeypatch):
    """Network guard: a failing download is normalised to DemDownloadError.

    No real network access -- ``pooch.retrieve`` is patched to raise.
    """
    import pooch

    def _boom(*_a, **_k):
        raise OSError("simulated offline: name resolution failed")

    monkeypatch.setattr(pooch, "retrieve", _boom)
    with pytest.raises(dem_ingest.DemDownloadError):
        dem_ingest.fetch_glo30(bbox=(76.0, 27.0, 77.0, 28.0), out_dir=tmp_path)

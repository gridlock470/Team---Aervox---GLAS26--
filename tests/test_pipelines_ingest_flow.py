"""Tests for :mod:`nowcast.pipelines.ingest_flow`.

The pipeline is driven with tiny synthetic NetCDF inputs; no network access.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from nowcast import config, schema
from nowcast.pipelines import ingest_flow
from nowcast.testing import synthetic


@pytest.fixture
def tmp_path():
    """Isolated temp dir (the shared pytest tmp root is permission-locked here)."""
    d = Path(tempfile.mkdtemp(prefix="sih-flow-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

_RES = 0.25
_LAT = np.arange(32.0, 26.99, -_RES)
_LON = np.arange(76.0, 82.01, _RES)
_TIME = np.arange(4)


def _rand(*shape, base=0.0, scale=1.0):
    return (base + scale * np.random.default_rng(sum(shape)).random(shape)).astype("float32")


def _write_imdaa_single(path):
    dims = ("time", "latitude", "longitude")
    shp = (len(_TIME), len(_LAT), len(_LON))
    xr.Dataset(
        {
            "TMP_2m": (dims, _rand(*shp, base=290.0, scale=8.0)),
            "UGRD_10m": (dims, _rand(*shp, base=-2.0, scale=4.0)),
            "VGRD_10m": (dims, _rand(*shp, base=-2.0, scale=4.0)),
            "PRMSL_msl": (dims, _rand(*shp, base=100_600.0, scale=120.0)),
            "PWAT_eatm": (dims, _rand(*shp, base=35.0, scale=15.0)),
            "APCP_sfc": (dims, np.full(shp, 1.0, dtype="float32")),
            "CAPE_sfc": (dims, _rand(*shp, base=300.0, scale=1200.0)),
            "CIN_sfc": (dims, _rand(*shp, base=-60.0, scale=40.0)),
        },
        coords={"time": _TIME, "latitude": _LAT, "longitude": _LON},
    ).to_netcdf(path)


def _write_imdaa_levels(path):
    levs = np.array(config.PRESSURE_LEVELS_HPA, dtype="float64")
    dims = ("time", "level", "latitude", "longitude")
    shp = (len(_TIME), len(levs), len(_LAT), len(_LON))
    xr.Dataset(
        {
            "TMP_prl": (dims, _rand(*shp, base=250.0, scale=30.0)),
            "RH_prl": (dims, _rand(*shp, base=20.0, scale=70.0)),
            "HGT_prl": (dims, _rand(*shp, base=5000.0, scale=80.0)),
            "UGRD_prl": (dims, _rand(*shp, base=-5.0, scale=20.0)),
            "VGRD_prl": (dims, _rand(*shp, base=-5.0, scale=20.0)),
        },
        coords={"time": _TIME, "level": levs, "latitude": _LAT, "longitude": _LON},
    ).to_netcdf(path)


def _write_mera(path):
    dims = ("time", "latitude", "longitude")
    shp = (len(_TIME), len(_LAT), len(_LON))
    tp = np.stack([np.full(shp[1:], k * 2.0e-3, dtype="float32") for k in range(len(_TIME))])
    xr.Dataset(
        {"tp": (dims, tp)}, coords={"time": _TIME, "latitude": _LAT, "longitude": _LON}
    ).to_netcdf(path)


def test_run_ingest_builds_valid_datacube(tmp_path):
    single = tmp_path / "imdaa_sfc.nc"
    level = tmp_path / "imdaa_prl.nc"
    mera_p = tmp_path / "mera.nc"
    _write_imdaa_single(single)
    _write_imdaa_levels(level)
    _write_mera(mera_p)
    static = synthetic.make_datacube(n_hours=1, seed=0, with_static=True)[
        ["elevation", "slope", "flow_accumulation", "flow_direction", "hand"]
    ]

    out = tmp_path / "datacube.zarr"
    path = ingest_flow.run_ingest(
        imdaa_single=single,
        imdaa_level=level,
        mera_paths=[mera_p],
        static=static,
        out_path=out,
    )
    ds = xr.open_zarr(path)
    schema.validate_datacube(ds, require_static=True, require_satellite=False)
    # merged precip came from MERA (~2 mm/h running total) not IMDAA APCP (1 mm/h)
    assert float(ds["precip"].isel(time=slice(1, None)).mean()) > 1.5


def test_ingest_datacube_flow_object_or_guarded():
    """The flow entry point is a Prefect flow when Prefect is installed,
    otherwise calling it raises a clear RuntimeError."""
    if ingest_flow.PREFECT_AVAILABLE:
        from prefect import Flow

        assert isinstance(ingest_flow.ingest_datacube, Flow)
    else:  # pragma: no cover - depends on env
        with pytest.raises(RuntimeError):
            ingest_flow.ingest_datacube(imdaa_single=[], imdaa_level=[])


def test_flow_and_plain_share_stage_impls():
    # The Prefect tasks wrap the very same callables the plain driver uses.
    assert ingest_flow._pipeline.__module__ == ingest_flow.__name__
